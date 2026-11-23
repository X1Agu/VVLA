"""Lifecycle revision of the unqualified true-batch candidate.

The v1 math/geometry/module-view helpers remain exact imports. This version
owns partial cold initialization before any validation/import/native load can
fail, loads pinned native ops inside the managed constructor, and prevents row
RNG/H2D work after close or an incomplete owner. The original managed runtime
must still run dependency_origins; no public B1 metadata is used for batches.
"""
from __future__ import annotations

import threading

import torch

from . import true_batch_v1 as v1
from .true_batch_v1 import (ALLOWED_BATCHES, RowNoise, clone_parameter_shared_view,
                           prepare_real_rows, split_real_batches, unapply_owned_rows)


class TrueBatchEnginePool(v1.TrueBatchEnginePool):
    def __init__(self, flow, *args, split_stage=False, **kwargs):
        # RuntimeBundle._construct_engine retains __new__ before __init__.
        # Every field required for retryable close must exist immediately.
        self.flow, self.device = flow, None
        self._engines, self._views, self._view_receipts = {}, {}, {}
        self._construction_ms = {}
        self._source_guard = None
        self._lock = threading.Lock()
        self.closed = self._closing = False
        self._failed, self._init_ready = False, False
        self.graph_diagnostics = None
        try:
            super().__init__(flow, *args, **kwargs)
            from vvla_runtime.execution import native
            native.load_native()
            # Managed runtime's actual loaded dependency/ABI origin gate follows.
            self._init_ready = True
            self.recipe['candidate'] = 'p2-true-batch-v2'
            if split_stage:
                self.allowed_batches = (1, 2, 4)
                self._factory = v1._engine_type(split_stage=True)
                self.recipe.pop('default_recipe_changed', None)
                self.recipe.update(candidate='split-stage', allowed_batches=list(self.allowed_batches),
                                   execution_path='eager-prefix/cuda-graph-action')
            self.recipe['lifecycle'] = 'partial cold ownership; native load; retryable close; live-only row entry'
        except BaseException:
            self._failed = True
            raise

    def __call__(self, *args, **kwargs):
        if not self._init_ready or self._closing or self._failed or self.closed:
            raise RuntimeError('True-batch owner is incomplete, failed, closing or closed')
        return super().__call__(*args, **kwargs)

    def close(self):
        if self.closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close during a true-batch call')
        try:
            self._closing = self._failed = True
            # None means the device already selected by managed RuntimeBundle;
            # it supports an error before flow/device validation completed.
            torch.cuda.synchronize(self.device)
            for engine in self._engines.values():
                graph = getattr(engine, '_graph', None)
                if graph is not None:
                    graph.reset()
            # On *any* drain/reset error all strong references above survive.
            self._engines.clear()
            self._views.clear()
            self.flow = self._source_guard = None
            self._init_ready = False
            self.closed = True
        finally:
            self._closing = False
            self._lock.release()


class RawTrueBatchPolicy(v1.RawTrueBatchPolicy):
    def __init__(self, official_policy, engine_pool, *, runtime_bundle=None):
        self._closed = self._closing = False
        self._failed = False
        self._runtime_bundle = runtime_bundle
        super().__init__(official_policy, engine_pool)
        # Retain this lock through the inherited method's own lock scope.
        self._lock = threading.RLock()

    def _check_live(self):
        if (self._closed or self._closing or self._failed or self.engine is None or self.engine.closed
                or self.engine._failed or not self.engine._init_ready):
            raise RuntimeError('Row entry requires a complete live true-batch runtime')

    def reset(self, identity, seed):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot reset during another row operation')
        try:
            self._check_live()  # Before constructing a CUDA Generator.
            return super().reset(identity, seed)
        finally:
            self._lock.release()

    def infer(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent row operations are unsupported')
        try:
            self._check_live()  # Before FeatureTransform/H2D/noise draw.
            return super().infer(*args, **kwargs)
        finally:
            self._lock.release()

    def infer_seeded(self, observations, identities, seeds):
        # Reset and consume each seed in one transaction. Two callers may
        # otherwise reuse the same row identities between reset() and infer().
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent row operations are unsupported')
        try:
            self._check_live()
            for identity, seed in zip(identities, seeds, strict=True):
                super().reset(identity, seed)
            return super().infer(observations, identities)
        finally:
            self._lock.release()

    def reset_all(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot reset during another row operation')
        try:
            self._check_live()
            self.noise.generators.clear()
            self.noise.calls.clear()
            self.calls.clear()
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close during another row operation')
        try:
            self._closing = True
            self._failed = True  # Also covers RuntimeBundle's *outer* failed drain.
            if self._runtime_bundle is not None:
                self._runtime_bundle.close()
            else:
                # An external managed owner must also close its RuntimeBundle
                # to release the process lease. Passing it here is preferred.
                self.engine.close()
            self.noise.generators.clear()
            self.noise.calls.clear()
            self.calls.clear()
            self.last_noise = None
            self.policy = self.engine = self._runtime_bundle = None
            self._closed = True
        finally:
            self._closing = False
            self._lock.release()
