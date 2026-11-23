"""Actual RuntimeBundle cold-fault/lease/close tests with CUDA calls mocked.

No checkpoint, CUDA context, native operator or B>1 execution is qualified.
"""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from vvla_lingbot.experimental import true_batch_v1 as v1, true_batch_v2 as v2


class FakeFlow:
    def parameters(self):
        yield types.SimpleNamespace(device=torch.device('cuda:0'))


@pytest.fixture
def harness(monkeypatch):
    state = types.SimpleNamespace(guard_error=False, factory_error=False, native_error=False,
                                  drain_error=False, native_calls=0, drains=[])

    class Guard:
        def __init__(self, flow):
            if state.guard_error:
                raise RuntimeError('injected source guard constructor fault')
            self.flow = flow

        def check(self):
            return None

    def factory():
        if state.factory_error:
            raise RuntimeError('injected cold factory import fault')
        return type('UnusedEngine', (), {})

    def load_native():
        state.native_calls += 1
        if state.native_error:
            raise RuntimeError('injected pinned native load fault')

    def drain(device=None):
        state.drains.append(device)
        if state.drain_error:
            raise RuntimeError('injected CUDA drain fault')

    guard_module = types.ModuleType('vvla_lingbot.lite_tuning.state_guard')
    guard_module.FrozenStateGuard = Guard
    native = types.ModuleType('vvla_runtime.execution.native')
    native.load_native = load_native
    import vvla_runtime.execution
    monkeypatch.setattr(vvla_runtime.execution, 'native', native)
    monkeypatch.setitem(sys.modules, guard_module.__name__, guard_module)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(v1, '_engine_type', factory)
    monkeypatch.setattr(torch.cuda, 'synchronize', drain)
    from vvla_lingbot import runtime
    assert not runtime._LEASE.locked()

    def make_bundle():
        bundle = runtime.RuntimeBundle()
        bundle.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=drain, device=lambda device: nullcontext()))
        bundle.cuda_started = True
        bundle.device = 'cuda:0'
        return bundle

    return state, runtime, make_bundle


@pytest.mark.parametrize('fault', ['mode', 'guard', 'factory', 'native'])
def test_actual_managed_cold_failure_retains_then_retires_and_releases_lease(harness, fault):
    state, runtime, make_bundle = harness
    if fault != 'mode':
        setattr(state, fault + '_error', True)
    bundle, flow = make_bundle(), FakeFlow()
    with pytest.raises((RuntimeError, ValueError)):
        runtime._construct_engine(bundle, v2.TrueBatchEnginePool, flow,
                                  mode='official-bf16' if fault == 'mode' else 'framework-bf16')
    pool = bundle.engine
    assert pool.flow is flow and pool._failed and not pool.closed and not pool._init_ready
    assert bundle._lease_held and not bundle.closed
    bundle.close()
    assert pool.closed and pool.flow is None and not pool._views and not pool._engines
    assert bundle.closed and bundle.engine is None and not bundle._lease_held
    assert not runtime._LEASE.locked() and state.drains


def test_native_load_precedes_ready_and_partial_graph_reset_failure_is_retryable(harness):
    state, runtime, make_bundle = harness
    bundle = make_bundle()
    runtime._construct_engine(bundle, v2.TrueBatchEnginePool, FakeFlow())
    pool = bundle.engine
    assert pool._init_ready and state.native_calls == 1
    fail_reset = [True]

    def reset():
        if fail_reset[0]:
            raise RuntimeError('injected graph reset fault')

    retained_view = object()
    retained_engine = types.SimpleNamespace(_graph=types.SimpleNamespace(reset=reset))
    pool._views[2], pool._engines[2] = retained_view, retained_engine
    with pytest.raises(RuntimeError, match='graph reset fault'):
        bundle.close()
    assert not bundle.closed and bundle._lease_held and not pool.closed
    assert pool._views[2] is retained_view and pool._engines[2] is retained_engine and pool.flow is not None
    with pytest.raises(RuntimeError, match='failed'):
        pool()
    fail_reset[0] = False
    bundle.close()
    assert pool.closed and not pool._views and not pool._engines and not runtime._LEASE.locked()


def test_row_wrapper_failed_outer_drain_blocks_before_generator_or_h2d(harness, monkeypatch):
    state, runtime, make_bundle = harness
    bundle = make_bundle()
    runtime._construct_engine(bundle, v2.TrueBatchEnginePool, FakeFlow())
    pool = bundle.engine
    policy = object()
    raw = v2.RawTrueBatchPolicy(policy, pool, runtime_bundle=bundle)
    generator_reference = object()
    raw.noise.generators['kept'] = generator_reference
    raw.noise.calls['kept'] = raw.calls['kept'] = 0
    raw.last_noise = torch.ones(1)
    touched = []
    monkeypatch.setattr(v1.RowNoise, 'reset', lambda *args: touched.append('generator'))
    monkeypatch.setattr(v1, 'prepare_real_rows', lambda *args, **kwargs: touched.append('h2d'))
    state.drain_error = True
    with pytest.raises(RuntimeError, match='drain fault'):
        raw.close()
    # Outer RuntimeBundle drain failed before Pool.close could set pool._failed.
    assert not pool._failed and raw._failed and not raw._closed
    assert raw.policy is policy and raw.noise.generators['kept'] is generator_reference
    assert raw.last_noise is not None and bundle._lease_held
    with pytest.raises(RuntimeError, match='live'):
        raw.reset('a', 7)
    with pytest.raises(RuntimeError, match='live'):
        raw.infer([{}], ['a'])
    assert touched == []
    state.drain_error = False
    raw.close()
    assert raw._closed and not raw.noise.generators and raw.last_noise is None
    assert raw.policy is raw.engine is raw._runtime_bundle is None and not runtime._LEASE.locked()
    raw.close()
    with pytest.raises(RuntimeError, match='live'):
        raw.reset('a', 7)
    with pytest.raises(RuntimeError, match='live'):
        raw.infer([{}], ['a'])
    assert touched == []
