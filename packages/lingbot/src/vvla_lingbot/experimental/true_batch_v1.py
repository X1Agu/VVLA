"""Unqualified true B1/2/4/8 candidate; explicit batch-domain validation without runtime source rewriting.

Each B owns a complete BatchedEngine and one graph/key. Module structures,
buffers, vision caches and MoE workspaces are independent; checkpoint Parameters
are shared by identity and guarded against mutation. This is not B1 GraphEntry
pooling. Only an unused official flow may be supplied to the constructor.

The upstream ragged attention, row positions, paged KV and sparse mathematics
are retained. Broadening its B assertion alone is not qualification: the GPU
harness must independently prove row isolation, routes, actual graph-owned
noise, owned outputs, memory and task quality for every advertised B.
"""
from __future__ import annotations

import copy
from pathlib import Path
import threading
import time

import torch

ALLOWED_BATCHES = (1, 2, 4, 8)
VISION_FIELDS = ('pos_embeds', 'position_embeddings', 'cu_seqlens',
                 'visual_split_sizes', 'visual_max_seqlen')


def _named_parameters(module):
    return dict(module.named_parameters(remove_duplicate=False))


def _storage(tensor):
    if tensor.numel() == 0:
        return None
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def clone_parameter_shared_view(flow):
    """Clone mutable structure before tuning; never clone a 6B weight tensor.

    deepcopy's memo preserves Parameter aliases, while every ordinary mutable
    module/config/buffer is copied. A returned view is not a second runtime or
    process lease. The pool owns all views until a successful CUDA drain.
    """
    parameters = _named_parameters(flow)
    if not parameters:
        raise ValueError('A fresh official parameter-bearing flow is required')
    for module in flow.modules():
        if module.training:
            raise RuntimeError('This candidate requires the original evaluation-only flow')
        if getattr(module, '_robby_moe_workspace', None) is not None:
            raise RuntimeError('Clone before any official sparse workspace is created')
        if type(module).__name__ in ('CachedAdaNorm', 'CachedRouterLinear',
                                     'FusedLlamaMLP', 'QuantizedLinear'):
            raise RuntimeError('Clone the official flow before installing any Engine')
    qwe = getattr(flow, 'qwenvl_with_expert', None)
    if qwe is not None and any(getattr(qwe, name, None) is not None for name in VISION_FIELDS):
        raise RuntimeError('Clone before the first official vision metadata preparation')
    view = copy.deepcopy(flow, {id(p): p for p in parameters.values()})
    copied = _named_parameters(view)
    if parameters.keys() != copied.keys() or any(copied[n] is not p for n, p in parameters.items()):
        raise RuntimeError('Checkpoint Parameter aliases were not exactly preserved')
    original_modules = dict(flow.named_modules(remove_duplicate=False))
    copied_modules = dict(view.named_modules(remove_duplicate=False))
    if original_modules.keys() != copied_modules.keys() or any(
            copied_modules[n] is m for n, m in original_modules.items()):
        raise RuntimeError('Mutable module structure is shared across batch owners')
    buffers = dict(flow.named_buffers(remove_duplicate=False))
    copied_buffers = dict(view.named_buffers(remove_duplicate=False))
    if buffers.keys() != copied_buffers.keys() or any(
            buffers[n] is copied_buffers[n] or
            (_storage(buffers[n]) is not None and _storage(buffers[n]) == _storage(copied_buffers[n]))
            for n in buffers):
        raise RuntimeError('A mutable registered buffer is shared with the source flow')
    if hasattr(flow, 'config') and view.config is flow.config:
        raise RuntimeError('A mutable model configuration is shared')
    return view, dict(parameter_names=len(parameters),
                      unique_parameter_objects=len({id(p) for p in parameters.values()}),
                      all_parameter_objects_and_data_ptrs_shared=True,
                      mutable_module_objects_disjoint=True,
                      registered_buffer_storages_disjoint=True,
                      checkpoint_tensor_copies=0)


def _engine_type(split_stage=False):
    if split_stage:
        from vvla_lingbot.split_stage import SplitStageBatchedEngine as base
    else:
        from vvla_lingbot.lite_tuning.engine import TunedBatchedEngine as base

    class FixedTrueBatchEngine(base):

        def _validate_batch_inputs(self, images, im, tokens, lm, state, grid):
            b = self._p2_fixed_batch
            values = (images, im, tokens, lm, state, grid)
            if any(not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] != b
                   or x.device != state.device for x in values):
                raise ValueError('Every true row tensor must use the exact fixed B and device')
            if (state.ndim != 2 or not state.is_cuda or state.dtype != torch.bfloat16
                    or images.dtype != torch.bfloat16 or images.ndim < 3
                    or im.shape != (b, 3) or im.dtype != torch.bool
                    or images.shape[1] != 3 or not im.all().item()
                    or tokens.ndim != 2 or tokens.dtype != torch.int64
                    or lm.shape != tokens.shape or lm.dtype != torch.bool
                    or grid.shape != (b, 3, 3) or grid.dtype != torch.int64):
                raise ValueError('Expected true BF16 rows, three cameras and original per-row metadata')
            if state.shape[1] != self.flow.state_proj.in_features:
                raise ValueError('State width differs from the official checkpoint')

    return FixedTrueBatchEngine


def _tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _tensors(item, prefix + '/' + str(name))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from _tensors(item, prefix + '/' + str(i))


class TrueBatchEnginePool:
    """Fresh runtime engine factory, full guard, combined BF16, exact true B.

    RuntimeBundle may own this object in place of TunedEngine. Do not create a
    normal Engine first. Each B retains its *whole* engine, including ps/max_p/
    kps/prefix_cu and vision caches. No cross-B GraphEntry restoration is used.
    All callers must supply their actual per-row noise explicitly.
    """
    allowed_batches = ALLOWED_BATCHES

    def __init__(self, flow, mode='framework-bf16', calibration=None, audit=None,
                 variant='combined', graph_diagnostics_path=None, denoise_precision='bf16', posttopk_sum_order=None):
        from vvla_lingbot.lite_tuning.state_guard import FrozenStateGuard
        if mode != 'framework-bf16' or variant != 'combined' or graph_diagnostics_path is not None:
            raise ValueError('This candidate is combined BF16 only, with no diagnostic timing path')
        self.denoise_precision, self.posttopk_sum_order = denoise_precision, posttopk_sum_order
        self.flow, self.mode = flow, mode
        self.audit = Path(audit) if audit else None
        self.device = next(flow.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('A fresh CUDA official flow is required')
        self._source_guard = FrozenStateGuard(flow)
        self._factory = _engine_type()
        self._engines, self._views, self._view_receipts = {}, {}, {}
        self._lock = threading.Lock()
        self.closed = False
        self._failed = False
        self._construction_ms = {}
        self.last_call = {}
        self.recipe = dict(candidate='p2-true-batch-v1', qualified=False,
                           allowed_batches=list(ALLOWED_BATCHES), default_recipe_changed=False,
                           variant='combined', mode=mode, graph_capacity_per_batch=1,
                           tail_policy='Use the exact real B; no dummy rows',
                           shared='read-only original checkpoint Parameter objects',
                           isolated='module/config/buffers/vision/workspaces/KV/graph per B')

        if denoise_precision != 'bf16':
            self.recipe.update(denoise_precision=denoise_precision, posttopk_sum_order=posttopk_sum_order)

    def _get(self, b):
        if b not in self._engines:
            # Prepared views run their own complete FrozenStateGuard. The
            # untouched source is checked again before deriving any new view.
            self._source_guard.check()
            started = time.perf_counter()
            view, receipt = clone_parameter_shared_view(self.flow)
            self._views[b], self._view_receipts[b] = view, receipt
            # Retain the partial engine before initialization can enqueue CUDA.
            engine = self._factory.__new__(self._factory)
            self._engines[b] = engine
            engine._p2_fixed_batch = b
            path = self.audit / ('batch-' + str(b)) if self.audit else None
            options = {'denoise_precision': self.denoise_precision} if self.denoise_precision != 'bf16' else {}
            self._factory.__init__(engine, view, 'framework-bf16', audit=path,
                                   variant='combined', graph_diagnostics_path=None, **options)
            if self.posttopk_sum_order is not None:
                from vvla_lingbot.routing import install_post_topk
                install_post_topk(engine, sum_order=self.posttopk_sum_order)
            engine.recipe['p2_true_batch'] = dict(self.recipe, fixed_batch=b, ownership=receipt)
            engine._write_recipe()
            self._construction_ms[b] = (time.perf_counter() - started) * 1000
        return self._engines[b]

    @torch.no_grad()
    def __call__(self, images, img_masks, tokens, lang_masks, state,
                 noise=None, image_grid_thw=None):
        if self.closed:
            raise RuntimeError('True-batch owner is closed')
        if self._failed:
            raise RuntimeError('A failed partial owner must close before another inference')
        b = state.shape[0]
        if b not in self.allowed_batches or noise is None:
            raise ValueError(f'Expected a real width in {self.allowed_batches} and explicit per-row noise')
        expected = (b, self.flow.config.n_action_steps, self.flow.config.max_action_dim)
        if (tuple(noise.shape) != expected or noise.dtype != torch.bfloat16
                or noise.device != state.device):
            raise ValueError('Explicit actual BF16 row noise has the wrong shape/device')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent calls require a scheduler and separate execution ownership')
        try:
            new_view = b not in self._engines
            engine = self._get(b)
            output = engine(images, img_masks, tokens, lang_masks, state,
                            noise=noise, image_grid_thw=image_grid_thw)
            self.last_call = dict(engine.last_call, actual_tensor_batch=b, real_batch=b,
                                  graph_capacity_per_batch=1, new_batch_owner=new_view,
                                  batch_owner_construction_ms=self._construction_ms[b] if new_view else 0.0)
            return output
        except BaseException:
            # A partially initialized engine remains owned for retryable close.
            self._failed = True
            raise
        finally:
            self._lock.release()

    def audit_ownership(self):
        """Mechanism-only audit; may synchronize, so never run inside timing."""
        self._source_guard.check()
        owners = {}
        for b, engine in self._engines.items():
            if getattr(engine, '_signature', None) is None:
                raise RuntimeError('Audit after each participating B has actually prepared')
            engine._state_signature()
            if tuple((id(t), t.data_ptr(), t._version) for t in engine._constant_tensors) != engine._constant_token:
                raise RuntimeError('Prepared per-B constants changed')
            if len(engine.ps) != b or len(engine.kps) != b or engine.batch != b:
                raise RuntimeError('Ragged row metadata does not match actual B')
            table = engine.block_table.cpu()
            if (table.shape[0] != b or table.unique().numel() != table.numel()
                    or engine.kcache.shape[1] != table.numel()
                    or engine.vcache.shape != engine.kcache.shape):
                raise RuntimeError('Paged KV blocks alias or have the wrong real batch extent')
            cu = engine.prefix_cu.cpu().tolist()
            if cu != [0, *(__import__('itertools').accumulate(engine.ps))]:
                raise RuntimeError('Prefix cu_seqlens differs from actual per-row token lengths')
            values = dict(static=engine.static, noise=engine.noise, output=engine.output,
                          kcache=engine.kcache, vcache=engine.vcache,
                          constants=engine._constant_tensors,
                          vision={n: getattr(engine.qwe, n) for n in VISION_FIELDS},
                          moe=[layer.mlp.experts._robby_moe_workspace for layer in engine.action.layers])
            storages = {}
            for name, tensor in _tensors(values):
                key = _storage(tensor)
                if key is not None:
                    storages.setdefault(key, []).append(name)
            owners[b] = storages
        for b, storages in owners.items():
            for other, other_storages in owners.items():
                if b < other and set(storages) & set(other_storages):
                    raise RuntimeError('Mutable execution storage shared across B owners')
        return dict(actual_prepared_batches=sorted(owners),
                    mutable_execution_storages_disjoint=True,
                    source_parameter_guard_passed=True,
                    views=self._view_receipts,
                    field_scope='whole engine; ragged prefix/KV/vision/MoE/static/noise/output')

    def close(self):
        if self.closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close during a true-batch call')
        try:
            # No owner reference is dropped before drain and all graph resets succeed.
            torch.cuda.synchronize(self.device)
            for engine in self._engines.values():
                graph = getattr(engine, '_graph', None)
                if graph is not None:
                    graph.reset()
            self._engines.clear()
            self._views.clear()
            self.flow = self._source_guard = None
            self.closed = True
        finally:
            self._lock.release()


class RowNoise:
    """Persistent condition generators; draw the same B1 shape per real row."""
    def __init__(self, device):
        self.device, self.generators, self.calls = device, {}, {}

    def reset(self, identity, seed):
        if type(identity) is not str or not identity or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Each real row needs a nonempty identity and uint32 seed')
        self.generators[identity] = torch.Generator(device=self.device).manual_seed(seed)
        self.calls[identity] = 0

    def draw(self, identities, *, h=50, width=55):
        if (len(identities) not in ALLOWED_BATCHES or len(set(identities)) != len(identities)
                or any(identity not in self.generators for identity in identities)):
            raise ValueError('An exact true batch of initialized distinct real identities is required')
        noise = torch.cat([torch.randn((1, h, width), device=self.device,
                                      dtype=torch.bfloat16, generator=self.generators[identity])
                           for identity in identities], dim=0)
        for identity in identities:
            self.calls[identity] += 1
        return noise


def prepare_real_rows(policy, observations, *, device):
    """Use official FeatureTransform per row and official ragged collation."""
    if len(observations) not in ALLOWED_BATCHES:
        raise ValueError('Split tails into exact true1/2/4/8; never pad a dummy row')
    applied = [policy._prepare_model_input(observation) for observation in observations]
    if any(row.keys() != applied[0].keys() for row in applied):
        raise ValueError('Official prepared row fields differ')
    batch = {}
    for key in applied[0]:
        values = [row[key] for row in applied]
        batch[key] = policy._pad_and_stack_tensors(values) if isinstance(values[0], torch.Tensor) else values
    arguments = [batch[name].to(device=device, dtype=dtype) for name, dtype in (
        ('images', torch.bfloat16), ('img_masks', torch.bool), ('lang_tokens', torch.int64),
        ('lang_masks', torch.bool), ('state', torch.bfloat16), ('image_grid_thw', torch.int64))]
    if any(value.shape[0] != len(observations) for value in arguments):
        raise RuntimeError('Official collation changed the real tensor batch')
    return applied, arguments


def unapply_owned_rows(policy, applied, normalized_actions):
    """Official physical-action transform; own CPU FP32[B,50,14] storage."""
    import numpy as np
    actions_cpu = normalized_actions.to(dtype=torch.float32, device='cpu')
    result = policy._unapply_batched_actions(applied, actions_cpu)
    actions = np.array(result['action'], dtype=np.float32, copy=True, order='C')
    if actions.shape != (len(applied), 50, 14) or not np.isfinite(actions).all():
        raise RuntimeError('The original physical action contract failed')
    return dict(result, action=actions)


def split_real_batches(observations, identities):
    """Preserve every real row once; split e.g.15 into8+4+2+1."""
    if len(observations) != len(identities) or not observations:
        raise ValueError('Equal nonempty real observations/identities are required')
    if (any(type(identity) is not str or not identity for identity in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError('No dummy or duplicate row identities')
    offset = 0
    while offset < len(observations):
        b = next(size for size in reversed(ALLOWED_BATCHES) if size <= len(observations) - offset)
        yield observations[offset:offset+b], identities[offset:offset+b]
        offset += b


class RawTrueBatchPolicy:
    """Actual raw-observation entry, without the historical B8/dummy server.

    The supplied official policy must already have performed its pinned reset
    and FeatureTransform construction. The managed runtime retains the engine
    pool/process lease. This wrapper does not acquire a second runtime lease.
    """
    def __init__(self, official_policy, engine_pool):
        self.policy, self.engine = official_policy, engine_pool
        self.noise = RowNoise(engine_pool.device)
        self.calls = {}
        self._lock = threading.Lock()
        self.last_call, self.last_noise = {}, None

    def reset(self, identity, seed):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot reset a row during inference')
        try:
            self.noise.reset(identity, seed)
            self.calls[identity] = 0
        finally:
            self._lock.release()

    @torch.no_grad()
    def infer(self, observations, identities, *, explicit_noise=None,
              expected_calls=None, record_noise=False):
        if (len(observations) not in ALLOWED_BATCHES or len(observations) != len(identities)
                or len(set(identities)) != len(identities)
                or any(identity not in self.calls for identity in identities)):
            raise ValueError('Every exact real row needs an initialized distinct identity')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent requests require a separately qualified scheduler')
        try:
            if expected_calls is not None and (
                    len(expected_calls) != len(identities) or
                    any(self.calls[identity] != count for identity, count in zip(identities, expected_calls))):
                raise ValueError('Per-condition request order differs from the caller protocol')
            started = time.perf_counter()
            applied, values = prepare_real_rows(self.policy, observations, device=self.engine.device)
            actual_noise = explicit_noise
            if actual_noise is None:
                actual_noise = self.noise.draw(identities,
                    h=self.engine.flow.config.n_action_steps,
                    width=self.engine.flow.config.max_action_dim)
            image, im, tokens, lm, state, grid = values
            normalized = self.engine(image, im, tokens, lm, state,
                                     noise=actual_noise, image_grid_thw=grid)
            result = unapply_owned_rows(self.policy, applied, normalized)
            elapsed = (time.perf_counter() - started) * 1000
            for identity in identities:
                self.calls[identity] += 1
            self.last_call = dict(self.engine.last_call, raw_to_owned_cpu_ms=elapsed,
                                  row_identities=list(identities),
                                  row_calls=[self.calls[identity] for identity in identities],
                                  explicit_noise=explicit_noise is not None)
            # Evidence copy is separately requested and lies after the declared
            # raw->owned-action timing span. Formal timing must leave it false.
            self.last_noise = actual_noise.detach().cpu().clone() if record_noise else None
            return result
        finally:
            self._lock.release()
