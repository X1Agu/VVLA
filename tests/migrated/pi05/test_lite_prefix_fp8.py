"""Remote CPU owner contracts using explicit meta/fake packing, not evidence.

Native arithmetic/CUDA capture contracts live in test_lite_prefix_fp8_native.py.
This file never presents its fictional evidence as a usable production report.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import prefix_fp8 as owner
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache, tensor_identity
from vvla_repro_pi05.models.pi05.precision_pi05 import FusedPrefixMLP


def evidence():
    return dict(backend_report='/fictional/never-runtime.json', backend_report_sha256='a' * 64,
                bf16_control={key: 'b' * 64 for key in
                    ('recipe_sha256', 'config_sha256', 'identity_sha256', 'content_sha256')})


class MetaPrefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        prefix = self.paligemma_with_expert.paligemma = nn.Module()
        prefix.model = nn.Module()
        prefix.model.language_model = nn.Module()
        layers = prefix.model.language_model.layers = nn.ModuleList()
        for _ in range(18):
            original = nn.Module()
            original.gate_proj = nn.Linear(2048, 16384, bias=False, device='meta', dtype=torch.bfloat16)
            original.up_proj = nn.Linear(2048, 16384, bias=False, device='meta', dtype=torch.bfloat16)
            original.down_proj = nn.Linear(16384, 2048, bias=False, device='meta', dtype=torch.bfloat16)
            original.act_fn = nn.GELU(approximate='tanh')
            layer = nn.Module()
            layer.mlp = FusedPrefixMLP(original)
            layers.append(layer)
        self.other = nn.Linear(2, 2, device='meta', dtype=torch.float32)
        self._graph_runners = weakref.WeakSet()
        self._inference_cache = Pi05InferenceCache()
        self.eval()

    def optimization_flags(self):
        return (True, True, True, True, True)

    def invalidate_inference_caches(self, **kwargs):
        self._inference_cache.clear()


@pytest.fixture
def rig(monkeypatch):
    def geometry(module, device, shape, *, installing=False):
        assert type(module) is nn.Linear and device.type == 'meta'
        assert module.weight.dtype == torch.bfloat16 and tuple(module.weight.shape) == shape
        assert module.bias is None and not module.training
        if installing:
            assert 'forward' not in module.__dict__
    def pack(weight, keepalive):
        return torch.empty_like(weight, dtype=torch.float8_e4m3fn), torch.empty((weight.shape[0], 1), device='meta')
    monkeypatch.setattr(owner, '_validate_module', geometry)
    monkeypatch.setattr(owner, '_pack_weight', pack)
    monkeypatch.setattr(owner, '_tensor_record', lambda t: dict(shape=list(t.shape), stride=list(t.stride()),
        dtype=str(t.dtype).removeprefix('torch.'), content_sha256='c' * 64))
    backend = dict(quantizer_files_sha256={name: 'd' * 64 for name in owner.native.SOURCE_FILES})
    raw = b'fictional CPU test evidence; not a backend report'
    monkeypatch.setattr(owner, '_load_evidence', lambda config, device: (raw, deepcopy(backend)))
    drains = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: drains.append('drained'))
    return SimpleNamespace(model=MetaPrefix(), backend=backend, raw=raw, drains=drains)


def installed(rig):
    handle = owner.install_prefix_fp8(rig.model, evidence())
    runner = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    handle.bind_runner(runner)
    return handle, runner


def prepare(handle, runner, *, changed=False):
    model = runner.model
    signature = tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers()))
    if changed:
        signature += ('explicit change',)
    generation = model._inference_cache.refresh(signature)
    with runner._call_lock:
        handle.prepare(generation)
        assert handle.compilation_members(generation) == handle.sites


def test_all51_preserve_original_parameters_state_dict_fused_boundary_and_complete_manifest(rig):
    model = rig.model
    original = tuple((name, id(t)) for name, t in model.named_parameters())
    keys = tuple(model.state_dict())
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        manifest = handle.validate_precision_manifest()
        assert len(manifest['sites']) == 51 and manifest['sites'][-1]['name'] == 'prefix.16.down_proj'
        assert manifest['lifecycle']['live_checks'] == owner.LIVE_CHECKS
        assert manifest['precision_boundaries'] == owner.PRECISION_BOUNDARIES
        assert handle.raw_report == rig.raw and handle._pending == []
        assert tuple((name, id(t)) for name, t in model.named_parameters()) == original
        assert tuple(model.state_dict()) == keys
        assert all(type(layer.mlp) is FusedPrefixMLP and not layer.mlp.concatenate
                   for layer in owner._layers(model))
        assert all('forward' not in m.__dict__ for m in (
            owner._layers(model)[17].mlp.gate_proj, owner._layers(model)[17].mlp.up_proj,
            owner._layers(model)[17].mlp.down_proj))
        manifest['sites'][0]['name'] = 'tampered copy'
        assert handle.manifest()['sites'][0]['name'] == 'prefix.0.gate_proj'
    finally:
        runner.close()
        handle.uninstall()
    assert not hasattr(model, owner.INSTALLATION)
    assert all('forward' not in module.__dict__ for module in owner._selected_modules(model))


@pytest.mark.parametrize('mutation', ('source_inplace', 'packed_inplace', 'scale_inplace', 'packed_replace',
    'module_replace', 'site_order', 'forward', 'other_forward', 'other_precision'))
def test_every_live_mutation_rejects_old_identity_without_repacking_and_cleanup_remains_possible(rig, mutation):
    handle, runner = installed(rig)
    prepare(handle, runner)
    site = handle.sites[0]
    packed_ids = tuple(id(s.quantized) for s in handle.sites)
    with torch.no_grad():
        if mutation == 'source_inplace': site.weight.add_(1)
        elif mutation == 'packed_inplace': site.quantized.copy_(torch.empty_like(site.quantized))
        elif mutation == 'scale_inplace': site.scale.add_(1)
        elif mutation == 'packed_replace': site.quantized = torch.empty_like(site.quantized)
        elif mutation == 'module_replace': owner._layers(rig.model)[0].mlp.gate_proj = nn.Linear(2048, 16384, bias=False, device='meta', dtype=torch.bfloat16).eval()
        elif mutation == 'site_order': handle.sites = tuple(reversed(handle.sites))
        elif mutation == 'forward': site.module.forward = lambda x: x
        elif mutation == 'other_forward': rig.model.other.forward = lambda x: x
        elif mutation == 'other_precision': rig.model.other.to(dtype=torch.bfloat16)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        prepare(handle, runner)
    if mutation not in ('packed_replace', 'site_order'):
        assert tuple(id(s.quantized) for s in handle.sites) == packed_ids
    runner.close()
    handle.uninstall()
    assert not hasattr(rig.model, owner.INSTALLATION)


def test_unique_owner_lock_generation_and_retirement_are_required(rig):
    handle, runner = installed(rig)
    with pytest.raises(RuntimeError, match='unique locked'):
        handle.prepare(rig.model._inference_cache.generation)
    prepare(handle, runner)
    second = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    with pytest.raises(RuntimeError, match='unique locked'):
        prepare(handle, runner)
    second.close()
    with runner._call_lock:
        with pytest.raises(RuntimeError, match='stale'):
            handle.compilation_members(handle._generation + 1)
    runner._device = torch.device('meta')
    with pytest.raises(RuntimeError, match='Drain native'):
        handle.uninstall()
    runner.clear()
    handle.uninstall()
    runner.close()


def test_explicit_invalidation_rejects_even_equal_signature_that_could_hide_data_write(rig):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        original = tuple(id(s.quantized) for s in handle.sites)
        rig.model.invalidate_inference_caches()
        with pytest.raises(RuntimeError, match='epoch invalidated'):
            prepare(handle, runner)
        assert tuple(id(s.quantized) for s in handle.sites) == original
        with pytest.raises(RuntimeError, match='epoch invalidated'):
            prepare(handle, runner, changed=True)
    finally:
        runner.close()
        handle.uninstall()


def test_epoch_rejects_global_math_or_compile_route_drift(rig, monkeypatch):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        monkeypatch.setattr(owner, '_math_settings', lambda: ('changed',))
        with pytest.raises(RuntimeError, match='compile/math settings'):
            prepare(handle, runner)
    finally:
        runner.close()
        handle.uninstall()


def test_partial_pack_failure_restores_all_forwards_and_drains_before_drop(rig, monkeypatch):
    original = owner._Site.initialize
    count = []
    def fail(site):
        count.append(site.index)
        if site.index == 2:
            site.owner._pending.append(torch.empty(1, device='meta'))
            raise RuntimeError('injected packing failure')
        original(site)
    monkeypatch.setattr(owner._Site, 'initialize', fail)
    with pytest.raises(RuntimeError, match='injected packing'):
        owner.install_prefix_fp8(rig.model, evidence())
    assert count == [0, 1, 2] and rig.drains == ['drained']
    assert not hasattr(rig.model, owner.INSTALLATION)
    assert all('forward' not in m.__dict__ for m in owner._selected_modules(rig.model))


def test_partial_pack_drain_failure_retains_reachable_owner_until_retry(rig, monkeypatch):
    def fail(site):
        site.owner._pending.append(torch.empty(1, device='meta'))
        raise RuntimeError('packing failure')
    monkeypatch.setattr(owner._Site, 'initialize', fail)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: (_ for _ in ()).throw(RuntimeError('drain failed')))
    with pytest.raises(RuntimeError, match='packing failure') as failure:
        owner.install_prefix_fp8(rig.model, evidence())
    assert any('drain failed' in note for note in failure.value.__notes__)
    handle = getattr(rig.model, owner.INSTALLATION)
    assert handle.enabled and handle._pending and len(handle._installed_sites) == 1
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    handle.uninstall()
    assert not hasattr(rig.model, owner.INSTALLATION) and not handle._pending


def test_failed_or_changed_backend_bytes_reject_before_runtime_or_model_work(tmp_path, monkeypatch):
    data = evidence()
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status': 'complete'}))
    data['backend_report'] = str(path)
    monkeypatch.setattr(owner.native, 'validate_runtime', lambda device: pytest.fail('must not reach CUDA'))
    with pytest.raises(ValueError, match='original report bytes'):
        owner._load_evidence(data, torch.device('cuda'))


class CudaProjectionModel(MetaPrefix):
    """Actual 51-site ownership, one representative projection per graph call.

    This is deliberately a lifecycle contract, not a real-policy FP8 result.
    The frozen real fixture and full closed loop remain separate acceptance.
    """
    def prepare_inference_context(self, images, image_masks, lang_masks, steps, **kwargs):
        signature = tuple(tensor_identity(t) for t in (*self.parameters(), *self.buffers()))
        generation = self._inference_cache.refresh(signature)
        getattr(self, owner.INSTALLATION).prepare(generation)
        return dict(token=('test51', generation), packing=None, empty_pattern=(False,))

    def prepare_attention_metadata(self, *args, **kwargs):
        return ()

    def prepare_position_context(self, *args, **kwargs):
        return None

    def sample_actions(self, images, image_masks, tokens, masks, *, noise, **kwargs):
        return owner._layers(self)[0].mlp.gate_proj(noise)


def test_cuda_actual51_owner_native_runner_compile_crossstream_owned_output_epoch_rejection(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Root queues actual CUDA owner contracts')
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('Production owner is SM120-only')
    # Explicit test-only report substitution. Arithmetic is the real package
    # implementation; this is never evidence usable by the policy loader.
    monkeypatch.setattr(owner, '_load_evidence', lambda config, device: (
        b'fictional CUDA lifecycle test only', dict(quantizer_files_sha256={
            name: 'd' * 64 for name in owner.native.SOURCE_FILES})))
    model = CudaProjectionModel().to_empty(device='cuda').eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index % 11 - 5) * .001)
    handle = owner.install_prefix_fp8(model, evidence())
    runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
    handle.bind_runner(runner)
    args = dict(images=[torch.zeros((1, 1), device='cuda')],
        image_masks=[torch.ones((1,), device='cuda', dtype=torch.bool)],
        lang_tokens=torch.ones((1, 2), device='cuda', dtype=torch.int64),
        lang_masks=torch.ones((1, 2), device='cuda', dtype=torch.bool),
        noise=torch.ones((1, 576, 2048), device='cuda', dtype=torch.bfloat16), num_steps=10)
    try:
        with torch.inference_mode():
            first = runner(**args)
            saved = first.clone()
            with torch.cuda.stream(torch.cuda.Stream()):
                second = runner(**args)
            torch.cuda.synchronize()
            assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
            assert first.data_ptr() != second.data_ptr()
            assert torch.equal(first.view(torch.uint8), saved.view(torch.uint8))
            assert runner.stats['captures'] == 1 and runner.stats['stream_handoffs'] >= 1
            # Same-pointer mutation must refuse the old immutable identity;
            # automatic graph recapture cannot silently reuse stale packing.
            handle.sites[0].weight.add_(.01)
            with pytest.raises(RuntimeError, match='epoch invalidated'):
                runner(**args)
        runner.clear()
        handle.uninstall()
        assert all('forward' not in m.__dict__ for m in owner._selected_modules(model))
    finally:
        runner.close()
        if handle.enabled:
            handle.uninstall()
