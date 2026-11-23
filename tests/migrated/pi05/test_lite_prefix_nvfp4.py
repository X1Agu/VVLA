"""Remote CPU owner contracts using explicit meta/fake packing, not evidence.

Native arithmetic/CUDA capture contracts live in test_lite_prefix_nvfp4_native.py.
This file never presents its fictional evidence as a usable production report.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import prefix_nvfp4 as owner
from nvfp4_evidence_fixtures import make_evidence_fixture, contract
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache, tensor_identity
from vvla_repro_pi05.models.pi05.precision_pi05 import FusedPrefixMLP


def evidence():
    return make_evidence_fixture()[3]


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
    def pack(weight, calibration, keepalive):
        n, k = weight.shape
        values = (torch.empty((n, k // 2), dtype=torch.uint8, device='meta'),
                  torch.empty((n, k // 16), dtype=torch.float8_e4m3fn, device='meta'),
                  *(torch.empty((1,), dtype=torch.float32, device='meta') for _ in range(3)))
        keepalive.extend(values)
        return values
    monkeypatch.setattr(owner, '_validate_module', geometry)
    monkeypatch.setattr(owner, '_pack_weight', pack)
    scalar_sha = __import__('hashlib').sha256(bytes.fromhex('0000803f')).hexdigest()
    monkeypatch.setattr(owner, '_tensor_record', lambda t: dict(shape=list(t.shape), stride=list(t.stride()),
        dtype=str(t.dtype).removeprefix('torch.'), content_sha256=scalar_sha if tuple(t.shape) == (1,) else 'c' * 64))
    monkeypatch.setattr(owner, '_scalar_value', lambda t: dict(value=1.0, fp32_bits='0x3f800000'))
    manifest, report, calibration, config = make_evidence_fixture()
    backend = manifest['backend']
    runtime = SimpleNamespace(validate=lambda device: None)
    raw, raw_calibration = contract.canonical(report), contract.canonical(calibration)
    monkeypatch.setattr(owner, '_load_evidence', lambda config, device, native_path: (
        raw, raw_calibration, deepcopy(calibration), deepcopy(backend), runtime))
    drains = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: drains.append('drained'))
    return SimpleNamespace(model=MetaPrefix(), backend=backend, raw=raw, raw_calibration=raw_calibration,
                           calibration=calibration, runtime=runtime, drains=drains)


def installed(rig):
    handle = owner.install_prefix_nvfp4(rig.model, evidence(), "/fictional/native")
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
        assert handle.raw_calibration == rig.raw_calibration
        contract.validate_quantization_manifest(manifest)
        contract.validate_calibration_binding(rig.calibration, manifest, manifest['backend']['hardware'])
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
        elif mutation == 'scale_inplace': site.scale.copy_(torch.empty_like(site.scale))
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
        owner.install_prefix_nvfp4(rig.model, evidence(), "/fictional/native")
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
        owner.install_prefix_nvfp4(rig.model, evidence(), "/fictional/native")
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
    monkeypatch.setattr(owner.native, 'load_native_backend', lambda *args: pytest.fail('must not reach CUDA'))
    with pytest.raises(ValueError, match='original report bytes'):
        owner._load_evidence(data, torch.device('cuda'), '/fictional/native')


@pytest.mark.parametrize('field', ('input_global_scale', 'weight_global_scale', 'alpha'))
@pytest.mark.parametrize('mutation', ('inplace', 'replacement'))
def test_each_persistent_global_scale_and_alpha_is_versioned_owned_and_live(rig, field, mutation):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        site = handle.sites[0]
        tensor = getattr(site, field)
        if mutation == 'inplace':
            tensor.add_(1)
        else:
            setattr(site, field, torch.empty_like(tensor))
        with pytest.raises(RuntimeError, match='packing epoch|ownership changed'):
            prepare(handle, runner)
    finally:
        runner.close()
        handle.uninstall()


@pytest.mark.parametrize('field', ('raw_report', 'raw_calibration', 'evidence', 'runtime', 'manifest'))
def test_evidence_and_backend_owner_replacement_cannot_relabel_existing_packing(rig, field):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        if field in ('raw_report', 'raw_calibration'):
            setattr(handle, field, getattr(handle, field) + b' ')
        elif field == 'evidence':
            handle.evidence['bf16_control']['recipe_sha256'] = '0' * 64
        elif field == 'runtime':
            handle.runtime = SimpleNamespace(validate=lambda device: None)
        else:
            handle._manifest['calibration']['quality_validated'] = True
        with pytest.raises(RuntimeError, match='binding changed|owner was replaced|provenance or manifest'):
            handle.manifest()
    finally:
        runner.close()
        handle.uninstall()


def test_explicit_noise_input_geometry_rejects_b2_and_unsupported_capacity_without_fallback(rig, monkeypatch):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        site = handle.sites[0]
        # Meta is only a CPU lifecycle fixture, not an autocast backend.
        monkeypatch.setattr(torch, 'is_autocast_enabled', lambda *args: False)
        monkeypatch.setattr(owner.native, 'scaled_linear', lambda *args: pytest.fail('no BF16 or native fallback'))
        with torch.no_grad():
            for shape in ((2, 576, 2048), (1, 512, 2048)):
                with pytest.raises(ValueError, match='explicit B1'):
                    site.call(site.module, torch.empty(shape, device='meta', dtype=torch.bfloat16))
    finally:
        runner.close()
        handle.uninstall()


def test_current_thread_must_have_prepared_epoch(rig, monkeypatch):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        handle._thread = -1
        with runner._call_lock:
            with pytest.raises(RuntimeError, match='generation/thread'):
                handle.compilation_members(handle._generation)
    finally:
        runner.close()
        handle.uninstall()


def test_conflicting_fp8_owner_is_rejected_before_packing(rig):
    rig.model._lite_prefix_fp8_installation = object()
    with pytest.raises(ValueError, match='mutually exclusive'):
        owner.install_prefix_nvfp4(rig.model, evidence(), '/fictional/native')
    assert not hasattr(rig.model, owner.INSTALLATION)


def test_source_weight_hash_must_match_frozen_actual_calibration_before_packing(rig, monkeypatch):
    monkeypatch.setattr(owner, '_tensor_record', lambda t: dict(content_sha256='0' * 64))
    monkeypatch.setattr(owner, '_pack_weight', lambda *args: pytest.fail('must reject before packing'))
    with pytest.raises(ValueError, match='calibration checkpoint identity'):
        owner.install_prefix_nvfp4(rig.model, evidence(), '/fictional/native')
    assert rig.drains == ['drained'] and not hasattr(rig.model, owner.INSTALLATION)


def test_cold_calibration_bytes_are_checked_before_native_import(tmp_path, monkeypatch):
    manifest, report, calibration, config = make_evidence_fixture()
    backend_path, calibration_path = tmp_path / 'backend.json', tmp_path / 'calibration.json'
    backend_path.write_bytes(contract.canonical(report))
    calibration_path.write_bytes(contract.canonical(calibration) + b' ')
    config.update(backend_report=str(backend_path), calibration=str(calibration_path))
    monkeypatch.setattr(owner.native, 'load_native_backend', lambda *args: pytest.fail('no import before byte binding'))
    with pytest.raises(ValueError, match='calibration original bytes'):
        owner._load_evidence(config, torch.device('cuda'), '/fictional/native')


