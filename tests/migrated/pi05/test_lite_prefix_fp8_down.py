"""Remote owner contracts, with fictional meta packing/evidence explicitly injected.

CUDA cases use real SM89 primitives and the native graph owner, but are synthetic
lifecycle checks. They cannot authorize backend evidence or policy quality.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import prefix_fp8_down as owner
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
from test_lite_prefix_fp8 import MetaPrefix, evidence

ONE_SHA = hashlib.sha256(b'\x00\x00\x80\x3f').hexdigest()


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
    def record(tensor):
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
            dtype=str(tensor.dtype).removeprefix('torch.'),
            content_sha256=ONE_SHA if tensor.shape == (1,) else 'c' * 64)
    monkeypatch.setattr(owner, '_validate_module', geometry)
    monkeypatch.setattr(owner, '_pack_weight', pack)
    monkeypatch.setattr(owner, '_tensor_record', record)
    monkeypatch.setattr(owner.native, 'validate_unit_scale', lambda one: dict(passed=True, raw_sha256=ONE_SHA))
    backend = dict(quantizer_files_sha256={name: 'd' * 64 for name in owner.native.SOURCE_FILES})
    raw = b'fictional meta-test evidence, never a usable executed report'
    monkeypatch.setattr(owner, '_load_evidence', lambda config, device: (raw, deepcopy(backend)))
    drains = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: drains.append('drained'))
    return SimpleNamespace(model=MetaPrefix(), backend=backend, raw=raw, drains=drains)


def installed(rig):
    handle = owner.install_prefix_fp8_down(rig.model, evidence())
    runner = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    handle.bind_runner(runner)
    return handle, runner


def prepare(handle, runner):
    model = runner.model
    signature = tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers()))
    generation = model._inference_cache.refresh(signature)
    with runner._call_lock:
        handle.prepare(generation)
        members = handle.compilation_members(generation)
        assert members == (*handle.sites, handle.unit)
        assert len(members) == 18


def test_exact17_down_original_parameters_gate_up_and_scalar_manifest_preserved(rig):
    model = rig.model
    parameters = tuple((name, id(t)) for name, t in model.named_parameters())
    keys = tuple(model.state_dict())
    gate_up = tuple((getattr(layer.mlp, name), getattr(layer.mlp, name).weight)
        for layer in owner._layers(model)[:17] for name in ('gate_proj', 'up_proj'))
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        report = handle.validate_precision_manifest()
        assert report['implementation'] == 'pi05-prefix-down-scalar-fp8-sm89-v1'
        assert report['scope'] == 'prefix_mlp_down_layers_0_16'
        assert [site['name'] for site in report['sites']] == [f'prefix.{i}.down_proj' for i in range(17)]
        assert report['scalar_one'] == dict(shape=[1], stride=[1], dtype='float32', content_sha256=ONE_SHA)
        assert report['quantizer']['gate_up'] == 'unchanged_BF16'
        assert report['quantizer']['intermediate_dtype'] == 'float32'
        assert report['lifecycle']['live_checks'] == owner.LIVE_CHECKS
        assert report['precision_boundaries'] == owner.PRECISION_BOUNDARIES
        assert tuple((name, id(t)) for name, t in model.named_parameters()) == parameters
        assert tuple(model.state_dict()) == keys
        assert all(module.weight is weight and 'forward' not in module.__dict__ for module, weight in gate_up)
        assert not torch.is_inference(handle.unit.value) and type(handle.unit.value._version) is int
        assert handle._signature[1] == tensor_identity(handle.unit.value)
        assert handle.raw_report == rig.raw and not handle._pending
        report['scalar_one']['content_sha256'] = '0' * 64
        assert handle.manifest()['scalar_one']['content_sha256'] == ONE_SHA
    finally:
        runner.close()
        handle.uninstall()
    assert not hasattr(model, owner.INSTALLATION)
    assert handle.unit is None
    assert all('forward' not in m.__dict__ for m in owner._selected_modules(model))


@pytest.mark.parametrize('mutation', ['source_inplace', 'packed_inplace', 'scale_inplace', 'packed_replace',
    'scalar_inplace', 'scalar_replace', 'scalar_member_replace', 'scalar_dtype', 'site_order',
    'gate_inplace', 'gate_parameter_replace', 'gate_module_same_weight', 'gate_forward',
    'other_precision', 'other_forward', 'compiled_disabled'])
def test_source_private_scalar_gate_up_and_nonprefix_mutations_fail_closed(rig, mutation):
    handle, runner = installed(rig)
    prepare(handle, runner)
    site, gate = handle.sites[0], owner._layers(rig.model)[0].mlp.gate_proj
    with torch.no_grad():
        if mutation == 'source_inplace': site.weight.add_(1)
        elif mutation == 'packed_inplace': site.quantized.copy_(torch.empty_like(site.quantized))
        elif mutation == 'scale_inplace': site.scale.add_(1)
        elif mutation == 'packed_replace': site.quantized = torch.empty_like(site.quantized)
        elif mutation == 'scalar_inplace': handle.unit.value.add_(1)
        elif mutation == 'scalar_replace': handle.unit.value = torch.ones_like(handle.unit.value)
        elif mutation == 'scalar_member_replace': handle.unit = SimpleNamespace()
        elif mutation == 'scalar_dtype': handle.unit.value = handle.unit.value.to(torch.bfloat16)
        elif mutation == 'site_order': handle.sites = tuple(reversed(handle.sites))
        elif mutation == 'gate_inplace': gate.weight.add_(1)
        elif mutation == 'gate_parameter_replace': gate.weight = nn.Parameter(torch.empty_like(gate.weight))
        elif mutation == 'gate_module_same_weight':
            replacement = nn.Linear(2048, 16384, bias=False, device='meta', dtype=torch.bfloat16).eval()
            replacement.weight = gate.weight
            owner._layers(rig.model)[0].mlp.gate_proj = replacement
        elif mutation == 'gate_forward': gate.forward = lambda x: x
        elif mutation == 'other_precision': rig.model.other.to(dtype=torch.bfloat16)
        elif mutation == 'other_forward': rig.model.other.forward = lambda x: x
        elif mutation == 'compiled_disabled': runner.compile_sample_actions = False
    try:
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            prepare(handle, runner)
    finally:
        runner.close()
        handle.uninstall()


def test_scalar_owner_is_a_required_prepared_compile_member(rig, monkeypatch):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        site = handle.sites[0]
        # This contract exercises the Python scope guard with a CPU stand-in;
        # meta is not an autocast device accepted by the real call guard.
        site.device = torch.device('cpu')
        value = torch.empty((1, 576, 16384), dtype=torch.bfloat16)
        monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: True)
        token = object()
        site._prepared_compile_token = token
        with torch.no_grad(), pytest.raises(RuntimeError, match='prepared owner scope'):
            site.call(site.module, value)
        handle.unit._prepared_compile_token = token
        calls = []
        def linear(x, q, s, one):
            calls.append(one)
            return torch.empty((576, 2048), dtype=torch.bfloat16), None, None
        monkeypatch.setattr(owner.native, 'scaled_linear', linear)
        with torch.no_grad():
            result = site.call(site.module, value)
        assert result.shape == (1, 576, 2048) and len(calls) == 1 and calls[0] is handle.unit.value
    finally:
        runner.close()
        handle.uninstall()


def test_unique_owner_full_generation_and_clear_before_uninstall(rig):
    handle, runner = installed(rig)
    with pytest.raises(RuntimeError, match='unique locked'):
        handle.prepare(rig.model._inference_cache.generation)
    prepare(handle, runner)
    with runner._call_lock:
        with pytest.raises(RuntimeError, match='stale'):
            handle.compilation_members(handle._generation + 1)
        original_thread = handle._thread
        handle._thread += 1
        with pytest.raises(RuntimeError, match='generation/thread'):
            handle.compilation_members(handle._generation)
        handle._thread = original_thread
    second = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    with pytest.raises(RuntimeError, match='unique locked'):
        prepare(handle, runner)
    second.close()
    rig.model.invalidate_inference_caches()
    with pytest.raises(RuntimeError, match='live manifest generation changed'):
        handle.validate_precision_manifest()
    with pytest.raises(RuntimeError, match='epoch invalidated'):
        prepare(handle, runner)
    runner._device = torch.device('meta')
    with pytest.raises(RuntimeError, match='Drain native'):
        handle.uninstall()
    runner.clear()
    handle.uninstall()
    runner.close()


def test_live_manifest_detects_same_parameters_nonprefix_module_replacement(rig):
    handle, runner = installed(rig)
    try:
        prepare(handle, runner)
        original = rig.model.other
        replacement = nn.Linear(2, 2, device='meta', dtype=torch.float32).eval()
        replacement.weight, replacement.bias = original.weight, original.bias
        rig.model.other = replacement
        # Parameter tensor signatures are identical; the live module topology
        # check must still reject the unpublished implementation identity.
        with pytest.raises(RuntimeError, match='precision/module scope changed'):
            handle.validate_precision_manifest()
    finally:
        runner.close()
        handle.uninstall()


def test_owner_independently_rereads_sha_and_rejects_other_variant_before_runtime(tmp_path, monkeypatch):
    data = evidence()
    path = tmp_path / 'report.json'
    raw = json.dumps({'quantizer_variant': 'triton-eager-rn-native-sm120-v1'}).encode()
    path.write_bytes(raw)
    data['backend_report'] = str(path)
    monkeypatch.setattr(owner.native, 'validate_runtime', lambda device: pytest.fail('must not query runtime'))
    with pytest.raises(ValueError, match='original report bytes'):
        owner._load_evidence(data, torch.device('cuda'))
    data['backend_report_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='explicit SM89'):
        owner._load_evidence(data, torch.device('cuda'))


def test_cold_scalar_content_failure_drains_before_forwards_are_patched(rig, monkeypatch):
    monkeypatch.setattr(owner.native, 'validate_unit_scale', lambda one: dict(passed=True, raw_sha256='0' * 64))
    with pytest.raises(ValueError, match='exact cold content proof'):
        owner.install_prefix_fp8_down(rig.model, evidence())
    assert rig.drains == ['drained'] and not hasattr(rig.model, owner.INSTALLATION)
    assert all('forward' not in m.__dict__ for m in owner._selected_modules(rig.model))


def test_cold_scalar_sync_failure_keeps_owned_allocation_until_retry(rig, monkeypatch):
    def fail(one):
        raise RuntimeError('cold scalar read failed')
    monkeypatch.setattr(owner.native, 'validate_unit_scale', fail)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: (_ for _ in ()).throw(RuntimeError('drain failed')))
    with pytest.raises(RuntimeError, match='cold scalar read failed') as failed:
        owner.install_prefix_fp8_down(rig.model, evidence())
    assert any('drain failed' in note for note in failed.value.__notes__)
    handle = getattr(rig.model, owner.INSTALLATION)
    assert handle._pending[0] is handle.unit.value and handle.enabled
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    handle.uninstall()
    assert handle.unit is None and not handle._pending


def test_partial_weight_pack_failure_cleans_shared_scalar_and_original_forwards(rig, monkeypatch):
    original, calls = owner._Site.initialize, []
    def fail(site):
        calls.append(site.index)
        if site.index == 2:
            site.owner._pending.append(torch.empty((1,), device='meta'))
            raise RuntimeError('partial down packing failed')
        original(site)
    monkeypatch.setattr(owner._Site, 'initialize', fail)
    with pytest.raises(RuntimeError, match='partial down packing'):
        owner.install_prefix_fp8_down(rig.model, evidence())
    assert calls == [0, 1, 2] and rig.drains == ['drained']
    assert not hasattr(rig.model, owner.INSTALLATION)


class CudaDownProjectionModel(MetaPrefix):
    """17 actual owned sites; one representative down projection is executed."""
    def prepare_inference_context(self, images, image_masks, lang_masks, steps, **kwargs):
        signature = tuple(tensor_identity(t) for t in (*self.parameters(), *self.buffers()))
        generation = self._inference_cache.refresh(signature)
        getattr(self, owner.INSTALLATION).prepare(generation)
        return dict(token=('test17down', generation), packing=None, empty_pattern=(False,))

    def prepare_attention_metadata(self, *args, **kwargs):
        return ()

    def prepare_position_context(self, *args, **kwargs):
        return None

    def sample_actions(self, images, image_masks, tokens, masks, *, noise, **kwargs):
        return owner._layers(self)[0].mlp.down_proj(noise)


@pytest.mark.parametrize('changed', ['weight', 'scalar'])
def test_cuda_actual17_owner_compiled_capture_crossstream_refresh_and_owned_output(monkeypatch, changed):
    if not torch.cuda.is_available():
        pytest.skip('Root queues CUDA owner contracts')
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip('Production down owner is explicit SM89 only')
    monkeypatch.setattr(owner, '_load_evidence', lambda config, device: (
        b'fictional CUDA lifecycle report, not usable evidence',
        dict(quantizer_files_sha256={name: 'd' * 64 for name in owner.native.SOURCE_FILES})))
    model = CudaDownProjectionModel().to_empty(device='cuda').eval()
    with torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            parameter.fill_((i % 11 - 5) * .001)
    handle = owner.install_prefix_fp8_down(model, evidence())
    runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
    handle.bind_runner(runner)
    args = dict(images=[torch.zeros((1, 1), device='cuda')],
        image_masks=[torch.ones((1,), device='cuda', dtype=torch.bool)],
        lang_tokens=torch.ones((1, 2), device='cuda', dtype=torch.int64),
        lang_masks=torch.ones((1, 2), device='cuda', dtype=torch.bool),
        noise=torch.linspace(-1, 1, 576 * 16384, device='cuda').reshape(1, 576, 16384).to(torch.bfloat16),
        num_steps=10)
    try:
        with torch.inference_mode():
            first = runner(**args)
            expected, _, _ = owner.native.scaled_linear(args['noise'].view(576, 16384),
                handle.sites[0].quantized, handle.sites[0].scale, handle.unit.value)
            saved = first.clone()
            with torch.cuda.stream(torch.cuda.Stream()):
                second = runner(**args)
            torch.cuda.synchronize()
            assert torch.equal(first.view(torch.uint8), expected.view(1, 576, 2048).view(torch.uint8))
            assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
            assert first.data_ptr() != second.data_ptr()
            assert torch.equal(first.view(torch.uint8), saved.view(torch.uint8))
            assert runner.stats['captures'] == 1 and runner.stats['stream_handoffs'] >= 1
            old = second.clone()
            args['noise'].mul_(.75)
            refreshed = runner(**args)
            expected_refresh, _, _ = owner.native.scaled_linear(args['noise'].view(576, 16384),
                handle.sites[0].quantized, handle.sites[0].scale, handle.unit.value)
            torch.cuda.synchronize()
            assert torch.equal(second.view(torch.uint8), old.view(torch.uint8))
            assert torch.isfinite(refreshed).all()
            assert torch.equal(refreshed.view(torch.uint8), expected_refresh.view(1, 576, 2048).view(torch.uint8))
            if changed == 'weight':
                handle.sites[0].weight.add_(.01)
            else:
                handle.unit.value.add_(1)
            with pytest.raises(RuntimeError, match='epoch invalidated|scalar-one immutable identity'):
                runner(**args)
        runner.clear()
        handle.uninstall()
        assert all('forward' not in m.__dict__ for m in owner._selected_modules(model))
    finally:
        runner.close()
        if handle.enabled:
            handle.uninstall()
