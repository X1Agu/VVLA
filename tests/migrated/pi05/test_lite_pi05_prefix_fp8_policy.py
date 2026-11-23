"""Remote CPU policy wiring with explicit fictional installers, never evidence."""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import pi05


def evidence():
    return dict(backend_report='/fictional/backend.json', backend_report_sha256='a' * 64,
        bf16_control={key: 'b' * 64 for key in ('recipe_sha256', 'config_sha256', 'identity_sha256', 'content_sha256')})


def model_processor():
    model = torch.nn.Linear(2, 2).eval()
    model.action_horizon, model.action_dim = 50, 32
    model._graph_runners = []
    processor = SimpleNamespace(config=SimpleNamespace(chunk_size=50, num_inference_steps=10,
        action_dim=7, attention_backend='torch_fa2'))
    return model, processor


def module(monkeypatch, name, **members):
    fake = ModuleType('vvla_repro_pi05.lite.' + name)
    fake.__dict__.update(members)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)


def rig(monkeypatch, *, failure=None):
    events, handles = [], []
    class Prefix:
        def __init__(self, model, value):
            self.model, self.evidence, self.raw_report = model, deepcopy(value), b'original fictional bytes'
            self.runner = None
            self.document = {'precision': 'mixed_bf16_fp8_prefix', 'sites': ['fictional']}
        def bind_runner(self, runner):
            events.append('prefix.bind')
            self.runner = runner
            if failure == 'bind':
                raise RuntimeError('injected bind')
        def manifest(self):
            return deepcopy(self.document)
        def validate_precision_manifest(self):
            events.append('prefix.validate')
            return self.manifest()
        def uninstall(self):
            if self.runner is not None:
                assert self.runner._closed
            events.append('prefix.uninstall')
            del self.model._lite_prefix_fp8_installation
    def install(model, value):
        events.append('prefix.install')
        handle = Prefix(model, value)
        handles.append(handle)
        model._lite_prefix_fp8_installation = handle
        if failure == 'retained_install':
            raise RuntimeError('injected retained install')
        return handle
    class Native:
        def manifest(self):
            return {'native': 'fictional'}
        def uninstall(self):
            assert 'prefix.uninstall' in events
            events.append('native.uninstall')
    def native_install(model, *, collector):
        events.append('native.install')
        return Native()
    class Runner:
        def __init__(self, model, **kwargs):
            events.append('runner.init')
            if failure == 'runner':
                raise RuntimeError('injected runner')
            self.model, self._closed = model, False
            model._graph_runners.append(self)
        def compilation_manifest(self):
            return {'options': {'triton.cudagraphs': False}}
        def close(self):
            events.append('runner.close')
            self._closed = True
    module(monkeypatch, 'prefix_fp8', install_prefix_fp8=install)
    module(monkeypatch, 'prefix_fp8_dispatch', install_prefix_fp8=install)
    module(monkeypatch, 'native_lifecycle', install_native_lifecycle_signature=native_install,
           load_native_lifecycle=lambda **kwargs: object())
    module(monkeypatch, 'graph', Pi05NativeGraphRunner=Runner)
    return events, handles


def enabled():
    return dict(device='cuda', cuda_graph=True, compile_sample_actions=True, native_lifecycle=True,
                prefix_fp8=True, prefix_fp8_evidence=evidence())


def test_disabled_option_preserves_existing_defaults_metadata_and_recipe_shape(monkeypatch):
    events, handles = rig(monkeypatch)
    absent = pi05._split_lite_overrides({})
    explicit = pi05._split_lite_overrides({'lite_prefix_fp8': False})
    assert absent == explicit and 'prefix_fp8' not in absent[1]
    model, processor = model_processor()
    policy = pi05.Pi05LitePolicy(model, processor, device='cpu', cuda_graph=False)
    try:
        data = policy.metadata()
        assert data['precision'] == 'bf16'
        assert 'prefix_fp8' not in data['lite_optimizations']
        assert 'quantization_manifest' not in data and 'recipe_components' not in data
        assert policy._prefix_fp8_installation is None and events == [] and handles == []
    finally:
        policy.close()


@pytest.mark.parametrize('value', [1, 0, None, {}, 'true'])
def test_prefix_option_rejects_nonboolean_without_installation(value):
    with pytest.raises(ValueError, match='must be a boolean'):
        pi05._split_lite_overrides({'lite_prefix_fp8': value})


@pytest.mark.parametrize('change', [dict(native_lifecycle=False), dict(compile_sample_actions=False),
                                   dict(cuda_graph=False, device='cpu')])
def test_dependencies_fail_before_optional_installation(monkeypatch, change):
    events, _ = rig(monkeypatch)
    options = enabled()
    options.update(change)
    with pytest.raises(ValueError, match='Prefix FP8 requires'):
        pi05.Pi05LitePolicy(*model_processor(), **options)
    assert events == []


def test_evidence_only_allowed_for_enabled_option_and_exact_complete_schema(monkeypatch):
    events, _ = rig(monkeypatch)
    with pytest.raises(ValueError, match='only allowed'):
        pi05.Pi05LitePolicy(*model_processor(), device='cpu', cuda_graph=False, prefix_fp8_evidence=evidence())
    options = enabled()
    options['prefix_fp8_evidence']['bf16_control'].pop('identity_sha256')
    with pytest.raises(ValueError, match='fields'):
        pi05.Pi05LitePolicy(*model_processor(), **options)
    assert events == []


def test_install_binding_metadata_manifest_live_hook_original_bytes_and_close_order(monkeypatch):
    events, handles = rig(monkeypatch)
    model, processor = model_processor()
    original = tuple((name, id(value)) for name, value in model.named_parameters())
    policy = pi05.Pi05LitePolicy(model, processor, **enabled())
    assert events == ['prefix.install', 'native.install', 'runner.init', 'prefix.bind']
    data = policy.metadata()
    assert data['precision'] == 'mixed_bf16_fp8_prefix'
    assert data['quantization_manifest'] == handles[0].manifest()
    components = data['recipe_components']
    assert components['bf16_control'] == evidence()['bf16_control']
    assert components['base_recipe_sha256'] == evidence()['bf16_control']['recipe_sha256']
    assert components['quantization_manifest_sha256'] == pi05._json_hash(data['quantization_manifest'])
    assert data['recipe_sha256'] == pi05._json_hash(components)
    assert policy.validate_precision_manifest() == data['quantization_manifest']
    assert policy.precision_backend_report() is handles[0].raw_report
    data['quantization_manifest']['sites'].clear()
    assert policy.metadata()['quantization_manifest']['sites'] == ['fictional']
    assert tuple((name, id(value)) for name, value in model.named_parameters()) == original
    policy.close()
    assert events[-3:] == ['runner.close', 'prefix.uninstall', 'native.uninstall']
    with pytest.raises(RuntimeError, match='No active'):
        policy.precision_backend_report()


def test_explicit_flashrt_reference_keeps_lite_execution_recipe_provenance(monkeypatch):
    _, handles = rig(monkeypatch)
    options = enabled()
    options['prefix_fp8_evidence']['reference_kind'] = 'flashrt_bf16'
    options['provenance'] = {'recipe_sha256': 'c' * 64}
    policy = pi05.Pi05LitePolicy(*model_processor(), **options)
    try:
        components = policy.metadata()['recipe_components']
        assert components['reference_kind'] == 'flashrt_bf16'
        assert components['base_recipe_scope'] == 'actual_lite_execution_recipe_before_reference_binding'
        assert components['base_recipe_sha256'] == 'c' * 64
        assert components['base_recipe_sha256'] != components['bf16_control']['recipe_sha256']
        assert components['bf16_control'] == evidence()['bf16_control']
        assert handles[0].evidence['reference_kind'] == 'flashrt_bf16'
    finally:
        policy.close()


@pytest.mark.parametrize('failure', ['retained_install', 'runner', 'bind'])
def test_constructor_failure_recovers_retained_owner_and_retires_before_cache_release(monkeypatch, failure):
    events, handles = rig(monkeypatch, failure=failure)
    model, processor = model_processor()
    failed = pi05.Pi05LitePolicy.__new__(pi05.Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        failed.__init__(model, processor, **enabled())
    assert failed._closed and failed._prefix_fp8_installation is None
    assert not hasattr(model, '_lite_prefix_fp8_installation')
    assert 'prefix.uninstall' in events
    if failure == 'bind':
        assert events.index('runner.close') < events.index('prefix.uninstall') < events.index('native.uninstall')
