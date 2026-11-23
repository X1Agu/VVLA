"""BF16 identity and real tiny-core lifetime tests; no CUDA qualification."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pickle

import pytest
import torch

from vvla_repro_pi05.lite import bf16_checkpoint_engine as bf16
from vvla_repro_pi05.lite import bf16_checkpoint_engine_evidence as evidence
from vvla_repro_pi05.lite import batch_checkpoint_engine as state_module
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _validate_compilation_options
from test_lite_fixed_checkpoint_engine import _Policy as TinyPolicy
from test_lite_host_layout import _LayoutModel, _inputs

ROOT = Path(__file__).resolve().parents[2] / 'fixtures'
ORIGINS = {
    'g4': ROOT / 'g4-identity.json',
    'l4': ROOT / 'l4-identity.json',
}


@pytest.mark.parametrize('profile', ['g4', 'l4'])
def test_original_bf16_domains_are_independent_of_fp8(profile):
    origin = evidence.load_origin(ORIGINS[profile], profile)
    assert origin['metadata']['precision'] == 'bf16'
    assert origin['config']['config_overrides']['lite_native_lifecycle_variant'] == ('v3' if profile == 'g4' else 'v2')
    assert not any(origin['config']['config_overrides'].get(k, False) for k in evidence.FORBIDDEN_OPTIONS)
    # The previous mixed-precision factory is deliberately not relaxed.
    with pytest.raises(ValueError, match='prefix51'):
        _validate_compilation_options(dict(fixed_checkpoint_engine=True, native_lifecycle=True,
            native_lifecycle_variant='v3', compile_sample_actions=True, host_layout_cache=True,
            vision_fc2_cublaslt=True, prefix_fp8=False), True)


def test_changed_or_symlinked_origin_and_foreign_profile_rejected(tmp_path):
    value = tmp_path / 'identity.json'
    value.write_bytes(ORIGINS['l4'].read_bytes() + b' ')
    with pytest.raises(ValueError, match='exact original'):
        evidence.load_origin(value, 'l4')
    link = tmp_path / 'link.json'
    link.symlink_to(ORIGINS['l4'])
    with pytest.raises(ValueError, match='exact original'):
        evidence.load_origin(link, 'l4')
    with pytest.raises(ValueError, match='lineages'):
        evidence.load_origin(ORIGINS['l4'], 'other')


class CpuBf16Policy(TinyPolicy):
    def __init__(self, profile):
        super().__init__()
        self.origin = evidence.load_origin(ORIGINS[profile], profile)
        self.config = SimpleNamespace(execution_recipe={})
        self.base = deepcopy(self.origin['metadata'])
        self._lite_optimizations = self.base['lite_optimizations']
        # An actual BF16 parameter in an explicitly fictional tiny CPU model.
        # This is a lifetime/dtype contract test, never checkpoint evidence.
        self.model.register_parameter('bf16_test_marker', torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)))

    def metadata(self):
        return deepcopy(dict(self.base, lite_optimizations=self._lite_optimizations))

    def close(self):
        assert not hasattr(self.model, state_module.INSTALLATION)
        self.graph_runner._closed = True
        self._closed = True
        self.closes += 1


@pytest.fixture(params=['g4', 'l4'])
def deployed(request, monkeypatch):
    profile = request.param
    received = []

    def construct(cls, **config):
        received.append(deepcopy(config))
        return CpuBf16Policy(profile)

    monkeypatch.setattr(Pi05LitePolicy, 'from_pretrained', classmethod(construct))
    monkeypatch.setattr(bf16, '_hardware', lambda device, card: dict(test_only='fictional CPU hardware', profile=card))
    engine = bf16.Bf16CheckpointEngine.from_pretrained('/checkpoint', tokenizer='/tokenizer',
        origin_identity=ORIGINS[profile], hardware_profile=profile)
    policy = engine._Bf16CheckpointEngine__policy
    yield policy, engine, received
    engine.close()


def test_factory_preserves_exact_original_options_and_encapsulation(deployed):
    policy, engine, received = deployed
    assert received[0]['config_overrides'] == policy.origin['config']['config_overrides']
    assert received[0]['precision'] == 'bf16' and received[0]['checkpoint'] == '/checkpoint'
    for name in ('model', 'processor', 'policy', 'graph_runner', 'config', 'parameters', 'state_dict', 'train', 'to'):
        assert not hasattr(engine, name)
        with pytest.raises(AttributeError):
            setattr(engine, name, object())
    assert not hasattr(engine, '__dict__')
    with pytest.raises(TypeError, match='serialized'):
        pickle.dumps(engine)
    first = engine.metadata()
    assert first['precision'] == first['bf16_precision_manifest']['precision'] == 'bf16'
    assert not first['bf16_precision_manifest']['quantized_owners']
    assert not first['bf16_deployment_manifest']['policy_quality_passed']
    assert first['recipe_sha256'] == evidence.digest(first['recipe_components'])
    first['bf16_deployment_manifest']['source_sha256'].clear()
    assert engine.metadata()['bf16_deployment_manifest']['source_sha256']
    assert received[0]['config_overrides'] == policy.origin['config']['config_overrides']


def test_seal_keeps_dynamic_real_context_and_explicit_full_audits(deployed):
    policy, engine, _ = deployed
    reference = _LayoutModel()
    initial = policy.model.complete_prepares
    outputs = []
    for valid, cameras, value in [(2, (True, False), 1.), (5, (True, True), 3.), (1, (False, True), -.5)]:
        inputs = _inputs(valid=valid, cameras=cameras, value=value)
        with policy._lock, policy.graph_runner._call_lock:
            actual = policy.model.prepare_inference_context(inputs['images'], inputs['image_masks'],
                inputs['lang_masks'], inputs['num_steps'], positions=False)
        wanted = reference.prepare_inference_context(inputs['images'], inputs['image_masks'],
            inputs['lang_masks'], inputs['num_steps'], positions=False)
        for key in ('indices', 'pads', 'positions'):
            assert torch.equal(actual['packing'][key], wanted['packing'][key])
        for model, context in ((policy.model, actual), (reference, wanted)):
            context['positions'] = model.prepare_position_context(inputs['image_masks'], inputs['lang_masks'], context['packing'])
        args = {k: inputs[k] for k in ('images', 'image_masks', 'lang_tokens', 'lang_masks', 'noise', 'num_steps')}
        assert torch.equal(policy.model.sample_actions(**args, inference_context=actual),
                           reference.sample_actions(**args, inference_context=wanted))
        outputs.append(actual['packing']['pads'].clone())
    assert policy.model.complete_prepares == initial
    engine.metadata()
    assert policy.model.complete_prepares == initial + 1
    assert any(not torch.equal(outputs[0], current) for current in outputs[1:])


@pytest.mark.parametrize('fault', ['epoch', 'autocast', 'math', 'detach', 'runner', 'source', 'weight', 'dtype', 'quantization', 'options', 'defaults', 'numerical_environment'])
def test_invalid_execution_domain_rejects_and_retirement_remains_available(deployed, monkeypatch, fault):
    policy, engine, _ = deployed
    if fault == 'epoch':
        policy.model._inference_cache.clear()
    elif fault == 'autocast':
        monkeypatch.setattr(torch, 'is_autocast_enabled', lambda *args: True)
    elif fault == 'math':
        monkeypatch.setattr(state_module, '_math_settings', lambda: ('changed',))
    elif fault == 'detach':
        delattr(policy.model, state_module.INSTALLATION)
    elif fault == 'runner':
        policy.graph_runner._closed = True
    elif fault == 'source':
        monkeypatch.setattr(bf16, '_sources', lambda: {'changed': 'f' * 64})
    elif fault == 'weight':
        with torch.no_grad():
            policy.model.weight.add_(1)
    elif fault == 'dtype':
        policy.model.bf16_test_marker.data = policy.model.bf16_test_marker.data.to(torch.float16)
    elif fault == 'quantization':
        policy._prefix_fp8_installation = object()
    elif fault == 'defaults':
        monkeypatch.setattr(torch, 'get_default_dtype', lambda: torch.float64)
    elif fault == 'numerical_environment':
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    else:
        policy._lite_optimizations['prefix_fp8'] = True
    with pytest.raises((RuntimeError, ValueError)):
        engine.metadata()
    engine.close()
    assert policy._closed
    with pytest.raises(RuntimeError, match='closed'):
        engine._live_policy()


def test_close_is_exclusive_retries_failures_and_releases_strong_owners(deployed, monkeypatch):
    policy, engine, _ = deployed
    with policy._lock:
        with pytest.raises(RuntimeError, match='executing'):
            engine.close()
    original = policy.close
    calls = []

    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('injected graph retirement failure')
        original()

    monkeypatch.setattr(policy, 'close', fail_once)
    with pytest.raises(RuntimeError, match='retirement failure'):
        engine.close()
    assert engine._Bf16CheckpointEngine__policy is policy and not policy._closed
    with pytest.raises(RuntimeError):
        engine._live_policy()
    engine.close()
    engine.close()
    assert len(calls) == 2 and policy.closes == 1
    assert engine._Bf16CheckpointEngine__policy is None and engine._Bf16CheckpointEngine__state is None


@pytest.mark.parametrize('kwargs', [dict(precision='mixed_bf16_fp8_prefix'), dict(max_graphs=True), dict(max_graphs=0)])
def test_invalid_factory_arguments_fail_before_model_load(kwargs, monkeypatch):
    monkeypatch.setattr(Pi05LitePolicy, 'from_pretrained', classmethod(lambda *args, **kw: pytest.fail('Loaded model')))
    with pytest.raises(ValueError):
        bf16.Bf16CheckpointEngine.from_pretrained('/checkpoint', tokenizer='/tokenizer',
            origin_identity=ORIGINS['l4'], hardware_profile='l4', **kwargs)
