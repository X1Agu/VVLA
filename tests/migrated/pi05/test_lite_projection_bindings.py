"""Remote CPU contracts for request-local QKV/MLP weight bindings."""
from collections import Counter
from functools import partial
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP

from vvla_repro_pi05.lite import fused_qkv as qkv, fused_expert_mlp as mlp


@pytest.fixture(scope='module')
def baseline():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/pi05-lite-bf16-20261105/profiling/projection_bindings_reference.py'
    spec = importlib.util.spec_from_file_location('projection_bindings_reference_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attention():
    result = nn.Module()
    result.head_dim = 4
    for name, width in [('q_proj', 16), ('k_proj', 4), ('v_proj', 4)]:
        setattr(result, name, nn.Linear(8, width, bias=False, dtype=torch.bfloat16))
    return result


def expert():
    return GemmaMLP(GemmaConfig(hidden_size=8, intermediate_size=16)).to(torch.bfloat16)


def outcome(function, argument):
    try:
        return 'return', function(argument)
    except Exception as error:
        return 'error', type(error), error.args


@pytest.mark.parametrize('change', ['projection_type', 'bias', 'input_width', 'kv_width', 'gqa_ratio',
                                  'head_dim', 'head_dim_zero', 'dtype', 'device', 'stride', 'weight_none',
                                  'missing_later', 'geometry_before_missing', 'dtype_before_missing',
                                  'stride_before_missing'])
def test_qkv_live_fields_and_original_exception_order(change, baseline):
    module = attention()
    assert qkv._projections(module) == baseline.qkv_projections(module)
    if change == 'projection_type': module.q_proj = nn.Identity()
    elif change == 'bias': module.q_proj.bias = nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    elif change == 'input_width': module.k_proj.in_features = 7
    elif change == 'kv_width': module.v_proj.out_features = 8
    elif change == 'gqa_ratio': module.q_proj.out_features = 15
    elif change == 'head_dim': module.head_dim = 3
    elif change == 'head_dim_zero': module.head_dim = 0
    elif change == 'dtype': module.v_proj.float()
    elif change == 'device': module.v_proj.to('meta')
    elif change == 'stride': module.k_proj.weight = nn.Parameter(torch.empty(8, 4, dtype=torch.bfloat16).T)
    elif change == 'weight_none': module.q_proj.weight = None
    elif change == 'missing_later': del module.v_proj.weight
    elif change == 'geometry_before_missing':
        module.q_proj.in_features = 9
        del module.q_proj.weight
    elif change == 'dtype_before_missing':
        module.q_proj.float()
        del module.k_proj.weight
    elif change == 'stride_before_missing':
        module.q_proj.weight = nn.Parameter(torch.empty(8, 16, dtype=torch.bfloat16).T)
        del module.k_proj.weight
    expected = outcome(baseline.qkv_projections, module)
    assert expected[0] == 'error'
    assert outcome(qkv._projections, module) == expected


@pytest.mark.parametrize('change', ['mlp_type', 'activation_type', 'activation_function', 'activation_forward',
                                  'projection_type', 'bias', 'forward', 'hook', 'pre_hook', 'gate_up_shape',
                                  'down_shape', 'dtype', 'device', 'stride', 'gate_none_before_missing',
                                  'shape_before_missing_down', 'dtype_with_missing_down'])
def test_mlp_live_fields_and_original_exception_order(change, baseline):
    module = expert()
    assert mlp._projections(module) == baseline.mlp_projections(module)
    if change == 'mlp_type': module = nn.Identity()
    elif change == 'activation_type': module.act_fn = nn.ReLU()
    elif change == 'activation_function': module.act_fn.act = partial(torch.nn.functional.gelu, approximate='none')
    elif change == 'activation_forward': module.act_fn.forward = lambda x: x
    elif change == 'projection_type': module.down_proj = nn.Identity()
    elif change == 'bias': module.gate_proj.bias = nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    elif change == 'forward': module.up_proj.forward = lambda x: x
    elif change == 'hook': module.down_proj._forward_hooks[7] = lambda *args: None
    elif change == 'pre_hook': module.down_proj._forward_pre_hooks[7] = lambda *args: None
    elif change == 'gate_up_shape': module.up_proj.weight = nn.Parameter(torch.empty(15, 8, dtype=torch.bfloat16))
    elif change == 'down_shape': module.down_proj.weight = nn.Parameter(torch.empty(8, 15, dtype=torch.bfloat16))
    elif change == 'dtype': module.up_proj.float()
    elif change == 'device': module.up_proj.to('meta')
    elif change == 'stride': module.up_proj.weight = nn.Parameter(torch.empty(8, 16, dtype=torch.bfloat16).T)
    elif change == 'gate_none_before_missing':
        module.gate_proj.weight = None
        del module.up_proj.weight
    elif change == 'shape_before_missing_down':
        module.up_proj.weight = nn.Parameter(torch.empty(15, 8, dtype=torch.bfloat16))
        del module.down_proj.weight
    elif change == 'dtype_with_missing_down':
        module.gate_proj.float()
        del module.down_proj.weight
    expected = outcome(baseline.mlp_projections, module)
    assert expected[0] == 'error'
    assert outcome(mlp._projections, module) == expected


@pytest.mark.parametrize('kind', ['qkv', 'mlp'])
def test_each_call_reads_current_weights_and_preserves_parameter_module_state_identity(kind, baseline):
    module = attention() if kind == 'qkv' else expert()
    current = qkv._projections if kind == 'qkv' else mlp._projections
    original = baseline.qkv_projections if kind == 'qkv' else baseline.mlp_projections
    module_ids = tuple(map(id, module.modules()))
    parameter_ids = tuple(map(id, module.parameters()))
    state = {key: value.clone() for key, value in module.state_dict().items()}
    for _ in range(2): assert current(module) == original(module)
    assert tuple(map(id, module.modules())) == module_ids
    assert tuple(map(id, module.parameters())) == parameter_ids
    assert state.keys() == module.state_dict().keys()
    for key, value in state.items(): assert torch.equal(value, module.state_dict()[key])
    projection = module.k_proj if kind == 'qkv' else module.up_proj
    saved = projection.weight
    projection.weight = nn.Parameter(saved.float())
    assert outcome(current, module) == outcome(original, module)
    assert outcome(current, module)[0] == 'error'
    projection.weight = saved
    assert current(module) == original(module)
    with torch.no_grad(): saved.add_(1)
    # These helpers validate geometry, not weight contents; the shared model
    # generation continues to own content-version invalidation.
    assert current(module) == original(module)


def test_exact_weight_read_reduction_at_18_real_layers(monkeypatch, baseline):
    from test_lite_expert_cublaslt import make_meta_case, close_case
    model, q_handle, m_handle, lt, runner = make_meta_case(layers=18)
    try:
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        original_getattr = nn.Module.__getattr__
        counts, mode = Counter(), ['baseline']
        def counted(module, name):
            if name == 'weight': counts[mode[0]] += 1
            return original_getattr(module, name)
        monkeypatch.setattr(nn.Module, '__getattr__', counted)
        for label, q_function, m_function in (
                ('baseline', baseline.qkv_projections, baseline.mlp_projections),
                ('candidate', qkv._projections, mlp._projections)):
            mode[0] = label
            for layer in layers:
                q_function(layer.self_attn)
                m_function(layer.mlp)
        assert counts == {'baseline': 18 * (12 + 15), 'candidate': 18 * (3 + 3)}
    finally:
        close_case(q_handle, m_handle, lt, runner)


@pytest.mark.parametrize('profile', ['g4_best_three', 'l4_exact_three'])
def test_actual_prepare_context_baseline_candidate_baseline_and_mutation_restore(monkeypatch, baseline, profile):
    from test_lite_expert_cublaslt import make_meta_case, close_case
    model, q_handle, m_handle, lt, runner = make_meta_case(profile=profile, layers=1)
    candidates = qkv._projections, mlp._projections
    args = dict(images=[torch.empty(1)], image_masks=[torch.ones(1, dtype=torch.bool)],
                lang_masks=torch.ones((1, 200), dtype=torch.bool), num_steps=10, positions=False,
                prepared_layout={'packing': None, 'empty_pattern': (False,)})
    try:
        outputs = []
        with runner._call_lock:
            for q_function, m_function in [(baseline.qkv_projections, baseline.mlp_projections),
                                            candidates, (baseline.qkv_projections, baseline.mlp_projections)]:
                monkeypatch.setattr(qkv, '_projections', q_function)
                monkeypatch.setattr(mlp, '_projections', m_function)
                outputs.append(model.prepare_inference_context(**args))
            assert outputs[0] == outputs[1] == outputs[2]
            layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
            saved = layer.mlp.up_proj.weight
            layer.mlp.up_proj.weight = nn.Parameter(saved.float())
            failures = []
            for q_function, m_function in [(baseline.qkv_projections, baseline.mlp_projections), candidates]:
                monkeypatch.setattr(qkv, '_projections', q_function)
                monkeypatch.setattr(mlp, '_projections', m_function)
                failures.append(outcome(lambda _: model.prepare_inference_context(**args), None))
            assert failures[0] == failures[1] and failures[0][0] == 'error'
            layer.mlp.up_proj.weight = saved
            restored = model.prepare_inference_context(**args)
            assert restored['token'][1] > outputs[-1]['token'][1]
    finally:
        monkeypatch.setattr(qkv, '_projections', candidates[0])
        monkeypatch.setattr(mlp, '_projections', candidates[1])
        close_case(q_handle, m_handle, lt, runner)
