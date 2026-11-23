"""Remote CPU contracts for three request-local Python variable bindings."""
from collections import Counter
from functools import partial
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP

from vvla_repro_pi05.lite import fused_adarms as adarms, fused_expert_mlp as mlp_helper
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05AdaRMSNorm


@pytest.fixture(scope='module')
def baseline():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/pi05-lite-bf16-20261105/profiling/validation_bindings_reference.py'
    spec = importlib.util.spec_from_file_location('validation_bindings_reference_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm(conditioned=True):
    return Pi05AdaRMSNorm(8, cond_dim=8 if conditioned else None).eval()


def outcome(function, *args):
    try:
        return ('return', function(*args))
    except Exception as exc:
        return ('error', type(exc), exc.args)


@pytest.mark.parametrize('conditioned', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_valid_norm_branches_and_return_device_match(conditioned, dtype, baseline):
    module = norm(conditioned).to(dtype)
    assert adarms._validate_norm(module) == baseline._validate_norm(module) == torch.device('cpu')


@pytest.mark.parametrize('change', ['type', 'dim_bool', 'dim_range', 'eps_nan', 'eps_zero',
                                  'dense_type', 'weight_shape', 'bias_none', 'bias_shape',
                                  'weight_dtype', 'bias_dtype', 'bias_device', 'double_invalid'])
def test_conditioned_norm_live_mutations_match_original_error_type_message_order(change, baseline):
    module = norm()
    assert adarms._validate_norm(module) == baseline._validate_norm(module)
    if change == 'type': module = nn.Linear(8, 8)
    elif change == 'dim_bool': module.dim = True
    elif change == 'dim_range': module.dim = 0
    elif change == 'eps_nan': module.eps = float('nan')
    elif change == 'eps_zero': module.eps = 0
    elif change == 'dense_type': module.dense = nn.Identity()
    elif change == 'weight_shape': module.dense.weight = nn.Parameter(torch.empty(23, 8))
    elif change == 'bias_none': module.dense.bias = None
    elif change == 'bias_shape': module.dense.bias = nn.Parameter(torch.empty(23))
    elif change == 'weight_dtype': module.dense.weight = nn.Parameter(module.dense.weight.double())
    elif change == 'bias_dtype': module.dense.bias = nn.Parameter(module.dense.bias.double())
    elif change == 'bias_device': module.dense.bias = nn.Parameter(module.dense.bias.to('meta'))
    elif change == 'double_invalid':
        module.dense.weight = nn.Parameter(torch.empty(23, 8))
        module.dense.bias = None
    expected = outcome(baseline._validate_norm, module)
    assert expected[0] == 'error'
    assert outcome(adarms._validate_norm, module) == expected


@pytest.mark.parametrize('change', ['weight_none', 'weight_shape', 'weight_dtype', 'dense_transition', 'eps'])
def test_unconditioned_norm_and_branch_transitions_are_live(change, baseline):
    module = norm(False)
    assert adarms._validate_norm(module) == baseline._validate_norm(module)
    if change == 'weight_none': module.weight = None
    elif change == 'weight_shape': module.weight = nn.Parameter(torch.empty(7))
    elif change == 'weight_dtype': module.weight = nn.Parameter(module.weight.double())
    elif change == 'dense_transition':
        module.cond_dim = 8
        module.dense = nn.Linear(8, 24)
    elif change == 'eps': module.eps = -1
    assert outcome(adarms._validate_norm, module) == outcome(baseline._validate_norm, module)


@pytest.mark.parametrize('change', ['type', 'rank', 'width', 'dtype', 'device', 'replacement', 'empty'])
def test_consumer_projection_contract_keeps_all_live_checks(change, baseline):
    projections = [nn.Linear(8, 16, bias=False, dtype=torch.bfloat16) for _ in range(2)]
    arguments = (8, torch.bfloat16, torch.device('cpu'))
    assert adarms._projection_contract(projections, *arguments) is None
    if change == 'type': projections[0] = nn.Identity()
    elif change == 'rank': projections[0].weight = nn.Parameter(torch.empty(8, dtype=torch.bfloat16))
    elif change == 'width': projections[0].weight = nn.Parameter(torch.empty(16, 7, dtype=torch.bfloat16))
    elif change == 'dtype': projections[0].float()
    elif change == 'device': projections[0].to('meta')
    elif change == 'replacement': projections[0] = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    elif change == 'empty': projections = []
    assert outcome(adarms._projection_contract, projections, *arguments) == outcome(
        baseline._projection_contract, projections, *arguments)


@pytest.mark.parametrize('change', ['activation', 'activation_function', 'activation_forward', 'projection',
                                  'projection_forward', 'forward_hook', 'pre_hook', 'remove_hook'])
def test_behavior_identity_is_fresh_for_each_call(change, baseline):
    module = GemmaMLP(GemmaConfig(hidden_size=8, intermediate_size=16))
    if change == 'remove_hook': module.gate_proj._forward_hooks[17] = lambda *args: None
    before = mlp_helper._behavior_identity(module)
    assert before == baseline._behavior_identity(module)
    if change == 'activation': module.act_fn = type(module.act_fn)()
    elif change == 'activation_function': module.act_fn.act = partial(torch.nn.functional.gelu, approximate='none')
    elif change == 'activation_forward': module.act_fn.forward = lambda x: x
    elif change == 'projection': module.up_proj = nn.Linear(8, 16, bias=False)
    elif change == 'projection_forward': module.gate_proj.forward = lambda x: x
    elif change == 'forward_hook': module.gate_proj._forward_hooks[17] = lambda *args: None
    elif change == 'pre_hook': module.gate_proj._forward_pre_hooks[17] = lambda *args: None
    elif change == 'remove_hook': module.gate_proj._forward_hooks.clear()
    after = mlp_helper._behavior_identity(module)
    assert after != before and after == baseline._behavior_identity(module)


def test_exact_attribute_counts_for_18_norm_36_consumer_36_behavior_workload(monkeypatch, baseline):
    with torch.device('meta'):
        norms = [Pi05AdaRMSNorm(1024, cond_dim=1024) for _ in range(18)]
        mlps = [GemmaMLP(GemmaConfig(hidden_size=1024, intermediate_size=4096)).to(torch.bfloat16)
                for _ in range(18)]
    consumers = [(module.up_proj, module.gate_proj) for module in mlps]
    original = nn.Module.__getattr__
    counts, active = Counter(), ['baseline']
    def counted(module, name):
        counts[active[0]] += 1
        return original(module, name)
    monkeypatch.setattr(nn.Module, '__getattr__', counted)
    for name, norm_fn, projection_fn, behavior_fn in (
            ('baseline', baseline._validate_norm, baseline._projection_contract, baseline._behavior_identity),
            ('candidate', adarms._validate_norm, adarms._projection_contract, mlp_helper._behavior_identity)):
        active[0] = name
        for module in norms: norm_fn(module)
        for pair in consumers: projection_fn(pair, 1024, torch.bfloat16, torch.device('meta'))
        for _ in range(2):
            for module in mlps: behavior_fn(module)
    assert counts == {'baseline': 576, 'candidate': 234}
