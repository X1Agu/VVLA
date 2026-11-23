"""CPU regressions for fast signatures and graph cache invalidation."""
from collections import OrderedDict
from types import MethodType

import pytest
import torch
from torch import nn

from vvla_runtime.execution.graph import TensorGraphRunner
from vvla_runtime.execution.state import ModelStateContract


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.register_buffer('offset', torch.zeros(3))
        self.register_buffer('optional', None)
        self.recipe = {'mode': 'bf16'}

    def forward(self, x):
        return self.projection(x) + self.offset

    def execution_signature(self):
        return self.recipe


def model_and_contract():
    model = TinyModel().eval()
    return model, ModelStateContract(model)


def test_stable_signature_reuses_token_and_keeps_validation_live():
    model, contract = model_and_contract()
    validations = []
    contract.validate = lambda: validations.append(True)
    first = contract.signature()
    assert contract.signature() is first
    assert len(validations) == 2


@pytest.mark.parametrize('kind', ['parameter', 'buffer', 'module'])
@pytest.mark.parametrize('change', ['add', 'delete', 'replace'])
def test_registration_mutation_refreshes_signature(kind, change):
    model, contract = model_and_contract()
    original = contract.signature()
    owner = model.projection if kind == 'parameter' else model
    name = {'parameter': 'weight', 'buffer': 'offset', 'module': 'projection'}[kind]
    new = {'parameter': lambda: nn.Parameter(torch.ones(3, 3)),
           'buffer': lambda: torch.ones(3),
           'module': lambda: nn.Linear(3, 3).eval()}[kind]
    if change == 'delete':
        delattr(owner, name)
    elif change == 'replace':
        setattr(owner, name, new())
    elif kind == 'parameter':
        owner.register_parameter('new_parameter', new())
    elif kind == 'buffer':
        owner.register_buffer('new_buffer', new())
    else:
        owner.add_module('new_module', new())
    changed = contract.signature()
    assert changed != original
    assert contract.signature() is changed


def test_ordered_module_reordering_is_not_treated_as_unchanged():
    model = nn.Sequential(OrderedDict([
        ('linear', nn.Linear(3, 3)), ('relu', nn.ReLU())
    ])).eval()
    contract = ModelStateContract(model)
    before = contract.signature()
    # Sequential semantics depend on insertion order even when identities match.
    module = model._modules.pop('linear')
    model._modules['linear'] = module
    assert contract.signature() != before


def test_equal_value_parameter_rebinding_still_invalidates():
    model, contract = model_and_contract()
    before = contract.signature()
    model.projection.weight = nn.Parameter(model.projection.weight.detach().clone())
    assert contract.signature() != before


@pytest.mark.parametrize('target', ['parameter', 'buffer'])
@pytest.mark.parametrize('change', ['value', 'shape', 'storage'])
def test_tensor_versions_invalidate_for_ordinary_mutations(target, change):
    model, contract = model_and_contract()
    tensor = model.projection.weight if target == 'parameter' else model.offset
    before = contract.signature()
    with torch.no_grad():
        if change == 'value':
            tensor.add_(1)
        elif change == 'shape':
            tensor.unsqueeze_(0)
        else:
            tensor.set_(torch.ones_like(tensor))
    assert contract.signature() != before


def test_training_recipe_and_inference_tensors_are_not_hidden_by_cache():
    model, contract = model_and_contract()
    before = contract.signature()
    model.projection.train()
    with pytest.raises(ValueError, match='submodules'):
        contract.signature()
    model.eval()
    assert contract.signature() == before
    model.recipe['mode'] = 'fp8'
    assert contract.signature() != before
    with torch.inference_mode():
        model.offset = torch.zeros(3)
    with pytest.raises(ValueError, match='version counters'):
        contract.signature()


def test_validate_rejection_still_retires_graph_runner_state():
    model, contract = model_and_contract()
    runner = TensorGraphRunner(model, revision='test', use_graph=False, state_contract=contract)
    runner(x=torch.zeros(1, 3))
    # CPU-owned dummy cache exercises retirement without allocating CUDA objects.
    runner.entries['stale'] = ([torch.zeros(1)], None, torch.zeros(1))
    def reject():
        raise RuntimeError('packed weight is stale')
    contract.validate = reject
    with pytest.raises(RuntimeError, match='packed weight'):
        runner(x=torch.zeros(1, 3))
    assert not runner.entries
    assert runner.stats['state_rejections'] == 1


def test_mutation_scope_retires_and_recaptures_current_state():
    model, contract = model_and_contract()
    runner = TensorGraphRunner(model, revision='test', use_graph=False, state_contract=contract)
    x = torch.ones(1, 3)
    before = runner(x=x)
    generation = runner._state_generation
    with runner.mutation_scope(), torch.no_grad():
        model.offset.add_(2)
        model.projection = nn.Linear(3, 3).eval()
    actual = runner(x=x)
    torch.testing.assert_close(actual, model(x))
    assert not torch.equal(actual, before)
    assert runner._state_generation == generation + 1
    assert runner.stats['explicit_mutations'] == 1
    runner.close()


def test_custom_enumerators_keep_uncached_behavior_even_when_installed_later():
    model, contract = model_and_contract()
    first = contract.signature()
    original = model.named_buffers
    model.hide_offset = False
    def named_buffers(self, *args, **kwargs):
        for name, tensor in original(*args, **kwargs):
            if not self.hide_offset or name != 'offset':
                yield name, tensor
    model.named_buffers = MethodType(named_buffers, model)
    assert contract.signature() == first
    model.hide_offset = True
    assert contract.signature() != first
    model.hide_offset = False
    assert contract.signature() == first


def test_shared_modules_and_parameters_preserve_duplicate_names():
    model = nn.Module()
    shared = nn.Linear(3, 3)
    model.left = model.right = shared
    model.eval()
    contract = ModelStateContract(model)
    before = contract.signature()
    names = [entry[1] for entry in before[2]]
    assert names == ['left.weight', 'left.bias', 'right.weight', 'right.bias']
    model.right = nn.Linear(3, 3).eval()
    assert contract.signature() != before



def test_normal_persistent_tensor_mutation_under_inference_mode_is_detected():
    model, contract = model_and_contract()
    before = contract.signature()
    with torch.inference_mode():
        model.offset.add_(1)
    assert contract.signature() != before


@pytest.mark.parametrize('change', ['dtype', 'clone', 'transpose'])
def test_module_apply_parameter_metadata_changes_without_version_bumps(change):
    model, contract = model_and_contract()
    before = contract.signature()
    parameter = model.projection.weight
    version = parameter._version
    # Touch only the projection: replacing root buffers would mask the
    # Parameter failure where PyTorch preserves identity and version.
    if change == 'dtype':
        model.projection.to(dtype=torch.float64)
    elif change == 'clone':
        model.projection._apply(lambda tensor: tensor.clone())
    else:
        model.projection._apply(lambda tensor: tensor.t() if tensor.ndim == 2 else tensor)
    assert model.projection.weight is parameter
    assert parameter._version == version
    changed = contract.signature()
    assert changed != before
    assert contract.signature() is changed


def test_module_to_meta_device_invalidates_without_a_gpu():
    model, contract = model_and_contract()
    before = contract.signature()
    model.projection.to(device='meta')
    assert contract.signature() != before
