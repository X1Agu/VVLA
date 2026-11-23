"""OFT native state candidate contracts; execute only in root's remote queue.

CPU tests build the small host extension explicitly. The deferred CUDA test uses
the real TensorGraphRunner, not a replacement runner or an external graph owner.
No benchmark/quality or latency assertion is made by these contracts.
"""
from collections import OrderedDict
import gc
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workers/oft/src'))
from vvla_runtime.execution.state import ModelStateContract
from vvla_runtime.execution.native_state import (
    NativeStateComparator, _compiler_provenance, load_native_state_comparator,
)


@pytest.fixture(scope='module')
def native():
    initialized = torch.cuda.is_initialized()
    result = load_native_state_comparator(verbose=True)
    assert torch.cuda.is_initialized() == initialized
    return result


def model_fixture():
    model = nn.Module()
    model.left = nn.Linear(4, 4, bias=False)
    model.right = nn.Sequential(nn.Linear(4, 4), nn.Identity())
    model.alias = model.left
    model.empty = nn.Identity()
    model.register_buffer('buffer', torch.arange(4, dtype=torch.float32))
    model.register_buffer('none_buffer', None)
    model.register_parameter('none_parameter', None)
    return model.eval()


def pair(model, native, **kwargs):
    reference = ModelStateContract(model, **kwargs)
    candidate = ModelStateContract(model, native_comparator=native, **kwargs)
    assert reference.signature() == candidate.signature()
    return reference, candidate


class NoTensorWork(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError('State comparison performed a tensor operation: ' + str(func))


def test_default_is_original_and_does_not_load(monkeypatch):
    import vvla_runtime.execution.native_state as loader
    monkeypatch.setattr(loader, 'load_native_state_comparator', lambda **kwargs: pytest.fail('lazy build'))
    assert inspect.signature(ModelStateContract).parameters['native_comparator'].default is None
    contract = ModelStateContract(model_fixture())
    token = contract.signature()
    assert contract.signature() is token and contract.native_comparator is None
    with pytest.raises(TypeError, match='explicitly loaded'):
        ModelStateContract(model_fixture(), native_comparator=True)


def test_manifest_and_equal_state_reuse_original_token_without_tensor_work(native):
    model = model_fixture()
    reference, candidate = pair(model, native)
    token = candidate._cached_token
    with NoTensorWork():
        for _ in range(3):
            assert native.compare(candidate) is True
            assert candidate.signature() is token
            assert candidate._registered_state_unchanged_python() is True
            assert candidate.signature() == reference.signature()
    manifest = native.manifest()
    assert manifest['default_enabled'] is False and manifest['with_cuda'] is False
    assert manifest['inventory_cache'] is False and manifest['tensor_content_reads'] is False
    assert 'stride' in manifest['tensor_fields']
    assert set(manifest['source_sha256']) == {'native_state.cpp', 'native_state.py', 'state.py'}
    assert len(manifest['binary_sha256']) == 64
    assert manifest['compiler_command']['executables'][-1]['role'] == 'compiler'
    manifest['source_sha256'].clear()
    assert native.manifest()['source_sha256']


@pytest.mark.parametrize('deep_mro', [False, True])
def test_builtin_object_dict_is_supported_without_fallback_or_reference_leak(native, deep_mro):
    """CPython 3.12 object.tp_dict is not its interpreter-owned namespace.

    The original direct PyDict_GetItemString(object.tp_dict, ...) crashed even
    for empty nn.Module. Require the real native result, not a None fallback;
    also check ownership of each new reference returned by PyType_GetDict.
    """
    class Middle(nn.Module):
        pass
    class Leaf(Middle):
        pass

    model = (Leaf() if deep_mro else nn.Module()).eval()
    if deep_mro:
        model.weight = nn.Parameter(torch.ones(2))
        model.register_buffer('buffer', torch.zeros(2))
    reference, candidate = pair(model, native)
    token = candidate._cached_token
    assert type(model).__mro__[-1] is object
    dictionaries = []
    seen = set()
    for kind in (type(model), torch.Tensor, nn.Parameter):
        for base in kind.__mro__:
            # Mappingproxy exposes its raw dictionary to the GC without
            # calling a custom type descriptor or relying on ctypes layout.
            referents = gc.get_referents(base.__dict__)
            assert len(referents) == 1 and type(referents[0]) is dict
            dictionary = referents[0]
            if id(dictionary) not in seen:
                seen.add(id(dictionary))
                dictionaries.append(dictionary)
    before = [sys.getrefcount(value) for value in dictionaries]
    candidate._registered_state_unchanged_python = lambda: pytest.fail('unexpected Python fallback')
    with NoTensorWork():
        for _ in range(32):
            assert native.compare(candidate) is True
            assert candidate.signature() is token
    assert [sys.getrefcount(value) for value in dictionaries] == before
    assert candidate.signature() == reference.signature()


@pytest.mark.parametrize('mutation', ['add', 'copy', 'pointer', 'shape', 'stride',
                                     'dtype', 'device', 'parameter', 'buffer'])
def test_every_live_tensor_identity_field_invalidates_and_matches_python(native, mutation):
    model = model_fixture()
    reference, candidate = pair(model, native)
    old = candidate._cached_token
    weight = model.left.weight
    with torch.no_grad():
        if mutation == 'add': weight.add_(1)
        elif mutation == 'copy': weight.copy_(weight.clone())
        elif mutation == 'pointer': weight.data = weight.data.clone()
        elif mutation == 'shape': weight.data = weight.data.view(2, 8)
        elif mutation == 'stride':
            pointer, version, shape = weight.data_ptr(), weight._version, weight.shape
            weight.data = weight.data.t()
            assert (weight.data_ptr(), weight._version, weight.shape) == (pointer, version, shape)
        elif mutation == 'dtype': model.to(torch.bfloat16)
        elif mutation == 'device': model.to('meta')
        elif mutation == 'parameter': model.left.weight = nn.Parameter(weight.clone())
        elif mutation == 'buffer': model.buffer.add_(1)
    assert native.compare(candidate) is False
    assert candidate._registered_state_unchanged_python() is False
    actual = candidate.signature()
    assert actual == reference.signature() and actual != old
    assert candidate.signature() is actual


@pytest.mark.parametrize('mutation', ['registry', 'order', 'none', 'alias', 'delete',
                                     'empty', 'module', 'type', 'type_name', 'type_module'])
def test_live_topology_and_registry_identity_refresh(native, mutation):
    class Local(nn.Module):
        pass
    model = model_fixture()
    model.extra = Local().eval()
    reference, candidate = pair(model, native)
    old = candidate._cached_token
    old_name, old_module = Local.__qualname__, Local.__module__
    try:
        if mutation == 'registry': model._modules = dict(model._modules)
        elif mutation == 'order': model._modules['left'] = model._modules.pop('left')
        elif mutation == 'none': model.register_parameter('new_none', None)
        elif mutation == 'alias': model.alias = model.right
        elif mutation == 'delete': del model._buffers['none_buffer']
        elif mutation == 'empty': model.empty = nn.Identity().eval()
        elif mutation == 'module': model.right = nn.Sequential(nn.Linear(4, 4), nn.Identity()).eval()
        elif mutation == 'type':
            class Other(nn.Module): pass
            model.extra.__class__ = Other
        elif mutation == 'type_name': Local.__qualname__ = 'ChangedLocal'
        elif mutation == 'type_module': Local.__module__ = 'changed_oft_test_module'
        assert native.compare(candidate) is False
        assert candidate._registered_state_unchanged_python() is False
        actual = candidate.signature()
        assert actual == reference.signature() and actual is not old
        assert candidate.signature() is actual
        # Replacing a dict with the same entries has always rebuilt an equal
        # token. The native candidate must not invent new invalidation policy.
        if mutation == 'registry': assert actual == old
    finally:
        Local.__qualname__, Local.__module__ = old_name, old_module


def test_training_and_inference_tensor_rejections_preserve_error_boundary(native):
    model = model_fixture()
    reference, candidate = pair(model, native)
    model.empty.train()
    for contract in (reference, candidate):
        with pytest.raises(ValueError, match='eval mode'): contract.signature()
    model.eval()
    with torch.inference_mode(): bad = torch.ones(4)
    model.register_buffer('bad', bad)
    for contract in (reference, candidate):
        with pytest.raises(ValueError, match='version counters'): contract.signature()
    del model.bad
    assert reference.signature() == candidate.signature()
    model.train()
    with pytest.raises(ValueError, match='Graph inference model'): candidate.signature()


def test_recipe_and_adapter_validation_stay_live(native):
    state = dict(recipe=1, validations=0, fail=False)
    def validate():
        state['validations'] += 1
        if state['fail']: raise RuntimeError('rotary basis changed')
    model = model_fixture()
    contract = ModelStateContract(model, native_comparator=native,
                                  validate=validate, recipe=lambda: {'version': state['recipe']})
    old = contract.signature()
    assert contract.signature() is old and state['validations'] == 2
    state['recipe'] = 2
    assert contract.signature() != old and state['validations'] == 3
    state['fail'] = True
    with pytest.raises(RuntimeError, match='rotary basis'): contract.signature()
    assert state['validations'] == 4


def test_custom_named_enumerator_and_instance_override_keep_full_python_path(native):
    class Custom(nn.Module):
        def named_parameters(self, *args, **kwargs):
            yield from super().named_parameters(*args, **kwargs)
    custom = Custom().eval()
    custom.weight = nn.Parameter(torch.ones(2))
    contract = ModelStateContract(custom, native_comparator=native)
    first = contract.signature()
    assert contract._cached_token is None
    assert contract.signature() == first and contract._cached_token is None
    ordinary = model_fixture()
    _, contract = pair(ordinary, native)
    ordinary.named_buffers = lambda *args, **kwargs: iter(())
    assert native.compare(contract) is False
    assert contract.signature() == ModelStateContract(ordinary).signature()
    assert contract._cached_token is None


def test_registry_subclass_and_custom_key_use_python_fallback(native):
    model = model_fixture()
    model._buffers = OrderedDict(model._buffers)
    _, contract = pair(model, native)
    assert native.compare(contract) is None
    assert contract._registered_state_unchanged() is True
    events = []
    class Key(str):
        __hash__ = str.__hash__
        def __ne__(self, other):
            events.append('ne')
            return super().__ne__(other)
    model = model_fixture()
    model._buffers = {Key(k): v for k, v in model._buffers.items()}
    _, contract = pair(model, native)
    events.clear()
    assert native.compare(contract) is None and events == []
    assert contract._registered_state_unchanged() is True and events


def test_custom_module_accessors_fall_back_before_any_original_getter(native):
    events = []
    class Custom(nn.Module):
        def __getattribute__(self, name):
            if name in ('training', '__dict__', '_parameters', '_modules', '_buffers'):
                events.append(name)
            return super().__getattribute__(name)
    model = Custom().eval()
    model.weight = nn.Parameter(torch.ones(2))
    _, contract = pair(model, native)
    events.clear()
    assert native.compare(contract) is None and events == []
    assert contract._registered_state_unchanged_python() is True
    expected = list(events)
    events.clear()
    assert contract._registered_state_unchanged() is True and events == expected


@pytest.mark.parametrize('kind', [torch.Tensor, nn.Parameter])
def test_exact_tensor_class_getattribute_override_is_not_bypassed(native, monkeypatch, kind):
    model = model_fixture()
    _, contract = pair(model, native)
    original = kind.__getattribute__
    events = []
    def changed(self, name):
        if name in ('_version', 'data_ptr', 'dtype', 'device', 'shape', 'stride'): events.append(name)
        return original(self, name)
    monkeypatch.setattr(kind, '__getattribute__', changed)
    assert native.compare(contract) is None and events == []
    assert contract._registered_state_unchanged() is True and events


def test_unknown_tensor_descriptor_not_invoked_by_eligibility(native, monkeypatch):
    model = model_fixture()
    _, contract = pair(model, native)
    events = []
    class Broken:
        def __get__(self, instance, owner):
            events.append('class' if instance is None else 'instance')
            raise RuntimeError('descriptor access')
    monkeypatch.setattr(nn.Parameter, '_version', Broken())
    assert native.compare(contract) is None and events == []
    model.empty.train()
    with pytest.raises(ValueError, match='eval mode'): contract._registered_state_unchanged()
    assert events == []  # original training rejection precedes all tensor reads
    model.eval()
    # The original Python _tensor_version wraps RuntimeError from the getter;
    # native eligibility must preserve that boundary and its original cause.
    with pytest.raises(ValueError, match='Persistent model tensors need version counters') as caught:
        contract._registered_state_unchanged()
    assert type(caught.value.__cause__) is RuntimeError
    assert str(caught.value.__cause__) == 'descriptor access'
    assert events == ['instance']


def test_tensor_subclass_and_instance_shadow_fallback(native):
    class Subclass(torch.Tensor): pass
    model = model_fixture()
    model.buffer = model.buffer.as_subclass(Subclass)
    _, contract = pair(model, native)
    assert native.compare(contract) is None
    assert contract._registered_state_unchanged() is True
    model = model_fixture()
    _, contract = pair(model, native)
    pointer = model.left.weight.data_ptr()
    model.left.weight.data_ptr = lambda: pointer
    assert native.compare(contract) is None
    assert contract._registered_state_unchanged() is True
    model.left.weight.data_ptr = lambda: pointer + 64
    assert contract._registered_state_unchanged() is False


def test_torch_function_mode_and_class_hook_do_not_bypass_python(native, monkeypatch):
    from torch.overrides import TorchFunctionMode
    model = model_fixture()
    _, contract = pair(model, native)
    events = []
    class Mode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            events.append(function)
            return function(*args, **(kwargs or {}))
    with Mode():
        assert native.compare(contract) is None and events == []
        assert contract._registered_state_unchanged_python() is True
        expected = list(events)
        events.clear()
        assert contract._registered_state_unchanged() is True and events == expected
    # Eligibility detects the actual class binding without invoking it.
    def unexpected(*args, **kwargs): raise RuntimeError('class torch-function hook')
    monkeypatch.setattr(nn.Parameter, '__torch_function__', unexpected)
    assert native.compare(contract) is None


def test_native_has_no_model_ownership_and_unexpected_errors_do_not_fallback(native):
    model = model_fixture()
    _, contract = pair(model, native)
    reference = weakref.ref(model)
    assert native.compare(contract) is True
    del model, contract, _
    assert reference() is None
    def fail(*args): raise RuntimeError('injected native failure')
    broken = NativeStateComparator(SimpleNamespace(compare=fail), {}, ())
    contract = ModelStateContract(model_fixture(), native_comparator=broken)
    contract.signature()
    contract._registered_state_unchanged_python = lambda: pytest.fail('unexpected silent fallback')
    with pytest.raises(RuntimeError, match='injected native'): contract.signature()


def test_compiler_receipt_resolves_actual_wrapper_and_driver(monkeypatch, tmp_path):
    import vvla_runtime.execution.native_state as loader
    wrapper, driver = tmp_path/'ccache', tmp_path/'c++'
    wrapper.write_bytes(b'wrapper fixture'); driver.write_bytes(b'compiler fixture')
    monkeypatch.delenv('CCACHE_PREFIX', raising=False)
    monkeypatch.delenv('CCACHE_PREFIX_CPP', raising=False)
    monkeypatch.setattr(loader.shutil, 'which', lambda word: str({'ccache': wrapper, 'c++': driver}[word]))
    receipt = _compiler_provenance('ccache c++')
    assert [row['role'] for row in receipt['executables']] == ['wrapper', 'compiler']
    assert receipt['executables'][1]['resolved_path'] == str(driver.resolve())
    assert receipt['executables'][1]['sha256'] == hashlib.sha256(driver.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='explicit compiler'): _compiler_provenance('ccache')
    monkeypatch.setenv('CCACHE_PREFIX', 'distcc')
    with pytest.raises(ValueError, match='prefix'): _compiler_provenance('c++')


def test_explicit_loader_build_failure_never_becomes_python_candidate(monkeypatch):
    import torch.utils.cpp_extension as cpp
    import vvla_runtime.execution.native_state as loader
    monkeypatch.setattr(loader, '_compiler_provenance', lambda command: {'fixture': True})
    def failure(**kwargs): raise RuntimeError('compiler unavailable')
    monkeypatch.setattr(cpp, 'load', failure)
    with pytest.raises(RuntimeError, match='candidate rejected'):
        load_native_state_comparator()


def test_real_cpu_runner_keeps_generation_validation_and_owned_output(native):
    from vvla_runtime.execution.graph import TensorGraphRunner
    model = nn.Linear(4, 4, bias=False).eval()
    contract = ModelStateContract(model, native_comparator=native)
    runner = TensorGraphRunner(model, revision='state-contract-test', state_contract=contract)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    try:
        first = runner(input=x)
        saved = first.clone()
        assert runner._state_generation == 1
        assert torch.equal(runner(input=x), first)
        with runner.mutation_scope():
            with torch.no_grad(): model.weight.add_(1)
        changed = runner(input=x)
        assert runner._state_generation == 2
        assert torch.equal(first, saved) and not torch.equal(changed, first)
        model.train()
        with pytest.raises(ValueError, match='Graph inference model'): runner(input=x)
        assert runner.stats['state_rejections'] == 1
    finally:
        runner.close()


def test_cuda_real_runner_recapture_cross_stream_and_retained_outputs(native):
    if not torch.cuda.is_available(): pytest.skip('root schedules real CUDA contract')
    from vvla_runtime.execution.graph import TensorGraphRunner
    model = nn.Linear(4, 4, bias=False, device='cuda', dtype=torch.bfloat16).eval()
    runner = TensorGraphRunner(model, revision='state-contract-cuda',
        state_contract=ModelStateContract(model, native_comparator=native))
    x = torch.arange(8, device='cuda', dtype=torch.bfloat16).reshape(2, 4)
    try:
        first = runner(input=x)
        first_cpu = first.cpu().clone()
        other = torch.cuda.Stream()
        with torch.cuda.stream(other): second = runner(input=x)
        other.synchronize()
        assert torch.equal(second.cpu(), first_cpu)
        assert first.data_ptr() != second.data_ptr()
        assert runner.stats['stream_handoffs'] >= 1
        with runner.mutation_scope():
            with torch.no_grad(): model.weight.add_(1)
        changed = runner(input=x)
        assert runner._state_generation == 2 and runner.stats['captures'] == 2
        assert not torch.equal(changed.cpu(), first_cpu)
        assert torch.equal(first.cpu(), first_cpu)
        # Retire work under the same runner lock, but leave the old token to
        # verify actual same-pointer/same-shape stride invalidation on next call.
        with runner._call_lock:
            runner._retire()
            model.weight.data = model.weight.data.t()
        runner(input=x)
        assert runner._state_generation == 3 and runner.stats['captures'] == 3
        assert runner.stats['state_invalidations'] == 1
        model.train()
        with pytest.raises(ValueError, match='Graph inference model'): runner(input=x)
        assert runner.entries == {} and runner.stats['state_rejections'] == 1
    finally:
        runner.close()
