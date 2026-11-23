"""No cached inventory: complete live tensor and topology checks on CPU/meta."""
import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity


def _model():
    model = nn.Module()
    model.left = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
    model.right = nn.Module()
    model.right.shared = model.left[0]
    model.right.alias = model.left[1].weight
    model.register_buffer("buffer", torch.arange(6).float().reshape(2, 3))
    model.right.register_buffer("alias_buffer", model.buffer, persistent=False)
    model.register_parameter("optional", None)
    model.register_buffer("empty", None)
    model.add_module("empty_child", None)
    return model


def _assert_native(model, signature):
    assert signature.parameters == tuple(map(tensor_identity, model.parameters()))
    assert signature.buffers == tuple(map(tensor_identity, model.buffers()))


def test_all_tensor_fields_native_order_shared_modules_and_no_tensor_ops():
    model = _model()
    class NoTensorWork(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"Signature executed tensor work: {func}")
    with NoTensorWork():
        signature = collect_lifecycle_signature(model)
        _assert_native(model, signature)
        assert collect_lifecycle_signature(model) == signature
    assert len(signature.parameters) == 3 and len(signature.buffers) == 1


@pytest.mark.parametrize("mutation", ["add", "copy", "data_pointer", "data_shape", "dtype", "device", "replace_parameter", "buffer_write"])
def test_detects_every_existing_tensor_identity_field(mutation):
    model = _model()
    original = collect_lifecycle_signature(model)
    weight = model.left[0].weight
    with torch.no_grad():
        if mutation == "add":
            weight.add_(1)
        elif mutation == "copy":
            weight.copy_(weight.clone())
        elif mutation == "data_pointer":
            weight.data = weight.data.clone()
        elif mutation == "data_shape":
            # Deliberately keep data_ptr and version unchanged: shape must still be read.
            before = weight.data_ptr(), weight._version
            weight.data = weight.data.view(2, 6)
            assert (weight.data_ptr(), weight._version) == before
        elif mutation == "dtype":
            model.to(dtype=torch.bfloat16)
        elif mutation == "device":
            model.to(device="meta")
        elif mutation == "replace_parameter":
            model.left[0].weight = nn.Parameter(weight.clone())
        elif mutation == "buffer_write":
            model.buffer.add_(1)
    changed = collect_lifecycle_signature(model)
    _assert_native(model, changed)
    assert changed != original


@pytest.mark.parametrize("mutation", ["empty_module", "same_parameter_module", "rename_child", "rewire_alias",
                                       "rename_parameter", "rename_buffer", "none_slot", "parameter_alias"])
def test_detects_topology_changes_even_when_native_tensor_signature_does_not(mutation):
    model = _model()
    model.extra = nn.Identity()
    original = collect_lifecycle_signature(model)
    if mutation == "empty_module":
        model.extra = nn.Identity()
    elif mutation == "same_parameter_module":
        old = model.left[1]
        replacement = nn.Linear(4, 2, bias=False)
        replacement.weight = old.weight
        model.left[1] = replacement
    elif mutation == "rename_child":
        model._modules["renamed"] = model._modules.pop("extra")
    elif mutation == "rewire_alias":
        model.right.shared = model.left[1]
    elif mutation == "rename_parameter":
        registry = model.left[1]._parameters
        registry["renamed"] = registry.pop("weight")
    elif mutation == "rename_buffer":
        model.right._buffers["renamed"] = model.right._buffers.pop("alias_buffer")
    elif mutation == "none_slot":
        model.register_parameter("new_empty_slot", None)
    elif mutation == "parameter_alias":
        model.right.register_parameter("another_alias", model.left[0].weight)
    changed = collect_lifecycle_signature(model)
    _assert_native(model, changed)
    assert changed.parameters == original.parameters and changed.buffers == original.buffers
    assert changed.topology != original.topology


def test_inference_tensor_without_version_and_cycle_match_original_semantics():
    model = _model()
    with torch.inference_mode():
        model.register_buffer("inference_tensor", torch.ones(3))
    model.right._modules["cycle"] = model
    signature = collect_lifecycle_signature(model)
    _assert_native(model, signature)
    assert signature.buffers[-1][2] is None


def test_real_full_meta_model_native_identity_parity():
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    signature = collect_lifecycle_signature(model)
    _assert_native(model, signature)
    assert len(signature.parameters) > 800 and len(signature.topology[0]) > 800


def _installable_model():
    model = _model().eval()
    model.invalidations = []
    model.invalidate_inference_caches = lambda **kwargs: model.invalidations.append(kwargs)
    return model


def test_collector_install_is_default_off_parameter_neutral_and_manifest_stable():
    from vvla_repro_pi05.lite.lifecycle import INSTALLATION, install_lifecycle_signature, LifecycleSignature
    model = _installable_model()
    assert not hasattr(model, INSTALLATION)
    original = collect_lifecycle_signature(model)
    state_keys = tuple(model.state_dict())
    handle = install_lifecycle_signature(model)
    assert getattr(model, INSTALLATION) is handle
    assert handle.signature() == handle() == original
    assert isinstance(handle.signature(), LifecycleSignature)
    assert tuple(model.state_dict()) == state_keys
    assert model.invalidations == [{'weights_changed': False}]
    before = handle.manifest()
    assert before['returned_fields'] == ['parameters', 'buffers', 'topology']
    assert not before['inventory_cache'] and not before['signature_cache']
    with torch.no_grad():model.left[0].weight.add_(1)
    assert handle.signature() != original and handle.manifest() == before
    handle.uninstall()
    assert not hasattr(model, INSTALLATION) and not handle.enabled
    assert model.invalidations == [{'weights_changed': False}] * 2
    with pytest.raises(RuntimeError, match='no longer active'):handle.signature()
    with pytest.raises(RuntimeError, match='no longer active'):handle.uninstall()


@pytest.mark.parametrize('mutation', ['parameter_write', 'new_parameter', 'buffer_write', 'new_buffer',
                                      'rename_parameter', 'none_slot', 'empty_module', 'rewire_shared'])
def test_installed_collector_revisits_all_tensor_and_registration_fields(mutation):
    from vvla_repro_pi05.lite.lifecycle import install_lifecycle_signature
    model = _installable_model()
    handle = install_lifecycle_signature(model)
    before = handle.signature()
    with torch.no_grad():
        if mutation == 'parameter_write':model.left[0].weight.add_(1)
        elif mutation == 'new_parameter':model.register_parameter('new_weight', nn.Parameter(torch.ones(3)))
        elif mutation == 'buffer_write':model.buffer.add_(1)
        elif mutation == 'new_buffer':model.register_buffer('new_buffer', torch.ones(2), persistent=False)
        elif mutation == 'rename_parameter':model.left[1]._parameters['renamed'] = model.left[1]._parameters.pop('weight')
        elif mutation == 'none_slot':model.register_parameter('new_none', None)
        elif mutation == 'empty_module':model.add_module('new_empty', nn.Identity())
        elif mutation == 'rewire_shared':model.right.shared = model.left[1]
    after = handle.signature()
    _assert_native(model, after)
    assert after == collect_lifecycle_signature(model) and after != before
    handle.uninstall()


def test_install_uninstall_require_idle_retired_graphs_and_eval_mode():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    from vvla_repro_pi05.lite.lifecycle import INSTALLATION, install_lifecycle_signature
    model = _installable_model()
    model.train()
    with pytest.raises(ValueError, match='eval mode'):install_lifecycle_signature(model)
    model.eval()
    with torch.inference_mode(),pytest.raises(ValueError, match='outside inference_mode'):
        install_lifecycle_signature(model)
    runner = Pi05NativeGraphRunner(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):install_lifecycle_signature(model)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='active'):install_lifecycle_signature(model)
    finally:runner._call_lock.release()
    handle = install_lifecycle_signature(model)
    with pytest.raises(ValueError, match='already installed'):install_lifecycle_signature(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):handle.uninstall()
    assert getattr(model, INSTALLATION) is handle and handle.enabled
    runner.close()
    handle.uninstall()
    assert not hasattr(model, INSTALLATION)


def test_signature_has_no_tensor_ops_cache_or_owning_model_reference():
    import gc
    import weakref
    from vvla_repro_pi05.lite.lifecycle import install_lifecycle_signature
    model = _installable_model()
    # Avoid an incidental test lambda closure retaining the model.
    model.invalidate_inference_caches = lambda **kwargs: None
    handle = install_lifecycle_signature(model)
    class NoTensorWork(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            raise AssertionError(f'Signature executed tensor work: {func}')
    with NoTensorWork():
        first,second = handle.signature(),handle.signature()
    assert first == second and first is not second
    assert set(handle.__dict__) == {'_model', 'enabled'}
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match='no longer active'):handle.signature()


def test_installed_collector_detects_external_handle_replacement():
    from vvla_repro_pi05.lite.lifecycle import INSTALLATION, install_lifecycle_signature
    model = _installable_model()
    handle = install_lifecycle_signature(model)
    setattr(model, INSTALLATION, object())
    with pytest.raises(RuntimeError, match='no longer active'):handle.signature()
    with pytest.raises(RuntimeError, match='no longer active'):handle.uninstall()


def test_shared_context_consumes_topology_and_refreshes_after_weight_and_module_changes():
    from test_lite_host_layout import _LayoutModel, _inputs, _context
    from vvla_repro_pi05.lite.lifecycle import install_lifecycle_signature
    model, inputs = _LayoutModel(), _inputs()
    _context(model, inputs)
    handle = install_lifecycle_signature(model)
    first = _context(model, inputs)['token']
    assert model._inference_cache.signature[-3:] == tuple(handle.signature())
    assert _context(model, inputs)['token'] == first
    model.add_module('empty_new_module', nn.Identity())
    second = _context(model, inputs)['token']
    assert second != first
    with torch.no_grad():
        model.weight.add_(1)
    assert _context(model, inputs)['token'] != second
    handle.uninstall()
    _context(model, inputs)
    assert model._inference_cache.signature[-2:] == (
        tuple(map(tensor_identity, model.parameters())), tuple(map(tensor_identity, model.buffers())))


def test_policy_option_is_default_off_and_close_retires_graph_before_collector():
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    assert not _split_lite_overrides({})[1]['single_pass_lifecycle']
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            _split_lite_overrides({'lite_single_pass_lifecycle': value})
    with torch.device('meta'):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend='torch_fa2'))
        set_bfloat16_layout(model)
    policy = Pi05LitePolicy(model, _policy().processor, device='cpu', cuda_graph=False,
                           single_pass_lifecycle=True)
    handle = policy._lifecycle_installation
    assert policy.metadata()['lite_optimizations']['single_pass_lifecycle']
    runner = Pi05NativeGraphRunner(model)
    runner.entries['live'] = object()
    policy.graph_runner = runner
    policy.close()
    assert runner._closed and not handle.enabled and policy._lifecycle_installation is None
    policy.close()
