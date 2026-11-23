"""Compile once on the scheduled CPU host; no GPU work or model execution.

Do not run locally for the current campaign. Root schedules this on an existing
Colab between measurements. A missing compiler/build error is a test failure,
not a skip or an implicit fallback to the Python collector.
"""
import gc
import weakref

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from vvla_repro_pi05.lite.lifecycle import LifecycleSignature, collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import (
    INSTALLATION, NativeLifecycleCollector, install_native_lifecycle_signature,
    load_native_lifecycle,
)


@pytest.fixture(scope="module")
def collector():
    initialized = torch.cuda.is_initialized()
    result = load_native_lifecycle(verbose=True)
    assert torch.cuda.is_initialized() == initialized
    return result


def model_fixture():
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
    model.eval()
    model.invalidations = []
    invalidations = model.invalidations
    model.invalidate_inference_caches = lambda **kwargs: invalidations.append(kwargs)
    return model


class NoTensorWork(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Signature executed tensor work: {func}")


def test_exact_reference_order_alias_none_and_no_tensor_work(collector):
    model = model_fixture()
    with NoTensorWork():
        expected = collect_lifecycle_signature(model)
        actual = collector.signature(model)
        assert actual == expected
        assert collector.signature(model) == actual
    assert isinstance(actual, LifecycleSignature)
    assert len(actual.parameters) == 3 and len(actual.buffers) == 1


@pytest.mark.parametrize("mutation", ["add", "copy", "data_pointer", "data_shape", "dtype", "device",
                                      "replace_parameter", "buffer_write", "new_parameter", "new_buffer"])
def test_all_live_tensor_fields_and_registry_additions(collector, mutation):
    model = model_fixture()
    before = collector.signature(model)
    weight = model.left[0].weight
    with torch.no_grad():
        if mutation == "add": weight.add_(1)
        elif mutation == "copy": weight.copy_(weight.clone())
        elif mutation == "data_pointer": weight.data = weight.data.clone()
        elif mutation == "data_shape":
            original = weight.data_ptr(), weight._version
            weight.data = weight.data.view(2, 6)
            assert (weight.data_ptr(), weight._version) == original
        elif mutation == "dtype": model.to(dtype=torch.bfloat16)
        elif mutation == "device": model.to(device="meta")
        elif mutation == "replace_parameter": model.left[0].weight = nn.Parameter(weight.clone())
        elif mutation == "buffer_write": model.buffer.add_(1)
        elif mutation == "new_parameter": model.register_parameter("added", nn.Parameter(torch.ones(1)))
        elif mutation == "new_buffer": model.register_buffer("added", torch.ones(1), persistent=False)
    after = collector.signature(model)
    assert after == collect_lifecycle_signature(model) and after != before


@pytest.mark.parametrize("mutation", ["empty_module", "same_parameter_module", "rename_child", "rewire_alias",
                                      "rename_parameter", "rename_buffer", "none_slot", "parameter_alias",
                                      "buffer_alias", "delete_none", "reorder"])
def test_all_live_topology_changes_with_unchanged_tensor_identity(collector, mutation):
    model = model_fixture()
    model.extra = nn.Identity()
    before = collector.signature(model)
    if mutation == "empty_module": model.extra = nn.Identity()
    elif mutation == "same_parameter_module":
        replacement = nn.Linear(4, 2, bias=False)
        replacement.weight = model.left[1].weight
        model.left[1] = replacement
    elif mutation == "rename_child": model._modules["renamed"] = model._modules.pop("extra")
    elif mutation == "rewire_alias": model.right.shared = model.left[1]
    elif mutation == "rename_parameter": model.left[1]._parameters["renamed"] = model.left[1]._parameters.pop("weight")
    elif mutation == "rename_buffer": model.right._buffers["renamed"] = model.right._buffers.pop("alias_buffer")
    elif mutation == "none_slot": model.register_parameter("new_empty", None)
    elif mutation == "parameter_alias": model.right.register_parameter("another_alias", model.left[0].weight)
    elif mutation == "buffer_alias": model.right.register_buffer("another_alias", model.buffer)
    elif mutation == "delete_none": del model._modules["empty_child"]
    elif mutation == "reorder": model._modules["empty_child"] = model._modules.pop("empty_child")
    after = collector.signature(model)
    assert after == collect_lifecycle_signature(model)
    assert before.parameters == after.parameters and before.buffers == after.buffers
    assert before.topology != after.topology


def test_cycle_empty_noncontiguous_offset_meta_and_inference_tensors(collector):
    model = model_fixture()
    model.right._modules["cycle"] = model
    model.register_buffer("zero", torch.empty(0))
    model.register_buffer("view", torch.arange(30).reshape(5, 6)[1:4, ::2])
    model.register_buffer("meta", torch.empty(2, 4, device="meta", dtype=torch.bfloat16))
    with torch.inference_mode(): model.register_buffer("inference", torch.ones(3))
    actual = collector.signature(model)
    assert actual == collect_lifecycle_signature(model)
    assert actual.buffers[-1][2] is None
    assert actual.buffers[-2][1] == 0


def test_full_pi05_meta_model_all_fields(collector):
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    actual = collector.signature(model)
    assert actual == collect_lifecycle_signature(model)
    assert len(actual.parameters) > 800 and len(actual.topology[0]) > 800


def test_supported_types_fail_closed(collector):
    from collections import OrderedDict
    model = model_fixture()
    model._buffers = OrderedDict(model._buffers)
    with pytest.raises(TypeError, match="ordinary dict"): collector.signature(model)
    model = model_fixture()
    class SpecialTensor(torch.Tensor): pass
    model.register_buffer("subclass", torch.ones(3).as_subclass(SpecialTensor))
    with pytest.raises(TypeError, match="exact Tensor/Parameter"): collector.signature(model)
    model = model_fixture()
    model.buffer.data_ptr = lambda: 0
    with pytest.raises(TypeError, match="shadowed"): collector.signature(model)


def test_install_default_off_live_mutation_manifest_and_retirement(collector):
    model = model_fixture()
    assert not hasattr(model, INSTALLATION)
    keys = tuple(model.state_dict())
    original = collect_lifecycle_signature(model)
    handle = install_native_lifecycle_signature(model, collector=collector)
    assert handle.signature() == handle() == original
    assert tuple(model.state_dict()) == keys
    assert model.invalidations == [{"weights_changed": False}]
    manifest = handle.manifest()
    assert not manifest["with_cuda"] and not manifest["inventory_cache"] and not manifest["signature_cache"]
    assert manifest["returned_fields"] == ["parameters", "buffers", "topology"]
    assert len(manifest["binary_sha256"]) == 64 and manifest["compiler"]
    with torch.no_grad(): model.left[0].weight.add_(1)
    assert handle.signature() == collect_lifecycle_signature(model) != original
    assert handle.manifest() == manifest
    handle.uninstall()
    assert not hasattr(model, INSTALLATION) and not handle.enabled
    assert model.invalidations == [{"weights_changed": False}] * 2
    with pytest.raises(RuntimeError, match="no longer active"): handle.signature()


def test_install_guard_and_python_exclusion(collector):
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    from vvla_repro_pi05.lite.lifecycle import install_lifecycle_signature
    model = model_fixture()
    model.train()
    with pytest.raises(ValueError, match="eval mode"): install_native_lifecycle_signature(model, collector=collector)
    model.eval()
    with torch.inference_mode(), pytest.raises(ValueError, match="outside inference_mode"):
        install_native_lifecycle_signature(model, collector=collector)
    python_handle = install_lifecycle_signature(model)
    with pytest.raises(ValueError, match="already installed"): install_native_lifecycle_signature(model, collector=collector)
    python_handle.uninstall()
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"): install_native_lifecycle_signature(model, collector=collector)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"): install_native_lifecycle_signature(model, collector=collector)
    finally: runner._call_lock.release()
    handle = install_native_lifecycle_signature(model, collector=collector)
    with pytest.raises(ValueError, match="already installed"): install_native_lifecycle_signature(model, collector=collector)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"): handle.uninstall()
    runner.close()
    handle.uninstall()


def test_no_cached_inventory_result_or_model_reference(collector):
    model = model_fixture()
    handle = install_native_lifecycle_signature(model, collector=collector)
    with NoTensorWork():
        first, second = handle.signature(), handle.signature()
    assert first == second and first is not second
    assert set(collector.__dict__) == {"_extension", "_manifest"}
    parameter = weakref.ref(model.left[0].weight)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None and parameter() is None
    with pytest.raises(RuntimeError, match="no longer active"): handle.signature()


def test_external_replacement_and_failed_install_cleanup(collector):
    model = model_fixture()
    handle = install_native_lifecycle_signature(model, collector=collector)
    setattr(model, INSTALLATION, object())
    with pytest.raises(RuntimeError, match="no longer active"): handle.signature()
    model = model_fixture()
    def fail(**kwargs): raise RuntimeError("invalidation failed")
    model.invalidate_inference_caches = fail
    with pytest.raises(RuntimeError, match="invalidation failed"):
        install_native_lifecycle_signature(model, collector=collector)
    assert not hasattr(model, INSTALLATION)


def test_mismatch_rejects_without_installation():
    class WrongExtension:
        def collect(self, model): return (), (), ()
    collector = NativeLifecycleCollector(WrongExtension(), {})
    model = model_fixture()
    with pytest.raises(RuntimeError, match="complete Python reference"):
        install_native_lifecycle_signature(model, collector=collector)
    assert not hasattr(model, INSTALLATION) and not model.invalidations


def test_build_failure_is_explicit_and_never_installs(monkeypatch):
    import torch.utils.cpp_extension
    def fail(**kwargs): raise RuntimeError("simulated compiler failure")
    monkeypatch.setattr(torch.utils.cpp_extension, "load", fail)
    model = model_fixture()
    with pytest.raises(RuntimeError, match="candidate rejected"):
        install_native_lifecycle_signature(model)
    assert not hasattr(model, INSTALLATION) and not model.invalidations


def test_real_context_consumes_all_native_fields_and_uninstall_restores_original(collector):
    from test_lite_host_layout import _LayoutModel, _inputs, _context
    from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
    model, inputs = _LayoutModel(), _inputs()
    _context(model, inputs)
    handle = install_native_lifecycle_signature(model, collector=collector)
    first = _context(model, inputs)["token"]
    assert model._inference_cache.signature[-3:] == tuple(handle.signature())
    assert _context(model, inputs)["token"] == first
    model.add_module("empty_new_module", nn.Identity())
    second = _context(model, inputs)["token"]
    assert second != first
    with torch.no_grad(): model.weight.add_(1)
    assert _context(model, inputs)["token"] != second
    handle.uninstall()
    _context(model, inputs)
    assert model._inference_cache.signature[-2:] == (
        tuple(map(tensor_identity, model.parameters())), tuple(map(tensor_identity, model.buffers())))


def test_policy_option_default_off_strict_bool_exclusive_and_graph_retirement(collector, monkeypatch):
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite import native_lifecycle
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides, _validate_compilation_options
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    assert not _split_lite_overrides({})[1]["native_lifecycle"]
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            _split_lite_overrides({"lite_native_lifecycle": value})
    _, both = _split_lite_overrides({"lite_native_lifecycle": True, "lite_single_pass_lifecycle": True})
    with pytest.raises(ValueError, match="only one lifecycle"):
        _validate_compilation_options(both, cuda_graph=False)
    monkeypatch.setattr(native_lifecycle, "load_native_lifecycle", lambda **kwargs: collector)
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    policy = Pi05LitePolicy(model, _policy().processor, device="cpu", cuda_graph=False, native_lifecycle=True)
    handle = policy._native_lifecycle_installation
    assert policy.metadata()["lite_optimizations"]["native_lifecycle"]
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    policy.graph_runner = runner
    policy.close()
    assert runner._closed and not handle.enabled and policy._native_lifecycle_installation is None
    policy.close()
