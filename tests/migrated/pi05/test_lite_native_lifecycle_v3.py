"""Explicit v3 reuse contracts; root schedules remote CPU builds/tests only."""
import gc
import inspect
import weakref

import pytest
import torch
from torch import nn

import test_lite_native_lifecycle as reference
import test_lite_native_lifecycle_v2 as v2_tests
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import (
    NativeLifecycleCollector, install_native_lifecycle_signature, load_native_lifecycle,
)


@pytest.fixture(scope="module")
def collectors():
    initialized = torch.cuda.is_initialized()
    result = {variant: load_native_lifecycle(variant=variant, verbose=True) for variant in ("v1", "v2", "v3")}
    assert torch.cuda.is_initialized() == initialized
    return result


def previous(collector, model):
    return tuple(collector.signature(model))


def test_default_v1_and_explicit_reuse_manifest(collectors):
    assert inspect.signature(load_native_lifecycle).parameters["variant"].default == "v1"
    for variant in ("v1", "v2"):
        assert collectors[variant].manifest()["signature_cache"] is False
    manifest = collectors["v3"].manifest()
    assert manifest["signature_cache"] is True and manifest["inventory_cache"] is False
    assert manifest["all_live_fields_read_every_call"] is True
    assert manifest["new_persistent_model_or_tensor_references"] is False
    assert "same native snapshot" in manifest["comparison_fallback"]
    assert "native_lifecycle_v3.cpp" in manifest["source_sha256"]


def test_equal_state_reuses_only_existing_three_subtuples(collectors):
    collector, model = collectors["v3"], reference.model_fixture()
    old = previous(collector, model)
    with reference.NoTensorWork():
        actual = collector.signature(model, previous=old)
    assert actual == collect_lifecycle_signature(model)
    assert all(actual[i] is old[i] for i in range(3))
    assert all(collector.signature(model)[i] is not old[i] for i in range(3))
    # Buffer and parameter dedup remain independent even for the same object.
    model.register_buffer("parameter_also_buffer", model.left[0].weight)
    changed = collector.signature(model, previous=old)
    assert changed == collect_lifecycle_signature(model) and changed != old
    assert changed.parameters[0] == changed.buffers[-1]


@pytest.mark.parametrize("mutation", ["add", "copy", "pointer", "shape", "dtype", "device", "replace", "buffer", "inference"])
def test_reuse_attempt_reads_every_live_tensor_field(collectors, mutation):
    model, collector = reference.model_fixture(), collectors["v3"]
    old = previous(collector, model)
    weight = model.left[0].weight
    with torch.no_grad():
        if mutation == "add": weight.add_(1)
        elif mutation == "copy": weight.copy_(weight.clone())
        elif mutation == "pointer": weight.data = weight.data.clone()
        elif mutation == "shape": weight.data = weight.data.view(2, 6)
        elif mutation == "dtype": model.to(torch.bfloat16)
        elif mutation == "device": model.to("meta")
        elif mutation == "replace": model.left[0].weight = nn.Parameter(weight.clone())
        elif mutation == "buffer": model.buffer.add_(1)
        elif mutation == "inference":
            with torch.inference_mode(): model.register_buffer("new_inference", torch.ones(3))
    actual = collector.signature(model, previous=old)
    assert actual == collect_lifecycle_signature(model) and actual != old
    assert all(actual[i] is not old[i] for i in range(3))
    fresh = tuple(actual)
    assert all(collector.signature(model, previous=fresh)[i] is fresh[i] for i in range(3))


@pytest.mark.parametrize("mutation", ["alias", "none", "delete", "rename", "reorder", "type", "empty", "cycle"])
def test_reuse_attempt_reads_topology_slots_order_none_aliases_and_types(collectors, mutation):
    model, collector = reference.model_fixture(), collectors["v3"]
    old = previous(collector, model)
    if mutation == "alias": model.right.another = model.left[0]
    elif mutation == "none": model.register_parameter("another_none", None)
    elif mutation == "delete": del model._modules["empty_child"]
    elif mutation == "rename": model._buffers["renamed"] = model._buffers.pop("buffer")
    elif mutation == "reorder": model._modules["left"] = model._modules.pop("left")
    elif mutation == "type":
        class Different(nn.Module): pass
        model.right.__class__ = Different
    elif mutation == "empty": model.empty_new = nn.Identity()
    elif mutation == "cycle": model.right._modules["cycle"] = model
    actual = collector.signature(model, previous=old)
    assert actual == collect_lifecycle_signature(model) and actual != old


def test_early_mismatch_cannot_hide_later_unsupported_tensor_or_shadow(collectors):
    collector, model = collectors["v3"], reference.model_fixture()
    old = previous(collector, model)
    model.add_module("changed_early", nn.Identity())
    class Special(torch.Tensor): pass
    model.register_buffer("late_invalid", torch.ones(1).as_subclass(Special))
    with pytest.raises(TypeError, match="exact Tensor/Parameter"):
        collector.signature(model, previous=old)
    model._buffers.pop("late_invalid")
    model.buffer.data_ptr = lambda: 0
    with pytest.raises(TypeError, match="shadowed"):
        collector.signature(model, previous=old)


def test_same_snapshot_materialization_reads_dict_once_and_tensor_after_dfs(collectors):
    collector = collectors["v3"]
    calls = []
    class Trigger(nn.Module):
        def __getattribute__(self, name):
            if name == "__dict__":
                state = object.__getattribute__(self, "__dict__")
                if state.get("active"):
                    calls.append("dict")
                    with torch.no_grad(): state["target"]().add_(1)
            return super().__getattribute__(name)
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(1))
    model.trigger = Trigger()
    object.__setattr__(model.trigger, "target", weakref.ref(model.weight))
    old = previous(collector, model)
    object.__setattr__(model.trigger, "active", True)
    actual = collector.signature(model, previous=old)
    assert calls == ["dict"]  # Mismatch must not collect the model a second time.
    assert actual.parameters[0][2] == model.weight._version != old[0][0][2]
    object.__setattr__(model.trigger, "active", False)
    assert actual == collect_lifecycle_signature(model)


def test_custom_key_equality_is_deferred_to_normal_tuple_comparison(collectors):
    events = []
    class Key(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            events.append("key eq")
            return str.__eq__(self, other)
    model, collector = nn.Module(), collectors["v3"]
    model._parameters[Key("weight")] = nn.Parameter(torch.ones(1))
    old = previous(collector, model)
    value = model._parameters.pop("weight")
    model._parameters[Key("weight")] = value
    events.clear()
    actual = collector.signature(model, previous=old)
    assert not events and actual.topology is not old[2]
    assert tuple(actual) == old
    assert events == ["key eq"]


def test_custom_metaclass_equality_is_not_called_by_fast_comparison(collectors):
    events = []
    class Meta(type):
        __hash__ = type.__hash__
        def __eq__(cls, other):
            events.append("type eq")
            return True
    class Left(nn.Module, metaclass=Meta): pass
    class Right(nn.Module, metaclass=Meta): pass
    model, collector = Left(), collectors["v3"]
    model.register_buffer("b", torch.ones(1))
    old = previous(collector, model)
    model.__class__ = Right
    events.clear()
    actual = collector.signature(model, previous=old)
    assert not events and actual.topology is not old[2]
    assert tuple(actual) == old and events == ["type eq"]


def test_malformed_previous_never_invokes_python_protocols(collectors):
    events = []
    class Previous:
        def __eq__(self, other): events.append("eq"); raise AssertionError("early eq")
        def __iter__(self): events.append("iter"); raise AssertionError("early iter")
    model, collector = reference.model_fixture(), collectors["v3"]
    old = previous(collector, model)
    malformed = [Previous(), list(old), (old[0],), (Previous(), old[1], old[2])]
    for number in (-1, 2**200, True):
        fields = (number, *old[0][0][1:])
        malformed.append(((fields, *old[0][1:]), old[1], old[2]))
    for item in malformed:
        assert collector.signature(model, previous=item) == collect_lifecycle_signature(model)
    assert not events


def test_no_persistent_model_tensor_inventory_or_result_references(collectors):
    collector, model = collectors["v3"], reference.model_fixture()
    old = previous(collector, model)
    actual = collector.signature(model, previous=old)
    parameter_ref, model_ref = weakref.ref(model.left[0].weight), weakref.ref(model)
    assert set(collector.__dict__) == {"_extension", "_manifest"}
    del model
    gc.collect()
    assert parameter_ref() is None and model_ref() is None
    assert actual == old  # Immutable identities do not own live weights/modules.


def test_pending_children_still_have_temporary_owners(collectors):
    v2_tests.test_borrowed_stack_retains_children_until_live_dict_callback_finishes({"v3": collectors["v3"]})


def test_original_noncontiguous_meta_inference_and_cycle_contract(collectors):
    reference.test_cycle_empty_noncontiguous_offset_meta_and_inference_tensors(collectors["v3"])


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_older_variants_ignore_previous_keyword(collectors, variant):
    collector, model = collectors[variant], reference.model_fixture()
    handle = install_native_lifecycle_signature(model, collector=collector)
    expected = handle.signature()
    full = (None,) * 6 + tuple(expected)
    actual = handle.signature(previous_full_signature=full)
    assert actual == expected and all(actual[i] is not expected[i] for i in range(3))
    assert collector.signature(model, previous=object()) == expected
    handle.uninstall()


def test_core_handoff_reuse_and_outer_flags_period_precision_still_refresh(collectors, monkeypatch):
    from test_lite_host_layout import _LayoutModel, _inputs, _context
    model, inputs, collector = _LayoutModel(), _inputs(), collectors["v3"]
    handle = install_native_lifecycle_signature(model, collector=collector)
    calls = []
    original = collector.signature
    def recorded(model, *, previous=None):
        result = original(model, previous=previous)
        calls.append((previous, result))
        return result
    monkeypatch.setattr(collector, "signature", recorded)
    _context(model, inputs)
    initial = model._inference_cache.signature
    generation = model._inference_cache.generation
    _context(model, inputs)
    assert model._inference_cache.generation == generation
    assert all(calls[-1][1][i] is initial[-3:][i] for i in range(3))
    changes = [lambda: setattr(model.config, "min_period", 0.008),
               lambda: setattr(model, "optimization_flags", lambda: (True, False, True, True, False)),
               lambda: monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")]
    for change in changes:
        old = model._inference_cache.signature
        generation = model._inference_cache.generation
        count = len(calls)
        change()
        _context(model, inputs)
        assert len(calls) == count + 1  # No skip because flags/precision already changed.
        assert model._inference_cache.generation > generation
        assert all(model._inference_cache.signature[-3:][i] is old[-3:][i] for i in range(3))
    generation = model._inference_cache.generation
    with torch.no_grad(): model.weight.add_(1)
    _context(model, inputs)
    assert model._inference_cache.generation > generation
    generation = model._inference_cache.generation
    model.add_module("new_empty", nn.Identity())
    _context(model, inputs)
    assert model._inference_cache.generation > generation
    handle.uninstall()
    _context(model, inputs)
    assert len(model._inference_cache.signature) == 8  # Original two-field path restored.


def test_policy_v3_explicit_guard_and_default_unchanged():
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides, _validate_compilation_options
    assert inspect.signature(Pi05LitePolicy.__init__).parameters["native_lifecycle_variant"].default == "v1"
    _, default = _split_lite_overrides({})
    assert default["native_lifecycle_variant"] == "v1" and not default["native_lifecycle"]
    _, explicit = _split_lite_overrides({"lite_native_lifecycle": True, "lite_native_lifecycle_variant": "v3"})
    _validate_compilation_options(explicit, cuda_graph=True)
    explicit["native_lifecycle"] = False
    with pytest.raises(ValueError, match="requires lite_native_lifecycle"):
        _validate_compilation_options(explicit, cuda_graph=True)
