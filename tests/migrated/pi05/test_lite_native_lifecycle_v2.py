"""Explicit allocation variant; scheduled remote CPU compile/tests only.

No production option selects v2 by default. The complete Python representation
and existing reference mutation/lifecycle tests remain authoritative.
"""
import gc
import inspect
import weakref

import pytest
import torch
from torch import nn

import test_lite_native_lifecycle as reference_tests
from test_lite_skip_weight_init import probe
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import load_native_lifecycle


@pytest.fixture(scope="module")
def collectors():
    initialized = torch.cuda.is_initialized()
    result = {name: load_native_lifecycle(variant=name, verbose=True) for name in ("v1", "v2")}
    assert torch.cuda.is_initialized() == initialized
    return result


def test_v2_explicit_default_stays_v1_and_manifest_tracks_distinct_sources(collectors):
    assert inspect.signature(load_native_lifecycle).parameters["variant"].default == "v1"
    for name, collector in collectors.items():
        manifest = collector.manifest()
        assert manifest["collector_variant"] == name
        assert not manifest["inventory_cache"] and not manifest["signature_cache"]
        source = "native_lifecycle.cpp" if name == "v1" else "native_lifecycle_v2.cpp"
        assert source in manifest["source_sha256"]
    assert collectors["v1"].manifest()["module_name"] != collectors["v2"].manifest()["module_name"]
    with pytest.raises(ValueError, match="v1 or v2"):
        load_native_lifecycle(variant="unknown")


def test_three_modes_exact_same_live_objects_and_full_fields(collectors):
    model = reference_tests.model_fixture()
    expected = collect_lifecycle_signature(model)
    with reference_tests.NoTensorWork():
        assert collectors["v1"].signature(model) == expected
        assert collectors["v2"].signature(model) == expected
    with torch.no_grad(): model.left[0].weight.data = model.left[0].weight.data.view(2, 6)
    model.right.add_module("alias_with_none", None)
    updated = collect_lifecycle_signature(model)
    assert updated != expected
    assert collectors["v1"].signature(model) == collectors["v2"].signature(model) == updated


@pytest.mark.parametrize("case", [
    "test_exact_reference_order_alias_none_and_no_tensor_work",
    "test_cycle_empty_noncontiguous_offset_meta_and_inference_tensors",
    "test_full_pi05_meta_model_all_fields",
    "test_supported_types_fail_closed",
    "test_install_default_off_live_mutation_manifest_and_retirement",
    "test_install_guard_and_python_exclusion",
    "test_no_cached_inventory_result_or_model_reference",
    "test_external_replacement_and_failed_install_cleanup",
    "test_real_context_consumes_all_native_fields_and_uninstall_restores_original",
])
def test_v2_preserves_existing_reference_contract(collectors, case):
    getattr(reference_tests, case)(collectors["v2"])


@pytest.mark.parametrize("mutation", ["add", "copy", "data_pointer", "data_shape", "dtype", "device",
                                      "replace_parameter", "buffer_write", "new_parameter", "new_buffer"])
def test_v2_all_existing_tensor_mutations(collectors, mutation):
    reference_tests.test_all_live_tensor_fields_and_registry_additions(collectors["v2"], mutation)


@pytest.mark.parametrize("mutation", ["empty_module", "same_parameter_module", "rename_child", "rewire_alias",
                                      "rename_parameter", "rename_buffer", "none_slot", "parameter_alias",
                                      "buffer_alias", "delete_none", "reorder"])
def test_v2_all_existing_topology_mutations(collectors, mutation):
    reference_tests.test_all_live_topology_changes_with_unchanged_tensor_identity(collectors["v2"], mutation)


def test_per_call_arena_overflow_keeps_all_nodes_and_dynamic_replacement(collectors):
    # Node-based visited + parameter sets alone exceed the 64 KiB stack arena;
    # upstream growth must preserve semantics and free all transient storage.
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(4096)])
    model.alias = model.layers[0]
    model.register_buffer("buffer", torch.ones(3))
    model.add_module("optional", None)
    expected = collect_lifecycle_signature(model)
    assert len(expected.parameters) == 4096
    assert collectors["v1"].signature(model) == collectors["v2"].signature(model) == expected
    with torch.no_grad(): model.layers[-1].weight.add_(1)
    model.layers[0] = nn.Linear(2, 1, bias=False)
    changed = collect_lifecycle_signature(model)
    assert changed != expected
    assert collectors["v1"].signature(model) == collectors["v2"].signature(model) == changed
    parameter_ref = weakref.ref(model.layers[-1].weight)
    model_ref = weakref.ref(model)
    del model
    gc.collect()
    assert model_ref() is None and parameter_ref() is None


def test_borrowed_stack_retains_children_until_live_dict_callback_finishes(collectors):
    class Rewire(nn.Module):
        def __getattribute__(self, name):
            if name == "__dict__":
                state = object.__getattribute__(self, "__dict__")
                parent_ref = state.get("parent_ref")
                if parent_ref is not None:
                    parent_ref()._modules["last"] = None
                    assert state["target_ref"]() is not None, "pending child lost its owner"
            return super().__getattribute__(name)
    for collector in collectors.values():
        model = nn.Module()
        model.first = nn.Identity()
        model.trigger = Rewire()
        model.last = nn.Linear(2, 2)
        target = weakref.ref(model.last)
        object.__setattr__(model.trigger, "target_ref", target)
        object.__setattr__(model.trigger, "parent_ref", weakref.ref(model))
        actual = collector.signature(model)
        assert len(actual.topology[0]) == 4 and len(actual.parameters) == 2
        assert model.last is None
        gc.collect()
        assert target() is None  # No inventory remains after the call.


def test_many_new_models_cannot_reuse_prior_inventory_or_versions(collectors):
    for index in range(24):
        model = reference_tests.model_fixture()
        model.register_buffer("different_each_time", torch.arange(index + 1))
        expected = collect_lifecycle_signature(model)
        assert collectors["v1"].signature(model) == collectors["v2"].signature(model) == expected
        parameter_ref = weakref.ref(model.left[0].weight)
        del model
        gc.collect()
        assert parameter_ref() is None


def test_policy_variant_defaults_v1_and_native_off_never_loads_extension(monkeypatch):
    from types import SimpleNamespace
    from vvla_repro_pi05.lite import native_lifecycle
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    assert inspect.signature(Pi05LitePolicy.__init__).parameters['native_lifecycle_variant'].default == 'v1'
    core, options = _split_lite_overrides({'pack_prefix': True})
    assert core == {'pack_prefix': True}
    assert options['native_lifecycle'] is False and options['native_lifecycle_variant'] == 'v1'
    def forbidden(**kwargs):
        raise AssertionError('Disabled native lifecycle attempted extension loading')
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', forbidden)
    policy = Pi05LitePolicy(nn.Linear(2, 2), SimpleNamespace(config=SimpleNamespace()),
                            device='cpu', cuda_graph=False)
    assert policy._lite_optimizations['native_lifecycle_variant'] == 'v1'
    assert policy._native_lifecycle_installation is None
    policy.close()


@pytest.mark.parametrize('variant,enabled', [(None, True), (True, True), (1, True), ('v4', True),
                                            ({}, True), ([], True), ('v2', False)])
def test_invalid_policy_variant_rejects_constructor_and_loader_before_extension_or_model(monkeypatch, tmp_path,
                                                                                         variant, enabled):
    from types import SimpleNamespace
    from vvla_repro_pi05.lite import pi05, native_lifecycle
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid native variant reached allocation or extension loading')
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', forbidden)
    monkeypatch.setattr(pi05, '_construct_pi05_model', forbidden)
    monkeypatch.setattr(pi05, '_resolve_checkpoint', lambda *args: tmp_path)
    monkeypatch.setattr(pi05, 'load_lite_config', lambda path: (Pi05Config(), {}))
    policy = pi05.Pi05LitePolicy.__new__(pi05.Pi05LitePolicy)
    with pytest.raises(ValueError, match='native_lifecycle_variant'):
        policy.__init__(nn.Linear(2, 2), SimpleNamespace(config=SimpleNamespace()),
                        device='cpu', cuda_graph=False, native_lifecycle=enabled, native_lifecycle_variant=variant)
    assert policy._closed and policy._native_lifecycle_installation is None
    with pytest.raises(ValueError, match='native_lifecycle_variant'):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, config_overrides={
            'lite_native_lifecycle': enabled, 'lite_native_lifecycle_variant': variant})


def test_real_factory_passes_variant_records_native_manifest_and_separates_recipe(collectors, probe, monkeypatch, tmp_path):
    from dataclasses import asdict
    import json
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from test_lite_skip_weight_init import _Tokenizer
    from vvla_repro_pi05.lite import pi05, native_lifecycle
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    config = Pi05Config(chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8,
                        num_inference_steps=2, image_resolution=(28, 28), tokenizer_max_length=8,
                        input_features={'observation.state': {'shape': [8]},
                                        'observation.images.front': {'shape': [3, 28, 28]}},
                        output_features={'action': {'shape': [8]}},
                        norm_stats={name: {'mode': 'mean_std', 'mean': [0] * 8, 'std': [1] * 8}
                                    for name in ('state', 'action')})
    (tmp_path / 'config.json').write_text(json.dumps(asdict(config)))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: _Tokenizer())
    calls = []
    def loaded(*, variant):
        calls.append(variant)
        return collectors[variant]
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', loaded)
    policies = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Reuse the loader audit's context, which keeps the tiny dimensions in
        # effect for both reference construction and all from_pretrained calls.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'), probe.small_dimensions():
            reference = Pi05ForActionPrediction(config)
            save_file({name: value.detach().clone() for name, value in reference.named_parameters()},
                      str(tmp_path / 'model.safetensors'))
            overrides = [{}, {'lite_native_lifecycle': True},
                         {'lite_native_lifecycle': True, 'lite_native_lifecycle_variant': 'v2'}]
            for options in overrides:
                policies.append(pi05.create_adapter(dict(
                    checkpoint=str(tmp_path), tokenizer=str(tmp_path), device='cpu', cuda_graph=False,
                    attention_backend='eager', profile='bf16', config_overrides=options)))
            assert calls == ['v1', 'v2']
            metadata = [policy.metadata() for policy in policies]
            assert len({value['recipe_sha256'] for value in metadata}) == 3
            for value in metadata[1:]:
                for key in ('checkpoint_sha256', 'preprocessing_sha256', 'resolved_config', 'source_sha256'):
                    assert value[key] == metadata[0][key]
            for policy, variant in zip(policies[1:], ('v1', 'v2'), strict=True):
                options = policy.metadata()['lite_optimizations']
                assert options['native_lifecycle_variant'] == variant
                assert options['native_lifecycle_manifest'] == policy._native_lifecycle_installation.manifest()
                assert options['native_lifecycle_manifest']['collector_variant'] == variant
                assert options['native_lifecycle_manifest']['source_sha256'] == collectors[variant].manifest()['source_sha256']
                options['native_lifecycle_manifest']['collector_variant'] = 'mutated'
                assert policy.metadata()['lite_optimizations']['native_lifecycle_manifest']['collector_variant'] == variant
                handle = policy._native_lifecycle_installation
                policy.close()
                assert not handle.enabled and policy._native_lifecycle_installation is None
    finally:
        for policy in policies:
            policy.close()
        torch.set_num_threads(previous_threads)
