"""Host mask candidate: parity, immutable templates, and the real lifecycle hook."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vvla_repro_pi05.lite.layout import HostLayoutCache, HostMaskLayout, copy_layout_tree
from vvla_repro_pi05.lite.pi05 import _split_lite_overrides
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache, copy_position_context
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vvla_repro_pi05.models.pi05.prefix_packing_pi05 import copy_prefix_packing


class _LayoutModel(torch.nn.Module):
    """Tiny tensors, but the production context/lifecycle and metadata methods."""
    prepare_inference_context = Pi05ForActionPrediction.prepare_inference_context
    invalidate_inference_caches = Pi05ForActionPrediction.invalidate_inference_caches
    prepare_prefix_packing = Pi05ForActionPrediction.prepare_prefix_packing
    prepare_attention_metadata = Pi05ForActionPrediction.prepare_attention_metadata
    action_horizon = 2

    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, device=device))
        self.vision = torch.nn.Linear(1, 1, device=device)
        self.config = SimpleNamespace(min_period=0.004, max_period=4.0, prefix_pack_bucket_size=4)
        self._inference_cache = Pi05InferenceCache()
        self.position_builds = self.empty_builds = 0
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(image_size=2, patch_size=1)),
                                      model=SimpleNamespace(vision_tower=self.vision)),
            embed_image=self._embed_image, prefix_attention=object(), attention_backend="torch_fa2", paged_attention=False,
        )
        self.eval()

    def optimization_flags(self):
        return True, False, False, True, False

    def _embed_image(self, image):
        self.empty_builds += 1
        return image.sum().reshape(1, 1, 1) * self.weight

    def prepare_position_context(self, image_masks, lang_masks, prefix_packing=None):
        self.position_builds += 1
        packing = prefix_packing if prefix_packing is not None else self.prepare_prefix_packing(image_masks, lang_masks)
        return {"prefix_positions": packing["positions"],
                "suffix_positions": packing["pads"].sum(1)[:, None] + torch.arange(2, device=lang_masks.device),
                "parameter_dependent": self.weight.detach().clone()}

    def sample_actions(self, images, image_masks, lang_tokens, lang_masks, noise, num_steps,
                       attention_metadata=None, inference_context=None):
        if inference_context is None:
            inference_context = self.prepare_inference_context(images, image_masks, lang_masks, num_steps, positions=False)
            inference_context["positions"] = self.prepare_position_context(image_masks, lang_masks,
                                                                             inference_context["packing"])
        context = inference_context
        metadata = attention_metadata or self.prepare_attention_metadata(
            image_masks, lang_masks, fixed_capacity=True, prefix_packing=context["packing"],
        )
        return (noise * self.weight + sum(image.mean() for image in images) + lang_tokens.float().sum()
                + context["packing"]["indices"].float().sum() + metadata[1].extra["cu_k"].float().sum()
                + context["positions"]["suffix_positions"].float().sum()
                + context["positions"]["parameter_dependent"])


def _inputs(valid=2, cameras=(True, False), device="cpu", value=1.0):
    images = [torch.full((1, 3, 2, 2), value if present else -1.0, device=device, dtype=torch.float32)
              for present in cameras]
    image_masks = [torch.tensor([present], device=device) for present in cameras]
    tokens = torch.full((1, 6), int(value), device=device, dtype=torch.int64)
    mask_values = tuple(i < valid for i in range(6))
    masks = torch.tensor([mask_values], device=device)
    host = HostMaskLayout.from_processor(cameras, mask_values, images, image_masks, masks)
    return dict(images=images, image_masks=image_masks, lang_tokens=tokens, lang_masks=masks,
                noise=torch.full((1, 2, 4), value, device=device), num_steps=2, host_layout=host)


def _get(cache, model, inputs):
    return cache.get(model, inputs["host_layout"], inputs["images"], inputs["image_masks"], inputs["lang_masks"])


def _context(model, inputs, template=None):
    kwargs = {} if template is None else {"prepared_layout": template.prepared_layout}
    return model.prepare_inference_context(inputs["images"], inputs["image_masks"], inputs["lang_masks"],
                                           inputs["num_steps"], positions=False, **kwargs)


def _assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    else:
        assert actual == expected


def test_option_is_independent_default_off_and_strict_boolean():
    core, options = _split_lite_overrides({"lite_host_layout_cache": True, "pack_prefix": True})
    assert core == {"pack_prefix": True}
    assert options == {"elide_identity_gathers": False, "host_layout_cache": True,
                       "fused_adarms": False, "skip_weight_init": False, "fused_expert_qkv": False,
                       "fused_expert_gate_up": False, "fused_expert_gelu": False,
                       "fused_gated_residual": False, "compile_sample_actions": False,
                       "fused_adarms_reduction": False, "suffix_attention": None,
                       "compile_preserve_action_linears": False, "batched_vision": False, "fused_vision_qkv": False, "action_norm_cache": False, "suffix_fa2": None, "single_pass_lifecycle": False, "native_lifecycle": False, "native_lifecycle_variant": "v1", "expert_cublaslt": None, "gpu_uint8_prepare": False, "fused_residual_adarms": False, "request_identity_lookup": False, "packed_input_metadata": False, "vision_fc2_cublaslt": False}
    assert not _split_lite_overrides({})[1]["host_layout_cache"]
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="must be a boolean"):
            _split_lite_overrides({"lite_host_layout_cache": value})


def test_processor_emits_exact_host_mask_bytes_without_changing_prepared_inputs():
    # Reuse the canonical real processor fixture, not an alternate tokenizer path.
    from test_lite_pi05 import _obs, _policy
    processor = _policy().processor
    ordinary = processor.build_model_inputs(_obs())
    *prepared, host = processor.build_model_inputs(_obs(), return_host_layout=True)
    for old, new in zip(ordinary[:2], prepared[:2], strict=True):
        for a, b in zip(old, new, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(ordinary[2:], prepared[2:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    old_bytes = torch.cat([m[:, None] for m in prepared[1]] + [prepared[3]], 1).numpy().tobytes()
    assert host.mask_bytes == old_bytes
    with pytest.raises(ValueError, match="does not belong"):
        host.validate(prepared[0], prepared[1], prepared[3].clone())


def test_cpu_layout_parity_and_A_B_A_graph_copies_do_not_pollute_templates():
    model, cache = _LayoutModel(), HostLayoutCache()
    a, b = _inputs(2), _inputs(3)
    ta, tb = _get(cache, model, a), _get(cache, model, b)
    for inputs, template in ((a, ta), (b, tb)):
        reference = _context(model, inputs)
        prepared = _context(model, inputs, template)
        assert reference["token"] == prepared["token"] and reference["empty_pattern"] == prepared["empty_pattern"]
        _assert_tree_equal(reference["packing"], prepared["packing"])
        template.materialize(model, inputs["image_masks"], inputs["lang_masks"])
    original_a = copy_layout_tree(ta.packing), copy_layout_tree(ta.positions)
    graph_packing, graph_positions = copy_layout_tree(ta.packing), copy_layout_tree(ta.positions)
    copy_prefix_packing(graph_packing, tb.packing)
    copy_position_context(graph_positions, tb.positions)
    _assert_tree_equal(ta.packing, original_a[0])
    _assert_tree_equal(ta.positions, original_a[1])
    assert _get(cache, model, _inputs(2, value=7)) is ta
    ta.materialize(model, a["image_masks"], a["lang_masks"])
    assert model.position_builds == 2
    copy_prefix_packing(graph_packing, ta.packing)
    copy_position_context(graph_positions, ta.positions)
    _assert_tree_equal(graph_packing, original_a[0])
    _assert_tree_equal(graph_positions, original_a[1])
    assert graph_packing["positions"].data_ptr() != ta.packing["positions"].data_ptr()


def test_cpu_layout_cache_bounds_camera_pattern_and_capacity_changes():
    model, cache = _LayoutModel(), HostLayoutCache(capacity=2)
    a = _get(cache, model, _inputs(2))
    b = _get(cache, model, _inputs(2, cameras=(False, True)))
    c = _get(cache, model, _inputs(5))
    assert a.empty_pattern == (False, True) and b.empty_pattern == (True, False)
    assert a.packing["pads"].shape == (1, 8) and c.packing["pads"].shape == (1, 12)
    assert len(cache.entries) == 2
    assert _get(cache, model, _inputs(2)) is not a


def test_real_lifecycle_preserves_parameter_updates_and_avoids_warm_scalar_mask_reads():
    model, cache, inputs = _LayoutModel(), HostLayoutCache(), _inputs()
    template = _get(cache, model, inputs)
    first = _context(model, inputs, template)

    class Operators(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.seen = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.seen.append(str(func))
            return func(*args, **(kwargs or {}))

    operators = Operators()
    with operators:
        assert _get(cache, model, inputs) is template
        second = _context(model, inputs, template)
    assert first["token"] == second["token"]
    assert operators.seen == []  # No GPU packing, mask reductions, or value reads.
    with torch.no_grad():
        model.weight.add_(1)
    changed = _context(model, inputs, template)
    assert changed["token"] != first["token"]
    assert model.empty_builds == 2
    model.train()
    with pytest.raises(ValueError, match="eval"):
        _context(model, inputs, template)


@pytest.mark.parametrize("corruption", ["fields", "dtype", "shape", "device", "pattern", "flags"])
def test_core_rejects_invalid_prepared_layout_geometry(corruption):
    model, inputs = _LayoutModel(), _inputs()
    template = _get(HostLayoutCache(), model, inputs)
    layout = copy_layout_tree(template.prepared_layout)
    if corruption == "fields":
        layout["extra"] = True
    elif corruption == "dtype":
        layout["packing"]["indices"] = layout["packing"]["indices"].int()
    elif corruption == "shape":
        layout["packing"]["pads"] = layout["packing"]["pads"][:, :1]
    elif corruption == "device":
        layout["packing"]["positions"] = layout["packing"]["positions"].to("meta")
    elif corruption == "pattern":
        layout["empty_pattern"] = (False, 1)
    else:
        layout["packing"] = None
    with pytest.raises(ValueError, match="[Pp]repared"):
        model.prepare_inference_context(inputs["images"], inputs["image_masks"], inputs["lang_masks"],
                                         2, positions=False, prepared_layout=layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("identity", [False, True])
def test_host_layout_graph_A_B_A_empty_camera_stream_and_parameter_invalidation(identity):
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    model = _LayoutModel("cuda")
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True, elide_identity_gathers=identity)
    retained = None
    for valid, cameras, value in ((2, (True, False), 1.0), (3, (True, False), 2.0),
                                   (2, (True, False), 3.0), (2, (False, True), 4.0),
                                   (5, (True, False), 5.0), (2, (True, False), 6.0)):
        inputs = _inputs(valid, cameras, "cuda", value)
        expected = model.sample_actions(**{k: v for k, v in inputs.items() if k != "host_layout"})
        actual = runner(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if retained is None:
            retained = actual, actual.clone()
        else:
            torch.testing.assert_close(*retained, rtol=0, atol=0)
    assert runner.stats["captures"] == 3
    assert runner.stats["metadata_updates"] == 2
    before = tuple(runner._host_layouts.entries.values())
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        actual = runner(**_inputs(device="cuda"))
    other.synchronize()
    assert runner.stats["stream_handoffs"] == 1
    with torch.no_grad():
        model.weight.add_(1)
    inputs = _inputs(device="cuda")
    actual = runner(**inputs)
    expected = model.sample_actions(**{k: v for k, v in inputs.items() if k != "host_layout"})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runner.last_call["cold"] and runner.stats["invalidations"] == 1
    assert not any(new is old for new in runner._host_layouts.entries.values() for old in before)
    runner.close()
    assert not runner._host_layouts.entries


def test_adapter_fused_adarms_is_independent_and_close_retires_graph_before_uninstall():
    from test_lite_fused_adarms import SmallExpert
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy

    core, options = _split_lite_overrides({"lite_fused_adarms": True})
    assert core == {} and options == {
        "elide_identity_gathers": False, "host_layout_cache": False, "fused_adarms": True,
        "skip_weight_init": False, "fused_expert_qkv": False,
        "fused_expert_gate_up": False, "fused_expert_gelu": False,
        "fused_gated_residual": False, "compile_sample_actions": False,
        "fused_adarms_reduction": False, "suffix_attention": None,
        "compile_preserve_action_linears": False, "batched_vision": False, "fused_vision_qkv": False, "action_norm_cache": False, "suffix_fa2": None, "single_pass_lifecycle": False, "native_lifecycle": False, "native_lifecycle_variant": "v1", "expert_cublaslt": None, "gpu_uint8_prepare": False, "fused_residual_adarms": False, "request_identity_lookup": False, "packed_input_metadata": False, "vision_fc2_cublaslt": False,
    }
    model = SmallExpert()
    model.action_horizon, model.action_dim = 2, 4
    policy = Pi05LitePolicy(model, _policy().processor, device="cpu", cuda_graph=False, fused_adarms=True)
    handle = policy._adarms_installation
    metadata = policy.metadata()
    assert metadata["lite_optimizations"]["fused_adarms"]
    assert not metadata["lite_optimizations"]["host_layout_cache"]
    assert metadata["lite_optimizations"]["fused_adarms_manifest"] == handle.manifest()
    # A real runner's lifecycle can be tested on CPU before it captures anything.
    # Its event/device remain unset, so close performs no CUDA operation.
    runner = Pi05NativeGraphRunner(model)
    runner.entries["sentinel"] = object()
    policy.graph_runner = runner
    policy.close()
    assert runner._closed and not runner.entries
    assert not handle.enabled and model.invalidations == [False, False]
    assert policy._adarms_installation is None
    policy.close()  # Idempotent; never uninstalls an already-retired handle.
