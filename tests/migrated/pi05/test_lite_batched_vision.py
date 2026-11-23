"""B1 camera batching: slot/cache contracts and deferred native CUDA checks."""

import copy
from contextlib import contextmanager
import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite.batched_vision import INSTALLATION, install_batched_vision
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction, PaliGemmaWithActionExpertPi05
from test_lite_skip_weight_init import probe


class ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.config = SimpleNamespace(vision_config=SimpleNamespace(
            image_size=2, patch_size=1, num_channels=3,
        ))
        self.paligemma.model = nn.Module()
        self.paligemma.model.vision_tower = nn.Linear(3, 8, dtype=torch.bfloat16)
        self.language = nn.Embedding(16, 8, dtype=torch.bfloat16)
        self.batches = []

    def embed_image(self, image):
        self.batches.append(image.shape[0])
        pixels = image.flatten(2).transpose(1, 2).bfloat16()
        return self.paligemma.model.vision_tower(pixels).float()

    def embed_language_tokens(self, tokens):
        return self.language(tokens)


class ToyModel(nn.Module):
    embed_prefix = Pi05ForActionPrediction.embed_prefix

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = ToyInner()
        self.config = SimpleNamespace(max_cameras=3)
        self.vlm_width = 8
        self.invalidations = []
        self.eval()

    def invalidate_inference_caches(self, *, weights_changed=True):
        self.invalidations.append(weights_changed)


def toy_inputs():
    return {
        "images": [torch.arange(12).reshape(1, 3, 2, 2).float() * 0.1 + i for i in range(3)],
        "image_masks": [torch.tensor([True]), torch.tensor([False]), torch.tensor([True])],
        "lang_tokens": torch.tensor([[1, 2, 3, 0]]),
        "lang_masks": torch.tensor([[True, True, True, False]]),
    }


@pytest.mark.parametrize("cached_bits", range(8))
def test_every_cache_pattern_keeps_slot_order_masks_and_owned_prefix(cached_bits):
    model = ToyModel()
    inputs = toy_inputs()
    inner = model.paligemma_with_expert
    with torch.no_grad():
        cached = [inner.embed_image(image).clone() if cached_bits & (1 << i) else None
                  for i, image in enumerate(inputs["images"])]
        originals = [None if value is None else value.clone() for value in cached]
        expected = model.embed_prefix(**inputs, empty_features=cached)
    modules = {name: id(value) for name, value in model.named_modules()}
    parameters = {name: id(value) for name, value in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    handle = install_batched_vision(model)
    manifest = handle.manifest()
    inner.batches.clear()
    with torch.no_grad():
        actual = model.embed_prefix(**inputs, empty_features=cached)
    uncached = 3 - cached_bits.bit_count()
    assert inner.batches == ([uncached] if uncached else [])
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert actual[0].dtype == torch.float32
    assert actual[1].tolist() == [[True] * 4 + [False] * 4 + [True] * 4 + [True, True, True, False]]
    assert not actual[2].any()
    assert {name: id(value) for name, value in model.named_modules()} == modules
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    for result, reference in zip(cached, originals, strict=True):
        if result is not None:
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    # Returned prefix is owned even when every image feature came from a cache.
    frozen = actual[0].clone()
    with torch.no_grad():
        inputs["images"][0].add_(1)
        inner.language.weight.add_(0.1)
        model.embed_prefix(**inputs, empty_features=cached)
    torch.testing.assert_close(actual[0], frozen, rtol=0, atol=0)
    assert manifest == handle.manifest()
    assert manifest["weight_cache_bytes"] == manifest["activation_cache_bytes"] == 0
    handle.uninstall()
    assert "embed_prefix" not in model.__dict__ and not hasattr(model, INSTALLATION)
    assert model.invalidations == [False, False]


def test_no_empty_features_batches_even_false_mask_slots_without_intercamera_mixing():
    model, inputs = ToyModel(), toy_inputs()
    handle = install_batched_vision(model)
    with torch.no_grad():
        first = model.embed_prefix(**inputs)
        assert model.paligemma_with_expert.batches == [3]
        # Changing a masked camera must still change its own unobserved prefix
        # values, while leaving all other cameras and their attention isolated.
        inputs["images"][1].add_(10)
        second = model.embed_prefix(**inputs)
    assert not torch.equal(first[0][:, 4:8], second[0][:, 4:8])
    torch.testing.assert_close(first[0][:, :4], second[0][:, :4], rtol=0, atol=0)
    torch.testing.assert_close(first[0][:, 8:], second[0][:, 8:], rtol=0, atol=0)
    handle.uninstall()


@pytest.mark.parametrize("fault", ["batch", "image_count", "image_dtype", "camera_mask", "language_mask", "cache_count", "cache_shape", "geometry"])
def test_unsupported_geometry_rejected_before_encoding(fault):
    model, inputs = ToyModel(), toy_inputs()
    handle = install_batched_vision(model)
    if fault == "batch":
        inputs["images"] = [image.repeat(2, 1, 1, 1) for image in inputs["images"]]
        inputs["image_masks"] = [mask.repeat(2) for mask in inputs["image_masks"]]
        inputs["lang_tokens"] = inputs["lang_tokens"].repeat(2, 1)
        inputs["lang_masks"] = inputs["lang_masks"].repeat(2, 1)
    elif fault == "image_count":
        inputs["images"].pop()
    elif fault == "image_dtype":
        inputs["images"][1] = inputs["images"][1].bfloat16()
    elif fault == "camera_mask":
        inputs["image_masks"][0] = torch.ones(1)
    elif fault == "language_mask":
        inputs["lang_masks"] = torch.ones(1, 3, dtype=torch.bool)
    elif fault == "cache_count":
        inputs["empty_features"] = [None, None]
    elif fault == "cache_shape":
        inputs["empty_features"] = [None, None, torch.ones(1, 3, 8)]
    elif fault == "geometry":
        model.config.max_cameras = 2
    with torch.no_grad(), pytest.raises(ValueError):
        model.embed_prefix(**inputs)
    assert model.paligemma_with_expert.batches == []
    handle.uninstall()


def test_mutation_graph_guards_double_install_tamper_and_invalidation_failure(monkeypatch):
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = ToyModel()
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        install_batched_vision(model)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_batched_vision(model)
    finally:
        runner._call_lock.release()
    with torch.inference_mode(), pytest.raises(ValueError, match="outside inference_mode"):
        install_batched_vision(model)
    original_invalidate = model.invalidate_inference_caches
    def fail(**kwargs):
        raise RuntimeError("intentional invalidation failure")
    monkeypatch.setattr(model, "invalidate_inference_caches", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        install_batched_vision(model)
    assert "embed_prefix" not in model.__dict__ and not hasattr(model, INSTALLATION)
    monkeypatch.setattr(model, "invalidate_inference_caches", original_invalidate)
    handle = install_batched_vision(model)
    with pytest.raises(ValueError, match="already installed"):
        install_batched_vision(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        handle.uninstall()
    runner.clear()
    monkeypatch.setattr(model, "invalidate_inference_caches", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        handle.uninstall()
    assert getattr(model, INSTALLATION) is handle and handle.enabled
    monkeypatch.setattr(model, "invalidate_inference_caches", original_invalidate)
    patched = model.embed_prefix
    model.embed_prefix = MethodType(Pi05ForActionPrediction.embed_prefix, model)
    with pytest.raises(RuntimeError, match="replaced externally"):
        handle.uninstall()
    model.embed_prefix = patched
    runner.close()
    handle.uninstall()


def test_grad_and_training_fallback_uses_original_per_camera_path():
    model, inputs = ToyModel(), toy_inputs()
    for image in inputs["images"]:
        image.requires_grad_(True)
    expected = model.embed_prefix(**inputs)[0]
    handle = install_batched_vision(model)
    inner = model.paligemma_with_expert
    inner.batches.clear()
    actual = model.embed_prefix(**inputs)[0]
    assert inner.batches == [1, 1, 1]
    variables = (*inputs["images"], *model.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), variables)
    actual_grads = torch.autograd.grad(actual.sum(), variables)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    model.train()
    inner.batches.clear()
    with torch.no_grad():
        model.embed_prefix(**inputs)
    assert inner.batches == [1, 1, 1]
    model.eval()
    handle.uninstall()


def _small_core(probe, device):
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.precision_pi05 import ExecutionRecipe, install_execution_recipe

    with probe.small_dimensions():
        model = Pi05ForActionPrediction(Pi05Config(
            chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8,
            num_inference_steps=2, attention_backend="eager", vision_bf16=True,
            cache_empty_camera_features=True, precompute_time_and_positions=True,
            prefix_kv_only=True, pack_prefix=True, prefix_pack_bucket_size=4,
        )).eval()
    with torch.no_grad():
        for module in model.modules():
            if type(module) is probe.core.Pi05AdaRMSNorm:
                module.dense.weight.normal_(std=0.04)
                module.dense.bias.normal_(std=0.04)
    set_bfloat16_layout(model)
    model.to(device)
    install_execution_recipe(model, ExecutionRecipe(fuse_vision_residual_cast=True))
    inputs = {
        "images": [torch.randn(1, 3, 28, 28, device=device), torch.randn(1, 3, 28, 28, device=device),
                   -torch.ones(1, 3, 28, 28, device=device)],
        "image_masks": [torch.tensor([True], device=device), torch.tensor([True], device=device), torch.tensor([False], device=device)],
        "lang_tokens": torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]], device=device),
        "lang_masks": torch.tensor([[True, True, True, True, False, False, False, False]], device=device),
        "noise": torch.randn(1, 4, 8, device=device),
    }
    return model, inputs


def test_real_siglip_two_camera_batch_preserves_norm_dtype_features_and_actions(probe, monkeypatch):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(54)
            model, inputs = _small_core(probe, "cpu")
            inner = model.paligemma_with_expert
            with torch.no_grad():
                context = model.prepare_inference_context(inputs["images"], inputs["image_masks"], inputs["lang_masks"], 2)
                empty = context["empty_features"]
                assert [x is None for x in empty] == [True, True, False]
                expected_prefix = model.embed_prefix(**{k: v for k, v in inputs.items() if k != "noise"}, empty_features=empty)
                expected = model.sample_actions(**inputs)
            norms = {name: (id(module), module.weight.dtype, module.bias.dtype)
                     for name, module in inner.paligemma.model.vision_tower.named_modules()
                     if isinstance(module, nn.LayerNorm)}
            parameters = {name: id(value) for name, value in model.named_parameters()}
            handle = install_batched_vision(model)
            calls = []
            original = inner.embed_image
            def spy(pixels):
                calls.append(tuple(pixels.shape))
                return original(pixels)
            monkeypatch.setattr(inner, "embed_image", spy)
            with torch.no_grad():
                actual_prefix = model.embed_prefix(**{k: v for k, v in inputs.items() if k != "noise"}, empty_features=empty)
            assert calls == [(2, 3, 28, 28)]
            for result, reference in zip(actual_prefix, expected_prefix, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            actual = model.sample_actions(**inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert {name: id(value) for name, value in model.named_parameters()} == parameters
            assert {name: (id(module), module.weight.dtype, module.bias.dtype)
                    for name, module in inner.paligemma.model.vision_tower.named_modules()
                    if isinstance(module, nn.LayerNorm)} == norms
            assert all(weight == bias == torch.float32 for _, weight, bias in norms.values())
            handle.uninstall()
            torch.testing.assert_close(model.sample_actions(**inputs), expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def _metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = (a - b).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "different_fraction": (a != b).float().mean().item(), "rmse": delta.square().mean().sqrt().item()}


def test_batched_native_prefix_is_cpu_fullgraph_traceable(probe):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, inputs = _small_core(probe, "cpu")
        handle = install_batched_vision(model)
        with torch.inference_mode():
            context = model.prepare_inference_context(inputs["images"], inputs["image_masks"], inputs["lang_masks"], 2)
            empty = context["empty_features"]
            arguments = {k: v for k, v in inputs.items() if k != "noise"}
            compiled = torch.compile(model.embed_prefix, backend="eager", fullgraph=True, dynamic=False)
            actual = compiled(**arguments, empty_features=empty)
            reference = model.embed_prefix(**arguments, empty_features=empty)
            for result, expected in zip(actual, reference, strict=True):
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        del compiled
        handle.uninstall()
    finally:
        torch.set_num_threads(previous_threads)


def _native_vision_model(device, *, width=1152, layers=2, image_size=224):
    """Actual SigLIP at production width; reduced depth keeps contracts bounded."""
    from transformers import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from vvla_repro_pi05.lite.dtype import _vision_norms_fp32

    config = SiglipVisionConfig(hidden_size=width, intermediate_size=4304, num_hidden_layers=layers,
                               num_attention_heads=16, image_size=image_size, patch_size=14,
                               attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = ToyModel()
    inner = model.paligemma_with_expert
    inner.paligemma.config.vision_config = config
    inner.paligemma.model.vision_tower = SiglipVisionModel(config).to(dtype=torch.bfloat16)
    _vision_norms_fp32(inner.paligemma.model.vision_tower)
    inner.paligemma.model.multi_modal_projector = nn.Linear(width, 2048, dtype=torch.bfloat16)
    inner.language = nn.Embedding(16, 2048, dtype=torch.bfloat16)
    inner.vision_output_dtype = torch.float32
    inner.embed_image = MethodType(PaliGemmaWithActionExpertPi05.embed_image, inner)
    model.vlm_width = 2048
    return model.to(device).eval()


@contextmanager
def _full_fp32_reference_math():
    """Independent reference only; restore process math settings afterwards."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _independent_fp32_camera_features(inner, images):
    """Promote the same BF16 model values and pixels; encode each row alone.

    This is a precision oracle, not the original checkpoint in FP32: BF16
    projection weights are promoted exactly, while original FP32 norm affine
    values stay FP32. Inputs first undergo native BF16 pixel rounding, then are
    promoted. Every subsequent layer/attention/projector intermediate is FP32.
    """
    vision = copy.deepcopy(inner.paligemma.model.vision_tower).float().eval()
    projector = copy.deepcopy(inner.paligemma.model.multi_modal_projector).float().eval()
    with torch.inference_mode(), _full_fp32_reference_math():
        result = [projector(vision(image.bfloat16().float()).last_hidden_state) for image in images]
    return result


def _cross_batch_evidence(batch, serial, fp32_rows):
    """Persist the failed former elementwise gate instead of widening it."""
    difference = (batch.float() - serial.float()).abs()
    matches = torch.isclose(batch, serial, rtol=0.008, atol=0.008)
    worst = difference.flatten().topk(min(8, difference.numel())).indices
    rows = []
    for index in worst.tolist():
        camera, within = divmod(index, batch.shape[1] * batch.shape[2])
        token, channel = divmod(within, batch.shape[2])
        rows.append({"camera": camera, "token": token, "channel": channel,
                     "batched_bf16": batch[camera, token, channel].item(),
                     "serial_bf16": serial[camera, token, channel].item(),
                     "independent_fp32": fp32_rows[camera][0, token, channel].item()})
    return {
        "quality_acceptance": False,
        "former_cross_batch_gate": {"rtol": 0.008, "atol": 0.008,
                                    "scope": "two observed camera features only; cached slot and text excluded",
                                    "mismatches": (~matches).sum().item(), "elements": matches.numel()},
        "batched_vs_serial": _metrics(batch, serial), "largest_differences": rows,
        "per_camera": [{"camera": i, "serial_vs_fp32": _metrics(serial[i:i + 1], reference),
                        "batch_vs_fp32": _metrics(batch[i:i + 1], reference)}
                       for i, reference in enumerate(fp32_rows)],
    }


def test_independent_fp32_reference_preserves_source_values_and_reports_each_camera():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = _native_vision_model("cpu", width=32, layers=1, image_size=28)
        inner = model.paligemma_with_expert
        images = [torch.randn(1, 3, 28, 28) for _ in range(2)]
        parameters = {name: (id(value), value.dtype, value.clone()) for name, value in model.named_parameters()}
        precision = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        fp32 = _independent_fp32_camera_features(inner, images)
        assert all(value.dtype == torch.float32 and value.shape == (1, 4, 2048) for value in fp32)
        assert precision == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with torch.inference_mode():
            serial = torch.cat([inner.embed_image(image) for image in images])
            batch = inner.embed_image(torch.cat(images))
        evidence = _cross_batch_evidence(batch, serial, fp32)
        assert evidence["quality_acceptance"] is False and len(evidence["per_camera"]) == 2
        assert all(row["serial_vs_fp32"]["rmse"] > 0 for row in evidence["per_camera"])
        for name, value in model.named_parameters():
            identity, dtype, before = parameters[name]
            assert id(value) == identity and value.dtype == dtype
            torch.testing.assert_close(value, before, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA camera batching deferred to GPU queue owner")
def test_cuda_actual_width_siglip_b2_batch_features_and_direct_capture(record_property):
    torch.manual_seed(827)
    model = _native_vision_model("cuda")
    inner = model.paligemma_with_expert
    attention = inner.paligemma.model.vision_tower.encoder.layers[0].self_attn
    assert (attention.embed_dim, attention.num_heads, attention.head_dim) == (1152, 16, 72)
    inputs = toy_inputs()
    inputs["images"] = [torch.randn(1, 3, 224, 224, device="cuda") for _ in range(2)] + [-torch.ones(1, 3, 224, 224, device="cuda")]
    inputs["image_masks"] = [torch.tensor([True], device="cuda"), torch.tensor([True], device="cuda"), torch.tensor([False], device="cuda")]
    inputs["lang_tokens"] = inputs["lang_tokens"].cuda()
    inputs["lang_masks"] = inputs["lang_masks"].cuda()
    with torch.inference_mode():
        empty = [None, None, inner.embed_image(inputs["images"][2])]
        reference = model.embed_prefix(**inputs, empty_features=empty)
        # This oracle never calls the candidate helper: it is the original
        # native B2 SigLIP/projector with cameras explicitly split into rows.
        native_batch = inner.embed_image(torch.cat(inputs["images"][:2], dim=0))
        native_expected = torch.cat([native_batch[0:1], native_batch[1:2], empty[2],
                                     inner.embed_language_tokens(inputs["lang_tokens"])], dim=1)
        serial = reference[0][:, :512].reshape(2, 256, 2048)
    fp32_rows = _independent_fp32_camera_features(inner, inputs["images"][:2])
    assert all(torch.isfinite(value).all() for value in fp32_rows)
    record_property("cross_batch_precision_evidence", json.dumps(
        _cross_batch_evidence(native_batch, serial, fp32_rows), sort_keys=True))
    handle = install_batched_vision(model)
    with torch.inference_mode():
        candidate = model.embed_prefix(**inputs, empty_features=empty)
        record_property("actual_width_b2_feature_error", json.dumps(_metrics(candidate[0], reference[0]), sort_keys=True))
        assert candidate[0].dtype == torch.float32 and torch.isfinite(candidate[0]).all()
        # Exact helper semantics. Native B2 vs native B1 numerical differences
        # are recorded above; this is not a relaxed cross-batch quality gate.
        torch.testing.assert_close(candidate[0], native_expected, rtol=0, atol=0)
        for result, expected in zip(candidate[1:], reference[1:], strict=True):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model.embed_prefix(**inputs, empty_features=empty)
            model.embed_prefix(**inputs, empty_features=empty)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = model.embed_prefix(**inputs, empty_features=empty)
        graph.replay()
        torch.testing.assert_close(captured[0], candidate[0], rtol=0, atol=0)
        owned = captured[0].clone()
        inputs["images"][0].mul_(0.5).add_(0.1)
        inputs["lang_tokens"][0, 0] = 2
        refreshed = model.embed_prefix(**inputs, empty_features=empty)
        graph.replay()
        torch.testing.assert_close(captured[0], refreshed[0], rtol=0, atol=0)
        assert not torch.equal(owned, captured[0])
        torch.testing.assert_close(owned, candidate[0], rtol=0, atol=0)
    del graph
    handle.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA camera batching deferred to GPU queue owner")
def test_cuda_native_graph_mask_refresh_empty_pattern_and_weight_invalidation(probe, record_property):
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    torch.manual_seed(891)
    model, inputs = _small_core(probe, "cuda")
    native_forward = model.embed_prefix
    expected = model.sample_actions(**inputs)
    handle = install_batched_vision(model)
    runner = Pi05NativeGraphRunner(model)
    try:
        actual = runner(**inputs, num_steps=2)
        record_property("small_native_action_error", json.dumps(_metrics(actual, expected), sort_keys=True))
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
        saved = actual.clone()
        # The same number of valid tokens preserves packed capacity and tests
        # metadata refresh rather than merely adding a new graph.
        inputs["lang_masks"][0, :5] = torch.tensor([True, False, True, True, True], device="cuda")
        with torch.inference_mode():
            eager = model.sample_actions(**inputs)
        changed = runner(**inputs, num_steps=2)
        torch.testing.assert_close(changed, eager, rtol=0, atol=0)
        assert runner.last_call["metadata_updated"]
        # A real camera becomes an empty cached slot: batch geometry changes
        # from two to one uncached image and requires another graph entry.
        original_camera = inputs["images"][1].clone()
        inputs["images"][1].fill_(-1)
        inputs["image_masks"][1].fill_(False)
        changed = runner(**inputs, num_steps=2)
        torch.testing.assert_close(changed, model.sample_actions(**inputs), rtol=0, atol=0)
        assert runner.stats["captures"] >= 2
        inputs["images"][1].copy_(original_camera)
        inputs["image_masks"][1].fill_(True)
        inputs["lang_masks"][0] = torch.tensor([True, True, True, True, False, False, False, False], device="cuda")
        restored = runner(**inputs, num_steps=2)
        torch.testing.assert_close(restored, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        # Empty image features must be recomputed after a projector weight edit;
        # this helper owns no cache that could bypass the original generation.
        with torch.no_grad():
            projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear
            projector.weight.add_(0.01)
        updated = runner(**inputs, num_steps=2)
        assert runner.stats["invalidations"] >= 1
        torch.testing.assert_close(updated, model.sample_actions(**inputs), rtol=0, atol=0)
        assert getattr(native_forward, "__func__") is Pi05ForActionPrediction.embed_prefix
    finally:
        runner.close()
        handle.uninstall()
