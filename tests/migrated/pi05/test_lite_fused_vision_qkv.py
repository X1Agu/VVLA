"""Biased SigLIP QKV: CPU lifecycle and deferred production-width CUDA tests."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipAttention

from vvla_repro_pi05.lite.fused_vision_qkv import INSTALLATION, PACKER, install_vision_qkv
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from test_lite_skip_weight_init import probe


class SmallVision(nn.Module):
    prepare_inference_context = Pi05ForActionPrediction.prepare_inference_context

    def __init__(self, *, width=32, heads=4, layers=2, device="cpu", backend="eager"):
        super().__init__()
        self.config = SimpleNamespace(min_period=0.004, max_period=4.0)
        self._inference_cache = Pi05InferenceCache()
        self.paligemma_with_expert = nn.Module()
        inner = self.paligemma_with_expert
        inner.paligemma = nn.Module()
        inner.paligemma.model = nn.Module()
        tower = inner.paligemma.model.vision_tower = nn.Module()
        tower.encoder = nn.Module()
        tower.encoder.layers = nn.ModuleList()
        config = SiglipVisionConfig(hidden_size=width, num_attention_heads=heads, attention_dropout=0.0)
        config._attn_implementation = backend
        for _ in range(layers):
            layer = nn.Module()
            layer.self_attn = SiglipAttention(config).to(device=device, dtype=torch.bfloat16)
            tower.encoder.layers.append(layer)
        self.eval()

    @property
    def layers(self):
        return self.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers

    def optimization_flags(self):
        return False, False, False, False, False

    def invalidate_inference_caches(self, **kwargs):
        self._inference_cache.clear()


def prepare(model):
    device = next(model.parameters()).device
    context = model.prepare_inference_context(
        [torch.ones(1, 3, 2, 2, device=device)], [torch.ones(1, device=device, dtype=torch.bool)],
        torch.ones(1, 4, device=device, dtype=torch.bool), 2, positions=False,
    )
    # Also valid while this standalone helper precedes policy/core integration;
    # the real hook is required before empty image encoding and tested below.
    getattr(model, INSTALLATION).prepare(context["token"][1])
    return context


def test_policy_option_parse_and_combined_compile_close_without_cuda_allocation(monkeypatch, tmp_path):
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config

    core, options = pi05._split_lite_overrides({})
    assert core == {} and options["fused_vision_qkv"] is False
    core, options = pi05._split_lite_overrides({"lite_fused_vision_qkv": True})
    assert core == {} and options["fused_vision_qkv"] is True
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid vision QKV option reached model allocation")
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    with pytest.raises(ValueError):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, config_overrides={"lite_fused_vision_qkv": 1})

    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2", vision_bf16=True))
        set_bfloat16_layout(model)
    model.eval()
    policy = pi05.Pi05LitePolicy(model, _policy().processor, device="cuda", cuda_graph=True,
                                 compile_sample_actions=True, compile_preserve_action_linears=True,
                                 batched_vision=True, fused_vision_qkv=True)
    vision, qkv = policy._vision_installation, policy._vision_qkv_installation
    linears, runner = policy._preserved_linears_installation, policy.graph_runner
    metadata = policy.metadata()["lite_optimizations"]
    assert metadata["fused_vision_qkv"] is True
    assert metadata["fused_vision_qkv_manifest"]["bias_dtype"] == "bfloat16"
    assert metadata["fused_vision_qkv_manifest"]["layers"][0] == {
        "layer": 0, "width": 1152, "heads": 16, "head_dim": 72,
    }
    assert runner._device is None  # Meta construction never allocated CUDA state.
    policy.close()
    assert runner._closed and not vision.enabled and not qkv.enabled and not linears.enabled
    assert "embed_prefix" not in model.__dict__
    assert all("forward" not in layer.self_attn.__dict__
               for layer in model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers)
    policy.close()


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_cpu_biased_projection_and_attention_parity_for_independent_batch_rows(batch, backend):
    torch.manual_seed(19)
    model = SmallVision(backend=backend)
    attention = model.layers[0].self_attn
    x = torch.randn(batch, 7, 32).bfloat16()
    mask = torch.zeros(batch, 1, 7, 7)
    mask[..., -1] = -1e9
    with torch.no_grad():
        expected = attention(x, attention_mask=mask)
        projections = tuple(getattr(attention, name)(x) for name in ("q_proj", "k_proj", "v_proj"))
    modules = {name: id(value) for name, value in model.named_modules()}
    parameters = {name: id(value) for name, value in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    handle = install_vision_qkv(model)
    prepare(model)
    with torch.no_grad():
        actual_projections = getattr(attention, PACKER).project(x)
        actual = attention(x, attention_mask=mask)
    for result, reference in zip(actual_projections, projections, strict=True):
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for result, reference in zip(actual, expected, strict=True):
        if result is None:
            assert reference is None
        else:
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert {name: id(value) for name, value in model.named_modules()} == modules
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    manifest = handle.manifest()
    assert manifest["bias_dtype"] == "bfloat16"
    assert manifest["packed_weight_bytes"] == 2 * 3 * 32 * 32 * 2
    assert manifest["packed_bias_bytes"] == 2 * 3 * 32 * 2
    handle.uninstall()
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.layers)
    assert not hasattr(model, INSTALLATION)


def test_original_biased_linear_boundary_differs_from_a_separate_bf16_bias_add():
    torch.manual_seed(557)
    x, weight, bias = (torch.randn(3, 7, 32).bfloat16(), torch.randn(32, 32).bfloat16(), torch.randn(32).bfloat16())
    combined = F.linear(x, weight, bias)
    separately_rounded = F.linear(x, weight) + bias
    assert torch.count_nonzero(combined != separately_rounded) > 0
    with pytest.raises(RuntimeError, match="same dtype"):
        F.linear(x, weight, bias.float())


@pytest.mark.parametrize("heads", [12, 16])
def test_cpu_real_width_accepts_both_head_geometries_without_changing_them(heads):
    model = SmallVision(width=1152, heads=heads, layers=1)
    attention = model.layers[0].self_attn
    x = torch.randn(2, 3, 1152).bfloat16()
    with torch.no_grad():
        expected = attention(x)[0]
    handle = install_vision_qkv(model)
    prepare(model)
    with torch.no_grad():
        torch.testing.assert_close(attention(x)[0], expected, rtol=0, atol=0)
    assert handle.manifest()["layers"] == [{"layer": 0, "width": 1152, "heads": heads, "head_dim": 1152 // heads}]
    handle.uninstall()


def test_cache_generation_refresh_preserves_addresses_and_tracks_weight_and_bias():
    model = SmallVision()
    attention = model.layers[0].self_attn
    handle = install_vision_qkv(model)
    first_context = prepare(model)
    packer = getattr(attention, PACKER)
    addresses = packer._weight.data_ptr(), packer._bias.data_ptr()
    manifest = handle.manifest()
    x = torch.randn(2, 5, 32).bfloat16()
    prepare(model)
    assert handle.refreshes == 1
    with torch.no_grad():
        attention.q_proj.weight.add_(0.125)
        attention.v_proj.bias.add_(0.5)
        with pytest.raises(RuntimeError, match="source changed"):
            attention(x)
        with pytest.raises(RuntimeError, match="source changed"):
            handle.prepare(first_context["token"][1])
        second_context = prepare(model)
        assert first_context["token"] != second_context["token"] and handle.refreshes == 2
        torch.testing.assert_close(attention(x)[0], SiglipAttention.forward(attention, x)[0], rtol=0, atol=0)
        # Explicit invalidation is required for nonstandard .data writes.
        attention.k_proj.bias.data.add_(0.2)
        attention.k_proj.weight.data.mul_(0.75)
        model.invalidate_inference_caches()
        prepare(model)
        torch.testing.assert_close(attention(x)[0], SiglipAttention.forward(attention, x)[0], rtol=0, atol=0)
    assert (packer._weight.data_ptr(), packer._bias.data_ptr()) == addresses
    assert handle.refreshes == 3 and handle.manifest() == manifest
    handle.uninstall()


@pytest.mark.parametrize("fault", ["fp32_bias", "absent_bias", "head_geometry", "projection_hook", "projection_forward"])
def test_unsupported_projection_contract_fails_before_any_patch(fault):
    model = SmallVision()
    attention = model.layers[1].self_attn
    if fault == "fp32_bias":
        attention.q_proj.bias = nn.Parameter(attention.q_proj.bias.float())
    elif fault == "absent_bias":
        attention.k_proj.bias = None
    elif fault == "head_geometry":
        attention.head_dim += 1
    elif fault == "projection_hook":
        attention.v_proj.register_forward_hook(lambda module, args, output: output)
    elif fault == "projection_forward":
        attention.q_proj.forward = lambda x: x
    with pytest.raises(ValueError):
        install_vision_qkv(model)
    assert not hasattr(model, INSTALLATION)
    assert all("forward" not in layer.self_attn.__dict__ and not hasattr(layer.self_attn, PACKER) for layer in model.layers)


def test_all_geometry_validates_before_copy_and_late_hooks_or_cache_tampering_fail():
    model = SmallVision()
    handle = install_vision_qkv(model)
    prepare(model)
    saved = [(p._weight.clone(), p._bias.clone()) for p in handle._packers]
    with torch.no_grad():
        model.layers[0].self_attn.q_proj.weight.add_(1)
    original_bias = model.layers[1].self_attn.v_proj.bias
    model.layers[1].self_attn.v_proj.bias = nn.Parameter(original_bias.float())
    with pytest.raises(ValueError, match="BF16 weight AND bias"):
        prepare(model)
    for packer, (weight, bias) in zip(handle._packers, saved, strict=True):
        torch.testing.assert_close(packer._weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(packer._bias, bias, rtol=0, atol=0)
    model.layers[1].self_attn.v_proj.bias = original_bias
    prepare(model)
    hook = model.layers[0].self_attn.q_proj.register_forward_hook(lambda module, args, output: output)
    with pytest.raises(ValueError, match="without forward hooks"):
        prepare(model)
    hook.remove()
    with torch.no_grad():
        handle._packers[0]._bias.add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        prepare(model)
    handle.uninstall()


def test_training_gradients_and_autocast_fall_back_without_prepared_cache():
    model = SmallVision(layers=1)
    attention = model.layers[0].self_attn
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    reference = attention(x)[0]
    handle = install_vision_qkv(model)
    actual = attention(x)[0]
    variables = (x, *attention.parameters())
    reference_grad = torch.autograd.grad(reference.float().sum(), variables)
    actual_grad = torch.autograd.grad(actual.float().sum(), variables)
    for result, expected in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    attention.train()
    with torch.no_grad():
        torch.testing.assert_close(attention(x)[0], SiglipAttention.forward(attention, x)[0], rtol=0, atol=0)
    attention.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(attention(x)[0], SiglipAttention.forward(attention, x)[0], rtol=0, atol=0)
    handle.uninstall()


def test_graph_retirement_guard_and_foreign_attention_patch():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = SmallVision()
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        install_vision_qkv(model)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_vision_qkv(model)
    finally:
        runner._call_lock.release()
    with torch.inference_mode(), pytest.raises(ValueError, match="outside inference_mode"):
        install_vision_qkv(model)
    handle = install_vision_qkv(model)
    prepare(model)
    with pytest.raises(ValueError, match="already installed"):
        install_vision_qkv(model)
    attention = model.layers[0].self_attn
    own_forward = attention.forward
    attention.forward = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="replaced externally"):
        prepare(model)
    with pytest.raises(RuntimeError, match="replaced externally"):
        handle.uninstall()
    attention.forward = own_forward
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        handle.uninstall()
    runner.close()
    handle.uninstall()


def test_real_siglip_batched_vision_composition_and_empty_cache_generation_hook(probe):
    from test_lite_batched_vision import _small_core
    from vvla_repro_pi05.lite.batched_vision import install_batched_vision

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, inputs = _small_core(probe, "cpu")
        expected = model.sample_actions(**inputs)
        modules = {name: id(value) for name, value in model.named_modules()}
        vision = install_batched_vision(model)
        qkv = install_vision_qkv(model)
        # This relies on the actual shared lifecycle hook: empty image encoding
        # itself uses QKV and would fail if packing ran after cache preparation.
        actual = model.sample_actions(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert qkv.refreshes == 1
        assert {name: id(value) for name, value in model.named_modules()} == modules
        generation = qkv._generation
        with torch.no_grad():
            model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers[0].self_attn.q_proj.bias.add_(0.1)
        changed = model.sample_actions(**inputs)
        assert qkv._generation != generation and qkv.refreshes == 2
        qkv.uninstall()
        vision.uninstall()
        torch.testing.assert_close(model.sample_actions(**inputs), changed, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_prepared_native_compile_scope_and_stale_parameter_rejection():
    from vvla_repro_pi05.lite.compilation import compile_prepared_run
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = SmallVision(layers=1)
    handle = install_vision_qkv(model)
    context = prepare(model)
    owner = Pi05NativeGraphRunner(model)
    owner._token = context["token"]
    x = torch.randn(2, 5, 32).bfloat16()
    def run():
        return model.layers[0].self_attn(x)[0]
    compiled = compile_prepared_run(model, run, backend="eager", owner=owner, context=context)
    try:
        with torch.inference_mode():
            owner._call_lock.acquire()
            try:
                torch.testing.assert_close(compiled(), run(), rtol=0, atol=0)
                assert handle._packers[0]._prepared_compile_token is None
                model.layers[0].self_attn.k_proj.bias.add_(0.1)
                with pytest.raises(RuntimeError, match="changed"):
                    compiled()
            finally:
                owner._call_lock.release()
    finally:
        compiled.retire()
        owner.close()
        handle.uninstall()


def _metrics(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "different_fraction": (actual != expected).float().mean().item(),
            "rmse": delta.square().mean().sqrt().item()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA vision QKV deferred to GPU queue owner")
@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("heads", [12, 16])
def test_cuda_width1152_both_head_geometries_projection_and_attention_parity(batch, backend, heads, record_property):
    torch.manual_seed(587)
    model = SmallVision(width=1152, heads=heads, layers=1, device="cuda", backend=backend)
    attention = model.layers[0].self_attn
    x = torch.randn(batch, 256, 1152, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(batch, 1, 256, 256, device="cuda", dtype=torch.bfloat16)
    mask[..., -7:] = -1e9
    with torch.inference_mode():
        reference_qkv = tuple(getattr(attention, name)(x) for name in ("q_proj", "k_proj", "v_proj"))
        reference = attention(x, attention_mask=mask)
    handle = install_vision_qkv(model)
    prepare(model)
    with torch.inference_mode():
        candidate_qkv = getattr(attention, PACKER).project(x)
        candidate = attention(x, attention_mask=mask)
        for name, result, expected in zip(("q", "k", "v"), candidate_qkv, reference_qkv, strict=True):
            record_property(name + "_error", json.dumps(_metrics(result, expected), sort_keys=True))
            torch.testing.assert_close(result, expected, rtol=0.008, atol=0.008)
        record_property("attention_output_error", json.dumps(_metrics(candidate[0], reference[0]), sort_keys=True))
        assert torch.isfinite(candidate[0]).all() and candidate[0].dtype == torch.bfloat16
        torch.testing.assert_close(candidate[0], reference[0], rtol=0.008, atol=0.008)
        if candidate[1] is not None:
            torch.testing.assert_close(candidate[1], reference[1], rtol=0.008, atol=0.008)
    handle.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA vision QKV deferred to GPU queue owner")
def test_cuda_packed_weight_bias_addresses_capture_refresh_and_owned_result():
    model = SmallVision(width=1152, heads=12, layers=1, device="cuda", backend="sdpa")
    attention = model.layers[0].self_attn
    handle = install_vision_qkv(model)
    prepare(model)
    packer = getattr(attention, PACKER)
    addresses = packer._weight.data_ptr(), packer._bias.data_ptr()
    x = torch.randn(2, 256, 1152, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            attention(x); attention(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = attention(x)[0]
        graph.replay()
        initial = attention(x)[0]
        torch.testing.assert_close(captured, initial, rtol=0, atol=0)
        owned = captured.clone()
        attention.k_proj.weight.add_(0.001)
        attention.q_proj.bias.add_(0.03)
        prepare(model)
        x.mul_(0.8)
        eager = attention(x)[0]
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(owned, initial, rtol=0, atol=0)
        assert not torch.equal(owned, captured)
    assert addresses == (packer._weight.data_ptr(), packer._bias.data_ptr())
    del graph
    handle.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA vision QKV deferred to GPU queue owner")
def test_cuda_real_core_camera_batches_B2_B3_B1_B2_and_weight_invalidation(probe, record_property):
    from test_lite_batched_vision import _small_core
    from vvla_repro_pi05.lite.batched_vision import install_batched_vision
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model, inputs = _small_core(probe, "cuda")
    reference = model.sample_actions(**inputs)
    vision = install_batched_vision(model)
    qkv = install_vision_qkv(model)
    runner = Pi05NativeGraphRunner(model)
    try:
        first = runner(**inputs, num_steps=2)
        saved = first.clone()
        record_property("small_core_action_error", json.dumps(_metrics(first, reference), sort_keys=True))
        torch.testing.assert_close(first, reference, rtol=0.008, atol=0.008)
        inputs["images"][2].normal_()
        inputs["image_masks"][2].fill_(True)
        three = runner(**inputs, num_steps=2)
        torch.testing.assert_close(three, model.sample_actions(**inputs), rtol=0, atol=0)
        camera_two = inputs["images"][1].clone()
        for slot in (1, 2):
            inputs["images"][slot].fill_(-1)
            inputs["image_masks"][slot].fill_(False)
        one = runner(**inputs, num_steps=2)
        torch.testing.assert_close(one, model.sample_actions(**inputs), rtol=0, atol=0)
        inputs["images"][1].copy_(camera_two)
        inputs["image_masks"][1].fill_(True)
        restored = runner(**inputs, num_steps=2)
        torch.testing.assert_close(restored, first, rtol=0, atol=0)
        assert runner.stats["captures"] >= 3
        with torch.no_grad():
            model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers[0].self_attn.v_proj.bias.add_(0.03)
        updated = runner(**inputs, num_steps=2)
        torch.testing.assert_close(updated, model.sample_actions(**inputs), rtol=0, atol=0)
        assert runner.stats["invalidations"] >= 1
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
    finally:
        runner.close()
        qkv.uninstall()
        vision.uninstall()
