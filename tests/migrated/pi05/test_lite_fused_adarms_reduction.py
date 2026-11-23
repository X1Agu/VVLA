"""Opt-in FP32 AdaRMS reduction: CPU contracts and deferred CUDA error checks."""

import json

import pytest
import torch

from test_lite_fused_adarms import SmallExpert, norm
from vvla_repro_pi05.lite.fused_adarms import fused_adarms, install_fused_adarms
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05AdaRMSNorm


@pytest.mark.parametrize("conditioned", [True, False])
@pytest.mark.parametrize("input_dtype,output_dtype", [
    (torch.float32, torch.bfloat16), (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
])
def test_cpu_reduction_option_retains_native_semantics_and_every_dtype_boundary(conditioned, input_dtype, output_dtype):
    module = norm(conditioned=conditioned)
    x = torch.linspace(-2, 2, 48).reshape(2, 3, 8).to(input_dtype)
    cond = torch.randn(2, 8) * 0.1
    with torch.inference_mode():
        modulation = module.dense(cond) if conditioned else None
        expected, gate = Pi05AdaRMSNorm.forward(module, x, cond, modulation)
        control, control_gate = fused_adarms(module, x, cond, modulation, output_dtype=output_dtype)
        actual, actual_gate = fused_adarms(module, x, cond, modulation, output_dtype=output_dtype, fuse_reduction=True)
    assert actual.dtype == output_dtype
    torch.testing.assert_close(actual, expected.to(output_dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual, control, rtol=0, atol=0)
    if conditioned:
        assert actual_gate.dtype == input_dtype
        torch.testing.assert_close(actual_gate, gate, rtol=0, atol=0)
        torch.testing.assert_close(actual_gate, control_gate, rtol=0, atol=0)
    else:
        assert actual_gate is control_gate is gate is None


def test_default_control_and_optional_manifest_preserve_modules_parameters_and_fp32_final_norm():
    model = SmallExpert()
    modules = {name: id(value) for name, value in model.named_modules()}
    parameters = {name: id(value) for name, value in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    control = install_fused_adarms(model)
    assert control.fuse_reduction is False
    control_manifest = control.manifest()
    assert control_manifest["reduction"] == "torch.mean(torch.square(x.float()), dim=-1, keepdim=True)"
    control.uninstall()
    candidate = install_fused_adarms(model, fuse_reduction=True)
    assert candidate.fuse_reduction is True
    manifest = candidate.manifest()
    assert manifest["version"] != control_manifest["version"]
    assert manifest["fuse_reduction"] is True
    assert "may differ from Torch" in manifest["reduction_tree"]
    assert manifest["square_dtype"] == manifest["reduction_dtype"] == manifest["rsqrt_dtype"] == "float32"
    assert manifest["enable_fp_fusion"] is False
    assert manifest["norms"][-1] == {"name": "expert.norm", "output_dtype": "torch.float32"}
    with pytest.raises(AttributeError):
        candidate.fuse_reduction = False
    assert {name: id(value) for name, value in model.named_modules()} == modules
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    expert = model.paligemma_with_expert.gemma_expert.model
    with torch.inference_mode():
        x, cond = torch.randn(2, 3, 8), torch.randn(2, 8)
        assert expert.layers[0].input_layernorm(x, cond)[0].dtype == torch.bfloat16
        assert expert.layers[0].post_attention_layernorm(x, cond)[0].dtype == torch.bfloat16
        assert expert.norm(x, cond)[0].dtype == torch.float32
    assert candidate.manifest() == manifest
    candidate.uninstall()
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert model.invalidations == [False] * 4


@pytest.mark.parametrize("value", [1, "true", None])
def test_reduction_option_is_strict_boolean_before_any_patch(value):
    model = SmallExpert()
    with pytest.raises(ValueError, match="must be a boolean"):
        install_fused_adarms(model, fuse_reduction=value)
    assert not any("forward" in module.__dict__ for module in model.modules())
    assert model.invalidations == []
    with pytest.raises(ValueError, match="must be a boolean"):
        fused_adarms(norm(), torch.ones(1, 8), torch.ones(1, 8), output_dtype=torch.bfloat16,
                     fuse_reduction=value)


def test_reduction_switch_requires_retiring_graphs_and_eval_mutation():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = SmallExpert()
    control = install_fused_adarms(model)
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        control.uninstall()
    with pytest.raises(ValueError, match="Clear all live"):
        install_fused_adarms(model, fuse_reduction=True)
    runner.clear()
    control.uninstall()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_fused_adarms(model, fuse_reduction=True)
    finally:
        runner._call_lock.release()
    with torch.inference_mode(), pytest.raises(ValueError, match="outside inference_mode"):
        install_fused_adarms(model, fuse_reduction=True)
    candidate = install_fused_adarms(model, fuse_reduction=True)
    runner.close()
    candidate.uninstall()


def test_reduction_option_preserves_training_gradients_and_cpu_fullgraph_trace():
    module = norm().train()
    x = torch.randn(2, 3, 8, requires_grad=True)
    cond = torch.randn(2, 8, requires_grad=True)
    expected, expected_gate = Pi05AdaRMSNorm.forward(module, x, cond)
    actual, gate = fused_adarms(module, x, cond, output_dtype=torch.bfloat16, fuse_reduction=True)
    variables = (x, cond, module.dense.weight, module.dense.bias)
    expected_gradients = torch.autograd.grad(expected.bfloat16().float().sum() + expected_gate.sum(), variables)
    actual_gradients = torch.autograd.grad(actual.float().sum() + gate.sum(), variables)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    module.eval()
    def forward(value, modulation):
        return fused_adarms(module, value, modulation=modulation,
                            output_dtype=torch.bfloat16, fuse_reduction=True)
    with torch.inference_mode():
        modulation = module.dense(cond.detach())
        compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=False)
        actual = compiled(x.detach(), modulation)
        expected = forward(x.detach(), modulation)
        for actual, expected in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _extreme_inputs(device, dtype):
    """Finite inputs, including square underflow, sum/square overflow and spikes."""
    row = torch.linspace(-1, 1, 1024, device=device, dtype=torch.float32)
    values = torch.stack([
        torch.zeros_like(row), torch.full_like(row, 1e-20), row * 1e8,
        torch.ones_like(row) + torch.arange(1024, device=device).remainder(7) * 2**-22,
        torch.where(torch.arange(1024, device=device) % 2 == 0, row * 1e8, row * 1e-6),
        torch.nn.functional.pad(torch.tensor([1e18], device=device), (0, 1023)),
        torch.full_like(row, 1e20), -torch.full_like(row, 1e20),
    ])
    return values.repeat(13, 1)[:100].reshape(2, 50, 1024).to(dtype)


def _error_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = (actual - expected).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "different_fraction": (actual != expected).float().mean().item()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction candidate deferred to GPU queue owner")
@pytest.mark.parametrize("scenario", ["random", "extreme"])
@pytest.mark.parametrize("input_dtype,output_dtype", [
    (torch.float32, torch.bfloat16), (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
])
def test_cuda_width1024_errors_against_native_and_original_tail(scenario, input_dtype, output_dtype, record_property):
    pytest.importorskip("triton")
    module = norm(1024, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(8821)
    x = (torch.randn((2, 50, 1024), device="cuda", generator=generator, dtype=input_dtype)
         if scenario == "random" else _extreme_inputs("cuda", input_dtype))
    # Direct modulation isolates the reduction from a differently scheduled GEMM.
    modulation = torch.randn((2, 3072), device="cuda", generator=generator) * 0.1
    with torch.inference_mode():
        expected, expected_gate = Pi05AdaRMSNorm.forward(module, x, modulation=modulation)
        expected = expected.to(output_dtype)
        control, control_gate = fused_adarms(module, x, modulation=modulation, output_dtype=output_dtype)
        actual, gate = fused_adarms(module, x, modulation=modulation, output_dtype=output_dtype, fuse_reduction=True)
        assert actual.dtype == output_dtype and gate.dtype == input_dtype
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        bf16_boundary = input_dtype == torch.bfloat16 or output_dtype == torch.bfloat16
        tolerance = 0.008 if bf16_boundary else 3e-6
        for name, reference in (("native", expected), ("torch_mean_tail", control)):
            record_property(name + "_error", json.dumps(_error_metrics(actual, reference), sort_keys=True))
            torch.testing.assert_close(actual.float(), reference.float(), rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)
        torch.testing.assert_close(gate, control_gate, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction candidate deferred to GPU queue owner")
@pytest.mark.parametrize("conditioned", [True, False])
def test_cuda_width1024_capture_updates_input_modulation_or_weight(conditioned):
    pytest.importorskip("triton")
    module = norm(1024, conditioned=conditioned, device="cuda")
    x = torch.randn(2, 50, 1024, device="cuda")
    modulation = torch.randn(2, 3072, device="cuda") * 0.1 if conditioned else None
    def run():
        return fused_adarms(module, x, modulation=modulation, output_dtype=torch.float32, fuse_reduction=True)
    with torch.inference_mode():
        actual, gate = run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(); run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, captured_gate = run()
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        x.mul_(0.75).add_(0.125)
        if conditioned:
            modulation.add_(0.02)
        else:
            module.weight.add_(0.02)
        refreshed, refreshed_gate = run()
        reference, _ = Pi05AdaRMSNorm.forward(module, x, modulation=modulation)
        graph.replay()
        torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
        torch.testing.assert_close(captured, reference, rtol=3e-6, atol=3e-6)
        if conditioned:
            torch.testing.assert_close(captured_gate, refreshed_gate, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction candidate deferred to GPU queue owner")
def test_cuda_reduction_graph_camera_token_masks_A_B_A_and_parameter_refresh():
    from test_lite_host_layout import _LayoutModel, _inputs
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    class Model(_LayoutModel):
        def __init__(self):
            super().__init__("cuda")
            self.adarms = norm(1024, device="cuda")
            self.register_buffer("ramp", torch.linspace(-1, 1, 1024, device="cuda"))
            self.eval()

        def sample_actions(self, images, image_masks, lang_tokens, lang_masks, noise,
                           *, _reference=False, **kwargs):
            base = super().sample_actions(images, image_masks, lang_tokens, lang_masks, noise, **kwargs)
            x = noise.repeat(1, 1, 256) + self.ramp[None, None]
            cond = lang_tokens.float().sum(-1, keepdim=True).expand(-1, 1024).contiguous() * 0.001
            if _reference:
                output, _ = Pi05AdaRMSNorm.forward(self.adarms, x, cond)
            else:
                output, _ = fused_adarms(self.adarms, x, cond, output_dtype=torch.float32, fuse_reduction=True)
            return base + output[..., :4]

    model = Model()
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True)
    retained = None
    with torch.inference_mode():
        for valid, cameras, value in ((2, (True, False), 1.0), (3, (True, False), 2.0),
                                      (2, (True, False), 3.0), (2, (False, True), 4.0),
                                      (5, (True, False), 5.0), (2, (True, False), 6.0)):
            inputs = _inputs(valid, cameras, "cuda", value)
            kwargs = {key: value for key, value in inputs.items() if key != "host_layout"}
            eager = model.sample_actions(**kwargs)
            reference = model.sample_actions(**kwargs, _reference=True)
            actual = runner(**inputs)
            torch.testing.assert_close(actual, eager, rtol=0, atol=0)
            torch.testing.assert_close(actual, reference, rtol=3e-6, atol=3e-6)
            if retained is None:
                retained = actual, actual.clone()
            else:
                torch.testing.assert_close(*retained, rtol=0, atol=0)
        assert runner.stats["captures"] == 3 and runner.stats["metadata_updates"] == 2
        model.adarms.dense.bias.add_(0.02)
        inputs = _inputs(device="cuda")
        actual = runner(**inputs)
        expected = model.sample_actions(**{key: value for key, value in inputs.items() if key != "host_layout"})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runner.last_call["cold"] and runner.stats["invalidations"] == 1
    runner.close()
