"""Exact Torch fallback and deferred CUDA bitwise/capture residual checks."""

import pytest
import torch

from vvla_repro_pi05.lite.fused_gated_residual import DISPATCH, fused_gated_residual, install_gated_residual, _validate_inputs
from vvla_repro_pi05.models.pi05.modeling_pi05 import _gated_residual


@pytest.mark.parametrize("projection_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 16), (2, 5, 16)])
def test_cpu_fallback_preserves_fp32_promotion_broadcast_and_gradients(projection_dtype, shape):
    residual = torch.randn(shape, requires_grad=True)
    out = torch.randn(shape).to(projection_dtype).requires_grad_()
    modulation = torch.randn(2, 48, requires_grad=True)
    gate = modulation.chunk(3, -1)[2]
    if len(shape) == 3:
        gate = gate.unsqueeze(1)
    _validate_inputs(residual, out, gate)
    expected, actual = _gated_residual(residual, out, gate), fused_gated_residual(residual, out, gate)
    assert expected.dtype == actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    variables = (residual, out, modulation)
    expected_grads = torch.autograd.grad(expected.sum(), variables)
    actual_grads = torch.autograd.grad(actual.sum(), variables)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(fused_gated_residual(residual, out, None), residual + out, rtol=0, atol=0)


def test_shape_contract_rejects_implicit_broadcast_or_changed_fp32_boundary():
    residual, out, gate = torch.ones(2, 5, 16), torch.ones(2, 5, 16).bfloat16(), torch.ones(2, 1, 16)
    _validate_inputs(residual, out, gate)
    for r, o, g in ((residual.bfloat16(), out, gate), (residual, out, gate.bfloat16()),
                    (residual, out[:, :1], gate), (residual, out, gate.expand(2, 3, 16))):
        with pytest.raises(ValueError):
            _validate_inputs(r, o, g)


def test_residual_is_pure_tensor_fullgraph_traceable_on_cpu():
    with torch.inference_mode():
        residual = torch.randn(2, 5, 16)
        out = torch.randn(2, 5, 16).bfloat16()
        gate = torch.randn(2, 1, 48).chunk(3, -1)[2]
        compiled = torch.compile(fused_gated_residual, backend="eager", fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(residual, out, gate), residual + gate * out, rtol=0, atol=0)


def test_install_preserves_parameters_modules_state_and_requires_idle_clear_graphs():
    from test_lite_fused_expert_mlp import SmallExpert
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = SmallExpert()
    parameters = {name: id(value) for name, value in model.named_parameters()}
    modules = {name: id(value) for name, value in model.named_modules()}
    state = tuple(model.state_dict())
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        install_gated_residual(model)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_gated_residual(model)
    finally:
        runner._call_lock.release()
    generation = model._inference_cache.generation
    handle = install_gated_residual(model)
    assert model._inference_cache.generation > generation
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert {name: id(value) for name, value in model.named_modules()} == modules
    assert tuple(model.state_dict()) == state
    residual, out, gate = torch.randn(2, 5, 16), torch.randn(2, 5, 16).bfloat16(), torch.randn(2, 1, 16)
    for layer in model.layers:
        torch.testing.assert_close(getattr(layer, DISPATCH)(residual, out, gate), residual + gate * out, rtol=0, atol=0)
    manifest = handle.manifest()
    assert manifest["enable_fp_fusion"] is False
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        handle.uninstall()
    assert handle.manifest() == manifest
    runner.close()
    handle.uninstall()
    assert all(not hasattr(layer, DISPATCH) for layer in model.layers)


def test_real_expert_layer_calls_both_local_residual_dispatches(monkeypatch):
    from types import SimpleNamespace
    from torch import nn
    import vvla_repro_pi05.models.pi05.modeling_pi05 as core

    class Norm(nn.Module):
        def forward(self, x, cond, modulation=None):
            return x, torch.full_like(x[:, :1], 0.5)

    layer = nn.Module()
    layer.input_layernorm = Norm()
    layer.post_attention_layernorm = Norm()
    layer.self_attn = nn.Module()
    layer.self_attn.head_dim = 4
    layer.self_attn.num_key_value_groups = 1
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(layer.self_attn, name, nn.Linear(4, 4, bias=False, dtype=torch.bfloat16))
    layer.mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    # Core matches its input to mlp.up_proj. Keep a nonregistered reference here.
    object.__setattr__(layer.mlp, "up_proj", layer.mlp)
    calls = []
    def local(residual, out, gate):
        calls.append((residual.dtype, out.dtype, gate.dtype))
        return fused_gated_residual(residual, out, gate)
    setattr(layer, DISPATCH, local)
    expert = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    monkeypatch.setattr(core, "apply_rotary_pos_emb", lambda q, k, *args, **kwargs: (q, k))
    monkeypatch.setattr(core, "_attend", lambda q, k, v, *args, **kwargs: q)
    with torch.no_grad():
        result = core._compute_layer_suffix_only(
            0, torch.ones(1, 3, 4), (torch.zeros(1, 1, 2, 4).bfloat16(), torch.zeros(1, 1, 2, 4).bfloat16()),
            None, None, expert, None, rotary=(None, None),
        )
    assert result.dtype == torch.float32
    assert calls == [(torch.float32, torch.bfloat16, torch.float32)] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tests deferred to experiment owner")
@pytest.mark.parametrize("projection_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("full_gate", [False, True])
def test_cuda_bitwise_mul_then_add_and_capture(projection_dtype, full_gate):
    pytest.importorskip("triton")
    shape = (2, 50, 1024)
    residual = torch.randn(shape, device="cuda")
    out = torch.randn(shape, device="cuda", dtype=projection_dtype)
    # Mimic the strided gate slice produced by AdaRMS modulation.chunk(3).
    modulation = torch.randn((2, 50 if full_gate else 1, 3072), device="cuda")
    gate = modulation.chunk(3, -1)[2]
    with torch.inference_mode():
        # This cancellation exposes an illicit FMA for both FP32 and BF16 out.
        gate[0, 0, 0] = 1.0 + 2**-23
        out[0, 0, 0] = 1.0 + 2**-7
        residual[0, 0, 0] = -(gate[0, 0, 0] * out[0, 0, 0])
        # Signed zero and FP32 subnormal preservation are also tested explicitly.
        residual[0, 0, 1] = -0.0
        gate[0, 0, 1] = -0.0
        out[0, 0, 1] = 1.0
        residual[0, 0, 2] = 2**-149
        gate[0, 0, 2] = 0.0
        out[0, 0, 2] = 1.0
        expected = _gated_residual(residual, out, gate)
        actual = fused_gated_residual(residual, out, gate)
        assert expected[0, 0, 0].item() == 0
        torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_gated_residual(residual, out, gate)
            fused_gated_residual(residual, out, gate)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fused_gated_residual(residual, out, gate)
        graph.replay()
        torch.testing.assert_close(captured.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
        residual.mul_(0.25)
        gate.add_(0.125)
        refreshed = _gated_residual(residual, out, gate)
        graph.replay()
        torch.testing.assert_close(captured.view(torch.int32), refreshed.view(torch.int32), rtol=0, atol=0)
