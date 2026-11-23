"""CPU lifecycle/semantic contracts; CUDA tests are separately opt-in by device."""
import threading
import weakref

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite.fused_adarms import fused_adarms, install_fused_adarms
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05AdaRMSNorm


def norm(dim=8, *, conditioned=True, device='cpu'):
    module = Pi05AdaRMSNorm(dim, cond_dim=dim if conditioned else None).to(device).eval()
    with torch.no_grad():
        if conditioned:
            module.dense.weight.copy_(torch.linspace(-0.25, 0.25, 3 * dim * dim, device=device).reshape(3 * dim, dim))
            module.dense.bias.copy_(torch.linspace(-0.1, 0.1, 3 * dim, device=device))
        else:
            module.weight.copy_(torch.linspace(-0.25, 0.25, dim, device=device))
    return module


class SmallExpert(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        expert = self.paligemma_with_expert.gemma_expert.model = nn.Module()
        expert.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.input_layernorm, layer.post_attention_layernorm = norm(device=device), norm(device=device)
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for name in ('q_proj', 'k_proj', 'v_proj'):
                setattr(layer.self_attn, name, nn.Linear(8, 8, device=device, dtype=torch.bfloat16))
            for name in ('up_proj', 'gate_proj'):
                setattr(layer.mlp, name, nn.Linear(8, 8, device=device, dtype=torch.bfloat16))
            expert.layers.append(layer)
        expert.norm = norm(device=device)
        self.action_out_proj = nn.Linear(8, 7, device=device, dtype=torch.float32)
        self.invalidations = []
        self._graph_runners = weakref.WeakSet()
        self.eval()

    def invalidate_inference_caches(self, *, weights_changed=True):
        self.invalidations.append(weights_changed)


@pytest.mark.parametrize('conditioned', [True, False])
@pytest.mark.parametrize('shape', [(2, 8), (2, 3, 8)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('output_dtype', [torch.float32, torch.bfloat16])
def test_cpu_fallback_preserves_reference_rounding_and_gate(conditioned, shape, dtype, output_dtype):
    module = norm(conditioned=conditioned)
    x = torch.linspace(-2, 2, torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
    cond = torch.linspace(-0.5, 0.5, shape[0] * 8).reshape(shape[0], 8)
    modulation = module.dense(cond) if conditioned else None
    with torch.inference_mode():
        for prepared in (None, modulation):
            expected, gate = Pi05AdaRMSNorm.forward(module, x, cond, prepared)
            actual, actual_gate = fused_adarms(module, x, cond, prepared, output_dtype=output_dtype)
            torch.testing.assert_close(actual, expected.to(output_dtype), atol=0, rtol=0)
            if conditioned:
                assert actual_gate.dtype == x.dtype
                torch.testing.assert_close(actual_gate, gate, atol=0, rtol=0)
            else:
                assert actual_gate is None


def test_install_uninstall_preserves_state_names_parameters_modules_and_final_fp32():
    model = SmallExpert()
    expert = model.paligemma_with_expert.gemma_expert.model
    parameters = {key: id(value) for key, value in model.named_parameters()}
    modules = {key: id(value) for key, value in model.named_modules()}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    x, cond = torch.randn(2, 3, 8), torch.randn(2, 8)
    expected, expected_gate = expert.layers[0].input_layernorm(x, cond)
    final_expected, _ = expert.norm(x, cond)
    installation = install_fused_adarms(model)
    assert model.invalidations == [False]
    assert installation.manifest()['enable_fp_fusion'] is False
    assert len(installation.manifest()['norms']) == 5
    actual, gate = expert.layers[0].input_layernorm(x, cond)
    final_actual, _ = expert.norm(x, cond)
    assert actual.dtype == torch.bfloat16 and final_actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(gate, expected_gate, atol=0, rtol=0)
    torch.testing.assert_close(final_actual, final_expected, atol=0, rtol=0)
    assert {key: id(value) for key, value in model.named_parameters()} == parameters
    assert {key: id(value) for key, value in model.named_modules()} == modules
    assert model.state_dict().keys() == state.keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    installation.uninstall()
    assert model.invalidations == [False, False]
    assert not installation.enabled
    restored, _ = expert.layers[0].input_layernorm(x, cond)
    assert restored.dtype == x.dtype
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)


def test_installation_is_atomic_and_final_projection_must_remain_fp32():
    model = SmallExpert()
    model.action_out_proj.to(torch.bfloat16)
    with pytest.raises(ValueError, match='consumer'):
        install_fused_adarms(model)
    assert not any('forward' in module.__dict__ for module in model.modules())
    assert model.invalidations == []


def test_graph_and_active_request_guards_apply_to_install_and_uninstall():
    model = SmallExpert()
    class Runner:
        entries = {'live': object()}
        _call_lock = threading.Lock()
    runner = Runner()
    model._graph_runners.add(runner)
    with pytest.raises(ValueError, match='Clear all live'):
        install_fused_adarms(model)
    runner.entries = {}
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='active'):
            install_fused_adarms(model)
    finally:
        runner._call_lock.release()
    installation = install_fused_adarms(model)
    runner.entries = {'live': object()}
    with pytest.raises(ValueError, match='Clear all live'):
        installation.uninstall()
    assert installation.enabled
    runner.entries = {}
    installation.uninstall()


def test_training_gradient_fallback_keeps_dense_parameters_differentiable():
    module = norm().train()
    x, cond = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True)
    expected, gate = Pi05AdaRMSNorm.forward(module, x, cond)
    expected_loss = expected.bfloat16().float().square().sum() + gate.square().sum()
    parameters = (x, cond, module.dense.weight, module.dense.bias)
    gradients = torch.autograd.grad(expected_loss, parameters)
    actual, gate = fused_adarms(module, x, cond, output_dtype=torch.bfloat16)
    actual_loss = actual.float().square().sum() + gate.square().sum()
    actual_gradients = torch.autograd.grad(actual_loss, parameters)
    for actual, expected in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_missing_condition_and_wrong_modulation_are_not_silently_ignored():
    module = norm()
    x = torch.ones(2, 3, 8)
    with pytest.raises(ValueError, match='conditioning'):
        fused_adarms(module, x, output_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='modulation'):
        fused_adarms(module, x, modulation=torch.zeros(1, 24), output_dtype=torch.bfloat16)
    # A supplied modulation takes precedence over cond, exactly like core.
    with torch.inference_mode():
        modulation = torch.randn(2, 24)
        expected = Pi05AdaRMSNorm.forward(module, x, None, modulation)[0]
        actual = fused_adarms(module, x, torch.randn(999), modulation, output_dtype=torch.float32)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA; never run by local preparation')
@pytest.mark.parametrize('conditioned', [True, False])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('output_dtype', [torch.float32, torch.bfloat16])
def test_cuda_reference_parity_and_capture_refresh(conditioned, dtype, output_dtype):
    pytest.importorskip('triton')
    module = norm(1024, conditioned=conditioned, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(193)
    x = torch.randn((2, 50, 1024), device='cuda', generator=generator, dtype=dtype)
    cond = torch.randn((2, 1024), device='cuda', generator=generator) * 0.1
    with torch.inference_mode():
        modulation = module.dense(cond) if conditioned else None
        expected, expected_gate = Pi05AdaRMSNorm.forward(module, x, cond, modulation)
        actual, gate = fused_adarms(module, x, cond, modulation, output_dtype=output_dtype)
        # Variance reduction is unchanged, but rsqrt lowering can differ by ULP.
        # BF16 receives its rounding quantum; FP32 final head stays tightly bound.
        bf16_rounding = dtype == torch.bfloat16 or output_dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected.to(output_dtype).float(),
                                   rtol=0.008 if bf16_rounding else 3e-6,
                                   atol=0.008 if bf16_rounding else 3e-6)
        if conditioned:
            torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)
            via_cond, _ = fused_adarms(module, x, cond, output_dtype=output_dtype)
            torch.testing.assert_close(via_cond, actual, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fused_adarms(module, x, cond, modulation, output_dtype=output_dtype)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, captured_gate = fused_adarms(module, x, cond, modulation, output_dtype=output_dtype)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        x.mul_(0.75).add_(0.1)
        if modulation is not None:
            modulation.add_(0.02)
        refreshed, refreshed_gate = fused_adarms(module, x, cond, modulation, output_dtype=output_dtype)
        graph.replay()
        torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
        if conditioned:
            torch.testing.assert_close(captured_gate, refreshed_gate, rtol=0, atol=0)
