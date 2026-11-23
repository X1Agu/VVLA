"""CPU lifecycle contracts and deferred CUDA parity/capture for expert MLP."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP

from vvla_repro_pi05.lite.fused_expert_mlp import INSTALLATION, install_expert_mlp
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from test_lite_skip_weight_init import probe


class SmallExpert(nn.Module):
    prepare_inference_context = Pi05ForActionPrediction.prepare_inference_context

    def __init__(self, *, device="cpu", width=16, intermediate=32, layers=2):
        super().__init__()
        self.config = SimpleNamespace(min_period=0.004, max_period=4.0)
        self._inference_cache = Pi05InferenceCache()
        self.paligemma_with_expert = nn.Module()
        inner = self.paligemma_with_expert
        inner.paligemma = nn.Module()
        inner.paligemma.model = nn.Module()
        inner.paligemma.model.vision_tower = nn.Linear(1, 1, device=device)
        inner.gemma_expert = nn.Module()
        inner.gemma_expert.model = nn.Module()
        inner.gemma_expert.model.layers = nn.ModuleList()
        config = GemmaConfig(hidden_size=width, intermediate_size=intermediate)
        for _ in range(layers):
            layer = nn.Module()
            layer.mlp = GemmaMLP(config).to(device=device, dtype=torch.bfloat16)
            inner.gemma_expert.model.layers.append(layer)
        self.eval()

    @property
    def layers(self):
        return self.paligemma_with_expert.gemma_expert.model.layers

    def optimization_flags(self):
        return False, False, False, False, False

    def invalidate_inference_caches(self, **kwargs):
        self._inference_cache.clear()


def prepare(model, *, device="cpu"):
    return model.prepare_inference_context(
        [torch.ones((1, 3, 2, 2), device=device)], [torch.ones(1, dtype=torch.bool, device=device)],
        torch.ones((1, 4), dtype=torch.bool, device=device), 10, positions=False,
    )


@pytest.mark.parametrize("concatenate,fuse_gelu_mul", [(True, False), (False, True), (True, True)])
def test_cpu_original_modules_parameters_state_and_independent_candidates(concatenate, fuse_gelu_mul):
    model = SmallExpert()
    mlp = model.layers[0].mlp
    parameters = {name: id(value) for name, value in model.named_parameters()}
    modules = {name: id(value) for name, value in model.named_modules()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(2, 5, 16).bfloat16()
    with torch.no_grad():
        expected = mlp(x)
        activated = mlp.act_fn(mlp.gate_proj(x))
        assert activated.dtype == (activated * mlp.up_proj(x)).dtype == torch.bfloat16
    handle = install_expert_mlp(model, concatenate=concatenate, fuse_gelu_mul=fuse_gelu_mul)
    prepare(model)
    manifest = handle.manifest()
    assert manifest["concatenate_gate_up"] is concatenate
    assert manifest["fuse_gelu_mul"] is fuse_gelu_mul
    assert manifest["packed_weight_bytes"] == (2 * 16 * 32 * 4 if concatenate else 0)
    with torch.no_grad():
        actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert {name: id(value) for name, value in model.named_modules()} == modules
    assert state.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    if concatenate:
        candidate = handle._candidates[0]
        packed = torch.nn.functional.linear(x, candidate._packed)
        gate, up = packed.chunk(2, -1)
        assert gate.stride(-1) == up.stride(-1) == 1
        assert gate.stride(-2) == up.stride(-2) == 64
        assert gate.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
    handle.uninstall()
    assert not hasattr(model, INSTALLATION) and not handle.enabled
    assert all("forward" not in layer.mlp.__dict__ for layer in model.layers)
    with torch.no_grad():
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)


def test_weight_updates_explicit_data_invalidation_and_packed_address_lifecycle():
    model = SmallExpert()
    mlp = model.layers[0].mlp
    handle = install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True)
    first = prepare(model)
    candidate = handle._candidates[0]
    address, identity = candidate._packed.data_ptr(), handle.manifest()
    x = torch.ones(1, 3, 16, dtype=torch.bfloat16)
    prepare(model)
    assert handle.refreshes == 1
    with torch.no_grad():
        mlp.gate_proj.weight.add_(0.125)
        with pytest.raises(RuntimeError, match="prepare_inference_context"):
            mlp(x)
        second = prepare(model)
        assert second["token"] != first["token"] and handle.refreshes == 2
        torch.testing.assert_close(mlp(x), GemmaMLP.forward(mlp, x), rtol=0, atol=0)
        mlp.up_proj.weight.data.add_(0.25)
        model.invalidate_inference_caches()
        prepare(model)
        torch.testing.assert_close(mlp(x), GemmaMLP.forward(mlp, x), rtol=0, atol=0)
    assert candidate._packed.data_ptr() == address
    assert handle.refreshes == 3 and handle.manifest() == identity
    handle.uninstall()


def test_bad_geometry_or_activation_rejected_atomically_and_cache_tampering_fails():
    model = SmallExpert()
    model.layers[1].mlp.up_proj = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compatible"):
        install_expert_mlp(model, concatenate=True)
    assert not hasattr(model, INSTALLATION)
    assert all("forward" not in layer.mlp.__dict__ for layer in model.layers)
    model = SmallExpert()
    model.layers[1].mlp.act_fn = nn.SiLU()
    with pytest.raises(ValueError, match="GELUTanh"):
        install_expert_mlp(model, fuse_gelu_mul=True)

    model = SmallExpert()
    handle = install_expert_mlp(model, concatenate=True)
    prepare(model)
    with torch.no_grad():
        handle._candidates[0]._packed.add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        prepare(model)
    handle.uninstall()


def test_changed_geometry_is_validated_before_any_cache_copy():
    model = SmallExpert()
    handle = install_expert_mlp(model, concatenate=True)
    prepare(model)
    saved = [candidate._packed.clone() for candidate in handle._candidates]
    with torch.no_grad():
        model.layers[0].mlp.gate_proj.weight.add_(1)
    model.layers[1].mlp.up_proj = nn.Linear(16, 64, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compatible"):
        prepare(model)
    for candidate, reference in zip(handle._candidates, saved, strict=True):
        torch.testing.assert_close(candidate._packed, reference, rtol=0, atol=0)
    handle.uninstall()


def test_graph_guard_training_gradient_fallback_and_explicit_enable():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    model = SmallExpert()
    for options in ({}, {"concatenate": 1}, {"fuse_gelu_mul": "true"}):
        with pytest.raises(ValueError):
            install_expert_mlp(model, **options)
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        install_expert_mlp(model, concatenate=True)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_expert_mlp(model, concatenate=True)
    finally:
        runner._call_lock.release()
    handle = install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True)
    mlp = model.layers[0].mlp
    x = torch.ones(1, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    # Gradient path does not consume an inference cache or detached packed weights.
    expected = GemmaMLP.forward(mlp, x)
    actual = mlp(x)
    variables = (x, *mlp.parameters())
    reference_grad = torch.autograd.grad(expected.float().sum(), variables)
    candidate_grad = torch.autograd.grad(actual.float().sum(), variables)
    for actual, expected in zip(candidate_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mlp.train()
    with torch.no_grad():
        torch.testing.assert_close(mlp(x), GemmaMLP.forward(mlp, x), rtol=0, atol=0)
    mlp.eval()
    runner.entries["live"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        handle.uninstall()
    runner.close()
    handle.uninstall()


def test_bf16_gelu_rounding_is_a_real_intermediate_boundary():
    gate = torch.linspace(-4, 4, 1024).bfloat16()
    up = torch.linspace(0.1, 3, 1024).bfloat16()
    rounded = torch.nn.functional.gelu(gate, approximate="tanh") * up
    delayed_rounding = (torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()).bfloat16()
    assert torch.count_nonzero(rounded != delayed_rounding) > 0


def test_gelu_only_is_pure_tensor_fullgraph_traceable_on_cpu():
    model = SmallExpert(layers=1)
    mlp = model.layers[0].mlp
    handle = install_expert_mlp(model, fuse_gelu_mul=True)
    assert handle.concatenate is False and handle.fuse_gelu_mul is True
    prepare(model)
    with torch.inference_mode():
        x = torch.randn(1, 5, 16).bfloat16()
        compiled = torch.compile(mlp, backend="eager", fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(x), mlp(x), rtol=0, atol=0)
        mlp.gate_proj.weight.add_(0.125)
        prepare(model)
        torch.testing.assert_close(compiled(x), GemmaMLP.forward(mlp, x), rtol=0, atol=0)
    del compiled
    handle.uninstall()


def test_small_real_core_composes_adarms_qkv_mlp_and_gated_residual_without_action_change(probe):
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.lite.fused_adarms import install_fused_adarms
    from vvla_repro_pi05.lite.fused_gated_residual import install_gated_residual
    from vvla_repro_pi05.lite.fused_qkv import install_expert_qkv
    from vvla_repro_pi05.models.pi05.config import Pi05Config

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), probe.small_dimensions():
            torch.manual_seed(54)
            model = Pi05ForActionPrediction(Pi05Config(
                chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8,
                num_inference_steps=2, attention_backend="eager",
            )).eval()
            # Default AdaRMS initializes gates to zero. Nonzero synthetic gates
            # make this check sensitive to the MLP/residual path being tested.
            with torch.no_grad():
                for module in model.modules():
                    if type(module) is probe.core.Pi05AdaRMSNorm:
                        module.dense.weight.normal_(std=0.04)
                        module.dense.bias.normal_(std=0.04)
            set_bfloat16_layout(model)
            inputs = {
                "images": [torch.randn(1, 3, 28, 28), torch.randn(1, 3, 28, 28), torch.zeros(1, 3, 28, 28)],
                "image_masks": [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)],
                "lang_tokens": torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]),
                "lang_masks": torch.tensor([[True, True, True, False, False, False, False, False]]),
                "noise": torch.randn(1, 4, 8),
            }
            expected = model.sample_actions(**inputs)
            handles = [install_fused_adarms(model), install_expert_qkv(model),
                       install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True), install_gated_residual(model)]
            actual = model.sample_actions(**inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for handle in reversed(handles):
                handle.uninstall()
            torch.testing.assert_close(model.sample_actions(**inputs), expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_actual_policy_records_independent_options_and_retires_graph_before_all_uninstalls(monkeypatch):
    from test_lite_fused_adarms import SmallExpert as SmallAdaExpert
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides

    for name in ("fused_expert_gate_up", "fused_expert_gelu", "fused_gated_residual"):
        assert _split_lite_overrides({})[1][name] is False
        core, options = _split_lite_overrides({"lite_" + name: True})
        assert core == {} and options[name] is True
        assert options["native_lifecycle_variant"] == "v1"
        assert not any(value for key, value in options.items()
                       if key not in (name, "native_lifecycle_variant"))
        with pytest.raises(ValueError, match="must be a boolean"):
            _split_lite_overrides({"lite_" + name: 1})
    model = SmallAdaExpert()
    model.action_horizon, model.action_dim = 2, 4
    for layer in model.paligemma_with_expert.gemma_expert.model.layers:
        layer.self_attn.head_dim = 4
        for name, width in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
            setattr(layer.self_attn, name, nn.Linear(8, width, bias=False, dtype=torch.bfloat16))
        layer.mlp = GemmaMLP(GemmaConfig(hidden_size=8, intermediate_size=16)).bfloat16()
    model.eval()
    policy = Pi05LitePolicy(
        model, _policy().processor, device="cpu", cuda_graph=False,
        fused_adarms=True, fused_expert_qkv=True, fused_expert_gate_up=True,
        fused_expert_gelu=True, fused_gated_residual=True,
    )
    metadata = policy.metadata()
    for name in ("fused_expert_gate_up", "fused_expert_gelu", "fused_gated_residual"):
        assert metadata["lite_optimizations"][name] is True
    assert metadata["lite_optimizations"]["fused_expert_mlp_manifest"] == policy._mlp_installation.manifest()
    assert metadata["lite_optimizations"]["fused_gated_residual_manifest"] == policy._residual_installation.manifest()
    runner = Pi05NativeGraphRunner(model)
    runner.entries["live"] = object()
    policy.graph_runner = runner
    events = []
    original_close = runner.close
    def close_graph():
        events.append("graph")
        original_close()
    monkeypatch.setattr(runner, "close", close_graph)
    handles = [("residual", policy._residual_installation), ("MLP", policy._mlp_installation),
               ("QKV", policy._qkv_installation), ("Ada", policy._adarms_installation)]
    for name, handle in handles:
        original = handle.uninstall
        def uninstall(_name=name, _original=original):
            assert runner._closed and not runner.entries
            events.append(_name)
            _original()
        monkeypatch.setattr(handle, "uninstall", uninstall)
    policy.close()
    assert events == ["graph", "residual", "MLP", "QKV", "Ada"]
    assert all(not handle.enabled for _, handle in handles)
    assert all(candidate._packed is None for candidate in handles[1][1]._candidates)
    policy.close()
    assert events == ["graph", "residual", "MLP", "QKV", "Ada"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tests deferred to experiment owner")
@pytest.mark.parametrize("concatenate,fuse_gelu_mul", [(True, False), (False, True), (True, True)])
def test_cuda_reference_parity_bf16_boundaries_and_capture(concatenate, fuse_gelu_mul):
    pytest.importorskip("triton")
    model = SmallExpert(device="cuda", width=1024, intermediate=4096, layers=1)
    mlp = model.layers[0].mlp
    handle = install_expert_mlp(model, concatenate=concatenate, fuse_gelu_mul=fuse_gelu_mul)
    prepare(model, device="cuda")
    with torch.inference_mode():
        x = torch.randn(1, 50, 1024, device="cuda", dtype=torch.bfloat16)
        expected = GemmaMLP.forward(mlp, x)
        actual = mlp(x)
        assert actual.dtype == expected.dtype == torch.bfloat16
        # BF16 GEMM shape and tanh lowering are not guaranteed bitwise identical.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.016, atol=0.002)
        if fuse_gelu_mul:
            from vvla_repro_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
            gate, up = mlp.gate_proj(x), mlp.up_proj(x)
            reference_hidden = mlp.act_fn(gate) * up
            fused_hidden = gelu_tanh_mul(gate, up)
            assert fused_hidden.dtype == torch.bfloat16
            torch.testing.assert_close(fused_hidden.float(), reference_hidden.float(), rtol=0.008, atol=1e-5)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mlp(x); mlp(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = mlp(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        x.mul_(0.5).add_(0.125)
        refreshed = mlp(x)
        graph.replay()
        torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
    del graph, captured
    torch.cuda.synchronize()
    handle.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tests deferred to experiment owner")
def test_cuda_native_graph_weight_refresh_stream_order_and_owned_output():
    from test_lite_host_layout import _LayoutModel, _inputs
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    class Model(_LayoutModel):
        def __init__(self):
            super().__init__("cuda")
            layer = nn.Module()
            layer.mlp = GemmaMLP(GemmaConfig(hidden_size=8, intermediate_size=16)).cuda().bfloat16()
            self.layers = nn.ModuleList([layer])
            self.paligemma_with_expert.gemma_expert = SimpleNamespace(model=SimpleNamespace(layers=self.layers))
            self.eval()

        def invalidate_inference_caches(self, **kwargs):
            self._inference_cache.clear()

        def sample_actions(self, *args, _reference=False, **kwargs):
            base = super().sample_actions(*args, **kwargs)
            x = torch.cat((kwargs["noise"], kwargs["noise"]), -1).bfloat16()
            mlp = self.layers[0].mlp
            output = GemmaMLP.forward(mlp, x) if _reference else mlp(x)
            return base + output[..., :4].float()

    model = Model()
    handle = install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True)
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True)
    with torch.inference_mode():
        inputs = _inputs(device="cuda")
        result = runner(**inputs)
        saved = result.clone()
        address = handle._candidates[0]._packed.data_ptr()
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            model.layers[0].mlp.gate_proj.weight.add_(0.125)
            updated = runner(**inputs)
        other.synchronize()
        expected = model.sample_actions(**{key: value for key, value in inputs.items() if key != "host_layout"},
                                        _reference=True)
        torch.testing.assert_close(updated, expected, rtol=0.016, atol=0.002)
        torch.testing.assert_close(result, saved, rtol=0, atol=0)
        assert handle._candidates[0]._packed.data_ptr() == address
        assert runner.stats["invalidations"] == 1 and runner.stats["stream_handoffs"] == 1
    runner.close()
    handle.uninstall()
