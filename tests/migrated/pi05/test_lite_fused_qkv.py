"""Standalone QKV CPU contracts and deferred CUDA parity/graph validation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
from vvla_repro_pi05.lite.fused_qkv import INSTALLATION, PACKER, install_expert_qkv
from vvla_repro_pi05.models.pi05.config import Pi05Config
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vvla_runtime.pointwise import rotary_pos_emb


def _model():
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    attention = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
    rng = torch.Generator().manual_seed(13)
    for module in (attention.q_proj, attention.k_proj, attention.v_proj):
        module.to_empty(device="cpu")
        with torch.no_grad():
            module.weight.copy_(torch.randn(module.weight.shape, generator=rng) / 32)
    return model.eval(), attention


def _prepare(model):
    return model.prepare_inference_context(
        [torch.ones((1, 3, 2, 2))], [torch.ones(1, dtype=torch.bool)],
        torch.ones((1, 6), dtype=torch.bool), 10, positions=False,
    )


def test_actual_topology_preserves_parameter_state_and_split_stride_rope_contracts():
    model, attention = _model()
    parameters = {name: id(value) for name, value in model.named_parameters()}
    state = tuple(model.state_dict())
    modules = {name: id(value) for name, value in model.named_modules()}
    handle = install_expert_qkv(model)
    _prepare(model)
    assert handle.manifest()["packed_weight_bytes"] == 90 * 1024**2
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert tuple(model.state_dict()) == state
    assert {name: id(value) for name, value in model.named_modules()} == modules
    x = torch.randn((1, 50, 1024)).bfloat16()
    with torch.no_grad():
        reference = tuple(module(x) for module in (attention.q_proj, attention.k_proj, attention.v_proj))
        parts = getattr(attention, PACKER).project(x)
        for actual, expected in zip(parts, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.stride(-1) == 1 and actual.stride(-2) == 2560
        assert len({value.untyped_storage().data_ptr() for value in parts}) == 1
        q, k, v = (value.view(1, 50, -1, 256).transpose(1, 2) for value in parts)
        qr, kr, vr = (value.view(1, 50, -1, 256).transpose(1, 2) for value in reference)
        angle = torch.randn((1, 50, 256))
        cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
        rotated, reference_rotated = rotary_pos_emb(q, k, cos, sin), rotary_pos_emb(qr, kr, cos, sin)
        for actual, expected in zip(rotated, reference_rotated, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        prefix = torch.zeros((1, 1, 64, 256), dtype=torch.bfloat16)
        torch.testing.assert_close(torch.cat((prefix, v), 2), torch.cat((prefix, vr), 2), rtol=0, atol=0)
    handle.uninstall()
    assert not hasattr(attention, PACKER) and not hasattr(model, INSTALLATION)


def test_normal_inplace_and_explicit_data_invalidation_rebuild_at_same_address():
    model, attention = _model()
    handle = install_expert_qkv(model)
    first = _prepare(model)
    packer = getattr(attention, PACKER)
    address, manifest = packer._packed.data_ptr(), handle.manifest()
    _prepare(model)
    assert handle.refreshes == 1
    x = torch.ones((1, 2, 1024), dtype=torch.bfloat16)
    with torch.no_grad():
        attention.q_proj.weight.add_(0.125)
        with pytest.raises(RuntimeError, match="prepare_inference_context"):
            packer.project(x)
        second = _prepare(model)
        assert second["token"] != first["token"] and handle.refreshes == 2
        torch.testing.assert_close(packer.project(x)[0], attention.q_proj(x), rtol=0, atol=0)
        # Unsupported .data writes are safe after the documented explicit epoch.
        attention.k_proj.weight.data.add_(0.25)
        model.invalidate_inference_caches()
        _prepare(model)
        torch.testing.assert_close(packer.project(x)[1], attention.k_proj(x), rtol=0, atol=0)
    assert packer._packed.data_ptr() == address and handle.refreshes == 3
    assert handle.manifest() == manifest  # Runtime counters do not alter recipe identity.
    handle.uninstall()


def test_bad_geometry_rejected_before_copy_and_packed_cache_tampering_fails_closed():
    model, attention = _model()
    handle = install_expert_qkv(model)
    _prepare(model)
    packer = getattr(attention, PACKER)
    saved = packer._packed.clone()
    attention.k_proj = nn.Linear(1024, 512, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="geometry"):
        _prepare(model)
    torch.testing.assert_close(packer._packed, saved, rtol=0, atol=0)
    handle.uninstall()

    model, attention = _model()
    handle = install_expert_qkv(model)
    _prepare(model)
    with torch.no_grad():
        getattr(attention, PACKER)._packed.add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        _prepare(model)
    handle.uninstall()


def test_install_and_uninstall_require_clear_idle_graphs_and_gradient_fallback():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    model, attention = _model()
    runner = Pi05NativeGraphRunner(model)
    runner.entries["sentinel"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        install_expert_qkv(model)
    runner.clear()  # CPU-only: runner has never acquired a CUDA stream/device.
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            install_expert_qkv(model)
    finally:
        runner._call_lock.release()
    handle = install_expert_qkv(model)
    x = torch.ones((1, 2, 1024), dtype=torch.bfloat16, requires_grad=True)
    # No prepared cache is needed for the untouched differentiable fallback.
    parts = getattr(attention, PACKER).project(x)
    assert all(part.requires_grad for part in parts)
    sum(part.float().sum() for part in parts).backward()
    assert x.grad is not None and attention.q_proj.weight.grad is not None
    runner.entries["sentinel"] = object()
    with pytest.raises(ValueError, match="Clear all live"):
        handle.uninstall()
    runner.close()
    handle.uninstall()


def test_actual_policy_independent_default_off_and_close_order():
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    assert _split_lite_overrides({})[1]["fused_expert_qkv"] is False
    core, options = _split_lite_overrides({"lite_fused_expert_qkv": True})
    assert core == {} and options["fused_expert_qkv"] is True
    assert options["native_lifecycle_variant"] == "v1"
    assert not any(value for key, value in options.items()
                   if key not in ("fused_expert_qkv", "native_lifecycle_variant"))
    model, _ = _model()
    policy = Pi05LitePolicy(model, _policy().processor, device="cpu", cuda_graph=False, fused_expert_qkv=True)
    handle = getattr(model, INSTALLATION)
    assert policy.metadata()["lite_optimizations"]["fused_expert_qkv"]
    runner = Pi05NativeGraphRunner(model)
    runner.entries["sentinel"] = object()
    policy.graph_runner = runner
    policy.close()
    assert runner._closed and not runner.entries and not handle.enabled
    assert all(packer._packed is None for packer in handle._packers)
    policy.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; run by experiment owner")
@pytest.mark.parametrize("batch", [1, 2])
def test_cuda_projection_stride_rope_attention_parity_and_capture(batch):
    from vvla_repro_pi05.lite.attention import Pi05FlashAttention, prepare_pi05_static_metadata
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = nn.Module()
            self.paligemma_with_expert.gemma_expert = nn.Module()
            expert = self.paligemma_with_expert.gemma_expert.model = nn.Module()
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.head_dim = 256
            for name, width in (("q_proj", 2048), ("k_proj", 256), ("v_proj", 256)):
                setattr(layer.self_attn, name, nn.Linear(1024, width, bias=False, device="cuda", dtype=torch.bfloat16))
            expert.layers = nn.ModuleList([layer])
            self.eval()
        def invalidate_inference_caches(self, **kwargs):
            pass

    model = Model()
    handle = install_expert_qkv(model)
    attention = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
    packer = getattr(attention, PACKER)
    with torch.inference_mode():
        handle.prepare(1)
        x = torch.randn((batch, 50, 1024), device="cuda", dtype=torch.bfloat16)
        expected = tuple(module(x) for module in (attention.q_proj, attention.k_proj, attention.v_proj))
        actual = packer.project(x)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0.008, atol=0.008)
        angles = torch.randn((batch, 50, 256), device="cuda")
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        prefix_k = torch.randn((batch, 1, 64, 256), device="cuda", dtype=torch.bfloat16)
        prefix_v = torch.randn_like(prefix_k)
        queries = torch.ones((batch, 50), device="cuda", dtype=torch.bool)
        prefix_keep = torch.arange(64, device="cuda")[None, :] < torch.arange(57, 57 + batch, device="cuda")[:, None]
        metadata = prepare_pi05_static_metadata(queries, torch.cat((prefix_keep, queries), 1), all_queries_valid=True)
        attention_kernel = Pi05FlashAttention(256**-0.5)
        def downstream(parts):
            q, k, v = (part.view(batch, 50, -1, 256).transpose(1, 2) for part in parts)
            q, k = rotary_pos_emb(q, k, cos, sin)
            joined_k, joined_v = torch.cat((prefix_k, k), 2), torch.cat((prefix_v, v), 2)
            output = attention_kernel(q.transpose(1, 2), joined_k.transpose(1, 2), joined_v.transpose(1, 2), metadata)
            return q, k, v, output
        def run():
            return downstream(packer.project(x))
        eager = run()
        reference_downstream = downstream(expected)
        for a, b in zip(eager, reference_downstream, strict=True):
            torch.testing.assert_close(a, b, rtol=0.025, atol=0.015)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(); run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        graph.replay()
        for a, b in zip(captured, eager, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x.mul_(0.5)
        expected = run()
        graph.replay()
        for a, b in zip(captured, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # This manual graph still references packer storage; release it before uninstall.
    del graph, captured, eager, expected, actual
    torch.cuda.synchronize()
    handle.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; run by experiment owner")
def test_cuda_graph_inplace_weight_refresh_stream_order_and_output_ownership():
    from test_lite_host_layout import _LayoutModel, _inputs
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    class Model(_LayoutModel):
        def __init__(self):
            super().__init__("cuda")
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.head_dim = 4
            for name, width in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
                setattr(layer.self_attn, name, nn.Linear(8, width, bias=False, device="cuda", dtype=torch.bfloat16))
            self.layers = nn.ModuleList([layer])
            self.paligemma_with_expert.gemma_expert = SimpleNamespace(model=SimpleNamespace(layers=self.layers))
            self.eval()
        def invalidate_inference_caches(self, **kwargs):
            self._inference_cache.clear()
        def sample_actions(self, *args, _reference=False, **kwargs):
            base = super().sample_actions(*args, **kwargs)
            x = torch.cat((kwargs["noise"], kwargs["noise"]), -1).bfloat16()
            attention = self.layers[0].self_attn
            projections = (tuple(module(x) for module in (attention.q_proj, attention.k_proj, attention.v_proj))
                           if _reference else getattr(attention, PACKER).project(x))
            return base + sum(value.float().sum(-1, keepdim=True) for value in projections)

    model = Model()
    handle = install_expert_qkv(model)
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True)
    with torch.inference_mode():
        inputs = _inputs(device="cuda")
        result = runner(**inputs)
        saved = result.clone()
        address = handle._packers[0]._packed.data_ptr()
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            model.layers[0].self_attn.q_proj.weight.add_(0.125)
            updated = runner(**inputs)
        other.synchronize()
        expected = model.sample_actions(**{k: v for k, v in inputs.items() if k != "host_layout"}, _reference=True)
        torch.testing.assert_close(updated, expected, rtol=0.008, atol=0.008)
        torch.testing.assert_close(result, saved, rtol=0, atol=0)
        assert runner.stats["invalidations"] == 1 and runner.stats["stream_handoffs"] == 1
        assert handle.refreshes == 2 and handle._packers[0]._packed.data_ptr() == address
    runner.close()
    handle.uninstall()
