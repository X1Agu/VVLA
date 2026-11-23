"""Small CUDA contracts; run before loading the full checkpoint on either card."""

import pytest
import torch

from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, device="cuda"))

    def prepare_inference_context(self, images, image_masks, lang_masks, steps, *, positions=True):
        return {
            "token": self.weight._version, "empty_pattern": (False,), "packing": None,
            "positions": self.prepare_position_context(image_masks, lang_masks) if positions else None,
        }

    def prepare_attention_metadata(self, *args, **kwargs):
        return None, None

    def prepare_position_context(self, image_masks, lang_masks, packing=None):
        return {"valid_count": image_masks[0].float().sum() + lang_masks.float().sum()}

    def sample_actions(self, images, image_masks, lang_tokens, lang_masks, noise, num_steps,
                       attention_metadata=None, inference_context=None):
        if inference_context is None:
            inference_context = self.prepare_inference_context(images, image_masks, lang_masks, num_steps)
        return (noise * self.weight + images[0].mean() + lang_tokens.float().sum()
                + inference_context["positions"]["valid_count"])


def _inputs(value=1.0, valid=2):
    return {
        "images": [torch.full((1, 3, 4, 4), value, device="cuda", dtype=torch.float32)],
        "image_masks": [torch.ones(1, dtype=torch.bool, device="cuda")],
        "lang_tokens": torch.tensor([[1, 2, 0, 0]], device="cuda"),
        "lang_masks": torch.tensor([[i < valid for i in range(4)]], device="cuda"),
        "noise": torch.full((1, 2, 4), value, device="cuda", dtype=torch.float32), "num_steps": 2,
    }


def test_native_graph_refresh_ownership_streams_and_invalidation():
    model = _SmallPolicy().eval()
    runner = Pi05NativeGraphRunner(model)
    first = runner(**_inputs())
    saved = first.clone()
    assert runner.last_call["cold"]
    with torch.no_grad():
        expected = model.sample_actions(**_inputs(2, valid=3))
    second = runner(**_inputs(2, valid=3))
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    assert not runner.last_call["cold"] and runner.last_call["metadata_updated"]
    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        third = runner(**_inputs(3))
    other_stream.synchronize()
    assert runner.stats["stream_handoffs"] == 1
    with torch.no_grad():
        torch.testing.assert_close(third, model.sample_actions(**_inputs(3)), rtol=0, atol=0)
        model.weight.add_(1)
    fourth = runner(**_inputs(3))
    assert runner.last_call["cold"] and runner.stats["invalidations"] == 1
    with torch.no_grad():
        torch.testing.assert_close(fourth, model.sample_actions(**_inputs(3)), rtol=0, atol=0)
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner(**_inputs())


@pytest.mark.parametrize("backend", ["torch_fa2", "flash_attn"])
def test_independent_fa2_matches_finite_mask_reference(backend):
    if backend == "flash_attn":
        pytest.importorskip("flash_attn")
    from vvla_repro_pi05.lite.attention import Pi05FlashAttention, prepare_pi05_static_metadata
    from vvla_repro_pi05.models.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE, _attend

    generator = torch.Generator(device="cuda").manual_seed(123)
    q = torch.randn((2, 4, 4, 32), generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 5, 2, 32), generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((2, 5, 2, 32), generator=generator, device="cuda", dtype=torch.bfloat16)
    qm = torch.tensor([[True, True, False, False], [True, False, True, False]], device="cuda")
    km = torch.tensor([[True, False, True, True, False], [False, True, True, True, True]], device="cuda")
    metadata = prepare_pi05_static_metadata(qm, km)
    mask = torch.where((qm[:, :, None] & km[:, None, :])[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
    reference = _attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask, 2, 32**-0.5).transpose(1, 2)
    attention = Pi05FlashAttention(32**-0.5, backend=backend)
    actual = attention(q, k, v, metadata)
    torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.015)
    assert torch.isfinite(actual).all()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            attention(q, k, v, metadata)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = attention(q, k, v, metadata)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    # Same allocated capacity, different live padding; the graph must consume
    # refreshed cumulative lengths/indices and retain finite masked outputs.
    updated = prepare_pi05_static_metadata(torch.ones_like(qm), km)
    for name, value in updated.extra.items():
        if isinstance(value, torch.Tensor):
            metadata.extra[name].copy_(value)
    expected = attention(q, k, v, metadata)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert torch.isfinite(captured).all()
