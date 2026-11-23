"""RTX correctness probes. No throughput, latency, or policy-quality claims."""

import pytest
import torch

from vvla_repro_pi05.models.pi05.attention_pi05 import Pi05VllmFA2Kernel, prepare_pi05_static_metadata
from vvla_repro_pi05.models.pi05.paged_pi05 import Pi05PagedKVCache

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def inputs():
    generator = torch.Generator(device="cuda").manual_seed(123)
    q = torch.randn(2, 17, 8, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(2, 17, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(2, 17, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    mask = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    mask[0, [3, 8, 16]] = False
    mask[1, [1, 4, 7, 12, 15]] = False
    return q, k, v, mask


def reference(q, k, v, mask):
    keys = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    values = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
    weights = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys.float()) * q.shape[-1] ** -0.5
    visible = mask[:, None, :, None] & mask[:, None, None, :]
    weights.masked_fill_(~visible, -2.3819763e38)
    probability = weights.softmax(-1).to(q.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probability.float(), values.float()).to(q.dtype)


def test_fa2_padding_and_cuda_graph_replay():
    q, k, v, mask = inputs()
    metadata = prepare_pi05_static_metadata(mask, mask)
    kernel = Pi05VllmFA2Kernel(q.shape[-1] ** -0.5)
    expected = reference(q, k, v, mask)
    actual = kernel(q, k, v, metadata)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    # Warm up on a side stream before capturing native FA2, then change the
    # inputs in place to prove replay does not return a previous request's data.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            kernel(q, k, v, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel(q, k, v, metadata)
    q.mul_(0.5)
    v.add_(0.25)
    graph.replay()
    torch.testing.assert_close(captured, reference(q, k, v, mask), atol=0.02, rtol=0.02)


def test_paged_cache_request_ownership_and_reassignment():
    q, k, v, mask = inputs()
    metadata = prepare_pi05_static_metadata(mask, mask)
    cache = Pi05PagedKVCache(mask, suffix_length=4, layers=1, kv_heads=1, head_dim=256, dtype=q.dtype)
    try:
        leased = [{block.block_id for block in row} for row in cache.leases]
        assert leased[0].isdisjoint(leased[1])
        actual = cache.attend(0, q, k, v, metadata, prefix=True)
        torch.testing.assert_close(actual, reference(q, k, v, mask), atol=0.02, rtol=0.02)
        address = cache.keys.data_ptr()
        new_mask = mask.flip(0).clone()
        cache.update(new_mask)
        assert cache.keys.data_ptr() == address
        metadata = prepare_pi05_static_metadata(new_mask, new_mask)
        actual = cache.attend(0, q.flip(0), k.flip(0), v.flip(0), metadata, prefix=True)
        torch.testing.assert_close(actual, reference(q.flip(0), k.flip(0), v.flip(0), new_mask), atol=0.02, rtol=0.02)
    finally:
        cache.release()
    assert cache.pool.get_num_free_blocks() == cache.num_blocks - 1
