"""DEFERRED: experiment owner runs these CUDA numerical/stride/graph checks.

Thresholds are kernel test gates, not policy-quality or latency acceptance.
Keep both eager/static metadata, real QKV striding, empty splits and graph mask
refresh coverage when tuning. No CUDA execution is claimed by CPU test passes.
"""

import pytest
import torch

from vvla_repro_pi05.lite.attention import Pi05FlashAttention, prepare_pi05_static_metadata, prepare_pi05_attention_metadata
from vvla_repro_pi05.lite.suffix_attention import Pi05SplitSuffixAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deferred to experiment owner')


def _inputs(prefix=576, *, seed=41, holes=True, strided=False, eager=False, all_prefix_masked=False):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    if strided:
        packed = torch.randn((1, 50, 2560), device='cuda', dtype=torch.bfloat16, generator=generator)
        q, sk, sv = (x.view(1, 50, -1, 256).transpose(1, 2) for x in packed.split((2048, 256, 256), -1))
        pk, pv = [torch.randn((1, 1, prefix, 512), device='cuda', dtype=torch.bfloat16, generator=generator)[..., :256]
                  for _ in range(2)]
    else:
        q = torch.randn((1, 50, 8, 256), device='cuda', dtype=torch.bfloat16, generator=generator).transpose(1, 2)
        pk, pv = [torch.randn((1, 1, prefix, 256), device='cuda', dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        sk, sv = [torch.randn((1, 1, 50, 256), device='cuda', dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    prefix_keep = torch.arange(prefix, device='cuda')[None] < max(0, prefix - 7)
    if holes:
        prefix_keep[:, 3::11] = False
    if all_prefix_masked:
        prefix_keep.zero_()
    queries = torch.ones((1, 50), dtype=torch.bool, device='cuda')
    keys = torch.cat((prefix_keep, queries), 1)
    if eager:
        visible = keys[:, None, None, :].expand(-1, 1, 50, -1)
        metadata = prepare_pi05_attention_metadata(torch.where(visible, 0.0, -2.3819763e38), prefix + 50)
    else:
        metadata = prepare_pi05_static_metadata(queries, keys, all_queries_valid=True)
    return (q, pk, pv, sk, sv, metadata), keys


def _reference(args, keys):
    q, pk, pv, sk, sv, metadata = args
    k, v = torch.cat((pk, sk), 2), torch.cat((pv, sv), 2)
    # Zero masked data before a matmul so deliberate NaN padding cannot poison
    # the independent FP32 reference through 0*NaN in its value product.
    k = torch.where(keys[:, None, :, None], k, 0)
    v = torch.where(keys[:, None, :, None], v, 0)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / 16
    logits.masked_fill_(~keys[:, None, None, :], float('-inf'))
    return torch.matmul(logits.softmax(-1), v.float()).bfloat16()


def _fa2(args):
    q, pk, pv, sk, sv, metadata = args
    return Pi05FlashAttention(1 / 16)(q.transpose(1, 2), torch.cat((pk, sk), 2).transpose(1, 2),
                                     torch.cat((pv, sv), 2).transpose(1, 2), metadata).transpose(1, 2)


@pytest.mark.parametrize('splits', [1, 2, 4, 8])
@pytest.mark.parametrize('block_q', [16, 32])
@pytest.mark.parametrize('case', ['dense', 'holes_strided', 'eager', 'empty_splits'])
def test_cuda_numerical_gqa_mapping_precision_boundaries_and_repeatability(splits, block_q, case, record_property):
    with torch.inference_mode():
        args, keys = _inputs(prefix=1 if case == 'empty_splits' else 576,
                             holes=case != 'dense', strided=case == 'holes_strided',
                             eager=case == 'eager', all_prefix_masked=case == 'empty_splits')
        # Deliberately poison masked prefix storage: direct indexed loads must
        # never read these values, including partial tiles and empty splits.
        args[1].masked_fill_(~keys[:, None, :args[1].shape[2], None], float('nan'))
        args[2].masked_fill_(~keys[:, None, :args[2].shape[2], None], float('nan'))
        backend = Pi05SplitSuffixAttention(splits=splits, block_q=block_q)
        actual, repeat = backend.forward_split(*args), backend.forward_split(*args)
        reference, fa2 = _reference(args, keys), _fa2(args)
        assert actual.shape == (1, 8, 50, 256) and actual.dtype == torch.bfloat16
        assert actual.transpose(1, 2).is_contiguous() and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, repeat, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.008)
        torch.testing.assert_close(actual, fa2, rtol=0.025, atol=0.008)
        record_property('max_abs_vs_fa2', float((actual.float() - fa2.float()).abs().max()))
        record_property('max_abs_vs_fp32', float((actual.float() - reference.float()).abs().max()))


@pytest.mark.parametrize('splits', [1, 4, 8])
def test_cuda_large_logits_do_not_overflow_online_softmax(splits):
    with torch.inference_mode():
        args, keys = _inputs(prefix=65, strided=True)
        args[0].mul_(4)
        args[1].mul_(4)
        args[3].mul_(4)
        actual = Pi05SplitSuffixAttention(splits=splits).forward_split(*args)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, _reference(args, keys), rtol=0.035, atol=0.02)
        torch.testing.assert_close(actual, _fa2(args), rtol=0.035, atol=0.02)


def _capture(backend, args):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.forward_split(*args)
        backend.forward_split(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = backend.forward_split(*args)
    graph.replay()
    return graph, output


@pytest.mark.parametrize('splits', [1, 2, 4, 8])
def test_cuda_capture_consumes_fresh_masks_kv_and_owns_two_independent_graph_workspaces(splits):
    with torch.inference_mode():
        backend = Pi05SplitSuffixAttention(splits=splits)
        args1, keys1 = _inputs(strided=True)
        before = backend.forward_split(*args1)
        retained = before.clone()
        graph1, output1 = _capture(backend, args1)
        torch.testing.assert_close(output1, before, rtol=0, atol=0)
        # All pointers/shapes stay fixed; change visibility count and positions,
        # query values and both prefix/suffix KV before replay.
        keys1[:, :576] = (torch.arange(576, device='cuda') % 5 != 0)[None]
        updated = prepare_pi05_static_metadata(torch.ones((1, 50), device='cuda', dtype=torch.bool),
                                                keys1, all_queries_valid=True)
        for name, value in updated.extra.items():
            if isinstance(value, torch.Tensor):
                args1[-1].extra[name].copy_(value)
        for index, tensor in enumerate(args1[:5]):
            tensor.mul_(0.75).add_(0.02 * index)
        expected1 = backend.forward_split(*args1)
        graph1.replay()
        torch.testing.assert_close(output1, expected1, rtol=0, atol=0)
        torch.testing.assert_close(output1, _reference(args1, keys1), rtol=0.025, atol=0.008)
        torch.testing.assert_close(before, retained, rtol=0, atol=0)
        args2, _ = _inputs(seed=97, strided=True)
        expected2 = backend.forward_split(*args2)
        graph2, output2 = _capture(backend, args2)
        assert output1.data_ptr() != output2.data_ptr()
        stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
        stream1.wait_stream(torch.cuda.current_stream())
        stream2.wait_stream(torch.cuda.current_stream())
        # The backend has no shared scratch. Distinct graphs may execute on
        # separate streams without one overwriting the other's partial sums.
        with torch.cuda.stream(stream1):
            for _ in range(3):
                graph1.replay()
        with torch.cuda.stream(stream2):
            for _ in range(3):
                graph2.replay()
        stream1.synchronize()
        stream2.synchronize()
        torch.testing.assert_close(output1, expected1, rtol=0, atol=0)
        torch.testing.assert_close(output2, expected2, rtol=0, atol=0)
