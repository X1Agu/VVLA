"""Deferred exact contracts for the uninstalled prefix GELU/FP8 candidate.

Root runs these remotely. There is no benchmark, owner installation or quality
claim here. The oracle is the currently selected production two-operation path.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import prefix_fp8_gelu as candidate
from vvla_repro_pi05.lite import prefix_fp8_native as native


def metadata(m=576):
    return SimpleNamespace(shape=(m, 16384), ndim=2, dtype=torch.bfloat16,
        device=torch.device('cuda:0'), requires_grad=False, stride=lambda: (16384, 1))


@pytest.mark.parametrize('m', (576, 640))
def test_exact_current_geometry_accepted_without_cuda_work(m):
    candidate.validate_inputs(metadata(m), metadata(m))


@pytest.mark.parametrize('index,change', [
    (0, dict(shape=(512, 16384))), (1, dict(shape=(640, 16384))),
    (0, dict(shape=(576, 2048))), (1, dict(ndim=3)),
    (0, dict(dtype=torch.float32)), (1, dict(requires_grad=True)),
    (0, dict(device=torch.device('cpu'))), (1, dict(device=torch.device('cuda:1'))),
    (0, dict(stride=lambda: (32768, 1))), (1, dict(stride=lambda: (1, 576))),
])
def test_changed_geometry_or_precision_rejected_without_cast_or_fallback(index, change):
    values = [metadata(), metadata()]
    values[index].__dict__.update(change)
    with pytest.raises(ValueError, match='Fused prefix GELU FP8 requires'):
        candidate.validate_inputs(*values)


def test_new_manifest_binds_actual_sources_and_does_not_claim_existing_authorization():
    value = candidate.implementation_manifest()
    root = Path(candidate.__file__).resolve().parent
    assert value['source_sha256'] == {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                     for name in candidate.SOURCE_FILES}
    assert value['reference_source_sha256']['models/pi05/fused_mlp_pi05.py'] == hashlib.sha256(
        (root.parent / 'models/pi05/fused_mlp_pi05.py').read_bytes()).hexdigest()
    assert value['quantizer_arithmetic'] == native.ARITHMETIC
    assert value['compile_options'] == native.COMPILE_OPTIONS
    assert value['intermediate_rounding'] == ['GELU_tanh_to_BF16_RTNE', 'rounded_GELU_times_up_to_BF16_RTNE']
    assert value['use_fast_accum'] is False and value['fallback_permitted'] is False
    assert value['performance_validated'] is value['quality_validated'] is False
    assert value['installation'] == 'none; explicit primitive only'
    assert 'sites' not in value and 'report_sha256' not in value
    value['required_runtime']['capability'][0] = 0
    assert candidate.implementation_manifest()['required_runtime']['capability'] == [12, 0]


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
def test_fake_fullgraph_keeps_one_converter_native_gemm_and_fresh_outputs(m):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    schema = torch.ops.vvla_repro_pi05_lite.prefix_gelu_fp8_quantize_rows_v1.default._schema
    assert all(item.alias_info is None for item in (*schema.arguments, *schema.returns))
    with FakeTensorMode():
        gate = torch.empty((m, 16384), device='cuda:0', dtype=torch.bfloat16)
        up = torch.empty_like(gate)
        weight = torch.empty((2048, 16384), device='cuda:0', dtype=torch.float8_e4m3fn)
        scale = torch.empty((2048, 1), device='cuda:0', dtype=torch.float32)
        y, qx, sx = candidate.scaled_down(gate, up, weight, scale)
        assert (y.shape, y.dtype, y.stride()) == ((m, 2048), torch.bfloat16, (2048, 1))
        assert (qx.shape, qx.dtype, qx.stride()) == ((m, 16384), torch.float8_e4m3fn, (16384, 1))
        assert (sx.shape, sx.dtype, sx.stride()) == ((m, 1), torch.float32, (1, 1))
        assert len({id(t) for t in (gate, up, weight, scale, y, qx, sx)}) == 7
        graph = make_fx(candidate.scaled_down)(gate, up, weight, scale)
    calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
    assert calls.count('vvla_repro_pi05_lite.prefix_gelu_fp8_quantize_rows_v1.default') == 1
    assert calls.count('vvla_repro_pi05_lite.prefix_fp8_native_scaled_mm_v1.default') == 1
    assert not any('prefix_fp8_quantize_rows_v1' in name or 'aten._scaled_mm' in name for name in calls)


def exact(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def selected_quantizer(gate, up):
    from vvla_repro_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
    return native.quantize_rows(gelu_tanh_mul(gate, up))


def selected_down(gate, up, weight, scale):
    qx, sx = selected_quantizer(gate, up)
    return native.native_scaled_mm(qx, weight, sx, scale), qx, sx


def pack_weight(weight):
    fp32 = weight.float()
    scale = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


_FAILED_CAPTURES = []


def require_sm120():
    if _FAILED_CAPTURES:
        raise RuntimeError('Earlier graph did not retire; no subsequent GPU work is authorized')
    if not torch.cuda.is_available():
        pytest.skip('Root queues CUDA contracts remotely')
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('This candidate explicitly targets the selected SM120 backend')


def inputs(m):
    generator = torch.Generator(device='cuda').manual_seed(9387 + m)
    gate = torch.randn((m, 16384), generator=generator, device='cuda', dtype=torch.bfloat16) * 3
    up = torch.randn(gate.shape, generator=generator, device='cuda', dtype=torch.bfloat16)
    # Include signed zero, floor-scale rows, and every representable finite BF16
    # gate value in [-16,16], with nontrivial up values. Avoid overflow inputs,
    # which are outside the selected finite-activation contract.
    bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    finite = bits.view(torch.bfloat16)
    finite = torch.where(torch.isfinite(finite) & (finite.abs() <= 16), finite, 0)
    gate[6:10].copy_(finite.reshape(4, 16384))
    gate[0].zero_()
    gate[1].fill_(-0.0)
    up[:2].fill_(1)
    gate[2].fill_(1e-38)
    gate[3].fill_(-1e-38)
    up[2:4].fill_(1)
    gate[4].fill_(1e-12)
    gate[5].copy_(torch.linspace(-10, 10, 16384, device='cuda').to(torch.bfloat16))
    return gate.contiguous(), up.contiguous()


def test_cuda_converter_matches_selected_gelu_and_quantizer_raw_bytes_and_ownership():
    require_sm120()
    with torch.inference_mode():
        gate, up = inputs(576)
        expected = selected_quantizer(gate, up)
        actual = candidate.gelu_quantize_rows(gate, up)
        exact(actual, expected)
        saved = tuple(t.clone() for t in actual)
        changed = candidate.gelu_quantize_rows(gate * .75, up * -.5)
        exact(changed, selected_quantizer(gate * .75, up * -.5))
        exact(actual, saved)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actual, changed, strict=True))
        assert all(value.data_ptr() not in {gate.data_ptr(), up.data_ptr()} for value in actual)
        torch.cuda.synchronize()


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
def test_cuda_compiled_capture_repeat_refresh_restore_matches_selected_down_exactly(m):
    require_sm120()
    with torch.inference_mode():
        gate, up = inputs(m)
        generator = torch.Generator(device='cuda').manual_seed(7103)
        weight = torch.randn((2048, 16384), generator=generator, device='cuda', dtype=torch.bfloat16) * .02
        qw, sw = pack_weight(weight)
        expected = tuple(t.clone() for t in selected_down(gate, up, qw, sw))
        eager = candidate.scaled_down(gate, up, qw, sw)
        exact(eager, expected)
        saved = tuple(t.clone() for t in eager)
        function = torch.compile(candidate.scaled_down, fullgraph=True, dynamic=False,
                                 options=dict(candidate.COMPILE_OPTIONS))
        original = tuple(t.clone() for t in (gate, up, qw, sw))
        stream, graph, warm = torch.cuda.Stream(), torch.cuda.CUDAGraph(), []
        stream.wait_stream(torch.cuda.current_stream())
        retained = [function, stream, graph, gate, up, qw, sw, weight, original, expected, eager, saved, warm]
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm.append(function(gate, up, qw, sw))
            stream.synchronize()
            for output in warm:
                exact(output, expected)
            with torch.cuda.graph(graph, stream=stream):
                captured = function(gate, up, qw, sw)
            retained.append(captured)
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            exact(captured, expected)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    graph.replay()
            stream.synchronize()
            exact(captured, expected)
            pointers = tuple(t.data_ptr() for t in (gate, up, qw, sw))
            with torch.cuda.stream(stream):
                gate.mul_(.75)
                up.mul_(-.5)
                changed_q, changed_s = pack_weight(weight * -.5)
                retained.extend((changed_q, changed_s))
                qw.copy_(changed_q)
                sw.copy_(changed_s)
                changed_reference = selected_down(gate, up, qw, sw)
                retained.append(changed_reference)
                graph.replay()
            stream.synchronize()
            exact(captured, changed_reference)
            exact(eager, saved)
            assert pointers == tuple(t.data_ptr() for t in (gate, up, qw, sw))
            with torch.cuda.stream(stream):
                for target, source in zip((gate, up, qw, sw), original, strict=True):
                    target.copy_(source)
                graph.replay()
            stream.synchronize()
            exact(captured, expected)
        finally:
            try:
                torch.cuda.synchronize()
                graph.reset()
            except BaseException:
                _FAILED_CAPTURES.append(retained)
                raise
