"""Production FP8 primitive contracts; root executes remotely, never locally.

These do not install 51 sites or certify closed-loop quality. New production
source requires its own backend report, even when the kernel bytes match v2.
"""
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import prefix_fp8_native as candidate


def tensor_metadata(shape, dtype, *, stride=None, device=None, requires_grad=False):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
        device=torch.device('cuda:0') if device is None else device,
        stride=lambda: (shape[1], 1) if stride is None else stride, requires_grad=requires_grad)


def operands(m=576, k=2048, n=16384):
    return [tensor_metadata((m, k), torch.float8_e4m3fn),
            tensor_metadata((n, k), torch.float8_e4m3fn),
            tensor_metadata((m, 1), torch.float32, stride=(1, 1)),
            tensor_metadata((n, 1), torch.float32, stride=(1, 1))]


def test_kernel_is_exact_v2_bytes_but_wrapper_is_new_package_owned_source():
    root = Path(candidate.__file__).resolve().parents[4]
    old = root / 'benchmarks/pi05-lite-bf16-20261105/profiling'
    package = Path(candidate.__file__).resolve().parent
    assert (package / 'prefix_fp8_native_kernel.py').read_bytes() == (old / 'prefix_fp8_quantizer_kernel.py').read_bytes()
    manifest = candidate.implementation_manifest()
    assert manifest['quantizer_files_sha256'] == {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in candidate.SOURCE_FILES}
    assert manifest['quantizer_files_sha256']['prefix_fp8_native.py'] != hashlib.sha256(
        (old / 'prefix_fp8_quantizer.py').read_bytes()).hexdigest()
    assert manifest['variant'] == 'triton-eager-rn-native-sm120-v1'
    assert manifest['required_runtime']['capability'] == [12, 0]
    assert manifest['fallback_permitted'] is False
    assert 'passed' not in manifest and 'sites' not in manifest
    manifest['required_runtime']['capability'][0] = 0
    assert candidate.implementation_manifest()['required_runtime']['capability'] == [12, 0]


def test_math_manifest_and_compile_options_keep_existing_bf16_global_lowering():
    from vvla_repro_pi05.lite.compilation import INDUCTOR_OPTIONS
    assert candidate.COMPILE_OPTIONS == INDUCTOR_OPTIONS
    assert 'max_autotune_gemm_backends' not in candidate.COMPILE_OPTIONS
    bits = struct.unpack('>I', struct.pack('>f', candidate.ARITHMETIC['scale_reciprocal_fp32']))[0]
    assert bits == 0x3b124925
    assert candidate.ARITHMETIC['normalization'] == 'FP32 div.rn(value, scale)'
    manifest = candidate.implementation_manifest()
    assert manifest['use_fast_accum'] is False and manifest['bias'] is None and manifest['scale_result'] is None


@pytest.mark.parametrize('change', [dict(device=torch.device('cpu')), dict(dtype=torch.float32),
    dict(ndim=3), dict(shape=(512, 2048)), dict(shape=(576, 1024)),
    dict(stride=lambda: (1, 576)), dict(requires_grad=True)])
def test_quantizer_geometry_rejects_without_cast_or_fallback(change):
    value = tensor_metadata((576, 2048), torch.bfloat16)
    value.__dict__.update(change)
    with pytest.raises(ValueError, match='quantizer requires'):
        candidate.validate_quantizer_input(value)


@pytest.mark.parametrize('slot,change', [
    (0, dict(shape=(512, 2048))), (0, dict(dtype=torch.bfloat16)),
    (1, dict(shape=(8192, 2048))), (1, dict(stride=lambda: (1, 16384))),
    (1, dict(device=torch.device('cuda:1'))), (2, dict(dtype=torch.bfloat16)),
    (2, dict(shape=(1, 576))), (3, dict(stride=lambda: (2, 1))),
    (3, dict(requires_grad=True)), (3, dict(ndim=1)),
])
def test_native_geometry_dtype_axes_device_and_stride_fail_closed(slot, change):
    values = operands()
    values[slot].__dict__.update(change)
    with pytest.raises(ValueError, match='Prefix FP8 native'):
        candidate.validate_scaled_mm_inputs(*values)


def test_runtime_rejects_sm89_unpinned_binary_and_autocast_before_kernel(monkeypatch):
    monkeypatch.setattr(torch, '__version__', candidate.REQUIRED_RUNTIME['torch'])
    monkeypatch.setattr(torch.version, 'cuda', candidate.REQUIRED_RUNTIME['cuda'])
    monkeypatch.setattr(torch.version, 'git_version', candidate.REQUIRED_RUNTIME['torch_git'])
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: False)
    candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 9))
    with pytest.raises(RuntimeError, match='SM120 only'):
        candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: True)
    with pytest.raises(RuntimeError, match='autocast'):
        candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.version, 'git_version', '0' * 40)
    with pytest.raises(RuntimeError, match='explicit Torch'):
        candidate.validate_runtime(torch.device('cuda:0'))


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_fake_fullgraph_has_only_two_opaque_ops_correct_layout_and_no_output_alias(m, k, n):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    for name in ('prefix_fp8_quantize_rows_v1', 'prefix_fp8_native_scaled_mm_v1'):
        schema = getattr(torch.ops.vvla_repro_pi05_lite, name).default._schema
        assert all(arg.alias_info is None for arg in schema.arguments)
        assert all(result.alias_info is None for result in schema.returns)
    with FakeTensorMode():
        x = torch.empty((m, k), device='cuda:0', dtype=torch.bfloat16)
        qw = torch.empty((n, k), device='cuda:0', dtype=torch.float8_e4m3fn)
        sw = torch.empty((n, 1), device='cuda:0', dtype=torch.float32)
        y, qx, sx = candidate.scaled_linear(x, qw, sw)
        assert y.shape == (m, n) and y.stride() == (n, 1) and y.dtype == torch.bfloat16
        assert qx.shape == x.shape and qx.stride() == (k, 1) and qx.dtype == torch.float8_e4m3fn
        assert sx.shape == (m, 1) and sx.stride() == (1, 1) and sx.dtype == torch.float32
        assert len({id(t) for t in (x, qw, sw, y, qx, sx)}) == 6
        graph = make_fx(candidate.scaled_linear)(x, qw, sw)
    calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
    assert calls.count('vvla_repro_pi05_lite.prefix_fp8_quantize_rows_v1.default') == 1
    assert calls.count('vvla_repro_pi05_lite.prefix_fp8_native_scaled_mm_v1.default') == 1
    assert not any('aten._scaled_mm' in name for name in calls)  # Stays inside opaque ATEN body.


def exact(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def eager_quantize(value):
    fp32 = value.float()
    scale = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def raw_reference(x, qw, sw):
    qx, sx = eager_quantize(x)
    return (torch._scaled_mm(qx, qw.T, sx, sw.T, bias=None, scale_result=None,
                            out_dtype=torch.bfloat16, use_fast_accum=False), qx, sx)


def boundary_input(m, k):
    bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    finite = bits.view(torch.bfloat16)
    finite = torch.where(torch.isfinite(finite), finite, 0)
    result = torch.ones((m, k), device='cuda', dtype=torch.bfloat16)
    result[6:6 + 65536 // k].copy_(finite.reshape(-1, k))
    result[0].zero_()
    result[1].fill_(-0.0)
    result[2].fill_(1e-38)
    result[3].fill_(-1e-38)
    result[4] = torch.linspace(-448, 448, k, device='cuda').to(torch.bfloat16)
    result[5].fill_(1e-12)
    return result


def require_sm120():
    if _FAILED_CAPTURES:
        raise RuntimeError('Earlier graph did not retire; no subsequent GPU contract may submit work')
    if not torch.cuda.is_available():
        pytest.skip('Root queues CUDA contracts separately')
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('This production variant explicitly targets SM120; SM89 rejection is a CPU contract')


@pytest.mark.parametrize('k', (2048, 16384))
def test_cuda_quantizer_all_finite_bf16_patterns_floor_signedzero_and_owned_outputs(k):
    require_sm120()
    with torch.inference_mode():
        x = boundary_input(576, k)
        expected = eager_quantize(x)
        actual = candidate.quantize_rows(x)
        exact(actual, expected)
        saved = tuple(t.clone() for t in actual)
        other = candidate.quantize_rows(x * .75)
        exact(other, eager_quantize(x * .75))
        exact(actual, saved)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actual, other, strict=True))
        torch.cuda.synchronize()


@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_cuda_native_scale_axes_have_independent_analytic_exact_reference(k, n):
    require_sm120()
    with torch.inference_mode():
        qx = torch.ones((576, k), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        qw = torch.ones((n, k), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sx = torch.pow(2., torch.arange(576, device='cuda', dtype=torch.float32).remainder(9) - 4)[:, None].contiguous()
        sw = torch.pow(2., torch.arange(n, device='cuda', dtype=torch.float32).remainder(7) - 3)[:, None].contiguous()
        expected = (k * sx * sw.T).to(torch.bfloat16)
        actual = candidate.native_scaled_mm(qx, qw, sx, sw)
        exact((actual,), (expected,))
        assert torch.isfinite(actual).all()
        torch.cuda.synchronize()


# Only used if CUDA retirement itself fails: retain graph/operands/outputs and
# stop this contract, rather than reset a graph that may still own async work.
_FAILED_CAPTURES = []


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_cuda_native_fullgraph_capture_repeat_input_weight_scale_refresh_and_owned_outputs(m, k, n):
    require_sm120()
    with torch.inference_mode():
        generator = torch.Generator(device='cuda').manual_seed(7531 + m + k)
        x = torch.randn((m, k), generator=generator, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn((n, k), generator=generator, device='cuda', dtype=torch.bfloat16) * .02
        qw, sw = eager_quantize(weight)
        reference = tuple(t.clone() for t in raw_reference(x, qw, sw))
        eager = candidate.scaled_linear(x, qw, sw)
        exact(eager, reference)
        saved = tuple(t.clone() for t in eager)
        function = torch.compile(candidate.scaled_linear, fullgraph=True, dynamic=False,
                                 options=dict(candidate.COMPILE_OPTIONS))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph, captured, warm = torch.cuda.CUDAGraph(), None, []
        original = (x.clone(), qw.clone(), sw.clone())
        # Clones above are created on the caller stream after the initial wait.
        stream.wait_stream(torch.cuda.current_stream())
        retained = [function, stream, graph, x, qw, sw, weight, original, eager, saved, reference, warm]
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm.append(function(x, qw, sw))
            stream.synchronize()
            for value in warm:
                exact(value, reference)
            with torch.cuda.graph(graph, stream=stream):
                captured = function(x, qw, sw)
            retained.append(captured)
            # Capture records work; its output buffers are valid only after
            # the first replay has actually executed on the captured stream.
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            exact(captured, reference)
            pointers = tuple(t.data_ptr() for t in (x, qw, sw))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    graph.replay()
            stream.synchronize()
            exact(captured, reference)
            with torch.cuda.stream(stream):
                x.mul_(.75)
                changed_q, changed_s = eager_quantize(weight * -.5)
                retained.extend((changed_q, changed_s))
                qw.copy_(changed_q)
                sw.copy_(changed_s)
                changed_reference = raw_reference(x, qw, sw)
                retained.append(changed_reference)
                graph.replay()
            stream.synchronize()
            exact(captured, changed_reference)
            exact(eager, saved)
            assert pointers == tuple(t.data_ptr() for t in (x, qw, sw))
            with torch.cuda.stream(stream):
                for target, source in zip((x, qw, sw), original, strict=True):
                    target.copy_(source)
                graph.replay()
            stream.synchronize()
            exact(captured, reference)
        finally:
            try:
                torch.cuda.synchronize()
            except BaseException:
                _FAILED_CAPTURES.append(retained)
                raise
            try:
                graph.reset()
            except BaseException:
                _FAILED_CAPTURES.append(retained)
                raise
