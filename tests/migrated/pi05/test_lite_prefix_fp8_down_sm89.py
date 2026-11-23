"""Package down-only core contracts; root executes remotely, never locally."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import prefix_fp8_down_sm89 as candidate


def metadata(shape, dtype, *, stride=None):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=torch.device('cuda:0'),
        requires_grad=False, stride=lambda: stride or ((shape[1], 1) if len(shape) == 2 else (1,)))


def test_new_manifest_binds_actual_three_files_and_byte_identical_audited_kernels():
    package = Path(candidate.__file__).resolve().parent
    repo = Path(candidate.__file__).resolve().parents[4]
    old = repo / 'benchmarks/pi05-lite-bf16-20261105/profiling'
    for new, prior in (('prefix_fp8_down_sm89_quantizer_kernel.py', 'prefix_fp8_quantizer_sm89_kernel.py'),
                       ('prefix_fp8_down_sm89_postscale_kernel.py', 'prefix_fp8_scalar_sm89_kernel.py')):
        assert (package / new).read_bytes() == (old / prior).read_bytes()
    manifest = candidate.implementation_manifest()
    assert manifest['core_files_sha256'] == {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                                             for name in candidate.SOURCE_FILES}
    assert manifest['intended_scope'] == dict(component='prefix', projections=['down_proj'],
        layer_indices=list(range(17)), intended_sites=17, gate_up_precision='bfloat16')
    assert manifest['required_runtime']['capability'] == [8, 9]
    assert manifest['intermediate_dtype'] == 'float32' and manifest['output_dtype'] == 'bfloat16'
    assert manifest['scalar_one']['value'] == 1.0
    assert 'cold validate_unit_scale' in manifest['scalar_one']['ownership']
    assert manifest['fallback_permitted'] is False and 'passed' not in manifest and 'sites' not in manifest
    assert manifest['installation'].startswith('none;')
    manifest['intended_scope']['layer_indices'].clear()
    assert candidate.implementation_manifest()['intended_scope']['layer_indices'] == list(range(17))


def test_compile_options_are_policy_three_not_old_diagnostic_six():
    from vvla_repro_pi05.lite.compilation import INDUCTOR_OPTIONS
    assert candidate.COMPILE_OPTIONS == INDUCTOR_OPTIONS
    assert len(candidate.COMPILE_OPTIONS) == 3 and 'max_autotune_gemm_backends' not in candidate.COMPILE_OPTIONS


@pytest.mark.parametrize('change', [dict(device=torch.device('cpu')), dict(dtype=torch.float32),
    dict(ndim=3), dict(shape=(512, 16384)), dict(shape=(576, 2048)),
    dict(stride=lambda: (1, 576)), dict(requires_grad=True)])
def test_quantizer_rejects_gate_up_and_invalid_live_metadata(change):
    x = metadata((576, 16384), torch.bfloat16)
    x.__dict__.update(change)
    with pytest.raises(ValueError, match='Down quantizer'):
        candidate.validate_quantizer_input(x)


@pytest.mark.parametrize('field', ['qx_dtype', 'qx_stride', 'qw_shape', 'one_shape', 'one_dtype', 'device', 'grad', 'm'])
def test_mm_rejects_non_down_geometry_and_invalid_unit_metadata(field):
    qx, qw, one = (metadata((576, 16384), torch.float8_e4m3fn),
                   metadata((2048, 16384), torch.float8_e4m3fn), metadata((1,), torch.float32))
    if field == 'qx_dtype': qx.dtype = torch.bfloat16
    if field == 'qx_stride': qx.stride = lambda: (1, 576)
    if field == 'qw_shape': qw.shape = (16384, 2048)
    if field == 'one_shape': one.shape = (1, 1)
    if field == 'one_dtype': one.dtype = torch.bfloat16
    if field == 'device': one.device = torch.device('cuda:1')
    if field == 'grad': one.requires_grad = True
    if field == 'm': qx.shape = (577, 16384)
    with pytest.raises(ValueError): candidate.validate_mm(qx, qw, one)


@pytest.mark.parametrize('field', ['dtype', 'shape', 'sx_shape', 'sw_stride', 'device', 'grad'])
def test_postscale_rejects_hidden_bf16_or_wrong_axis(field):
    z, sx, sw = metadata((576, 2048), torch.float32), metadata((576, 1), torch.float32), metadata((2048, 1), torch.float32)
    if field == 'dtype': z.dtype = torch.bfloat16
    if field == 'shape': z.shape = (576, 16384)
    if field == 'sx_shape': sx.shape = (1, 576)
    if field == 'sw_stride': sw.stride = lambda: (2, 1)
    if field == 'device': sw.device = torch.device('cuda:1')
    if field == 'grad': z.requires_grad = True
    with pytest.raises(ValueError): candidate.validate_postscale(z, sx, sw)


def fake_runtime(**changes):
    fields = dict(__version__=candidate.REQUIRED_RUNTIME['torch'],
        version=SimpleNamespace(cuda='13.0', git_version=candidate.REQUIRED_RUNTIME['torch_git']),
        cuda=SimpleNamespace(get_device_capability=lambda device: (8, 9), is_current_stream_capturing=lambda: False),
        is_autocast_enabled=lambda device: False,
        is_inference=lambda value: False,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        float32=torch.float32, uint8=torch.uint8)
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize('field', ['version', 'capability', 'tf32', 'autocast', 'cpu'])
def test_runtime_fail_closed_without_mutating_torch_dynamic_properties(monkeypatch, field):
    fake = fake_runtime()
    device = torch.device('cuda:0')
    if field == 'version': fake.__version__ = '2.12.0'
    if field == 'capability': fake.cuda.get_device_capability = lambda device: (12, 0)
    if field == 'tf32': fake.backends.cuda.matmul.allow_tf32 = True
    if field == 'autocast': fake.is_autocast_enabled = lambda device: True
    if field == 'cpu': device = torch.device('cpu')
    monkeypatch.setattr(candidate, 'torch', fake)
    with pytest.raises(RuntimeError): candidate.validate_runtime(device)


@pytest.mark.parametrize('raw', [b'\x00\x00\x80\x3f', b'\x00\x00\x00\x40', b'\x00\x00\x80\xbf', b'\x00\x00\xc0\x7f'])
def test_cold_unit_scale_checks_actual_bytes_and_returns_detached_evidence(monkeypatch, raw):
    fake = fake_runtime()
    monkeypatch.setattr(candidate, 'torch', fake)
    one = metadata((1,), torch.float32)
    one._version = 0
    one.detach = lambda: one
    one.view = lambda dtype: one
    one.cpu = lambda: one
    one.tolist = lambda: list(raw)
    if raw != b'\x00\x00\x80\x3f':
        with pytest.raises(ValueError, match='exact FP32 positive one'):
            candidate.validate_unit_scale(one)
    else:
        evidence = candidate.validate_unit_scale(one)
        assert evidence['passed'] and evidence['raw_sha256'] == hashlib.sha256(raw).hexdigest()
        assert all(value is not one for value in evidence.values())
        fake.is_inference = lambda value: True
        with pytest.raises(ValueError, match='ordinary versioned Tensor'):
            candidate.validate_unit_scale(one)
        fake.is_inference = lambda value: False
        fake.cuda.is_current_stream_capturing = lambda: True
        with pytest.raises(RuntimeError, match='during capture'):
            candidate.validate_unit_scale(one)


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
def test_fake_fullgraph_three_opaque_ops_no_alias_and_no_physical_queries(monkeypatch, m):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    def forbidden(*args): raise AssertionError('Fake path cannot inspect hardware')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', forbidden)
    names = ('prefix_down_sm89_quantize_rows_v1', 'prefix_down_sm89_scalar_mm_fp32_v1', 'prefix_down_sm89_postscale_bf16_v1')
    for name in names:
        schema = getattr(torch.ops.vvla_repro_pi05_lite, name).default._schema
        assert all(item.alias_info is None for item in (*schema.arguments, *schema.returns))
    with FakeTensorMode():
        x = torch.empty((m, 16384), device='cuda:0', dtype=torch.bfloat16)
        qw = torch.empty((2048, 16384), device='cuda:0', dtype=torch.float8_e4m3fn)
        sw = torch.empty((2048, 1), device='cuda:0', dtype=torch.float32)
        one = torch.empty((1,), device='cuda:0', dtype=torch.float32)
        y, qx, sx = candidate.scaled_linear(x, qw, sw, one)
        assert y.shape == (m, 2048) and y.dtype == torch.bfloat16
        assert qx.shape == x.shape and qx.dtype == torch.float8_e4m3fn
        assert sx.shape == (m, 1) and sx.dtype == torch.float32
        assert all(t._base is None for t in (y, qx, sx))
        graph = make_fx(candidate.scaled_linear)(x, qw, sw, one)
    calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
    assert all(calls.count('vvla_repro_pi05_lite.' + name + '.default') == 1 for name in names)
    assert not any('aten._scaled_mm' in name for name in calls)


def exact(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def eager_quantize(value):
    fp32 = value.float()
    scale = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def raw_reference(x, qw, sw, one):
    qx, sx = eager_quantize(x)
    z = torch._scaled_mm(qx, qw.T, one, one, out_dtype=torch.float32, use_fast_accum=False)
    assert z.dtype == torch.float32
    return ((z * sx) * sw.T).to(torch.bfloat16), qx, sx


_FAILED_CAPTURES = []


@pytest.fixture
def sm89():
    if _FAILED_CAPTURES: raise RuntimeError('Prior capture not retired; no more submissions')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip('Root queues this new production-core diagnostic on L4 only')
    old, precision = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    try: yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = old


def test_cuda_quantizer_all_finite_bf16_codes_and_owned_outputs(sm89):
    with torch.inference_mode():
        bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
        finite = bits.view(torch.bfloat16)
        finite = torch.where(torch.isfinite(finite), finite, 0)
        x = torch.ones((576, 16384), device='cuda', dtype=torch.bfloat16)
        x[:4].copy_(finite.reshape(4, 16384))
        x[4].zero_(); x[5].fill_(-0.); x[6].fill_(1e-38); x[7].fill_(1e-12)
        actual = candidate.quantize_rows(x)
        exact(actual, eager_quantize(x))
        saved = tuple(t.clone() for t in actual)
        changed = candidate.quantize_rows(x * .75)
        exact(changed, eager_quantize(x * .75)); exact(actual, saved)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actual, changed, strict=True))
        torch.cuda.synchronize()


def test_cuda_cold_unit_scale_fresh_ownership_content_and_analytic_axes(sm89):
    with torch.inference_mode():
        one = candidate.create_unit_scale(torch.device('cuda:0'))
        other = candidate.create_unit_scale(torch.device('cuda:0'))
        assert not torch.is_inference(one) and not torch.is_inference(other)
        old_version = other._version
        assert one.data_ptr() != other.data_ptr() and candidate.validate_unit_scale(one)['passed']
        qx = torch.ones((576, 16384), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        qw = torch.ones((2048, 16384), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sx = torch.pow(2., torch.arange(576, device='cuda', dtype=torch.float32).remainder(9) - 4)[:, None].contiguous()
        sw = torch.pow(2., torch.arange(2048, device='cuda', dtype=torch.float32).remainder(7) - 3)[:, None].contiguous()
        z = candidate.scalar_mm_fp32(qx, qw, one)
        exact((z,), (torch.full((576, 2048), 16384., device='cuda', dtype=torch.float32),))
        exact((candidate.rowwise_postscale(z, sx, sw),), ((16384. * sx * sw.T).to(torch.bfloat16),))
        other.fill_(2.)
        assert other._version > old_version
        with pytest.raises(ValueError, match='positive one'): candidate.validate_unit_scale(other)
        assert candidate.validate_unit_scale(one)['passed']
        torch.cuda.synchronize()


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
def test_cuda_fullgraph_capture_refresh_restore_same_pointer_and_owned_outputs(sm89, m):
    with torch.inference_mode():
        generator = torch.Generator(device='cuda').manual_seed(7721 + m)
        x = torch.randn((m, 16384), device='cuda', dtype=torch.bfloat16, generator=generator)
        w = torch.randn((2048, 16384), device='cuda', dtype=torch.bfloat16, generator=generator) * .02
        qw, sw = eager_quantize(w)
        one = candidate.create_unit_scale(torch.device('cuda:0'))
        assert candidate.validate_unit_scale(one)['passed']
        reference = tuple(t.clone() for t in raw_reference(x, qw, sw, one))
        eager = candidate.scaled_linear(x, qw, sw, one)
        exact(eager, reference)
        saved = tuple(t.clone() for t in eager)
        function = torch.compile(candidate.scaled_linear, fullgraph=True, dynamic=False, options=dict(candidate.COMPILE_OPTIONS))
        stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
        original = tuple(t.clone() for t in (x, qw, sw))
        stream.wait_stream(torch.cuda.current_stream())
        warm, captured = [], None
        retained = [function, stream, graph, x, qw, sw, one, w, reference, eager, saved, original, warm]
        try:
            with torch.cuda.stream(stream):
                for _ in range(2): warm.append(function(x, qw, sw, one))
            stream.synchronize()
            for result in warm: exact(result, reference)
            with torch.cuda.graph(graph, stream=stream): captured = function(x, qw, sw, one)
            retained.append(captured)
            with torch.cuda.stream(stream): graph.replay()
            stream.synchronize(); exact(captured, reference)
            pointers = tuple(t.data_ptr() for t in (x, qw, sw, one))
            with torch.cuda.stream(stream):
                for _ in range(3): graph.replay()
                x.mul_(.75)
                changed_q, changed_s = eager_quantize(w * -.5)
                retained.extend((changed_q, changed_s))
                qw.copy_(changed_q); sw.copy_(changed_s)
                changed_reference = raw_reference(x, qw, sw, one)
                retained.append(changed_reference)
                graph.replay()
            stream.synchronize(); exact(captured, changed_reference); exact(eager, saved)
            assert tuple(t.data_ptr() for t in (x, qw, sw, one)) == pointers
            with torch.cuda.stream(stream):
                for target, source in zip((x, qw, sw), original, strict=True): target.copy_(source)
                graph.replay()
            stream.synchronize(); exact(captured, reference)
            assert candidate.validate_unit_scale(one)['passed']
        finally:
            try: torch.cuda.synchronize()
            except BaseException:
                _FAILED_CAPTURES.append(retained); raise
            try: graph.reset()
            except BaseException:
                _FAILED_CAPTURES.append(retained); raise
