"""Native NVFP4 contracts authored locally, executed only by the remote queue.

CUDA cases require VLA_NVFP4_NATIVE_PATH explicitly. Primitive/capture evidence
is not a calibration acceptance, 51-site installation, or closed-loop quality.
"""
import base64
import csv
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import prefix_nvfp4_native as candidate


def metadata(shape, dtype, *, stride=None, device='cuda:0', requires_grad=False):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=torch.device(device),
        stride=lambda: ((shape[-1], 1) if len(shape) == 2 else (1,)) if stride is None else stride,
        requires_grad=requires_grad)


def operands(m=576, k=2048, n=16384):
    return [metadata((m, k // 2), torch.uint8), metadata((n, k // 2), torch.uint8),
            metadata(candidate.scale_shape(m, k), torch.float8_e4m3fn),
            metadata(candidate.scale_shape(n, k), torch.float8_e4m3fn), metadata((1,), torch.float32)]


def test_manifest_real_encoding_scales_fixed_binary_and_no_claim_of_installed_quality():
    manifest = candidate.implementation_manifest()
    assert manifest['source_files_sha256'] == {
        'prefix_nvfp4_native.py': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()}
    assert manifest['native_extension_sha256'] == 'ae9c8a293020e422e0b6b61d1024263dc476a772337f5d39d46d18baa7ce0ad9'
    assert manifest['native_wheel_sha256'] == '5e2ecdc5b0baf72921c288486417ca82778456a448e4fe60b62ee43c0dfc8611'
    assert manifest['packed_dtype'] == 'uint8' and manifest['block_size'] == 16
    assert manifest['block_scale_dtype'] == 'float8_e4m3fn' and manifest['global_scale_dtype'] == 'float32'
    assert manifest['fallback_permitted'] is False and 'quality claim' in manifest['installation']
    assert 'passed' not in manifest and 'sites' not in manifest
    assert 'every call' in manifest['scale_padding']
    from vvla_repro_pi05.lite.compilation import INDUCTOR_OPTIONS
    assert manifest['compile_options'] == INDUCTOR_OPTIONS
    manifest['required_runtime']['capability'][0] = 8
    assert candidate.implementation_manifest()['required_runtime']['capability'] == [12, 0]


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_actual_activation_and_weight_geometries_and_128x4_scale_padding(m, k, n):
    scale = metadata((1,), torch.float32)
    candidate.validate_quantizer_input(metadata((m, k), torch.bfloat16), scale)
    candidate.validate_quantizer_input(metadata((n, k), torch.bfloat16), scale, weight=True)
    candidate.validate_scaled_mm_inputs(*operands(m, k, n))
    assert candidate.scale_shape(m, k) == (640, k // 16)
    assert candidate.scale_shape(n, k) == (n, k // 16)


@pytest.mark.parametrize('change', [dict(device=torch.device('cpu')), dict(dtype=torch.float32),
    dict(ndim=3), dict(shape=(512, 2048)), dict(shape=(576, 1024)),
    dict(stride=lambda: (1, 576)), dict(requires_grad=True)])
def test_quantizer_rejects_unsupported_metadata_without_cast_or_fallback(change):
    x = metadata((576, 2048), torch.bfloat16)
    x.__dict__.update(change)
    with pytest.raises(ValueError, match='quantizer requires'):
        candidate.validate_quantizer_input(x, metadata((1,), torch.float32))


@pytest.mark.parametrize('slot,change', [
    (0, dict(shape=(512, 1024))), (0, dict(dtype=torch.float8_e4m3fn)),
    (1, dict(shape=(8192, 1024))), (1, dict(stride=lambda: (1, 16384))),
    (1, dict(device=torch.device('cuda:1'))), (2, dict(dtype=torch.float32)),
    (2, dict(shape=(576, 128))), (3, dict(stride=lambda: (2, 1))),
    (3, dict(requires_grad=True)), (3, dict(ndim=1)),
    (4, dict(shape=(1, 1))), (4, dict(dtype=torch.bfloat16)),
    (4, dict(stride=lambda: (0,))), (4, dict(requires_grad=True)),
])
def test_native_rejects_layout_scale_dtype_device_and_alias_stride_changes(slot, change):
    values = operands()
    values[slot].__dict__.update(change)
    with pytest.raises(ValueError, match='NVFP4'):
        candidate.validate_scaled_mm_inputs(*values)


def fake_runtime(monkeypatch):
    monkeypatch.setattr(torch, '__version__', candidate.REQUIRED_RUNTIME['torch'])
    monkeypatch.setattr(torch.version, 'cuda', candidate.REQUIRED_RUNTIME['cuda'])
    monkeypatch.setattr(torch.version, 'git_version', candidate.REQUIRED_RUNTIME['torch_git'])
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (12, 0))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(
        name='NVIDIA RTX PRO 6000 Blackwell Server Edition', uuid='GPU-TEST', total_memory=1))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: False)


def test_runtime_rejects_sm89_unpinned_torch_and_autocast_before_any_operator(monkeypatch):
    fake_runtime(monkeypatch)
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


def schemas():
    q = SimpleNamespace(_schema=torch._C.parse_schema(
        '_C::scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout, '
        '*, Tensor(a!) output, Tensor(b!) output_scale) -> ()'))
    mm = SimpleNamespace(_schema=torch._C.parse_schema(
        '_C::cutlass_scaled_fp4_mm(Tensor(a!) out, Tensor a, Tensor b, '
        'Tensor block_scale_a, Tensor block_scale_b, Tensor alpha) -> ()'))
    return q, mm


def test_supported_actual_out_parameter_schema_and_incompatible_abi_rejection():
    q, mm = schemas()
    actual = candidate._validate_operator_schemas(q, mm)
    assert 'Tensor(a!) output' in actual['quantizer']
    for wrong in [
        '_C::scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool layout, *, Tensor output, Tensor output_scale) -> ()',
        '_C::scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool layout, *, Tensor(a!) output, Tensor(b!) output_scale) -> Tensor',
    ]:
        with pytest.raises(RuntimeError, match='operator ABI'):
            candidate._validate_operator_schemas(SimpleNamespace(_schema=torch._C.parse_schema(wrong)), mm)


def make_native_tree(tmp_path, monkeypatch):
    root = tmp_path / 'target'
    files = {'vllm/__init__.py': b"__version__ = '0.32.0'\n",
        'vllm/version.py': b"__version__ = '0.32.0'\n", 'vllm/_C_stable_libtorch.abi3.so': b'FIXTURE ONLY',
        'vllm-0.32.0.dist-info/METADATA': b'Metadata-Version: 2.1\nName: vllm\nVersion: 0.32.0\n',
        'vllm-0.32.0.dist-info/WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp38-abi3-linux_x86_64\n'}
    record = io.StringIO()
    writer = csv.writer(record)
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        writer.writerow((name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip('='), len(value)))
    writer.writerow(('vllm-0.32.0.dist-info/RECORD', '', ''))
    (root / 'vllm-0.32.0.dist-info/RECORD').write_text(record.getvalue())
    monkeypatch.setattr(candidate, 'NATIVE_EXTENSION_SHA256', hashlib.sha256(b'FIXTURE ONLY').hexdigest())
    return root


def test_cold_source_checks_real_bytes_record_and_truthful_receipt_scope(tmp_path, monkeypatch):
    root = make_native_tree(tmp_path, monkeypatch)
    actual_root, binary, evidence = candidate._inspect_native_files(root)
    assert actual_root == root and binary == root / 'vllm/_C_stable_libtorch.abi3.so'
    assert evidence['preparation_receipt_sha256'] is None
    assert 'archive is not re-read' in evidence['archive_identity_scope']
    receipt = dict(schema=2, version='0.32.0', archive_sha256=candidate.NATIVE_WHEEL_SHA256,
                   files=deepcopy(evidence['installed_source_files']))
    (root / 'OFT_NATIVE_RECEIPT.json').write_text(json.dumps(receipt))
    assert candidate._inspect_native_files(root)[2]['preparation_receipt_sha256']
    receipt['archive_sha256'] = '0' * 64
    (root / 'OFT_NATIVE_RECEIPT.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='preparation receipt'):
        candidate._inspect_native_files(root)


@pytest.mark.parametrize('name', ['vllm/_C_stable_libtorch.abi3.so', 'vllm/version.py'])
def test_cold_changed_extension_or_python_source_fails_before_import(tmp_path, monkeypatch, name):
    root = make_native_tree(tmp_path, monkeypatch)
    (root / name).write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='fixed SM120|differs from RECORD'):
        candidate._inspect_native_files(root)


def test_loader_restores_search_path_poisoned_import_is_not_retried(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    root = make_native_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(candidate, '_BACKEND', None)
    monkeypatch.setattr(candidate, '_LOAD_FAILURE', None)
    monkeypatch.setattr(candidate, '_module_origins', lambda root: {})
    monkeypatch.setattr(candidate, '_operators', lambda: (_ for _ in ()).throw(RuntimeError('not loaded')))
    calls = []
    def fail(name):
        calls.append(name)
        assert sys.path[0] == str(root)
        raise ImportError('injected stable import failure')
    monkeypatch.setattr(candidate.importlib, 'import_module', fail)
    original = list(sys.path)
    with pytest.raises(ImportError, match='injected'):
        candidate.load_native_backend(root, 'cuda:0')
    assert sys.path == original and calls == ['vllm']
    with pytest.raises(RuntimeError, match='restart the process'):
        candidate.load_native_backend(root, 'cuda:0')
    assert calls == ['vllm']


def test_unknown_dispatcher_owner_and_uncold_use_are_rejected(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    root = make_native_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(candidate, '_BACKEND', None)
    monkeypatch.setattr(candidate, '_LOAD_FAILURE', None)
    monkeypatch.setattr(candidate, '_module_origins', lambda root: {})
    monkeypatch.delitem(sys.modules, 'vllm._C_stable_libtorch', raising=False)
    monkeypatch.setattr(candidate, '_operators', schemas)
    with pytest.raises(RuntimeError, match='no verified stable-extension owner'):
        candidate.load_native_backend(root, 'cuda:0')
    with pytest.raises(RuntimeError, match='Explicitly load'):
        candidate._require_backend(torch.device('cuda:0'))


def test_cold_load_is_idempotent_evidence_owned_and_device_operator_changes_rejected(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    root = make_native_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(candidate, '_BACKEND', None)
    monkeypatch.setattr(candidate, '_LOAD_FAILURE', None)
    monkeypatch.delitem(sys.modules, 'vllm._C_stable_libtorch', raising=False)
    package = SimpleNamespace(__version__='0.32.0', __file__=str(root / 'vllm/__init__.py'))
    extension = SimpleNamespace(__file__=str(root / 'vllm/_C_stable_libtorch.abi3.so'))
    q, mm = schemas()
    calls = []
    def imported(name):
        calls.append(name)
        module = package if name == 'vllm' else extension
        monkeypatch.setitem(sys.modules, name, module)
        return module
    def operators():
        if sys.modules.get('vllm._C_stable_libtorch') is not extension:
            raise RuntimeError('not loaded')
        return q, mm
    monkeypatch.setattr(candidate.importlib, 'import_module', imported)
    monkeypatch.setattr(candidate, '_operators', operators)
    monkeypatch.setattr(candidate, '_module_origins', lambda root: {})
    original = list(sys.path)
    first = candidate.load_native_backend(root, 'cuda:0')
    assert first is candidate.load_native_backend(root, 'cuda:0')
    assert sys.path == original and calls == ['vllm', 'vllm._C_stable_libtorch']
    manifest = first.manifest()
    assert manifest['import_succeeded'] and not manifest['execution_validated'] and not manifest['quality_validated']
    manifest['runtime']['capability'][0] = 0
    assert first.manifest()['runtime']['capability'] == [12, 0]
    with pytest.raises(RuntimeError, match='binding changed'):
        first.validate(torch.device('cuda:1'))
    monkeypatch.setattr(candidate, '_operators', lambda: (q, object()))
    with pytest.raises(RuntimeError, match='binding changed'):
        first.validate(torch.device('cuda:0'))


def test_foreign_preloaded_package_and_extended_namespace_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'target'
    root.mkdir()
    outside = tmp_path / 'foreign.py'
    outside.write_text('')
    monkeypatch.setitem(sys.modules, 'vllm.foreign_fixture', SimpleNamespace(__file__=str(outside)))
    with pytest.raises(ValueError, match='inside the explicit root'):
        candidate._module_origins(root)


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_fake_fullgraph_two_native_boundaries_and_fresh_result_layouts(m, k, n):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    for name in ('prefix_nvfp4_quantize_rows_v1', 'prefix_nvfp4_native_scaled_mm_v1'):
        schema = getattr(torch.ops.vvla_repro_pi05_lite, name).default._schema
        assert all(argument.alias_info is None for argument in schema.arguments)
        assert all(result.alias_info is None for result in schema.returns)
    with FakeTensorMode():
        x = torch.empty((m, k), device='cuda:0', dtype=torch.bfloat16)
        qw = torch.empty((n, k // 2), device='cuda:0', dtype=torch.uint8)
        sw = torch.empty(candidate.scale_shape(n, k), device='cuda:0', dtype=torch.float8_e4m3fn)
        scale, alpha = (torch.empty((1,), device='cuda:0', dtype=torch.float32) for _ in range(2))
        y, qx, sx = candidate.scaled_linear(x, qw, sw, scale, alpha)
        assert y.shape == (m, n) and y.stride() == (n, 1) and y.dtype == torch.bfloat16
        assert qx.shape == (m, k // 2) and qx.dtype == torch.uint8
        assert sx.shape == candidate.scale_shape(m, k) and sx.dtype == torch.float8_e4m3fn
        assert len({id(t) for t in (x, qw, sw, scale, alpha, y, qx, sx)}) == 8
        graph = make_fx(candidate.scaled_linear)(x, qw, sw, scale, alpha)
    calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
    assert calls.count('vvla_repro_pi05_lite.prefix_nvfp4_quantize_rows_v1.default') == 1
    assert calls.count('vvla_repro_pi05_lite.prefix_nvfp4_native_scaled_mm_v1.default') == 1
    assert not any(name.startswith('_C.') for name in calls)


def exact(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


_UNRETIRED = []


def require_backend():
    if _UNRETIRED:
        raise RuntimeError('A previous native CUDA contract did not retire; no later GPU submissions')
    if not torch.cuda.is_available():
        pytest.skip('Root schedules actual GPU contracts separately')
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('This explicit candidate is SM120 only')
    path = os.environ.get('VLA_NVFP4_NATIVE_PATH')
    if not path:
        pytest.fail('CUDA contracts require an explicit VLA_NVFP4_NATIVE_PATH')
    return candidate.load_native_backend(path, torch.device('cuda', torch.cuda.current_device()))


def direct_quantize(x, scale):
    m, k = x.shape
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    padded_m, cols = candidate.scale_shape(m, k)
    storage = torch.zeros((padded_m, cols // 4), dtype=torch.int32, device=x.device)
    torch.ops._C.scaled_fp4_quant.out(x, scale, True, output=q, output_scale=storage)
    return q, storage.view(torch.float8_e4m3fn)


def direct_reference(x, qw, sw, scale, alpha):
    qx, sx = direct_quantize(x, scale)
    result = torch.empty((x.shape[0], qw.shape[0]), device=x.device, dtype=torch.bfloat16)
    torch.ops._C.cutlass_scaled_fp4_mm(result, qx, qw, sx, sw, alpha)
    return result, qx, sx


@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_cuda_analytic_scale_axes_signed_e2m1_packing_and_fresh_outputs(k, n):
    backend = require_backend()
    assert backend.manifest()['execution_validated'] is False  # Import evidence cannot promote itself.
    with torch.no_grad():
        one = torch.ones((1,), dtype=torch.float32, device='cuda')
        candidate.validate_cold_scale(one)
        # Exactly representable six and power-of-two scales avoid using the
        # native candidate itself as this mathematical oracle.
        a = torch.pow(2., torch.arange(576, device='cuda').remainder(5).float() - 2) * 6
        b = torch.pow(2., torch.arange(n, device='cuda').remainder(7).float() - 3) * 6
        a[1::2].neg_()
        b[1::3].neg_()
        x = a[:, None].expand(576, k).to(torch.bfloat16).contiguous()
        weight = b[:, None].expand(n, k).to(torch.bfloat16).contiguous()
        qw, sw = candidate.quantize_weight(weight, one)
        assert qw._version >= 0 and sw._version >= 0
        actual = candidate.scaled_linear(x, qw, sw, one, one)
        expected = (k * a[:, None] * b[None, :]).to(torch.bfloat16)
        exact((actual[0],), (expected,))
        expected_q = torch.where(a[:, None] < 0, 255, 119).to(torch.uint8).expand(576, k // 2)
        exact((actual[1],), (expected_q,))
        saved = tuple(t.clone() for t in actual)
        other = candidate.scaled_linear(x * .5, qw, sw, one, one)
        exact(actual, saved)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actual, other, strict=True))
        assert torch.isfinite(actual[0]).all()
        torch.cuda.synchronize()


@pytest.mark.parametrize('m', candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_cuda_eager_compiled_capture_repeat_all_inputs_refresh_restore_and_owned_outputs(m, k, n):
    require_backend()
    with torch.no_grad():
        generator = torch.Generator(device='cuda').manual_seed(9171 + m + k)
        x = torch.randn((m, k), generator=generator, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn((n, k), generator=generator, device='cuda', dtype=torch.bfloat16) * .02
        scale = (2688. / x.float().abs().amax()).reshape(1)
        wg = (2688. / weight.float().abs().amax()).reshape(1)
        alpha = (1. / (scale * wg)).float()
        for value in (scale, wg, alpha):
            candidate.validate_cold_scale(value)
        qw, sw = candidate.quantize_weight(weight, wg)
    with torch.inference_mode():
        reference = tuple(t.clone() for t in direct_reference(x, qw, sw, scale, alpha))
        eager = candidate.scaled_linear(x, qw, sw, scale, alpha)
        exact(eager, reference)
        saved = tuple(t.clone() for t in eager)
        function = torch.compile(candidate.scaled_linear, fullgraph=True, dynamic=False,
                                 options=dict(candidate.COMPILE_OPTIONS))
        stream, graph, warm = torch.cuda.Stream(), torch.cuda.CUDAGraph(), []
        targets = (x, qw, sw, scale, alpha)
        original = tuple(t.clone() for t in targets)
        pointers = tuple(t.data_ptr() for t in targets)
        stream.wait_stream(torch.cuda.current_stream())
        retained = [function, stream, graph, targets, original, eager, saved, reference, warm, weight, wg]
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm.append(function(*targets))
            stream.synchronize()
            for output in warm:
                exact(output, reference)
            with torch.cuda.graph(graph, stream=stream):
                captured = function(*targets)
            retained.append(captured)
            with torch.cuda.stream(stream):
                graph.replay()  # Capture alone does not initialize output values.
            stream.synchronize()
            exact(captured, reference)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    graph.replay()
            stream.synchronize()
            exact(captured, reference)
            with torch.cuda.stream(stream):
                x.mul_(.75)
                scale.mul_(2.)
                alpha.copy_(1. / (scale * wg))
                changed_q, changed_s = direct_quantize(weight * -.5, wg)
                retained.extend((changed_q, changed_s))
                qw.copy_(changed_q)
                sw.copy_(changed_s)
                changed_reference = direct_reference(*targets)
                retained.append(changed_reference)
                graph.replay()
            stream.synchronize()
            exact(captured, changed_reference)
            exact(eager, saved)
            assert pointers == tuple(t.data_ptr() for t in targets)
            with torch.cuda.stream(stream):
                for target, source in zip(targets, original, strict=True):
                    target.copy_(source)
                graph.replay()
            stream.synchronize()
            exact(captured, reference)
        finally:
            try:
                torch.cuda.synchronize()
                graph.reset()
            except BaseException:
                _UNRETIRED.append(retained)
                raise
