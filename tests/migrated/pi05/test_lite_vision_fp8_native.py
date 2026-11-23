"""Root runs these remotely. No full checkpoint or policy installation."""
import hashlib
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import vision_fp8_native as candidate

_PROBE = None  # Preserve failed CUDA graph owners across parameterized cases.


def metadata(shape, dtype, *, stride=None):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=torch.device('cuda:0'),
        requires_grad=False, stride=lambda: ((shape[1], 1) if len(shape) == 2 else (1,)) if stride is None else stride)


def operands(k=1152, n=4304):
    return [metadata((512, k), torch.float8_e4m3fn), metadata((n, k), torch.float8_e4m3fn),
            metadata((512, 1), torch.float32, stride=(1, 1)),
            metadata((n, 1), torch.float32, stride=(1, 1)), metadata((n,), torch.bfloat16)]


def load_probe():
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    root = Path(__file__).resolve().parents[2]
    path = root / 'benchmarks/pi05-lite-bf16-20261105/profiling/probe_vision_fp8_native.py'
    spec = importlib.util.spec_from_file_location('_vision_fp8_contract_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PROBE = module
    return _PROBE


def test_manifest_is_new_source_biased_variant_not_old_prefix_authorization():
    from vvla_repro_pi05.lite.compilation import INDUCTOR_OPTIONS
    value = candidate.implementation_manifest()
    root = Path(candidate.__file__).resolve().parent
    assert value['source_files_sha256'] == {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                           for name in candidate.SOURCE_FILES}
    assert value['variant'] == 'vision-biased-rowwise-fp8-native-sm120-v1'
    assert value['supported_m'] == 512
    assert value['supported_geometries'] == [[1152, 4304], [4304, 1152]]
    assert value['compile_options'] == INDUCTOR_OPTIONS
    assert value['bias'] == 'explicit original contiguous BF16[N]; no post-BF16 add'
    assert value['use_fast_accum'] is False and value['fallback_permitted'] is False
    assert value['scale_result'] is None and 'passed' not in value and 'sites' not in value
    assert struct.unpack('>I', struct.pack('>f', value['quantizer_arithmetic']['scale_reciprocal_fp32']))[0] == 0x3b124925
    value['required_runtime']['capability'][0] = 0
    assert candidate.implementation_manifest()['required_runtime']['capability'] == [12, 0]


@pytest.mark.parametrize('change', [dict(device=torch.device('cpu')), dict(dtype=torch.float32),
    dict(ndim=3), dict(shape=(256, 1152)), dict(shape=(512, 2048)),
    dict(stride=lambda: (1153, 1)), dict(requires_grad=True)])
def test_quantizer_rejects_unsupported_shape_dtype_stride_and_grad(change):
    value = metadata((512, 1152), torch.bfloat16)
    value.__dict__.update(change)
    with pytest.raises(ValueError, match='quantizer requires'):
        candidate.validate_quantizer_input(value)


@pytest.mark.parametrize('slot,change', [(0, dict(shape=(256, 1152))),
    (1, dict(shape=(4096, 1152))), (1, dict(stride=lambda: (1, 4304))),
    (2, dict(dtype=torch.bfloat16)), (2, dict(shape=(1, 512))),
    (3, dict(device=torch.device('cuda:1'))), (4, dict(dtype=torch.float32)),
    (4, dict(shape=(4304, 1), ndim=2)), (4, dict(stride=lambda: (2,))),
    (4, dict(requires_grad=True)), (4, dict(device=torch.device('cpu')))])
def test_biased_gemm_rejects_unapproved_geometry_scale_bias_and_device(slot, change):
    values = operands()
    values[slot].__dict__.update(change)
    with pytest.raises(ValueError, match='Vision FP8'):
        candidate.validate_scaled_mm_inputs(*values)


def test_runtime_rejects_other_architecture_version_and_autocast(monkeypatch):
    monkeypatch.setattr(torch, '__version__', candidate.REQUIRED_RUNTIME['torch'])
    monkeypatch.setattr(torch.version, 'cuda', candidate.REQUIRED_RUNTIME['cuda'])
    monkeypatch.setattr(torch.version, 'git_version', candidate.REQUIRED_RUNTIME['torch_git'])
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: False)
    candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 9))
    with pytest.raises(RuntimeError, match='SM120 only'): candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: True)
    with pytest.raises(RuntimeError, match='autocast'): candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: False)
    monkeypatch.setattr(torch.version, 'git_version', '0' * 40)
    with pytest.raises(RuntimeError, match='explicit Torch'): candidate.validate_runtime(torch.device('cuda:0'))


def test_cold_weight_packing_rejects_inference_mode_without_gpu_initialization():
    with torch.inference_mode(), pytest.raises(ValueError, match='outside inference_mode'):
        candidate.quantize_weight(metadata((4304, 1152), torch.bfloat16))


@pytest.mark.parametrize('k,n', candidate.SUPPORTED_GEOMETRIES)
def test_fake_trace_keeps_bias_explicit_two_opaque_ops_and_fresh_outputs(k, n):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    for name in ('vision_fp8_quantize_rows_v1', 'vision_fp8_native_biased_mm_v1'):
        schema = getattr(torch.ops.vvla_repro_pi05_lite, name).default._schema
        assert all(arg.alias_info is None for arg in schema.arguments)
        assert all(output.alias_info is None for output in schema.returns)
    with FakeTensorMode():
        x = torch.empty((512, k), device='cuda:0', dtype=torch.bfloat16)
        qw = torch.empty((n, k), device='cuda:0', dtype=torch.float8_e4m3fn)
        sw = torch.empty((n, 1), device='cuda:0', dtype=torch.float32)
        bias = torch.empty((n,), device='cuda:0', dtype=torch.bfloat16)
        y, qx, sx = candidate.scaled_linear(x, qw, sw, bias)
        assert y.shape == (512, n) and y.dtype == torch.bfloat16 and y.stride() == (n, 1)
        assert qx.shape == x.shape and qx.dtype == torch.float8_e4m3fn
        assert sx.shape == (512, 1) and sx.dtype == torch.float32
        assert len({id(v) for v in (x, qw, sw, bias, y, qx, sx)}) == 7
        graph = make_fx(candidate.scaled_linear)(x, qw, sw, bias)
    calls = [node for node in graph.graph.nodes if node.op == 'call_function']
    names = [str(node.target) for node in calls]
    assert names.count('vvla_repro_pi05_lite.vision_fp8_quantize_rows_v1.default') == 1
    assert names.count('vvla_repro_pi05_lite.vision_fp8_native_biased_mm_v1.default') == 1
    gemm = next(node for node in calls if str(node.target).endswith('vision_fp8_native_biased_mm_v1.default'))
    assert len(gemm.args) == 5 and gemm.args[-1].op == 'placeholder'
    assert not any('aten._scaled_mm' in name or 'aten.add' in name for name in names)


def test_probe_exact_rejects_signed_zero_and_does_not_mask_bad_shapes():
    probe = load_probe()
    with pytest.raises(RuntimeError, match='raw-byte'):
        probe.exact(torch, (torch.tensor([0.]),), (torch.tensor([-0.]),), 'signed zero')
    with pytest.raises(RuntimeError, match='raw-byte'):
        probe.exact(torch, (torch.zeros(2),), (torch.zeros(1, 2),), 'shape')


def test_probe_cancellation_separates_correct_bias_boundary_on_cpu():
    dot = torch.tensor([1 + 2.**-8], dtype=torch.float32)
    bias = torch.tensor([-1.], dtype=torch.bfloat16)
    fused = (dot + bias.float()).to(torch.bfloat16)
    forbidden = (dot.to(torch.bfloat16).float() + bias.float()).to(torch.bfloat16)
    assert fused.item() == 2.**-8 and forbidden.item() == 0.


def test_probe_fp64_oracle_uses_full_k_and_separates_quantization_error():
    probe = load_probe()
    x = torch.ones((16, 24), dtype=torch.bfloat16)
    weight = torch.full((32, 24), .5, dtype=torch.bfloat16)
    bias = torch.full((32,), .25, dtype=torch.bfloat16)
    qx, sx = probe.eager_quantize(torch, x)
    qw, sw = probe.eager_quantize(torch, weight)
    result = torch.full((16, 32), 12.25, dtype=torch.bfloat16)
    evidence = probe.high_precision_tiles(torch, x, weight, bias, qw, sw, (result, qx, sx))
    assert evidence['quality_gate'] is False and len(evidence['tiles']) == 3
    assert all(row['full_k'] == 24 and row['oracle_dtype'] == 'float64' for row in evidence['tiles'])
    assert all(row['original_bf16_math']['max_abs'] == 0. for row in evidence['tiles'])
    assert all(row['quantized_operand_math']['max_abs'] >= 0. for row in evidence['tiles'])


def test_probe_frozen_output_and_reuse_are_rejected_before_loading_model(tmp_path):
    probe = load_probe()
    parent = tmp_path / 'formal'; parent.mkdir(); (parent / 'FROZEN.json').write_text('{}')
    with pytest.raises(ValueError, match='frozen evidence'): probe.safe_output(parent / 'diagnostic')
    existing = tmp_path / 'used'; existing.mkdir()
    with pytest.raises(ValueError, match='new diagnostic directory'): probe.safe_output(existing)


@pytest.mark.parametrize('shape', ((512, 1152, 4304), (512, 4304, 1152)))
def test_cuda_vision_biased_native_quantization_oracle_compile_capture_and_refresh(shape):
    if not torch.cuda.is_available(): pytest.skip('Root queues these CUDA contracts separately')
    if torch.cuda.get_device_capability() != (12, 0): pytest.skip('Explicit SM120 variant only')
    probe = load_probe()
    incoming = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
        result = probe.shape_contract(torch, candidate, shape)
        assert result['status'] == 'contracts_passed' and result['graph_retired'] is True
        assert result['quantization']['raw_byte_exact'] and result['analytic']['bias_cancellation_exact']
        assert result['execution']['same_pointer_refresh_exact'] and result['execution']['restoration_exact']
    finally:
        torch.set_float32_matmul_precision(incoming[1]); torch.backends.cuda.matmul.allow_tf32 = incoming[0]
