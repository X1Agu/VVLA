"""Package-owned SM89 prefix-down FP8 primitives, not a policy installation.

New wrapper/source identity requires new same-card executed backend evidence.
Only the 17 prefix down projections are in scope; gate/up remain BF16. The
caller cold-creates/verifies/owns immutable scalar one, original qW/scales and
their lifecycle. No tensor, graph, stream or model is cached here. Hot operators
read metadata only; finite inputs/positive scales are caller cold contracts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

VERSION = 'pi05-prefix-down-fp8-native-sm89-v1'
VARIANT = 'sm89-down-scalar-fp32-native-v1'
IMPLEMENTATION = 'sm89_down_native_opaque_scalar_fp32_postscale_v1'
QUANTIZE_OPERATOR = 'vvla_pi05_lite::prefix_down_sm89_quantize_rows_v1'
MM_OPERATOR = 'vvla_pi05_lite::prefix_down_sm89_scalar_mm_fp32_v1'
SCALE_OPERATOR = 'vvla_pi05_lite::prefix_down_sm89_postscale_bf16_v1'
SOURCE_FILES = ('prefix_fp8_down_sm89.py', 'prefix_fp8_down_sm89_quantizer_kernel.py',
                'prefix_fp8_down_sm89_postscale_kernel.py')
SUPPORTED_CAPACITIES = (576, 640)
SUPPORTED_GEOMETRIES = ((16384, 2048),)
REQUIRED_RUNTIME = dict(torch_api='2.13', cuda_major=13, capability=[8, 9])
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='PTX cvt.rn.satfinite.e4m3x2.f32; duplicated lane; low byte bitcast',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu')


def implementation_manifest():
    """Actual file bytes and requirements; neither runtime nor quality proof."""
    root = Path(__file__).resolve().parent
    return deepcopy(dict(version=VERSION, variant=VARIANT, implementation=IMPLEMENTATION,
        core_files_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        required_runtime=REQUIRED_RUNTIME, historical_capacities=list(SUPPORTED_CAPACITIES), row_alignment=16,
        supported_geometries=[list(v) for v in SUPPORTED_GEOMETRIES],
        intended_scope=dict(component='prefix', projections=['down_proj'], layer_indices=list(range(17)),
                            intended_sites=17, gate_up_precision='bfloat16'),
        quantizer_arithmetic=ARITHMETIC, compile_options=COMPILE_OPTIONS,
        operators=dict(quantizer=QUANTIZE_OPERATOR, gemm=MM_OPERATOR, postscale=SCALE_OPERATOR),
        dispatch='ATEN torch._scaled_mm singleton FP32 scales inside opaque op; native cuBLASLt',
        weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]',
        input_dtype='bfloat16', quantized_dtype='float8_e4m3fn', scale_dtype='float32',
        accumulator_dtype='float32', intermediate_dtype='float32', output_dtype='bfloat16',
        postscale='two ordered mul.rn.f32 sx[row] then sw[column]; sole BF16 output cast',
        scalar_one=dict(shape=[1], dtype='float32', value=1.0,
                        tensor_kind='ordinary versioned Tensor; inference tensors forbidden',
                        ownership='caller-owned immutable; cold validate_unit_scale required',
                        hot_validation='metadata only; caller binds private identity/compile lifetime'),
        use_fast_accum=False, allow_tf32=False, bias=None, scale_result=None,
        old_rowwise_output_exact_required=False, old_rowwise_no_bias_positive_zero_add_preserved=False,
        fallback_permitted=False, backend_evidence='separate benchmark qualification; no report required by runtime',
        installation='none; intended 17-site scope only, no policy or quality claim'))


def validate_runtime(device):
    if (device.type != 'cuda' or not str(torch.__version__).startswith('2.13.')
            or torch.version.cuda is None or torch.version.cuda.split('.')[0] != '13'
            or not hasattr(torch, '_scaled_mm')):
        raise RuntimeError('Prefix down FP8 requires the checked Torch2.13/CUDA13 API')
    if tuple(torch.cuda.get_device_capability(device)) != (8, 9):
        raise RuntimeError('Down-only FP8 uses the SM89 backend; no silent architecture substitution')
    if torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('Prefix down FP8 requires explicit dtypes and disabled TF32')


def _unit_metadata(one):
    if (one.device.type != 'cuda' or one.dtype != torch.float32 or one.shape != (1,)
            or one.stride() != (1,) or one.requires_grad):
        raise ValueError('Unit scale requires inference contiguous CUDA FP32[1]')


def validate_unit_scale(one):
    """Cold content check/evidence; never call inside capture or the hot loop."""
    _unit_metadata(one)
    validate_runtime(one.device)
    if torch.is_inference(one):
        raise ValueError('Unit scale must be an ordinary versioned Tensor for caller lifecycle checks')
    version = one._version
    if type(version) is not int or version < 0:
        raise ValueError('Unit scale requires a readable nonnegative mutation counter')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Unit scale cold validation cannot run during capture')
    raw = bytes(one.detach().view(torch.uint8).cpu().tolist())
    if raw != b'\x00\x00\x80\x3f':
        raise ValueError('Unit scale must contain exact FP32 positive one')
    return dict(passed=True, dtype='float32', shape=[1], stride=[1], value=1.0,
                inference_tensor=False, version_counter_available=True,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                ownership='caller-owned; no core retention', validation='cold exact bytes')


def create_unit_scale(device):
    """Cold constructor only; caller takes ownership, then MUST validate bytes.

    Returning before the blocking content check lets a caller retain the tensor
    if that check/synchronization fails. This function returns no passed proof.
    """
    validate_runtime(device)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Unit scale cold creation cannot run during capture')
    # A private inference Tensor has no mutation counter. The future owner must
    # be able to detect same-pointer content changes without per-request D2H.
    with torch.inference_mode(False):
        value = torch.ones((1,), dtype=torch.float32, device=device)
    return value


def validate_quantizer_input(value):
    if (value.device.type != 'cuda' or value.dtype != torch.bfloat16 or value.ndim != 2
            or (value.shape[0] < 1 or value.shape[0] % 16) or value.shape[1] != 16384
            or value.stride() != (16384, 1) or value.requires_grad):
        raise ValueError('Down quantizer requires inference contiguous CUDA BF16 [576|640,16384]')


def validate_mm(qx, qw, one):
    if any(t.device.type != 'cuda' or t.device != qx.device or t.requires_grad for t in (qx, qw, one)):
        raise ValueError('Down scalar GEMM requires same-device inference CUDA tensors')
    if (qx.ndim != 2 or qw.ndim != 2 or (qx.shape[0] < 1 or qx.shape[0] % 16)
            or qx.shape[1] != 16384 or qw.shape != (2048, 16384)
            or qx.dtype != torch.float8_e4m3fn or qw.dtype != torch.float8_e4m3fn
            or qx.stride() != (16384, 1) or qw.stride() != (16384, 1)):
        raise ValueError('Down scalar GEMM requires MK[576|640,16384]/NK[2048,16384] E4M3fn')
    _unit_metadata(one)


def validate_postscale(z, sx, sw):
    if any(t.device.type != 'cuda' or t.device != z.device or t.requires_grad for t in (z, sx, sw)):
        raise ValueError('Down postscale requires same-device inference CUDA tensors')
    if (z.ndim != 2 or (z.shape[0] < 1 or z.shape[0] % 16) or z.shape[1] != 2048
            or z.dtype != torch.float32 or z.stride() != (2048, 1)
            or sx.dtype != torch.float32 or sx.shape != (z.shape[0], 1) or sx.stride() != (1, 1)
            or sw.dtype != torch.float32 or sw.shape != (2048, 1) or sw.stride() != (1, 1)):
        raise ValueError('Down postscale requires actual FP32 [576|640,2048] and row/channel FP32 scales')


@torch.library.custom_op(QUANTIZE_OPERATOR, mutates_args=())
def quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_quantizer_input(value)
    validate_runtime(value.device)
    from .prefix_fp8_down_sm89_quantizer_kernel import launch
    q = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale = torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device)
    launch(value, q, scale)
    return q, scale


@quantize_rows.register_fake
def _fake_quantize(value):
    validate_quantizer_input(value)
    return (torch.empty_like(value, dtype=torch.float8_e4m3fn),
            torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device))


@torch.library.custom_op(MM_OPERATOR, mutates_args=())
def scalar_mm_fp32(qx: torch.Tensor, qw: torch.Tensor, one: torch.Tensor) -> torch.Tensor:
    validate_mm(qx, qw, one)
    validate_runtime(qx.device)
    z = torch._scaled_mm(qx, qw.T, one, one, out_dtype=torch.float32, use_fast_accum=False)
    if z.dtype != torch.float32 or z.shape != (qx.shape[0], 2048) or z.stride() != (2048, 1):
        raise RuntimeError('Down native scalar GEMM did not return actual contiguous FP32 output')
    return z


@scalar_mm_fp32.register_fake
def _fake_mm(qx, qw, one):
    validate_mm(qx, qw, one)
    return torch.empty((qx.shape[0], 2048), device=qx.device, dtype=torch.float32)


@torch.library.custom_op(SCALE_OPERATOR, mutates_args=())
def rowwise_postscale(z: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    validate_postscale(z, sx, sw)
    validate_runtime(z.device)
    from .prefix_fp8_down_sm89_postscale_kernel import launch
    output = torch.empty(z.shape, device=z.device, dtype=torch.bfloat16)
    launch(z, sx, sw, output)
    return output


@rowwise_postscale.register_fake
def _fake_postscale(z, sx, sw):
    validate_postscale(z, sx, sw)
    return torch.empty(z.shape, device=z.device, dtype=torch.bfloat16)


def native_scaled_mm(qx, qw, sx, sw, one):
    return rowwise_postscale(scalar_mm_fp32(qx, qw, one), sx, sw)


def scaled_linear(x, qw, sw, one):
    qx, sx = quantize_rows(x)
    return native_scaled_mm(qx, qw, sx, sw, one), qx, sx
