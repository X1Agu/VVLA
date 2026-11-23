"""Independent SM120 biased vision FP8 primitives; no model owner or fallback.

Only the two M512 SigLIP projections are admitted. Bias is an explicit BF16
operand of native scaled_mm, added in its FP32 epilogue before the output cast.
The existing SigLIP activation is outside this primitive and is not replaced.
An implementation manifest is source identity, never executed/quality evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

VERSION = 'pi05-vision-biased-fp8-native-sm120-v1'
VARIANT = 'vision-biased-rowwise-fp8-native-sm120-v1'
IMPLEMENTATION = 'vision_eager_rn_native_opaque_bias_v1'
QUANTIZE_OPERATOR = 'vvla_repro_pi05_lite::vision_fp8_quantize_rows_v1'
SCALED_MM_OPERATOR = 'vvla_repro_pi05_lite::vision_fp8_native_biased_mm_v1'
SOURCE_FILES = ('vision_fp8_native.py', 'vision_fp8_native_kernel.py')
SUPPORTED_M = 512
SUPPORTED_GEOMETRIES = ((1152, 4304), (4304, 1152))  # K, N; never padded model dimensions
REQUIRED_RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
    padding='masked zero loads neutral to absmax; masked output stores; original K unchanged')


def implementation_manifest():
    root = Path(__file__).resolve().parent
    return deepcopy(dict(version=VERSION, variant=VARIANT, implementation=IMPLEMENTATION,
        source_files_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES},
        required_runtime=REQUIRED_RUNTIME, supported_m=SUPPORTED_M,
        supported_geometries=[list(pair) for pair in SUPPORTED_GEOMETRIES],
        quantizer_arithmetic=ARITHMETIC, compile_options=COMPILE_OPTIONS,
        operators=dict(quantizer=QUANTIZE_OPERATOR, gemm=SCALED_MM_OPERATOR),
        dispatch='ATEN torch._scaled_mm inside opaque custom op',
        weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose',
        input_dtype='bfloat16', quantized_dtype='float8_e4m3fn', scale_dtype='float32',
        accumulator_dtype='float32', output_dtype='bfloat16',
        epilogue='FP32 accumulator * activation scale * weight scale + original BF16 bias -> sole BF16 cast',
        bias='explicit original contiguous BF16[N]; no post-BF16 add',
        activation='unchanged original SigLIP activation, outside this primitive',
        use_fast_accum=False, scale_result=None, fallback_permitted=False,
        cold_weight_packing='Torch eager FP32 abs/amax/divide448/clamp_min/divide/clamp/E4M3fn',
        installation='none; no policy, site installation, performance or quality claim'))


def validate_runtime(device):
    if (device.type != 'cuda' or str(torch.__version__) != REQUIRED_RUNTIME['torch']
            or torch.version.cuda != REQUIRED_RUNTIME['cuda']
            or torch.version.git_version != REQUIRED_RUNTIME['torch_git']):
        raise RuntimeError('Vision FP8 requires the explicit Torch2.13.0/cu130 runtime')
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        raise RuntimeError('Vision FP8 supports SM120 only; no architecture fallback')
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Vision FP8 requires explicit dtypes; autocast unsupported')


def validate_quantizer_input(value):
    if (value.device.type != 'cuda' or value.dtype != torch.bfloat16 or value.ndim != 2
            or value.shape[0] != SUPPORTED_M or value.shape[1] not in (1152, 4304)
            or value.stride() != (value.shape[1], 1) or value.requires_grad):
        raise ValueError('Vision FP8 quantizer requires contiguous inference CUDA BF16 [512,1152|4304]')


def validate_scaled_mm_inputs(qx, qw, sx, sw, bias):
    matrices = (qx, qw, sx, sw)
    if (any(t.ndim != 2 or t.device != qx.device or t.device.type != 'cuda' or t.requires_grad
            for t in matrices)
            or bias.ndim != 1 or bias.device != qx.device or bias.requires_grad):
        raise ValueError('Vision FP8 operands require same-device CUDA matrices and explicit inference bias')
    m, k = qx.shape
    n = qw.shape[0]
    if (m != SUPPORTED_M or (k, n) not in SUPPORTED_GEOMETRIES or qw.shape[1] != k
            or qx.dtype != torch.float8_e4m3fn or qw.dtype != torch.float8_e4m3fn
            or sx.dtype != torch.float32 or sw.dtype != torch.float32
            or qx.stride() != (k, 1) or qw.stride() != (k, 1)
            or sx.shape != (m, 1) or sw.shape != (n, 1)
            or sx.stride() != (1, 1) or sw.stride() != (1, 1)
            or bias.shape != (n,) or bias.dtype != torch.bfloat16 or bias.stride() != (1,)):
        raise ValueError('Vision FP8 geometry/dtype/scale/bias layout is unsupported')


def quantize_weight(weight):
    """Cold only: original NK BF16 -> owned, ordinary versioned E4M3/FP32 tensors."""
    if (torch.is_inference_mode_enabled() or weight.device.type != 'cuda'
            or torch.cuda.is_current_stream_capturing()):
        raise ValueError('Pack vision weights outside inference_mode and CUDA graph capture')
    if (weight.ndim != 2 or tuple(reversed(weight.shape)) not in SUPPORTED_GEOMETRIES
            or weight.dtype != torch.bfloat16 or weight.stride() != (weight.shape[1], 1)):
        raise ValueError('Cold packing requires original contiguous vision NK BF16 weights')
    validate_runtime(weight.device)
    with torch.no_grad():
        fp32 = weight.float()
        if not bool(torch.isfinite(fp32).all()):
            raise ValueError('Cannot pack nonfinite vision weights')
        scale = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
        quantized = (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        return quantized, scale.contiguous()


@torch.library.custom_op(QUANTIZE_OPERATOR, mutates_args=())
def quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_quantizer_input(value)
    validate_runtime(value.device)
    from .vision_fp8_native_kernel import launch
    quantized = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale = torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device)
    launch(value, quantized, scale)
    return quantized, scale


@quantize_rows.register_fake
def _fake_quantize_rows(value):
    validate_quantizer_input(value)
    return (torch.empty_like(value, dtype=torch.float8_e4m3fn),
            torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device))


@torch.library.custom_op(SCALED_MM_OPERATOR, mutates_args=())
def native_scaled_mm(qx: torch.Tensor, qw: torch.Tensor, sx: torch.Tensor,
                     sw: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    validate_scaled_mm_inputs(qx, qw, sx, sw, bias)
    validate_runtime(qx.device)
    return torch._scaled_mm(qx, qw.T, sx, sw.T, bias=bias, scale_result=None,
                            out_dtype=torch.bfloat16, use_fast_accum=False)


@native_scaled_mm.register_fake
def _fake_native_scaled_mm(qx, qw, sx, sw, bias):
    validate_scaled_mm_inputs(qx, qw, sx, sw, bias)
    return torch.empty((qx.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=qx.device)


def scaled_linear(value, quantized_weight, weight_scale, bias):
    qx, sx = quantize_rows(value)
    return native_scaled_mm(qx, quantized_weight, sx, weight_scale, bias), qx, sx
