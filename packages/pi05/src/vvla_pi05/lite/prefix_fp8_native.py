"""Historical SM120 rowwise FP8 kernels, exposed as package-owned operators.

The original quantization arithmetic and native GEMM boundary are preserved.
Aligned batch rows are accepted; qualification is recorded outside runtime.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

VERSION = 'pi05-prefix-fp8-native-sm120-v1'
VARIANT = 'triton-eager-rn-native-sm120-v1'
IMPLEMENTATION = 'triton_eager_rn_native_opaque_v1'
QUANTIZE_OPERATOR = 'vvla_pi05_lite::prefix_fp8_quantize_rows_v1'
SCALED_MM_OPERATOR = 'vvla_pi05_lite::prefix_fp8_native_scaled_mm_v1'
SOURCE_FILES = ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')
SUPPORTED_CAPACITIES = (576, 640)
SUPPORTED_GEOMETRIES = ((2048, 16384), (16384, 2048))  # (K, N)
REQUIRED_RUNTIME = dict(torch_api='2.13', cuda_major=13, capability=[12, 0])
# Same three options as the selected BF16 policy. ATEN dispatch is guaranteed
# inside the opaque native operator, not by changing all model GEMM lowering.
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu')


def implementation_manifest():
    """Actual file bytes plus declared requirements; not a passed-backend claim."""
    root = Path(__file__).resolve().parent
    return deepcopy(dict(version=VERSION, variant=VARIANT, implementation=IMPLEMENTATION,
        quantizer_files_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in SOURCE_FILES},
        required_runtime=REQUIRED_RUNTIME, historical_capacities=list(SUPPORTED_CAPACITIES), row_alignment=16,
        supported_geometries=[list(shape) for shape in SUPPORTED_GEOMETRIES],
        quantizer_arithmetic=ARITHMETIC, compile_options=COMPILE_OPTIONS,
        operators=dict(quantizer=QUANTIZE_OPERATOR, gemm=SCALED_MM_OPERATOR),
        dispatch='ATEN torch._scaled_mm inside opaque custom op',
        weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose',
        input_dtype='bfloat16', quantized_dtype='float8_e4m3fn',
        scale_dtype='float32', accumulator_dtype='float32', output_dtype='bfloat16',
        use_fast_accum=False, bias=None, scale_result=None, fallback_permitted=False,
        backend_evidence='separate benchmark qualification; no report required for runtime',
        installation='none; no policy, weight owner, or quality claim'))


def validate_runtime(device):
    """Check the operator ABI/architecture, without author build or GPU identity."""
    if (device.type != 'cuda' or not str(torch.__version__).startswith('2.13.')
            or torch.version.cuda is None or torch.version.cuda.split('.')[0] != '13'
            or not hasattr(torch, '_scaled_mm')):
        raise RuntimeError('Prefix FP8 requires the checked Torch 2.13/CUDA 13 scaled-mm API')
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        raise RuntimeError('Rowwise prefix FP8 uses the SM120 backend; select an SM89-specific recipe on Ada')
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Prefix FP8 requires explicit dtypes outside autocast')


def validate_quantizer_input(value):
    if (value.device.type != 'cuda' or value.dtype != torch.bfloat16 or value.ndim != 2
            or (value.shape[0] < 1 or value.shape[0] % 16) or value.shape[1] not in (2048, 16384)
            or value.stride() != (value.shape[1], 1) or value.requires_grad):
        raise ValueError('Prefix FP8 quantizer requires contiguous inference CUDA BF16 [M (multiple of 16),2048|16384]')


def validate_scaled_mm_inputs(qx, qw, sx, sw):
    tensors = (qx, qw, sx, sw)
    if any(t.ndim != 2 or t.device != qx.device or t.device.type != 'cuda' or t.requires_grad for t in tensors):
        raise ValueError('Prefix FP8 native operands must be same-device inference CUDA matrices')
    m, k = qx.shape
    n = qw.shape[0]
    if ((m < 1 or m % 16) or (k, n) not in SUPPORTED_GEOMETRIES or qw.shape[1] != k
            or qx.dtype != torch.float8_e4m3fn or qw.dtype != torch.float8_e4m3fn
            or sx.dtype != torch.float32 or sw.dtype != torch.float32
            or qx.stride() != (k, 1) or qw.stride() != (k, 1)
            or sx.shape != (m, 1) or sw.shape != (n, 1)
            or sx.stride() != (1, 1) or sw.stride() != (1, 1)):
        raise ValueError('Prefix FP8 native geometry/dtype/scale layout is unsupported')


@torch.library.custom_op(QUANTIZE_OPERATOR, mutates_args=())
def quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_quantizer_input(value)
    validate_runtime(value.device)
    from .prefix_fp8_native_kernel import launch
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
def native_scaled_mm(qx: torch.Tensor, qw: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    validate_scaled_mm_inputs(qx, qw, sx, sw)
    validate_runtime(qx.device)
    # PyTorch executes this body eagerly even when the caller is fullgraph
    # compiled. QW remains original NK; transpose is a view, never a new pack.
    return torch._scaled_mm(qx, qw.T, sx, sw.T, bias=None, scale_result=None,
                            out_dtype=torch.bfloat16, use_fast_accum=False)


@native_scaled_mm.register_fake
def _fake_native_scaled_mm(qx, qw, sx, sw):
    validate_scaled_mm_inputs(qx, qw, sx, sw)
    return torch.empty((qx.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=qx.device)


def scaled_linear(value, quantized_weight, weight_scale):
    """Expose all converter bytes in the same graph for strict backend tests."""
    qx, sx = quantize_rows(value)
    return native_scaled_mm(qx, quantized_weight, sx, weight_scale), qx, sx
