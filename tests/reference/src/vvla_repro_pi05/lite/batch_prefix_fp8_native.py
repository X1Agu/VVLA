"""Independent B32 SM120 prefix FP8 primitives; no B1 fallback.

This package-owned wrapper has a NEW source identity. The earlier diagnostic
backend report does not authorize it, even though its kernel is byte-identical.
Root must execute a new backend probe through these actual operators. Neither
the implementation manifest nor a successful import is runtime/quality evidence.

Inputs are finite, inference-only tensors; cold packing and the eventual owner
must verify finite positive scales. Hot operators check metadata without host
tensor reads. They return fresh outputs and retain no persistent tensor state.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

VERSION = 'pi05-b32-prefix-fp8-native-sm120-v1'
VARIANT = 'b32-triton-eager-rn-native-sm120-v1'
IMPLEMENTATION = 'b32_triton_eager_rn_native_opaque_v1'
QUANTIZE_OPERATOR = 'vvla_repro_pi05_lite::batch32_prefix_fp8_quantize_rows_v1'
SCALED_MM_OPERATOR = 'vvla_repro_pi05_lite::batch32_prefix_fp8_native_scaled_mm_v1'
SOURCE_FILES = ('batch_prefix_fp8_native.py', 'batch_prefix_fp8_native_kernel.py')
BATCH_SIZE = 32
PER_SLOT_CAPACITIES = (576, 640)
SUPPORTED_CAPACITIES = (18432, 20480)  # flattened true-B32 rows
SUPPORTED_GEOMETRIES = ((2048, 16384), (16384, 2048))  # (K, N)
REQUIRED_RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
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
        required_runtime=REQUIRED_RUNTIME, supported_capacities=list(SUPPORTED_CAPACITIES),
        supported_geometries=[list(shape) for shape in SUPPORTED_GEOMETRIES],
        batch_size=BATCH_SIZE, per_slot_capacities=list(PER_SLOT_CAPACITIES),
        quantizer_arithmetic=ARITHMETIC, compile_options=COMPILE_OPTIONS,
        operators=dict(quantizer=QUANTIZE_OPERATOR, gemm=SCALED_MM_OPERATOR),
        dispatch='ATEN torch._scaled_mm inside opaque custom op',
        weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose',
        input_dtype='bfloat16', quantized_dtype='float8_e4m3fn',
        scale_dtype='float32', accumulator_dtype='float32', output_dtype='bfloat16',
        use_fast_accum=False, bias=None, scale_result=None, fallback_permitted=False,
        backend_evidence='new same-card executed production-wrapper report required',
        installation='none; no policy, weight owner, or quality claim'))


def validate_runtime(device):
    """Fail closed on unsupported binaries/architecture; no tensor-content reads."""
    if (device.type != 'cuda' or str(torch.__version__) != REQUIRED_RUNTIME['torch']
            or torch.version.cuda != REQUIRED_RUNTIME['cuda']
            or torch.version.git_version != REQUIRED_RUNTIME['torch_git']):
        raise RuntimeError('Prefix FP8 native requires the explicit Torch2.13.0/cu130 runtime')
    if tuple(torch.cuda.get_device_capability(device)) != tuple(REQUIRED_RUNTIME['capability']):
        raise RuntimeError('Prefix FP8 native v1 supports SM120 only; no architecture fallback')
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Prefix FP8 native requires explicit dtypes; autocast unsupported')


def validate_quantizer_input(value):
    if (value.device.type != 'cuda' or value.dtype != torch.bfloat16 or value.ndim != 2
            or value.shape[0] not in SUPPORTED_CAPACITIES or value.shape[1] not in (2048, 16384)
            or value.stride() != (value.shape[1], 1) or value.requires_grad):
        raise ValueError('Prefix FP8 quantizer requires contiguous inference CUDA BF16 [18432|20480,2048|16384]')


def validate_scaled_mm_inputs(qx, qw, sx, sw):
    tensors = (qx, qw, sx, sw)
    if any(t.ndim != 2 or t.device != qx.device or t.device.type != 'cuda' or t.requires_grad for t in tensors):
        raise ValueError('Prefix FP8 native operands must be same-device inference CUDA matrices')
    m, k = qx.shape
    n = qw.shape[0]
    if (m not in SUPPORTED_CAPACITIES or (k, n) not in SUPPORTED_GEOMETRIES or qw.shape[1] != k
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
    from .batch_prefix_fp8_native_kernel import launch
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
