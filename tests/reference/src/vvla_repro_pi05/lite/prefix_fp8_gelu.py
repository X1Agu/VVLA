"""Explicit, uninstalled SM120 prefix GELU-to-FP8 conversion candidate.

The selected model and its existing backend authorization remain unchanged.
Callers must opt in by invoking these primitives. They must establish exact
eager/compiled/captured bytes against the selected two-operation path before
any future owner integration. No speed or closed-loop quality is claimed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from . import prefix_fp8_native as native


VERSION = 'pi05-prefix-gelu-fp8-fusion-sm120-v1'
OPERATOR = 'vvla_repro_pi05_lite::prefix_gelu_fp8_quantize_rows_v1'
SOURCE_FILES = ('prefix_fp8_gelu.py', 'prefix_fp8_gelu_kernel.py')
SUPPORTED_CAPACITIES = (576, 640)
WIDTH = 16384
OUTPUT_WIDTH = 2048
COMPILE_OPTIONS = dict(native.COMPILE_OPTIONS)


def implementation_manifest():
    root = Path(__file__).resolve().parent
    references = {
        'prefix_fp8_native.py': root / 'prefix_fp8_native.py',
        'prefix_fp8_native_kernel.py': root / 'prefix_fp8_native_kernel.py',
        'models/pi05/fused_mlp_pi05.py': root.parent / 'models/pi05/fused_mlp_pi05.py',
    }
    return deepcopy(dict(
        version=VERSION, operator=OPERATOR,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in SOURCE_FILES},
        reference_source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in references.items()},
        required_runtime=native.REQUIRED_RUNTIME, compile_options=COMPILE_OPTIONS,
        supported_capacities=list(SUPPORTED_CAPACITIES), width=WIDTH, output_width=OUTPUT_WIDTH,
        input_dtype='bfloat16', output_dtypes=['float8_e4m3fn', 'float32'],
        intermediate_rounding=['GELU_tanh_to_BF16_RTNE', 'rounded_GELU_times_up_to_BF16_RTNE'],
        quantizer_arithmetic=native.ARITHMETIC,
        gemm_operator=native.SCALED_MM_OPERATOR, use_fast_accum=False,
        fp_fusion=False, hidden_materialization=False, persistent_tensors=False,
        output_ownership='fresh_nonaliasing_outputs', fallback_permitted=False,
        installation='none; explicit primitive only',
        evidence_requirement='new executed exact candidate contracts and owner evidence; old backend report is insufficient',
        performance_validated=False, quality_validated=False,
    ))


def validate_inputs(gate, up):
    if any(t.device.type != 'cuda' or t.dtype != torch.bfloat16 or t.ndim != 2
           or t.requires_grad for t in (gate, up)):
        raise ValueError('Fused prefix GELU FP8 requires inference CUDA BF16 matrices')
    if (gate.device != up.device or gate.shape != up.shape
            or gate.shape[0] not in SUPPORTED_CAPACITIES or gate.shape[1] != WIDTH
            or gate.stride() != (WIDTH, 1) or up.stride() != (WIDTH, 1)):
        raise ValueError('Fused prefix GELU FP8 requires matching contiguous [576|640,16384] inputs')


@torch.library.custom_op(OPERATOR, mutates_args=())
def gelu_quantize_rows(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_inputs(gate, up)
    native.validate_runtime(gate.device)
    from .prefix_fp8_gelu_kernel import launch
    quantized = torch.empty(gate.shape, device=gate.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((gate.shape[0], 1), device=gate.device, dtype=torch.float32)
    launch(gate, up, quantized, scale)
    return quantized, scale


@gelu_quantize_rows.register_fake
def _fake_gelu_quantize_rows(gate, up):
    validate_inputs(gate, up)
    return (torch.empty(gate.shape, device=gate.device, dtype=torch.float8_e4m3fn),
            torch.empty((gate.shape[0], 1), device=gate.device, dtype=torch.float32))


def scaled_down(gate, up, quantized_weight, weight_scale):
    """Candidate conversion plus the unchanged opaque ATEN BF16-output GEMM.

    Flattening the B1 input is the eventual owner's responsibility. Explicit
    converter outputs allow independent raw-byte contracts in the same graph.
    """
    qx, sx = gelu_quantize_rows(gate, up)
    return native.native_scaled_mm(qx, quantized_weight, sx, weight_scale), qx, sx
