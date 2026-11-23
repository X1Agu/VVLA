"""Uint8 storage to correctly rounded FP32 unit pixels, without layout changes.

Extracted from the independently measured GPU image preparation diagnostic.
Only dense HWC storage viewed as [1,3,H,W] is passed by the owning helper.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _uint8_unit(RAW, UNIT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(RAW + offsets, offsets < N, 0).to(tl.float32)
    value = tl.div_rn(value, 255.0)
    tl.store(UNIT + offsets, value, offsets < N)


def uint8_to_unit(raw):
    if (raw.device.type != "cuda" or raw.dtype != torch.uint8 or raw.ndim != 4
            or raw.shape[0] != 1 or raw.shape[1] != 3 or min(raw.shape[2:]) <= 0
            or raw.stride(1) != 1 or raw.stride(3) != 3
            or raw.stride(2) != raw.shape[3] * 3):
        raise ValueError("GPU uint8 conversion requires dense HWC storage viewed as CUDA [1,3,H,W]")
    unit = torch.empty_like(raw, dtype=torch.float32, memory_format=torch.preserve_format)
    _uint8_unit[(triton.cdiv(raw.numel(), 1024),)](
        raw, unit, raw.numel(), 1024, enable_fp_fusion=False)
    return unit
