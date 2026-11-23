# SPDX-License-Identifier: Apache-2.0
"""GELU(tanh) x up -> native NVFP4 bytes, without a BF16 HBM intermediate.

Quantization follows vLLM's nvfp4_utils.cuh: two BF16 rounding boundaries,
16-element E4M3 scales, rcp.approx.ftz, and the CUTLASS swizzled scale layout.
This is opt-in. Native packed-byte and scale equality must be checked on the
target GPU before adopting it. All padded scale bytes are written every call.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rcp_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _gelu_mul_nvfp4(G, U, GS, Q, S, M: tl.constexpr, K: tl.constexpr,
                    PAD_M: tl.constexpr, PAD_K: tl.constexpr, GROUPS: tl.constexpr):
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = group // (PAD_K // 16)
    col_group = group % (PAD_K // 16)
    col = col_group[:, None] * 16 + tl.arange(0, 16)[None, :]
    valid = (row[:, None] < M) & (col < K)
    g = tl.load(G + row[:, None] * K + col, valid, 0).to(tl.float32)
    u = tl.load(U + row[:, None] * K + col, valid, 0).to(tl.float32)
    arg = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    activated = (0.5 * g * (1.0 + libdevice.tanh(arg))).to(tl.bfloat16).to(tl.float32)
    # Both conversions are observable in the old GELU-mul -> quant path.
    hidden = (activated * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(hidden), 1)
    global_scale = tl.load(GS).to(tl.float32)
    sf = global_scale * (amax * _rcp_ftz(6.0))
    sf8 = tl.minimum(sf, 448.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    sf32 = sf8.to(tl.float32)
    multiplier = tl.where(sf32 != 0, _rcp_ftz(sf32 * _rcp_ftz(global_scale)), 0.0)
    scaled = hidden * multiplier[:, None]
    even = tl.gather(scaled, tl.broadcast_to((2 * tl.arange(0, 8))[None, :], (GROUPS, 8)), 1)
    odd = tl.gather(scaled, tl.broadcast_to((2 * tl.arange(0, 8) + 1)[None, :], (GROUPS, 8)), 1)
    # PTX packs the first argument in the high nibble; element 0 is low.
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, $2, $1; cvt.u32.u8 $0, q; }",
        constraints="=r,f,f", args=[even, odd], dtype=tl.uint32, is_pure=True, pack=1,
    ).to(tl.uint8)
    qcol = col_group[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Q + row[:, None] * (K // 2) + qcol, packed, (row[:, None] < M) & (qcol < K // 2))
    # [ceil(M/128), ceil(K/64), 32, 4, 4] in byte units.
    sf_offset = ((row // 128) * (PAD_K // 64) + col_group // 4) * 512
    sf_offset += (row % 32) * 16 + ((row // 32) % 4) * 4 + col_group % 4
    sf_bytes = sf8.to(tl.uint8, bitcast=True)
    sf_bytes = tl.where((row < M) & (col_group < K // 16), sf_bytes, 0)
    tl.store(S + sf_offset, sf_bytes, row < PAD_M)


def gelu_mul_nvfp4(gate, up, global_scale):
    """Return native uint8 packed values and float8_e4m3fn swizzled scales.

    The caller validates hardware once when installing the recipe. Allocation
    is ordinary torch allocation and is owned by CUDA Graph during capture;
    there is no synchronization, tuning, tensor-value read, or global cache.
    """
    if (gate.shape != up.shape or gate.dtype != torch.bfloat16 or up.dtype != gate.dtype
            or gate.device != up.device or not gate.is_cuda or gate.ndim < 2
            or not gate.is_contiguous() or not up.is_contiguous()):
        raise ValueError("Fused NVFP4 requires matching contiguous CUDA BF16 gate/up tensors")
    if (global_scale.dtype != torch.float32 or global_scale.numel() != 1
            or global_scale.device != gate.device or not global_scale.is_contiguous()):
        raise ValueError("NVFP4 global scale must be one contiguous FP32 value on the input device")
    width = gate.shape[-1]
    if width == 0 or width % 32:
        raise ValueError("Fused NVFP4 width must be positive and divisible by 32")
    rows = gate.numel() // width
    padded_rows, padded_width = triton.cdiv(rows, 128) * 128, triton.cdiv(width, 64) * 64
    packed = torch.empty((rows, width // 2), device=gate.device, dtype=torch.uint8)
    scales = torch.empty((padded_rows, padded_width // 64), device=gate.device, dtype=torch.int32)
    if rows:
        groups = 64
        _gelu_mul_nvfp4[(triton.cdiv(padded_rows * (padded_width // 16), groups),)](
            gate, up, global_scale, packed, scales.view(torch.uint8), rows, width,
            padded_rows, padded_width, groups, num_warps=4, enable_fp_fusion=False,
        )
    return packed, scales.view(torch.float8_e4m3fn)
