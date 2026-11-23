# SPDX-License-Identifier: Apache-2.0
"""Opt-in Gemma prefix RMSNorm + BF16 cast, one program per residual row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_cast(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / WIDTH
    normalized = x * tl.rsqrt(variance + EPS)
    # Gemma uses (1 + weight), with both FP32 multiplications before BF16 cast.
    result = normalized * (1.0 + weight)
    tl.store(Y + row * WIDTH + col, result, col < WIDTH)


@triton.jit
def _gemma_rmsnorm_after_mean(X, V, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    variance = tl.load(V + row).to(tl.float32)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    normalized = x * tl.rsqrt(variance + EPS)
    result = normalized * (1.0 + weight)
    tl.store(Y + row * WIDTH + col, result, col < WIDTH)


def gemma_rmsnorm_bf16(x, weight, eps, reduction="triton", squared=None):
    # The installed wrapper checks semantics/dtypes/layout. No host tensor reads,
    # cache mutation, autotuning, or synchronization in the captured path.
    if reduction not in ("triton", "torch"):
        raise ValueError("Unsupported prefix RMSNorm reduction")
    # Retain the original mean reduction and its input layout. The optional
    # square comes from the same-request residual kernel, with FP32 rounding.
    if squared is not None and reduction != "torch":
        raise ValueError("Precomputed square requires the original torch mean")
    variance = (x.pow(2) if squared is None else squared).mean(-1, keepdim=True) if reduction == "torch" else None
    output = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        if variance is None:
            _gemma_rmsnorm_cast[(rows,)](x, weight, output, width, eps, triton.next_power_of_2(width),
                                       num_warps=4, enable_fp_fusion=False)
        else:
            _gemma_rmsnorm_after_mean[(rows,)](x, variance, weight, output, width, eps,
                                              triton.next_power_of_2(width), num_warps=4,
                                              enable_fp_fusion=False)
    return output
