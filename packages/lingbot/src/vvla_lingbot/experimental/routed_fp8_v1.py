"""Unqualified routed-only FP8 grouped candidate; original sparse routes retained.

R0 router/top-k/setup, counts/pack, shared expert and final output cast/add are
unchanged. New math is explicitly FP8 gate/up and down with dynamic per-token
and per-route activation scales, per-expert/output weight scales, FP32 dot and
dequantization, then the original SiLU*up*route -> BF16 inter and FP32 atomic
merge. All quantization/scale kernels count toward the candidate's full cost.
No dense-all-expert replacement or silent BF16 fallback is allowed.
"""
from __future__ import annotations


import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
SCALE_MIN = 1e-12


def tensor_token(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor._version)


@torch.no_grad()
def pack_weight(weight):
    """Pure weight-derived E/output scales; no activation fit or calibration."""
    if weight.ndim != 3 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError('Expected original contiguous BF16 expert/output/input weight')
    if not torch.isfinite(weight).all().item():
        raise ValueError('Original weights must be finite')
    scale = (weight.detach().float().abs().amax(dim=-1) / FP8_MAX).clamp_min(SCALE_MIN)
    quantized = (weight.detach().float() / scale[..., None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


@triton.jit
def _quantize_rows(X, Q, Scale, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0.0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), 0), 448.0), 1.0e-12)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Q + row * WIDTH + col, q.to(Q.dtype.element_ty), col < WIDTH)
    tl.store(Scale + row, scale)


@triton.jit
def _gate_up(
    X, XScale, Gate, GateScale, Up, UpScale, Counts, Rows, Slots, Route, Inter,
    T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
    I: tl.constexpr, MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr,
):
    expert, bid_m, bid_i = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + expert).to(tl.int32)
    start_m = bid_m * BLOCK_M
    if start_m >= count:
        return
    route_idx = start_m + tl.arange(0, BLOCK_M)
    offs_i = bid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = tl.arange(0, BLOCK_D)
    valid_m = route_idx < count
    rows = tl.load(Rows + expert * MAX_ROUTES + route_idx, valid_m, 0).to(tl.int32)
    slots = tl.load(Slots + expert * MAX_ROUTES + route_idx, valid_m, 0).to(tl.int32)
    acc_g = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        ds = d0 + offs_d
        x = tl.load(X + rows[:, None] * D + ds[None, :],
                    valid_m[:, None] & (ds[None, :] < D), 0.0)
        gate = tl.load(Gate + (expert * I + offs_i[None, :]) * D + ds[:, None],
                       (offs_i[None, :] < I) & (ds[:, None] < D), 0.0)
        up = tl.load(Up + (expert * I + offs_i[None, :]) * D + ds[:, None],
                     (offs_i[None, :] < I) & (ds[:, None] < D), 0.0)
        acc_g += tl.dot(x, gate, max_num_imprecise_acc=0)
        acc_u += tl.dot(x, up, max_num_imprecise_acc=0)
    xs = tl.load(XScale + rows, valid_m, 1.0)
    gs = tl.load(GateScale + expert * I + offs_i, offs_i < I, 1.0)
    us = tl.load(UpScale + expert * I + offs_i, offs_i < I, 1.0)
    # Explicit FP32 dequantization order belongs to this new FP8 recipe.
    acc_g = (acc_g * xs[:, None]) * gs[None, :]
    acc_u = (acc_u * xs[:, None]) * us[None, :]
    route = tl.load(Route + rows * TOPK + slots, valid_m, 0.0).to(tl.float32)
    silu = acc_g * (1.0 / (1.0 + tl.exp(-acc_g)))
    val = silu * acc_u * route[:, None]
    tl.store(Inter + ((rows[:, None] * TOPK + slots[:, None]) * I + offs_i[None, :]),
             val.to(Inter.dtype.element_ty), valid_m[:, None] & (offs_i[None, :] < I))


@triton.jit
def _down(
    Inter, InterScale, Down, DownScale, Counts, Rows, Slots, Out,
    T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
    I: tl.constexpr, MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_I: tl.constexpr,
):
    expert, bid_m, bid_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + expert).to(tl.int32)
    start_m = bid_m * BLOCK_M
    if start_m >= count:
        return
    route_idx = start_m + tl.arange(0, BLOCK_M)
    offs_d = bid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_i = tl.arange(0, BLOCK_I)
    valid_m = route_idx < count
    rows = tl.load(Rows + expert * MAX_ROUTES + route_idx, valid_m, 0).to(tl.int32)
    slots = tl.load(Slots + expert * MAX_ROUTES + route_idx, valid_m, 0).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        indices = i0 + offs_i
        x = tl.load(Inter + ((rows[:, None] * TOPK + slots[:, None]) * I + indices[None, :]),
                    valid_m[:, None] & (indices[None, :] < I), 0.0)
        weight = tl.load(Down + (expert * D + offs_d[None, :]) * I + indices[:, None],
                         (offs_d[None, :] < D) & (indices[:, None] < I), 0.0)
        acc += tl.dot(x, weight, max_num_imprecise_acc=0)
    xs = tl.load(InterScale + rows * TOPK + slots, valid_m, 1.0)
    ws = tl.load(DownScale + expert * D + offs_d, offs_d < D, 1.0)
    acc = (acc * xs[:, None]) * ws[None, :]
    tl.atomic_add(Out + rows[:, None] * D + offs_d[None, :], acc,
                  sem='relaxed', mask=valid_m[:, None] & (offs_d[None, :] < D))


