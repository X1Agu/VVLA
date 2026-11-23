"""Experimental BF16 PostTopK kernel, installed explicitly before graph capture.

The router projection, activation, bias and top-k stay in routing.py. This
kernel fuses gather, optional normalization/scale and casts. CPU contracts and
SM89 primitive checks do not establish full-model or SM120 qualification.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

SUM_ORDERS = ('left', 'pairwise_01_23', 'pairwise_02_13', 'pairwise_03_12')
DEFAULT_SUM_ORDER = 'pairwise_02_13'


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        'add.rn.f32 $0, $1, $2;', constraints='=f,f,f',
        args=[a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        'mul.rn.f32 $0, $1, $2;', constraints='=f,f,f',
        args=[a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _post_topk_kernel(
    Scores, Selected, Weights, SelectedI32,
    Gathered, Sum, Denominator, Normalized, Scaled,
    T: tl.constexpr, SCORE_STRIDE: tl.constexpr, ID_STRIDE: tl.constexpr,
    NORMALIZE: tl.constexpr, SCALE: tl.constexpr, APPLY_SCALE: tl.constexpr,
    SUM_ORDER: tl.constexpr, DIAGNOSTICS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = token < T
    i0 = tl.load(Selected + token * ID_STRIDE + 0, valid, 0).to(tl.int32)
    i1 = tl.load(Selected + token * ID_STRIDE + 1, valid, 0).to(tl.int32)
    i2 = tl.load(Selected + token * ID_STRIDE + 2, valid, 0).to(tl.int32)
    i3 = tl.load(Selected + token * ID_STRIDE + 3, valid, 0).to(tl.int32)
    a = tl.load(Scores + token * SCORE_STRIDE + i0, valid, 0).to(tl.float32)
    b = tl.load(Scores + token * SCORE_STRIDE + i1, valid, 0).to(tl.float32)
    c = tl.load(Scores + token * SCORE_STRIDE + i2, valid, 0).to(tl.float32)
    d = tl.load(Scores + token * SCORE_STRIDE + i3, valid, 0).to(tl.float32)

    if NORMALIZE:
        # Each PTX add is a separate RN boundary; no tl.sum/reassociation claim.
        if SUM_ORDER == 0:
            total = _add_rn(_add_rn(_add_rn(a, b), c), d)
        elif SUM_ORDER == 1:
            total = _add_rn(_add_rn(a, b), _add_rn(c, d))
        elif SUM_ORDER == 2:
            total = _add_rn(_add_rn(a, c), _add_rn(b, d))
        else:
            total = _add_rn(_add_rn(a, d), _add_rn(b, c))
        denominator = _add_rn(total, tl.full((), 1.0e-20, tl.float32))
        na = tl.div_rn(a, denominator)
        nb = tl.div_rn(b, denominator)
        nc = tl.div_rn(c, denominator)
        nd = tl.div_rn(d, denominator)
    else:
        na, nb, nc, nd = a, b, c, d

    if APPLY_SCALE:
        # A Python scalar is converted to FP32 for the original FP32 tensor mul.
        factor = tl.full((), SCALE, tl.float32)
        sa = _mul_rn(na, factor)
        sb = _mul_rn(nb, factor)
        sc = _mul_rn(nc, factor)
        sd = _mul_rn(nd, factor)
    else:
        sa, sb, sc, sd = na, nb, nc, nd

    base = token * 4
    tl.store(Weights + base + 0, sa, valid)
    tl.store(Weights + base + 1, sb, valid)
    tl.store(Weights + base + 2, sc, valid)
    tl.store(Weights + base + 3, sd, valid)
    tl.store(SelectedI32 + base + 0, i0, valid)
    tl.store(SelectedI32 + base + 1, i1, valid)
    tl.store(SelectedI32 + base + 2, i2, valid)
    tl.store(SelectedI32 + base + 3, i3, valid)

    if DIAGNOSTICS:
        tl.store(Gathered + base + 0, a, valid)
        tl.store(Gathered + base + 1, b, valid)
        tl.store(Gathered + base + 2, c, valid)
        tl.store(Gathered + base + 3, d, valid)
        if NORMALIZE:
            tl.store(Sum + token, total, valid)
            tl.store(Denominator + token, denominator, valid)
        tl.store(Normalized + base + 0, na, valid)
        tl.store(Normalized + base + 1, nb, valid)
        tl.store(Normalized + base + 2, nc, valid)
        tl.store(Normalized + base + 3, nd, valid)
        tl.store(Scaled + base + 0, sa, valid)
        tl.store(Scaled + base + 1, sb, valid)
        tl.store(Scaled + base + 2, sc, valid)
        tl.store(Scaled + base + 3, sd, valid)


def post_topk_setup(
    routing_scores, selected_experts, *, norm_topk_prob,
    routed_scaling_factor, sum_order=DEFAULT_SUM_ORDER, diagnostics=False, output_dtype=torch.bfloat16,
):
    """Return new contiguous 16-bit weights and int32 IDs, without input mutation.

    Inputs are the *uncorrected* FP32 routing_scores and original Torch top-k
    int64 IDs. Correction bias must affect selection only, as in the official
    forward. Values/IDs must obey the torch.topk contract (finite scores, IDs in
    [0,32), distinct four IDs per row). No CUDA scalar read or synchronization is
    added here, so the owning primitive gate must check that value contract.

    With diagnostics=True a third dictionary contains FP32 gathered/normalized/
    scaled [T,4], plus sum/denominator [T,1] only when NORMALIZE executes. These
    extra stores are evidence work and must stay outside formal timing.
    """
    if (not routing_scores.is_cuda or routing_scores.dtype != torch.float32
            or routing_scores.ndim != 2 or routing_scores.shape[1] != 32
            or routing_scores.shape[0] < 1 or routing_scores.stride(1) != 1):
        raise ValueError('Expected CUDA FP32 routing_scores[T,32], contiguous columns, T>=1')
    if (not selected_experts.is_cuda or selected_experts.device != routing_scores.device
            or selected_experts.dtype != torch.int64
            or tuple(selected_experts.shape) != (routing_scores.shape[0], 4)
            or selected_experts.stride(1) != 1):
        raise ValueError('Expected same-device original torch.topk int64 IDs[T,4]')
    if type(norm_topk_prob) is not bool or type(diagnostics) is not bool:
        raise TypeError('Normalization and diagnostic flags must be explicit bools')
    if isinstance(routed_scaling_factor, bool) or not isinstance(routed_scaling_factor, (int, float)):
        raise TypeError('Scaling factor must be the original Python numeric scalar')
    factor = float(routed_scaling_factor)
    if not math.isfinite(factor):
        raise ValueError('Finite original routing scale is required')
    if sum_order not in SUM_ORDERS:
        raise ValueError('Unknown explicit sum association')
    t = routing_scores.shape[0]
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("PostTopK output must use the declared 16-bit format")
    weights = torch.empty((t, 4), device=routing_scores.device, dtype=output_dtype)
    ids_i32 = torch.empty((t, 4), device=routing_scores.device, dtype=torch.int32)
    if diagnostics:
        debug = {name: torch.empty((t, 4), device=routing_scores.device, dtype=torch.float32)
                 for name in ('gathered', 'normalized', 'scaled')}
        debug.update({name: (torch.empty((t, 1), device=routing_scores.device, dtype=torch.float32)
                             if norm_topk_prob else None)
                      for name in ('sum', 'denominator')})
        arguments = [debug['gathered'], debug['sum'] if norm_topk_prob else weights,
                     debug['denominator'] if norm_topk_prob else weights,
                     debug['normalized'], debug['scaled']]
    else:
        arguments = [weights] * 5  # Compile-time-disabled diagnostic pointers.
    _post_topk_kernel[(triton.cdiv(t, 128),)](
        routing_scores, selected_experts, weights, ids_i32, *arguments,
        T=t, SCORE_STRIDE=routing_scores.stride(0), ID_STRIDE=selected_experts.stride(0),
        NORMALIZE=norm_topk_prob, SCALE=factor, APPLY_SCALE=factor != 1.0,
        SUM_ORDER=SUM_ORDERS.index(sum_order), DIAGNOSTICS=diagnostics,
        BLOCK_T=128, num_warps=4, enable_fp_fusion=False,
    )
    return (weights, ids_i32, debug) if diagnostics else (weights, ids_i32)



def install_overlay(engine, *, sum_order=DEFAULT_SUM_ORDER):
    from ..routing import install_post_topk
    return install_post_topk(engine, sum_order=sum_order)
