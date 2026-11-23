"""Candidate-only fusion of the selected BF16 prefix GELU and FP8 converter.

One CTA owns one row. Both BF16 rounding boundaries precede amax. This file
does not replace the selected converter or authorize a production installation.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu_mul_quantize_rows(G, U, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    offsets = row * K + column
    g = tl.load(G + offsets, column < K, other=0).to(tl.float32)
    u = tl.load(U + offsets, column < K, other=0).to(tl.float32)
    # Match models/pi05/fused_mlp_pi05.py, including its intermediate BF16
    # GELU rounding. No FMA or approximate-tanh substitution is authorized.
    arg = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    activated = (0.5 * g * (1.0 + libdevice.tanh(arg))).to(
        tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
    # The original gelu_tanh_mul stores this product into a BF16 hidden tensor.
    # Preserve that second rounding before the selected converter sees FP32.
    hidden = (activated * u).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
    amax = tl.max(tl.abs(hidden), axis=0)
    # Exact existing converter arithmetic: FP32 reciprocal448 is bits3b124925.
    scale = tl.maximum(amax * 0.0022321429569274187, 1.0e-12)
    normalized = tl.div_rn(hidden, scale)
    clamped = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    quantized = clamped.to(tl.float8e4nv, fp_downcast_rounding='rtne')
    tl.store(Q + offsets, quantized, column < K)
    tl.store(S + row, scale)


def launch(gate, up, quantized, scale):
    width = gate.shape[1]
    _gelu_mul_quantize_rows[(gate.shape[0],)](
        gate, up, quantized, scale, K=width, BLOCK=triton.next_power_of_2(width),
        num_warps=16, enable_fp_fusion=False,
    )
