"""Strict arithmetic for the independent prefix FP8 diagnostic; CUDA only."""
import triton
import triton.language as tl


@triton.jit
def _quantize_rows_eager_rn(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + column, column < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # Torch2.13 BinaryDivTrueKernel's CPU-scalar denominator branch computes
    # FP32(amax * FP32(double(1) / 448)); it is NOT RN division by448.
    # This literal is exact FP32 bits0x3b124925. The later tensor/tensor divide
    # requires div.rn, not an approximate reciprocal multiplication.
    scale = tl.maximum(amax * 0.0022321429569274187, 1.0e-12)
    normalized = tl.div_rn(x, scale)
    clamped = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    quantized = clamped.to(tl.float8e4nv, fp_downcast_rounding='rtne')
    tl.store(Q + row * K + column, quantized, column < K)
    tl.store(S + row, scale)


def launch(value, quantized, scale):
    width = value.shape[1]
    # A single CTA owns one complete row, so no intermediate scale buffer or
    # inter-kernel synchronization is needed. Both widths are exact powers2.
    _quantize_rows_eager_rn[(value.shape[0],)](
        value, quantized, scale, K=width, BLOCK=triton.next_power_of_2(width),
        num_warps=4 if width == 2048 else 16, enable_fp_fusion=False,
    )
