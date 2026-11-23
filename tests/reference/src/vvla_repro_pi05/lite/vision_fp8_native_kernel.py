"""Masked non-power-of-two vision row quantization; original K is unchanged."""
import triton
import triton.language as tl


@triton.jit
def _quantize_vision_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + column, column < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(value), axis=0)
    # Same explicit Torch eager scalar-divisor boundary, reciprocal bits3b124925.
    scale = tl.maximum(amax * 0.0022321429569274187, 1.0e-12)
    normalized = tl.div_rn(value, scale)
    quantized = tl.minimum(tl.maximum(normalized, -448.0), 448.0).to(tl.float8e4nv,
                                                                 fp_downcast_rounding='rtne')
    tl.store(Q + row * K + column, quantized, column < K)
    tl.store(S + row, scale)


def launch(value, quantized, scale):
    width = value.shape[1]
    if width not in (1152, 4304):
        raise ValueError('Only the two explicitly supported vision widths are permitted')
    _quantize_vision_rows[(value.shape[0],)](value, quantized, scale, K=width,
        BLOCK=triton.next_power_of_2(width), num_warps=4 if width == 1152 else 8,
        enable_fp_fusion=False)
