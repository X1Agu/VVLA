"""Triton pointwise AdaRMS tail; the original Torch variance is supplied."""
import triton
import triton.language as tl


@triton.jit
def adarms_after_mean(
    X, V, M, W, Y,
    WIDTH: tl.constexpr, TOKENS: tl.constexpr, EPS: tl.constexpr,
    CONDITIONED: tl.constexpr, INPUT_BF16: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * WIDTH + col, valid, 0).to(tl.float32)
    variance = tl.load(V + row).to(tl.float32)
    normalized = x * tl.rsqrt(variance + EPS)
    if CONDITIONED:
        batch = row // TOKENS
        scale = tl.load(M + batch * 3 * WIDTH + col, valid, 0).to(tl.float32)
        shift = tl.load(M + batch * 3 * WIDTH + WIDTH + col, valid, 0).to(tl.float32)
        result = normalized * (1.0 + scale) + shift
    else:
        weight = tl.load(W + col, valid, 0).to(tl.float32)
        result = normalized * (1.0 + weight)
    # Core returns normed.to(x.dtype), then the consumer casts to its projection
    # dtype. Retain the intermediate BF16 rounding if final projection is FP32.
    if INPUT_BF16:
        result = result.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * WIDTH + col, result, valid)


@triton.jit
def adarms_with_reduction(
    X, M, W, Y,
    WIDTH: tl.constexpr, TOKENS: tl.constexpr, EPS: tl.constexpr,
    CONDITIONED: tl.constexpr, INPUT_BF16: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * WIDTH + col, valid, 0).to(tl.float32)
    # Square, sum and normalization remain FP32. Padded columns contribute zero
    # and the divisor is the actual width, not the padded block size. The tree
    # reduction differs from Torch's mean; callers must record this candidate.
    squared = x * x
    variance = tl.sum(squared, axis=0) / WIDTH
    normalized = x * tl.rsqrt(variance + EPS)
    if CONDITIONED:
        batch = row // TOKENS
        scale = tl.load(M + batch * 3 * WIDTH + col, valid, 0).to(tl.float32)
        shift = tl.load(M + batch * 3 * WIDTH + WIDTH + col, valid, 0).to(tl.float32)
        result = normalized * (1.0 + scale) + shift
    else:
        weight = tl.load(W + col, valid, 0).to(tl.float32)
        result = normalized * (1.0 + weight)
    # Preserve the original norm result's dtype before its consumer conversion,
    # including BF16->FP32 for an FP32 final action projection.
    if INPUT_BF16:
        result = result.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * WIDTH + col, result, valid)
