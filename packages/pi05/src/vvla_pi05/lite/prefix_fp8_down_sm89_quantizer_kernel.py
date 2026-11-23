"""SM89 direct-FP32 conversion experiment; no changes to the v2 kernel.

Triton v3.7.1 ElementwiseOpToLLVM.cpp:486-502 uses FP16 RZ then FP8 RN
for SM89, losing residual bits near FP8 midpoints. PTX 8.1 supports direct
cvt.rn.satfinite.e4m3x2.f32 on SM89. The explicit packed instruction below
duplicates one input into both lanes and retains the low byte. This simple
diagnostic intentionally makes no conversion-throughput claim.
"""
import triton
import triton.language as tl


@triton.jit
def _float32_to_e4m3fn_direct_rn(value):
    # Both packed lanes contain the SAME FP32 operand, avoiding lane-order or
    # implicit vector-packing assumptions. Keep byte extraction integer-only;
    # the final bitcast must not re-enter Triton's numerical FP8 conversion.
    bits = tl.inline_asm_elementwise(
        asm="""{
            .reg .b16 packed;
            .reg .b32 extended;
            cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1;
            cvt.u32.u16 extended, packed;
            and.b32 $0, extended, 255;
        }""",
        constraints='=r,f', args=[value], dtype=tl.uint32,
        is_pure=True, pack=1,
    )
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _quantize_rows_sm89_direct_f32_rn(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + column, column < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # Exactly the v2 arithmetic: FP32 scalar-reciprocal multiply (0x3b124925),
    # scale floor, precise FP32 tensor division, then finite clamp.
    scale = tl.maximum(amax * 0.0022321429569274187, 1.0e-12)
    normalized = tl.div_rn(x, scale)
    clamped = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    quantized = _float32_to_e4m3fn_direct_rn(clamped)
    tl.store(Q + row * K + column, quantized, column < K)
    tl.store(S + row, scale)


@triton.jit
def _cast_float32_for_contract(X, Q, N: tl.constexpr, BLOCK: tl.constexpr):
    """Isolate the same conversion for exact FP32 midpoint/zero tests only."""
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + column, column < N, other=0)
    result = _float32_to_e4m3fn_direct_rn(value)
    tl.store(Q + column, result, column < N)


def launch(value, quantized, scale):
    width = value.shape[1]
    # Launch on PyTorch/Triton's current stream, using caller-owned fresh Q/S.
    # Returning the compiled object lets the diagnostic inspect actual PTX.
    return _quantize_rows_sm89_direct_f32_rn[(value.shape[0],)](
        value, quantized, scale, K=width, BLOCK=triton.next_power_of_2(width),
        num_warps=4 if width == 2048 else 16, enable_fp_fusion=False,
    )
