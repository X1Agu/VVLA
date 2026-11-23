import triton
import triton.language as tl


@triton.jit
def residual_cast_kernel(A, B, Y, F, N: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                         A0: tl.constexpr,A1: tl.constexpr,A2: tl.constexpr,
                         B0: tl.constexpr,B1: tl.constexpr,B2: tl.constexpr,BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row=i//D;batch=row//S;token=row%S;col=i%D
    a = tl.load(A + batch*A0 + token*A1 + col*A2, i < N, other=0).to(tl.float32)
    b = tl.load(B + batch*B0 + token*B1 + col*B2, i < N, other=0).to(tl.float32)
    rounded = (a + b).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Y + i, rounded, i < N)
    tl.store(F + i, rounded.to(tl.float32), i < N)
