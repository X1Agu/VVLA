"""FP32 gate * projection followed by FP32 residual add, with FMA disabled."""

import triton
import triton.language as tl


@triton.jit
def gated_residual_kernel(R, O, G, Y, N: tl.constexpr, WIDTH: tl.constexpr,
                         TOKENS: tl.constexpr, GB: tl.constexpr, GS: tl.constexpr,
                         BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = offset // WIDTH, offset % WIDTH
    batch, token = row // TOKENS, row % TOKENS
    residual = tl.load(R + offset, offset < N, 0).to(tl.float32)
    output = tl.load(O + offset, offset < N, 0).to(tl.float32)
    gate = tl.load(G + batch * GB + token * GS + column, offset < N, 0).to(tl.float32)
    product = gate * output
    result = residual + product
    tl.store(Y + offset, result, offset < N)
