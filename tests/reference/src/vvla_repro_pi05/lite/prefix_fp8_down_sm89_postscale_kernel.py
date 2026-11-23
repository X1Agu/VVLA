"""Two ordered FP32 RN multiplies, then the sole BF16 output cast."""
import triton
import triton.language as tl


@triton.jit
def _postscale(Z, SX, SW, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < M * N
    z = tl.load(Z + offset, valid, other=0).to(tl.float32)
    sx = tl.load(SX + offset // N, valid, other=1).to(tl.float32)
    sw = tl.load(SW + offset % N, valid, other=1).to(tl.float32)
    # Explicit RN instructions prohibit multiply reassociation/FTZ and preserve
    # the sequential FP32 boundary, including subnormals and signed zeros.
    first = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                    args=[z, sx], dtype=tl.float32, is_pure=True, pack=1)
    second = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                     args=[first, sw], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Y + offset, second.to(tl.bfloat16), valid)


def launch(z, sx, sw, output):
    m, n = z.shape
    return _postscale[(triton.cdiv(m * n, 256),)](
        z, sx, sw, output, M=m, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
