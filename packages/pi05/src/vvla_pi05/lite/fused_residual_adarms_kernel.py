"""Attention-output residual plus post-attention AdaRMS; FP32, no FMA."""
import triton
import triton.language as tl


@triton.jit
def _mul_rn(left, right):
    # Inductor may recompile user Triton kernels without the launch's
    # enable_fp_fusion=False. An explicit PTX instruction preserves this
    # rounding boundary through both direct launch and torch.compile.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[left, right], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[left, right], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def residual_adarms_kernel(
    R, O, G, M, UPDATED, NORMALIZED,
    WIDTH: tl.constexpr, TOKENS: tl.constexpr, EPS: tl.constexpr,
    GB: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    batch, token = row // TOKENS, row % TOKENS
    residual = tl.load(R + row * WIDTH + col, valid, 0).to(tl.float32)
    output = tl.load(O + row * WIDTH + col, valid, 0).to(tl.float32)
    gate = tl.load(G + batch * GB + token * GS + col, valid, 0).to(tl.float32)
    # Same two FP32 operations as gated_residual_kernel. enable_fp_fusion=False
    # preserves multiply rounding before add; no FMA or BF16 residual storage.
    product = _mul_rn(gate, output)
    updated = _add_rn(residual, product)
    tl.store(UPDATED + row * WIDTH + col, updated, valid)
    # Match adarms_with_reduction's row/block/warp geometry and exact operation
    # order. The original residual is still returned for the next MLP residual.
    squared = _mul_rn(updated, updated)
    variance = tl.sum(squared, axis=0) / WIDTH
    normalized = _mul_rn(updated, tl.rsqrt(_add_rn(variance, EPS)))
    scale = tl.load(M + batch * 3 * WIDTH + col, valid, 0).to(tl.float32)
    shift = tl.load(M + batch * 3 * WIDTH + WIDTH + col, valid, 0).to(tl.float32)
    result = _add_rn(_mul_rn(normalized, _add_rn(1.0, scale)), shift)
    tl.store(NORMALIZED + row * WIDTH + col, result, valid)
