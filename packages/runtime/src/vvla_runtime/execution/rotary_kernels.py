"""CUDA-only kernels. No reduction, in-place mutation, or semantic cache."""
import triton
import triton.language as tl


@triton.jit
def rope_kernel(Q, K, C, S, QO, KO, NQ: tl.constexpr, NK: tl.constexpr,
                SEQ: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                qb: tl.constexpr, qh: tl.constexpr, qs: tl.constexpr, qd: tl.constexpr,
                kb: tl.constexpr, kh: tl.constexpr, ks: tl.constexpr, kd: tl.constexpr,
                cb: tl.constexpr, cs: tl.constexpr, cd: tl.constexpr,
                sb: tl.constexpr, ss: tl.constexpr, sd: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_q = i < NQ
    local = tl.where(is_q, i, i - NQ)
    heads = tl.where(is_q, max(HQ, 1), HK)
    col = local % D
    head = local // D % heads
    seq = local // (D * heads) % SEQ
    batch = local // (D * heads * SEQ)
    rotated_col = (col + D // 2) % D
    qbase = batch * qb + head * qh + seq * qs
    kbase = batch * kb + head * kh + seq * ks
    q0 = tl.load(Q + qbase + col * qd, is_q & (i < NQ + NK), 0).to(tl.float32)
    q1 = tl.load(Q + qbase + rotated_col * qd, is_q & (i < NQ + NK), 0).to(tl.float32)
    k0 = tl.load(K + kbase + col * kd, (~is_q) & (i < NQ + NK), 0).to(tl.float32)
    k1 = tl.load(K + kbase + rotated_col * kd, (~is_q) & (i < NQ + NK), 0).to(tl.float32)
    x = tl.where(is_q, q0, k0)
    rotated = tl.where(col < D // 2, -tl.where(is_q, q1, k1), tl.where(is_q, q1, k1))
    c = tl.load(C + batch * cb + seq * cs + col * cd, i < NQ + NK, 0).to(tl.float32)
    s = tl.load(S + batch * sb + seq * ss + col * sd, i < NQ + NK, 0).to(tl.float32)
    # PyTorch materializes each product at the operand dtype before the add.
    left = (x * c).to(K.dtype.element_ty).to(tl.float32)
    right = (rotated * s).to(K.dtype.element_ty).to(tl.float32)
    y = left + right
    tl.store(QO + i, y, is_q & (i < NQ + NK))
    tl.store(KO + i - NQ, y, (~is_q) & (i < NQ + NK))


