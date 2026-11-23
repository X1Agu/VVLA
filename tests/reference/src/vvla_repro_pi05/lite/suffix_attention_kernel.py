"""B1/H8/HKV1/Q50/D256 BF16 suffix attention; no concatenated or gathered KV.

Both kernels are deterministic for a fixed recipe. Partial softmax numerators,
maxima (base-2 logit units) and denominators remain FP32 between stages. The
exp2 probability tiles are rounded to BF16 only at the Tensor Core PV boundary,
matching FA2's precision boundary, but a different split tree is not bitwise FA2.
"""

import triton
import triton.language as tl


@triton.jit
def suffix_partial(
    Q, PREFIX_K, PREFIX_V, SUFFIX_K, SUFFIX_V, INDICES, CU_K,
    NUMERATOR, MAXIMUM, DENOMINATOR,
    Q_HEAD_STRIDE: tl.constexpr, Q_TOKEN_STRIDE: tl.constexpr,
    PK_TOKEN_STRIDE: tl.constexpr, PV_TOKEN_STRIDE: tl.constexpr,
    SK_TOKEN_STRIDE: tl.constexpr, SV_TOKEN_STRIDE: tl.constexpr,
    PREFIX_CAPACITY: tl.constexpr, INDEX_COUNT: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    query_block, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dims = tl.arange(0, 256)
    offsets = tl.arange(0, BLOCK_K)
    q = tl.load(Q + head * Q_HEAD_STRIDE + rows[:, None] * Q_TOKEN_STRIDE + dims[None, :],
                rows[:, None] < 50, 0)
    valid_total = tl.load(CU_K + 1)
    capacity: tl.constexpr = PREFIX_CAPACITY + 50
    blocks: tl.constexpr = triton.cdiv(capacity, BLOCK_K)
    # Distribute whole K tiles evenly. S8 over ten tiles must not create three
    # idle high splits as ceil(10/8)-tiles-per-split partitioning would.
    begin, end = split * blocks // SPLITS, (split + 1) * blocks // SPLITS
    maximum = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_Q,), tl.float32)
    numerator = tl.zeros((BLOCK_Q, 256), tl.float32)
    for tile in range(begin, end):
        logical = tile * BLOCK_K + offsets
        live = (logical < valid_total) & (logical < INDEX_COUNT) & (logical < capacity)
        physical = tl.load(INDICES + logical, live, -1).to(tl.int32)
        prefix = live & (physical >= 0) & (physical < PREFIX_CAPACITY)
        suffix = live & (physical >= PREFIX_CAPACITY) & (physical < capacity)
        # Distinct masked loads avoid ever dereferencing an out-of-segment
        # pointer. In particular masked prefix NaNs never enter the dot product.
        kp = tl.load(PREFIX_K + physical[None, :] * PK_TOKEN_STRIDE + dims[:, None], prefix[None, :], 0)
        ks = tl.load(SUFFIX_K + (physical[None, :] - PREFIX_CAPACITY) * SK_TOKEN_STRIDE + dims[:, None], suffix[None, :], 0)
        k = tl.where(prefix[None, :], kp, ks)
        logits = tl.dot(q, k).to(tl.float32) * (1.4426950408889634 / 16.0)
        logits = tl.where((rows[:, None] < 50) & (prefix | suffix)[None, :], logits, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        # An entirely empty tile/split has m=-inf. Use finite zero only in the
        # subtraction so exp2(-inf)=0, never exp2(-inf-(-inf))=NaN.
        safe_maximum = tl.where(next_maximum == float("-inf"), 0.0, next_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        numerator = numerator * alpha[:, None]
        vp = tl.load(PREFIX_V + physical[:, None] * PV_TOKEN_STRIDE + dims[None, :], prefix[:, None], 0)
        vs = tl.load(SUFFIX_V + (physical[:, None] - PREFIX_CAPACITY) * SV_TOKEN_STRIDE + dims[None, :], suffix[:, None], 0)
        value = tl.where(prefix[:, None], vp, vs)
        numerator = tl.dot(probability.to(tl.bfloat16), value, numerator)
        maximum = next_maximum
    scalar_offset = (split * 8 + head) * 50 + rows
    tl.store(MAXIMUM + scalar_offset, maximum, rows < 50)
    tl.store(DENOMINATOR + scalar_offset, denominator, rows < 50)
    tl.store(NUMERATOR + scalar_offset[:, None] * 256 + dims[None, :], numerator, rows[:, None] < 50)


@triton.jit
def suffix_merge(NUMERATOR, MAXIMUM, DENOMINATOR, OUTPUT,
                 SPLITS: tl.constexpr, BLOCK_Q: tl.constexpr):
    query_block, head = tl.program_id(0), tl.program_id(1)
    rows = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dims = tl.arange(0, 256)
    split_ids = tl.arange(0, SPLITS)
    all_maxima = tl.load(MAXIMUM + (split_ids[:, None] * 8 + head) * 50 + rows[None, :],
                         rows[None, :] < 50, float("-inf"))
    maximum = tl.max(all_maxima, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    numerator = tl.zeros((BLOCK_Q, 256), tl.float32)
    denominator = tl.zeros((BLOCK_Q,), tl.float32)
    # Fixed merge order; no floating atomics or inter-CTA synchronization.
    for split in tl.static_range(SPLITS):
        scalar_offset = (split * 8 + head) * 50 + rows
        part_maximum = tl.load(MAXIMUM + scalar_offset, rows < 50, float("-inf"))
        part_denominator = tl.load(DENOMINATOR + scalar_offset, rows < 50, 0)
        part_numerator = tl.load(NUMERATOR + scalar_offset[:, None] * 256 + dims[None, :], rows[:, None] < 50, 0)
        alpha = tl.exp2(part_maximum - safe_maximum)
        numerator = numerator + alpha[:, None] * part_numerator
        denominator = denominator + alpha * part_denominator
    # Helpers guarantee at least one live key. A forged zero-key metadata row
    # should surface nonfinite output rather than silently invent zero attention.
    result = tl.where(denominator[:, None] > 0, numerator / denominator[:, None], float("nan"))
    output_offset = (rows[:, None] * 8 + head) * 256 + dims[None, :]
    tl.store(OUTPUT + output_offset, result.to(tl.bfloat16), rows[:, None] < 50)
