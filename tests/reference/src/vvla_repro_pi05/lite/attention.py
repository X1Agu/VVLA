# SPDX-License-Identifier: Apache-2.0
"""Standalone FA2 with the preserved pi05 mask and padded-query semantics.

The metadata contract matches models/pi05/attention_pi05.py. Execution uses
PyTorch 2.13's public varlen_attn (bundled FA2), or explicitly selected upstream
flash-attn, without Omni dispatch or vLLM.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AttentionMetadata:
    extra: dict


def annotate_identity_gathers(metadata):
    """Prove identity permutations once, before capture; never infer from length.

    The booleans are capture-time control flow. A graph owner must recapture if
    either changes when refreshing masks. Tensor contents/valid lengths still
    refresh normally when the proof stays unchanged.
    """
    if metadata is None:
        return
    m = metadata.extra
    for axis in ("q", "k"):
        indices = m[f"{axis}_indices"]
        size = m["batch"] * m[f"{axis}_length"]
        m[f"{axis}_identity"] = indices.numel() == size and torch.equal(
            indices, torch.arange(size, device=indices.device, dtype=indices.dtype)
        )


def identity_gather_signature(metadata):
    return tuple(None if item is None else (
        item.extra.get("q_identity", False), item.extra.get("k_identity", False)
    ) for item in metadata)


def select_packed_qkv(query, key, value, metadata, *, elide_identity_gathers=False):
    """Select the same rows, permitting views only for explicitly proven identity."""
    m = metadata.extra

    def select(tensor, axis):
        flat = tensor.flatten(0, 1)
        # Both FA2 backends require contiguous head columns. A gather supplies
        # that layout if the caller provided a strided last dimension.
        if elide_identity_gathers and m.get(f"{axis}_identity", False) and flat.stride(-1) == 1:
            return flat
        return flat.index_select(0, m[f"{axis}_indices"])

    return select(query, "q"), select(key, "k"), select(value, "k")


def prepare_pi05_attention_metadata(mask, key_length):
    from vvla_repro_pi05.models.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("pi05 FA2 expects additive [B,1,Q,K] attention mask")
    if not bool(((mask == 0) | (mask == OPENPI_ATTENTION_MASK_VALUE)).all()):
        raise ValueError("pi05 FA2 expects the exact OpenPI padding-mask values")
    visible = mask[:, 0, :, :key_length] == 0
    q_keep, k_keep = visible.any(-1), visible.any(-2)
    if not torch.equal(visible, q_keep[:, :, None] & k_keep[:, None, :]):
        raise ValueError("pi05 FA2 only supports separable noncausal visibility")
    if not bool(q_keep.any(1).all() & k_keep.any(1).all()):
        raise ValueError("Each pi05 request needs at least one visible query and key")
    zero = torch.zeros(1, dtype=torch.int32, device=mask.device)
    return AttentionMetadata(extra={
        "q_indices": q_keep.flatten().nonzero().flatten(),
        "k_indices": k_keep.flatten().nonzero().flatten(),
        "cu_q": torch.cat((zero, q_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
        "cu_k": torch.cat((zero, k_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
        "batch": q_keep.shape[0], "q_length": q_keep.shape[1], "k_length": key_length,
    })


def prepare_pi05_static_metadata(q_keep, k_keep, *, all_queries_valid=False):
    if q_keep.ndim != 2 or k_keep.ndim != 2 or q_keep.shape[0] != k_keep.shape[0]:
        raise ValueError("Attention masks must be [B,Q] and [B,K]")
    if q_keep.dtype != torch.bool or k_keep.dtype != torch.bool:
        raise ValueError("Attention masks must be boolean")
    if not bool(q_keep.any(1).all() & k_keep.any(1).all()):
        raise ValueError("Each pi05 request needs at least one visible query and key")
    if all_queries_valid and not bool(q_keep.all()):
        raise ValueError("all_queries_valid contradicts query padding")
    zero = torch.zeros(1, dtype=torch.int32, device=q_keep.device)
    return AttentionMetadata(extra={
        "q_indices": torch.argsort((~q_keep).flatten().to(torch.int32), stable=True),
        "k_indices": torch.argsort((~k_keep).flatten().to(torch.int32), stable=True),
        "cu_q": torch.cat((zero, q_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
        "cu_k": torch.cat((zero, k_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
        "q_keep": q_keep, "batch": q_keep.shape[0], "q_length": q_keep.shape[1],
        "k_length": k_keep.shape[1], "fixed_capacity": True, "all_queries_valid": all_queries_valid,
    })


class Pi05FlashAttention(nn.Module):
    def __init__(self, scale, *, backend="torch_fa2", elide_identity_gathers=False):
        super().__init__()
        self.scale = scale
        self.backend = backend
        self.elide_identity_gathers = elide_identity_gathers
        # Fail at construction rather than silently timing a fallback backend.
        if backend == "torch_fa2":
            from torch.nn.attention.varlen import varlen_attn
            self._kernel = varlen_attn
        elif backend == "flash_attn":
            from flash_attn import flash_attn_varlen_func
            self._kernel = flash_attn_varlen_func
        else:
            raise ValueError(f"Unsupported standalone FA2 backend: {backend}")

    def forward(self, query, key, value, metadata):
        if query.dtype not in (torch.float16, torch.bfloat16) or not query.is_cuda:
            raise ValueError("pi05 FA2 requires CUDA FP16/BF16 QKV")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("QKV dtypes must match")
        m = metadata.extra
        b, sq, h, d = query.shape
        sk = key.shape[1]
        if (b, sq, sk) != (m["batch"], m["q_length"], m["k_length"]):
            raise ValueError("Attention metadata does not match QKV shape")
        if self.elide_identity_gathers and "q_identity" not in m:
            # Eager callers prepare fresh metadata per request. Graph owners
            # explicitly annotate before warmup/capture and on mask refresh.
            annotate_identity_gathers(metadata)
        packed_q, packed_k, packed_v = select_packed_qkv(
            query, key, value, metadata, elide_identity_gathers=self.elide_identity_gathers,
        )
        if self.backend == "torch_fa2":
            # Public 2.13 API dispatches to aten._flash_attention_forward.
            # cu_seq values delimit the live prefix of fixed-capacity storage.
            packed = self._kernel(
                packed_q, packed_k, packed_v, m["cu_q"], m["cu_k"], sq, sk,
                scale=self.scale, window_size=(-1, -1), enable_gqa=h != key.shape[2],
            )
        else:
            packed = self._kernel(
                packed_q, packed_k, packed_v,
                max_seqlen_q=sq, cu_seqlens_q=m["cu_q"],
                max_seqlen_k=sk, cu_seqlens_k=m["cu_k"],
                causal=False, softmax_scale=self.scale, dropout_p=0.0,
            )
        fixed = m.get("fixed_capacity", False)
        if (fixed and m["all_queries_valid"]) or (not fixed and m["q_indices"].numel() == b * sq):
            return packed.reshape(b, sq, h, d)
        # Preserve the finite-mask eager behavior on completely masked query
        # rows. Do not multiply uninitialized packed tail values by zero.
        probability = torch.full((), 1.0 / sk, device=query.device, dtype=query.dtype)
        uniform = (value.float().sum(1) * probability.float()).to(query.dtype)
        uniform = uniform.repeat_interleave(h // key.shape[2], dim=1)
        out = uniform[:, None].expand(b, sq, h, d).contiguous()
        out.view(b * sq, h, d).index_copy_(0, m["q_indices"], packed)
        if fixed:
            out = torch.where(m["q_keep"][:, :, None, None], out, uniform[:, None])
        return out


def make_pi05_flash_attention(num_heads, num_kv_heads, head_dim, *, backend="torch_fa2"):
    if num_heads % num_kv_heads:
        raise ValueError("Q head count must be divisible by KV head count")
    return Pi05FlashAttention(head_dim**-0.5, backend=backend)
