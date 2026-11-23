# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit vLLM FA2 integration for pi05's noncausal separable padding masks.

Uses the shared Omni Attention dispatch with a model-owned adapter. The kernel
is vLLM's bundled FlashAttention-2, never the platform's automatic FA4 choice.
Metadata belongs to a forward invocation, not mutable shared request state.
"""

import torch
from torch import nn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention


def prepare_pi05_attention_metadata(mask: torch.Tensor, key_length: int) -> AttentionMetadata:
    from vvla_pi05.models.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("pi05 FA2 expects additive [B,1,Q,K] attention mask")
    if not bool(((mask == 0) | (mask == OPENPI_ATTENTION_MASK_VALUE)).all()):
        raise ValueError("pi05 FA2 expects the exact OpenPI padding-mask values")
    visible = mask[:, 0, :, :key_length] == 0
    q_keep = visible.any(dim=-1)
    k_keep = visible.any(dim=-2)
    # Pi05 prefix and suffix are noncausal blocks with separable padding.
    # Reject arbitrary pair masks rather than silently changing visibility.
    if not torch.equal(visible, q_keep[:, :, None] & k_keep[:, None, :]):
        raise ValueError("pi05 FA2 only supports separable noncausal visibility")
    if not bool(q_keep.any(dim=1).all() & k_keep.any(dim=1).all()):
        raise ValueError("Each pi05 request needs at least one visible query and key")
    b, sq = q_keep.shape
    q_counts = q_keep.sum(dim=1, dtype=torch.int32)
    k_counts = k_keep.sum(dim=1, dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=mask.device)
    return AttentionMetadata(
        extra={
            "q_indices": q_keep.flatten().nonzero().flatten(),
            "k_indices": k_keep.flatten().nonzero().flatten(),
            "cu_q": torch.cat((zero, q_counts.cumsum(0, dtype=torch.int32))),
            "cu_k": torch.cat((zero, k_counts.cumsum(0, dtype=torch.int32))),
            "batch": b,
            "q_length": sq,
            "k_length": key_length,
        }
    )


def prepare_pi05_static_metadata(q_keep, k_keep, *, all_queries_valid=False):
    """Fixed-capacity varlen buffers; cu_seqlens delimit only the live prefix.

    Stable sorting packs live rows in request order, followed by unused storage.
    FA2 never reads keys or computes queries beyond the final cu_seqlens value.
    Uncomputed query outputs are discarded before returning to the model.
    """
    if not bool(q_keep.any(dim=1).all() & k_keep.any(dim=1).all()):
        raise ValueError("Each pi05 request needs at least one visible query and key")
    if all_queries_valid and not bool(q_keep.all()):
        raise ValueError("all_queries_valid contradicts query padding")
    zero = torch.zeros(1, dtype=torch.int32, device=q_keep.device)
    return AttentionMetadata(
        extra={
            "q_indices": torch.argsort((~q_keep).flatten().to(torch.int32), stable=True),
            "k_indices": torch.argsort((~k_keep).flatten().to(torch.int32), stable=True),
            "cu_q": torch.cat((zero, q_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
            "cu_k": torch.cat((zero, k_keep.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32))),
            "q_keep": q_keep,
            "batch": q_keep.shape[0],
            "q_length": q_keep.shape[1],
            "k_length": k_keep.shape[1],
            "fixed_capacity": True,
            "all_queries_valid": all_queries_valid,
        }
    )


class Pi05VllmFA2Kernel(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, metadata: AttentionMetadata
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        if query.dtype not in (torch.float16, torch.bfloat16) or not query.is_cuda:
            raise ValueError("pi05 vLLM FA2 requires CUDA FP16/BF16 QKV; no silent fallback")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("QKV dtypes must match")
        m = metadata.extra
        if "paged_cache" in m:
            return m["paged_cache"].attend(m["layer_index"], query, key, value, metadata, prefix=m["prefix"])
        b, sq, h, d = query.shape
        sk = key.shape[1]
        if (b, sq, sk) != (m["batch"], m["q_length"], m["k_length"]):
            raise ValueError("Attention metadata does not match QKV shape")
        packed = flash_attn_varlen_func(
            query.flatten(0, 1).index_select(0, m["q_indices"]),
            key.flatten(0, 1).index_select(0, m["k_indices"]),
            value.flatten(0, 1).index_select(0, m["k_indices"]),
            max_seqlen_q=sq,
            cu_seqlens_q=m["cu_q"],
            max_seqlen_k=sk,
            cu_seqlens_k=m["cu_k"],
            causal=False,
            softmax_scale=self.scale,
            fa_version=2,
        )
        fixed_capacity = m.get("fixed_capacity", False)
        if (fixed_capacity and m["all_queries_valid"]) or (not fixed_capacity and m["q_indices"].numel() == b * sq):
            return packed.reshape(b, sq, h, d)
        # Eager uses a finite large negative additive mask. Completely masked
        # query rows therefore have uniform softmax over ALL keys. Preserve
        # that result, including BF16 rounding of the softmax probabilities.
        # These rows remain masked as keys at every later prefix layer.
        probability = torch.full((), 1.0 / sk, device=query.device, dtype=query.dtype)
        uniform = (value.float().sum(dim=1) * probability.float()).to(query.dtype)
        uniform = uniform.repeat_interleave(h // key.shape[2], dim=1)
        out = uniform[:, None].expand(b, sq, h, d).contiguous()
        out.view(b * sq, h, d).index_copy_(0, m["q_indices"], packed)
        if fixed_capacity:
            # The unused packed-query tail is uninitialized. Select it away;
            # never multiply it by zero (NaN * 0 would still be NaN).
            out = torch.where(m["q_keep"][:, :, None, None], out, uniform[:, None])
        return out


def make_pi05_flash_attention(num_heads: int, num_kv_heads: int, head_dim: int) -> Attention:
    scale = head_dim**-0.5
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        causal=False,
        softmax_scale=scale,
        qkv_layout="BSND",
        skip_sequence_parallel=True,
        disable_kv_quant=True,
        custom_attention=Pi05VllmFA2Kernel(scale),
    )
