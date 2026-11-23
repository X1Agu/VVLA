# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-wave paged KV using vLLM BlockPool, cache writes and paged FA2.

Each row owns disjoint pages across all layers. Prefix entries are written once
per observation; suffix entries are overwritten at every denoising step. Graphs
retain storage, never semantic KV across observations. The pool has no APC:
bidirectional vision-language prefixes cannot use causal token-prefix hashes.
"""

import math

import torch
from vllm import _custom_ops as ops
from vllm.v1.core.block_pool import BlockPool
from vllm.vllm_flash_attn import flash_attn_varlen_func


class Pi05PagedKVCache:
    def __init__(self, pads, suffix_length, layers, kv_heads, head_dim, dtype, block_size=16):
        if block_size not in (16, 32, 64):
            raise ValueError("pi05 paged FA2 block size must be 16, 32 or 64")
        if dtype not in (torch.float16, torch.bfloat16) or not pads.is_cuda:
            raise ValueError("pi05 paged FA2 requires CUDA FP16/BF16")
        self.batch, self.prefix_length = pads.shape
        self.suffix_length = suffix_length
        self.block_size = block_size
        self.max_blocks = math.ceil((self.prefix_length + suffix_length) / block_size)
        # Main-vLLM BlockPool reserves block zero as its null page.
        self.num_blocks = 1 + self.batch * self.max_blocks
        self.pool = BlockPool(self.num_blocks, enable_caching=False, hash_block_size=block_size)
        self.leases = []
        shape = (layers, self.num_blocks, block_size, kv_heads, head_dim)
        self.keys = torch.empty(shape, device=pads.device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.scale = torch.ones((), device=pads.device, dtype=torch.float32)
        self.block_table = torch.zeros(self.batch, self.max_blocks, device=pads.device, dtype=torch.int32)
        self.prefix_lengths = torch.empty(self.batch, device=pads.device, dtype=torch.int32)
        self.full_lengths = torch.empty_like(self.prefix_lengths)
        self.prefix_slots = torch.empty(self.batch * self.prefix_length, device=pads.device, dtype=torch.int64)
        self.suffix_slots = torch.empty(self.batch * suffix_length, device=pads.device, dtype=torch.int64)
        self.update(pads)

    def release(self):
        """Release ownership; captured storage remains allocated with its graph."""
        if self.leases:
            self.pool.free_blocks([block for row in self.leases for block in row])
            self.leases = []

    def update(self, pads):
        """Reassign pages outside capture; copy metadata into stable addresses."""
        if tuple(pads.shape) != (self.batch, self.prefix_length):
            raise ValueError("Paged cache geometry cannot change in place")
        counts = pads.sum(1, dtype=torch.int32)
        counts_cpu = counts.cpu().tolist()
        if min(counts_cpu) < 1:
            raise ValueError("Each request needs visible prefix tokens")
        needed = [math.ceil((n + self.suffix_length) / self.block_size) for n in counts_cpu]
        self.release()
        blocks = iter(self.pool.get_new_blocks(sum(needed)))
        self.leases = [[] for _ in needed]
        # Interleave physical pages to exercise block-table indirection even
        # for a steady batch. Logical sequences need not be contiguous in VRAM.
        for column in range(max(needed)):
            for row, n in enumerate(needed):
                if column < n:
                    self.leases[row].append(next(blocks))
        table = torch.zeros((self.batch, self.max_blocks), dtype=torch.int32)
        for row, lease in enumerate(self.leases):
            table[row, : len(lease)] = torch.tensor([block.block_id for block in lease], dtype=torch.int32)
        self.block_table.copy_(table)
        self.prefix_lengths.copy_(counts)
        self.full_lengths.copy_(counts + self.suffix_length)
        positions = pads.cumsum(1, dtype=torch.int64) - 1
        positions.clamp_(min=0)
        page = self.block_table.gather(1, positions // self.block_size).to(torch.int64)
        slots = page * self.block_size + positions % self.block_size
        self.prefix_slots.copy_(torch.where(pads, slots, -1).flatten())
        positions = counts[:, None].to(torch.int64) + torch.arange(self.suffix_length, device=pads.device)
        page = self.block_table.gather(1, positions // self.block_size).to(torch.int64)
        self.suffix_slots.copy_((page * self.block_size + positions % self.block_size).flatten())

    def write(self, layer, key, value, *, prefix):
        # Negative slots skip masked tokens. K/V layout is [B,S,Hkv,D].
        ops.reshape_and_cache_flash(
            key.flatten(0, 1).contiguous(),
            value.flatten(0, 1).contiguous(),
            self.keys[layer],
            self.values[layer],
            self.prefix_slots if prefix else self.suffix_slots,
            "auto",
            self.scale,
            self.scale,
        )

    def attend(self, layer, query, key, value, metadata, *, prefix):
        if query.dtype != self.keys.dtype or key.dtype != self.keys.dtype or value.dtype != self.values.dtype:
            raise ValueError("Paged QKV and cache dtypes must match")
        self.write(layer, key, value, prefix=prefix)
        m = metadata.extra
        b, sq, h, d = query.shape
        q = query.flatten(0, 1).index_select(0, m["q_indices"]) if prefix else query.flatten(0, 1)
        packed = flash_attn_varlen_func(
            q,
            self.keys[layer],
            self.values[layer],
            max_seqlen_q=sq,
            cu_seqlens_q=m["cu_q"],
            max_seqlen_k=self.prefix_length + (0 if prefix else self.suffix_length),
            seqused_k=self.prefix_lengths if prefix else self.full_lengths,
            block_table=self.block_table,
            causal=False,
            softmax_scale=d**-0.5,
            fa_version=2,
        )
        if not prefix:
            return packed.reshape(b, sq, h, d)
        # Preserve eager's finite-mask semantics for fully masked prefix Qs.
        probability = torch.full((), 1.0 / self.prefix_length, device=query.device, dtype=query.dtype)
        uniform = (value.float().sum(1) * probability.float()).to(query.dtype)
        uniform = uniform.repeat_interleave(h // key.shape[2], dim=1)
        out = torch.empty_like(query)
        out.view(b * sq, h, d).index_copy_(0, m["q_indices"], packed)
        return torch.where(m["q_keep"][:, :, None, None], out, uniform[:, None])

    def metrics(self):
        allocated = sum(map(len, self.leases))
        valid = sum(len(row) for row in self.leases)
        return {
            "block_size": self.block_size,
            "capacity_blocks": self.num_blocks - 1,
            "allocated_blocks": allocated,
            "free_blocks": self.pool.get_num_free_blocks(),
            "reserved_bytes": self.keys.numel() * self.keys.element_size() * 2,
            "leased_bytes": valid
            * self.block_size
            * self.keys.shape[0]
            * self.keys.shape[-2]
            * self.keys.shape[-1]
            * self.keys.element_size()
            * 2,
            "cross_observation_prefix_caching": False,
        }
