"""Opt-in B1 pi05 suffix GQA over separate prefix/suffix KV.

Only the explicit suffix dispatch uses this backend. Prefix attention and its
finite-mask padded-query behavior are untouched. The trusted metadata is the
unchanged standalone FA2 metadata: in B1, k_indices[:cu_k[1]] is the ordered
list of live physical keys in the conceptual prefix+suffix concatenation.
Triton reads that mapping directly, without concatenation, packing or host sync.

Scratch and output tensors belong to each invocation, not to this module.
During capture PyTorch's graph allocator owns their addresses; graph replay
overwrites every live scratch element before consuming it. The existing native
graph runner owns replay serialization/stream ordering and clones final policy
outputs. No KV, mask, scratch or output is cached across calls here. Eager calls
on separate streams have independent allocations. Change the recipe only by
clearing graphs, uninstalling and reinstalling through the model lifecycle.
"""

import hashlib
from pathlib import Path
import weakref

import torch
from torch import nn

from vvla_pi05.lite.attention import AttentionMetadata, Pi05FlashAttention
from vvla_pi05.lite.fused_expert_mlp import _mutation_guard

INSTALLATION = "_lite_suffix_attention_installation"
VERSION = "suffix-separate-kv-bf16-splitk-v1"


def _validate_options(splits, block_q):
    if type(splits) is not int or splits not in (1, 2, 4, 8):
        raise ValueError("Suffix split count must be one of 1, 2, 4, 8")
    if type(block_q) is not int or block_q not in (16, 32):
        raise ValueError("Suffix query tile must be 16 or 32")


def _validate_call(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata, *, require_cuda=True):
    """Shape/host metadata checks only; never read a device scalar on the host."""
    if q.shape != (1, 8, 50, 256):
        raise ValueError("Suffix split-K requires Q [1,8,50,256]")
    if (prefix_k.ndim != 4 or prefix_k.shape[:2] != (1, 1)
            or prefix_k.shape[-1] != 256 or prefix_k.shape[2] < 1
            or prefix_v.shape != prefix_k.shape
            or suffix_k.shape != (1, 1, 50, 256) or suffix_v.shape != suffix_k.shape):
        raise ValueError("Suffix split-K requires prefix KV [1,1,P,256] and suffix KV [1,1,50,256]")
    tensors = (q, prefix_k, prefix_v, suffix_k, suffix_v)
    if any(t.dtype != torch.bfloat16 or t.device != q.device or t.stride(-1) != 1
           or any(stride < 0 for stride in t.stride()) for t in tensors):
        raise ValueError("Suffix split-K requires same-device BF16 QKV with contiguous columns")
    if require_cuda and not q.is_cuda:
        raise ValueError("Suffix split-K requires CUDA; no silent CPU or FA2 fallback")
    if type(metadata) is not AttentionMetadata or "paged_cache" in metadata.extra:
        raise ValueError("Suffix split-K requires standalone nonpaged FA2 metadata")
    m = metadata.extra
    capacity = prefix_k.shape[2] + 50
    if (m.get("batch"), m.get("q_length"), m.get("k_length")) != (1, 50, capacity):
        raise ValueError("Suffix split-K metadata geometry differs from QKV")
    for name in ("q_indices", "k_indices", "cu_q", "cu_k"):
        value = m.get(name)
        dtype = torch.int64 if name.endswith("indices") else torch.int32
        if (not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != dtype
                or value.device != q.device or not value.is_contiguous()):
            raise ValueError("Suffix split-K invalid metadata tensor: " + name)
    if m["cu_q"].shape != (2,) or m["cu_k"].shape != (2,) or m["q_indices"].numel() != 50:
        raise ValueError("Suffix split-K requires one sequence with all 50 queries live")
    if m.get("fixed_capacity", False):
        # This flag is checked against q_keep by the original metadata helper.
        if m.get("all_queries_valid") is not True or m["k_indices"].numel() != capacity:
            raise ValueError("Suffix split-K requires all_queries_valid fixed-capacity metadata")
    elif not 1 <= m["k_indices"].numel() <= capacity:
        raise ValueError("Suffix split-K invalid eager key index count")
    # The original helpers prove nonempty keys, valid physical indices and
    # identity query order when all queries are live. Do not resynchronize those
    # already established data invariants in a 180-call captured/timed path.
    return m


def _allocate_workspace(device, splits):
    """No module-owned buffers: each call/graph pool owns these live tensors."""
    numerator = torch.empty((splits, 8, 50, 256), device=device, dtype=torch.float32)
    maximum = torch.empty((splits, 8, 50), device=device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = torch.empty((1, 50, 8, 256), device=device, dtype=torch.bfloat16)
    return numerator, maximum, denominator, output


def _launch(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata, splits, block_q):
    import triton
    from vvla_pi05.lite.suffix_attention_kernel import suffix_partial, suffix_merge

    numerator, maximum, denominator, output = _allocate_workspace(q.device, splits)
    grid = (triton.cdiv(50, block_q), 8, splits)
    suffix_partial[grid](
        q, prefix_k, prefix_v, suffix_k, suffix_v,
        metadata["k_indices"], metadata["cu_k"], numerator, maximum, denominator,
        Q_HEAD_STRIDE=q.stride(1), Q_TOKEN_STRIDE=q.stride(2),
        PK_TOKEN_STRIDE=prefix_k.stride(2), PV_TOKEN_STRIDE=prefix_v.stride(2),
        SK_TOKEN_STRIDE=suffix_k.stride(2), SV_TOKEN_STRIDE=suffix_v.stride(2),
        PREFIX_CAPACITY=prefix_k.shape[2], INDEX_COUNT=metadata["k_indices"].numel(),
        SPLITS=splits, BLOCK_Q=block_q, BLOCK_K=64, num_warps=4, num_stages=1,
    )
    suffix_merge[(triton.cdiv(50, block_q), 8)](
        numerator, maximum, denominator, output,
        SPLITS=splits, BLOCK_Q=block_q, num_warps=4,
    )
    return output.transpose(1, 2)


class Pi05SplitSuffixAttention(nn.Module):
    """Inference-only suffix dispatch; settings are immutable through public API."""

    def __init__(self, *, splits=4, block_q=16):
        super().__init__()
        _validate_options(splits, block_q)
        self._configuration = self._initial_configuration = (splits, block_q)
        self.eval()

    @property
    def splits(self):
        return self._configuration[0]

    @property
    def block_q(self):
        return self._configuration[1]

    def forward(self, *args, **kwargs):
        raise RuntimeError("Suffix split-K must use forward_split before core KV concatenation")

    def forward_split(self, q, prefix_k, prefix_v, suffix_k, suffix_v, metadata):
        if self._configuration != self._initial_configuration:
            raise RuntimeError("Suffix recipe changed; clear graphs and reinstall")
        if self.training or torch.is_grad_enabled() or torch.is_autocast_enabled(q.device.type):
            raise ValueError("Suffix split-K supports eval inference without gradients or autocast only")
        m = _validate_call(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata)
        return _launch(q, prefix_k, prefix_v, suffix_k, suffix_v, m, self.splits, self.block_q)

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / "models/pi05/modeling_pi05.py"
        paths = (Path(__file__), Path(__file__).with_name("suffix_attention_kernel.py"), core)
        return dict(
            version=VERSION, splits=self.splits, block_q=self.block_q, block_k=64,
            shape=dict(batch=1, query_heads=8, kv_heads=1, query_tokens=50, head_dim=256),
            dtype="bfloat16", scale=1.0 / 16, causal=False, dropout=0.0,
            accumulation="FP32 QK, online max/denominator/numerator and fixed-order split merge",
            probability_boundary="unnormalized exp2 scores cast to BF16 before Tensor Core PV, as in FA2",
            key_mapping="current k_indices[logical_k], valid when logical_k < device cu_k[1]",
            workspace="per-invocation PyTorch allocations; graph pool owns captured scratch; no module KV/mask cache",
            numerical_limit="FA2 bitwise equality not promised; changed split/reduction tree requires CUDA and paired-rollout validation",
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        )


class SuffixAttentionInstallation:
    def __init__(self, model, original, candidate):
        self._model = weakref.ref(model)
        self._original, self._candidate = original, candidate
        self.enabled = True

    def manifest(self):
        return dict(self._candidate.manifest(), enabled=self.enabled)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Suffix attention installation is no longer active")
        with _mutation_guard(model, "suffix attention"):
            inner = model.paligemma_with_expert
            if inner.suffix_attention is not self._candidate:
                raise RuntimeError("Suffix attention was modified externally")
            inner.suffix_attention = self._original
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_suffix_attention(model, *, splits=4, block_q=16):
    """Default-off candidate. Clear every live model graph before install/uninstall."""
    _validate_options(splits, block_q)
    with _mutation_guard(model, "suffix attention"):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Suffix attention candidate is already installed")
        inner = model.paligemma_with_expert
        original = inner.suffix_attention
        expert = inner.gemma_expert.config
        if ((expert.num_attention_heads, expert.num_key_value_heads, expert.head_dim) != (8, 1, 256)
                or getattr(model, "action_horizon", None) != 50):
            raise ValueError("Suffix candidate requires pi05 H8/HKV1/D256/action horizon50")
        if (type(original) is not Pi05FlashAttention or original.scale != 1.0 / 16
                or original.backend not in ("torch_fa2", "flash_attn")
                or "forward" in original.__dict__ or original._forward_hooks or original._forward_pre_hooks):
            raise ValueError("Suffix candidate requires the original independent FA2 suffix module")
        candidate = Pi05SplitSuffixAttention(splits=splits, block_q=block_q)
        inner.suffix_attention = candidate
        installation = SuffixAttentionInstallation(model, original, candidate)
        setattr(model, INSTALLATION, installation)
        model.invalidate_inference_caches(weights_changed=False)
        return installation
