"""Fixed-batch extension (default B32) of public Torch FA2 paged SplitKV.

Current compact metadata is expanded into one invocation-owned page per sample.
No KV, mask, observation or length is reused across requests. B1 implementation
and its evidence are unchanged; B32 needs its own numerical/capture qualification.
"""
import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn

from .batch_layout import validate_fixed_batch_size
from .attention import AttentionMetadata, Pi05FlashAttention
from .fused_expert_mlp import _mutation_guard
from .suffix_fa2 import _public_api, _validate_splits, SuffixFA2Installation

BATCH = 32
VERSION = "torch213-public-paged-b32-suffix-fa2-v1"


def _validate_call(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata, *, batch_size=32, require_cuda=True):
    batch = validate_fixed_batch_size(batch_size)
    if q.shape != (batch, 8, 50, 256):
        raise ValueError(f"B{batch} suffix requires Q[{batch},8,50,256]")
    if (prefix_k.ndim != 4 or prefix_k.shape[:2] != (batch, 1) or prefix_k.shape[-1] != 256
            or prefix_k.shape[2] < 1 or prefix_v.shape != prefix_k.shape
            or suffix_k.shape != (batch, 1, 50, 256) or suffix_v.shape != suffix_k.shape):
        raise ValueError(f"B{batch} suffix requires per-sample prefix and suffix KV")
    if any(t.dtype != torch.bfloat16 or t.device != q.device or t.stride(-1) != 1
           or any(stride < 0 for stride in t.stride()) for t in (q, prefix_k, prefix_v, suffix_k, suffix_v)):
        raise ValueError(f"B{batch} suffix requires same-device BF16 QKV with contiguous columns")
    if require_cuda and not q.is_cuda:
        raise ValueError(f"B{batch} suffix has no CPU or implicit attention fallback")
    if type(metadata) is not AttentionMetadata or "paged_cache" in metadata.extra:
        raise ValueError(f"B{batch} suffix requires original standalone fixed-capacity metadata")
    m = metadata.extra
    capacity = prefix_k.shape[2] + 50
    if ((m.get("batch"), m.get("q_length"), m.get("k_length")) != (batch, 50, capacity)
            or m.get("fixed_capacity") is not True or m.get("all_queries_valid") is not True):
        raise ValueError(f"B{batch} suffix metadata geometry/all-live queries differ")
    for name in ("q_indices", "k_indices", "cu_q", "cu_k"):
        value = m.get(name)
        dtype = torch.int64 if name.endswith("indices") else torch.int32
        if (not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != dtype
                or value.device != q.device or not value.is_contiguous()):
            raise ValueError("Invalid B32 suffix metadata tensor: " + name)
    if (m["cu_q"].shape != (batch+1,) or m["cu_k"].shape != (batch+1,)
            or m["q_indices"].numel() != batch*50 or m["k_indices"].numel() != batch*capacity):
        raise ValueError(f"B{batch} suffix requires all{batch} sequences in fixed storage")
    return m


def _pack_joint_pages(prefix_k, prefix_v, suffix_k, suffix_v, indices, cu_k, capacity, *, batch_size=32):
    """Compact global valid KV -> independent current per-sample pages.

    The original helper globally stable-sorts valid physical indices; cu_k
    bounds each sample's contiguous valid range within that compact array.
    Fixed-capacity tail indices are never exposed as valid keys.
    """
    batch = validate_fixed_batch_size(batch_size)
    if (any(tensor.shape[0] != batch for tensor in (prefix_k, prefix_v, suffix_k, suffix_v))
            or cu_k.shape != (batch + 1,)):
        raise ValueError("Paged suffix pack requires the owner's fixed batch geometry")
    k = torch.cat((prefix_k, suffix_k), dim=2).transpose(1, 2).flatten(0, 1).index_select(0, indices)
    v = torch.cat((prefix_v, suffix_v), dim=2).transpose(1, 2).flatten(0, 1).index_select(0, indices)
    page = ((capacity + 255) // 256) * 256
    position = torch.arange(page, device=k.device, dtype=torch.int64)[None, :]
    lengths = cu_k[1:] - cu_k[:-1]
    gather = (cu_k[:-1, None].to(torch.int64) + position).clamp(max=k.shape[0]-1).flatten()
    keep = (position < lengths[:, None])[:, :, None, None]
    shape = (batch, page, 1, 256)
    k = k.index_select(0, gather).reshape(shape)
    v = v.index_select(0, gather).reshape(shape)
    return torch.where(keep, k, 0), torch.where(keep, v, 0), lengths


class Pi05BatchFA2SplitSuffixAttention(nn.Module):
    def __init__(self, *, splits=8, batch_size=32):
        super().__init__()
        _validate_splits(splits)
        self._batch_size = self._installed_batch_size = validate_fixed_batch_size(batch_size)
        self._splits = self._installed_splits = splits
        self._kernel = self._installed_kernel = _public_api()
        self.eval()

    @property
    def batch_size(self):
        if self._batch_size != self._installed_batch_size:
            raise RuntimeError("Fixed suffix batch changed; retire and reinstall")
        return self._batch_size

    def forward(self, *args, **kwargs):
        raise RuntimeError("B32 suffix requires forward_split")

    def forward_split(self, q, prefix_k, prefix_v, suffix_k, suffix_v, metadata):
        batch = self.batch_size
        if self._splits != self._installed_splits or self._kernel is not self._installed_kernel:
            raise RuntimeError("B32 suffix recipe changed; retire and reinstall")
        if self.training or torch.is_grad_enabled() or torch.is_autocast_enabled(q.device.type):
            raise ValueError("B32 suffix requires eval inference without gradients/autocast")
        m = _validate_call(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata, batch_size=batch)
        capacity = m["k_length"]
        k, v, lengths = _pack_joint_pages(prefix_k, prefix_v, suffix_k, suffix_v,
                                         m["k_indices"], m["cu_k"], capacity, batch_size=batch)
        table = torch.div(m["cu_q"][:-1], 50, rounding_mode="floor").view(batch, 1)
        output = self._kernel(q.transpose(1, 2).flatten(0, 1), k, v, m["cu_q"], m["cu_k"], 50, capacity,
                              scale=1/16, window_size=(-1, -1), enable_gqa=True,
                              seqused_k=lengths, block_table=table, num_splits=self._splits)
        return output.reshape(batch, 50, 8, 256).transpose(1, 2)

    def manifest(self):
        batch = self.batch_size
        here = Path(__file__).resolve()
        api_file = Path(inspect.getfile(self._installed_kernel))
        return dict(version=f"torch213-public-paged-b{batch}-suffix-fa2-v1", batch=batch, splits=self._splits, dtype="bfloat16", scale=1/16,
                    shape=dict(query_heads=8, kv_heads=1, query_tokens=50, head_dim=256),
                    public_api="torch.nn.attention.varlen.varlen_attn", torch=torch.__version__,
                    layout=f"{batch} invocation-owned aligned pages from actual global compact k_indices/cu_k",
                    length="current cu_k differences per slot; no host scalar read",
                    workspace="fresh current Torch dataflow; native graph pool owns captured storage",
                    numerical_limit=f"SplitKV/reduction differs; B{batch} numerical and policy qualification required",
                    source_sha256={here.name:hashlib.sha256(here.read_bytes()).hexdigest(),
                                   "suffix_fa2.py":hashlib.sha256(here.with_name("suffix_fa2.py").read_bytes()).hexdigest(),
                                   "torch_varlen_api":hashlib.sha256(api_file.read_bytes()).hexdigest()})


def install_batch_suffix_fa2(model, *, splits=8, batch_size=32):
    validate_fixed_batch_size(batch_size)
    _validate_splits(splits)
    attribute = "_lite_suffix_fa2_installation"
    with _mutation_guard(model, "B32 suffix FA2"):
        if getattr(model, attribute, None) is not None:
            raise ValueError("Suffix attention is already installed")
        inner = model.paligemma_with_expert
        original, expert = inner.suffix_attention, inner.gemma_expert.config
        if ((expert.num_attention_heads, expert.num_key_value_heads, expert.head_dim) != (8, 1, 256)
                or model.action_horizon != 50 or type(original) is not Pi05FlashAttention
                or original.scale != 1/16 or original.backend not in ("torch_fa2", "flash_attn")
                or "forward" in original.__dict__ or original._forward_hooks or original._forward_pre_hooks):
            raise ValueError("B32 suffix requires unchanged original H8/HKV1/D256/Q50 standalone FA2")
        candidate = Pi05BatchFA2SplitSuffixAttention(splits=splits, batch_size=batch_size)
        inner.suffix_attention = candidate
        handle = SuffixFA2Installation(model, original, candidate)
        setattr(model, attribute, handle)
        model.invalidate_inference_caches(weights_changed=False)
        return handle
