# SPDX-License-Identifier: Apache-2.0
"""Default-off mature Torch FA2 SplitKV for pi05 suffix attention.

The public Torch2.13 paged varlen API reaches SplitKV for Sq50. Ordinary varlen
Sq50 ignores num_splits in the reviewed C++ dispatcher. One aligned page is a
layout only: current prefix/suffix are concatenated, packed using the original
metadata, and copied into fresh invocation-owned storage every call. No KV,
length, mask, result or workspace is memoized across requests. Pure Torch
packing operations permit Inductor fusion; CUDA graph pools own allocations.

Prefix attention and all model BF16/FP32 arithmetic boundaries are unchanged.
The existing explicit forward_split hook owns dispatch; installation is opt-in.
"""
import hashlib
import inspect
from pathlib import Path
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from vvla_repro_pi05.lite.attention import Pi05FlashAttention
from vvla_repro_pi05.lite.fused_expert_mlp import _mutation_guard
from vvla_repro_pi05.lite.suffix_attention import _validate_call

INSTALLATION = '_lite_suffix_fa2_installation'
VERSION = 'torch213-public-paged-suffix-fa2-v2'
SUPPORTED_SPLITS = (1, 2, 4, 8, 10)


def _validate_splits(splits):
    if type(splits) is not int or splits not in SUPPORTED_SPLITS:
        raise ValueError('Suffix FA2 split count must be one of 1, 2, 4, 8, 10')


def _public_api():
    from torch.nn.attention.varlen import varlen_attn
    if not str(torch.__version__).split('+')[0].startswith('2.13.'):
        raise RuntimeError('Suffix FA2 is pinned to the reviewed Torch2.13 API')
    if not {'num_splits', 'seqused_k', 'block_table'} <= set(inspect.signature(varlen_attn).parameters):
        raise RuntimeError('Public varlen_attn lacks the reviewed paged SplitKV API')
    return varlen_attn


def _pack_joint_pages(prefix_k, prefix_v, suffix_k, suffix_v, indices, capacity):
    """Pure tensor dataflow; logical order exactly equals original cat+gather.

    Fixed metadata has capacity indices; eager metadata can have only live
    indices. Both are padded to a page sized from physical capacity, never
    from a device length. Padding is masked by current device seqused_k.
    Full graph compilation may fuse cat/gather/pad. No host scalar reads.
    """
    k = torch.cat((prefix_k, suffix_k), dim=2).transpose(1, 2).flatten(0, 1)
    v = torch.cat((prefix_v, suffix_v), dim=2).transpose(1, 2).flatten(0, 1)
    k, v = k.index_select(0, indices), v.index_select(0, indices)
    page = ((capacity + 255) // 256) * 256
    return (F.pad(k, (0, 0, 0, 0, 0, page - k.shape[0])).unsqueeze(0),
            F.pad(v, (0, 0, 0, 0, 0, page - v.shape[0])).unsqueeze(0))


class Pi05FA2SplitSuffixAttention(nn.Module):
    """Stateless suffix dispatch, safe for eager, full-graph compile and capture.

    Compilation/capture support is an interface guarantee tested separately;
    model latency/quality acceptance remains the campaign owner's responsibility.
    """
    def __init__(self, *, splits=8):
        super().__init__()
        _validate_splits(splits)
        self._splits = self._installed_splits = splits
        self._kernel = self._installed_kernel = _public_api()
        self.eval()

    @property
    def splits(self):
        return self._splits

    def forward(self, *args, **kwargs):
        raise RuntimeError('Suffix FA2 must use forward_split before core KV concatenation')

    def forward_split(self, q, prefix_k, prefix_v, suffix_k, suffix_v, metadata):
        if self._splits != self._installed_splits or self._kernel is not self._installed_kernel:
            raise RuntimeError('Suffix FA2 recipe modified; clear graphs and reinstall')
        if self.training or torch.is_grad_enabled() or torch.is_autocast_enabled(q.device.type):
            raise ValueError('Suffix FA2 supports eval inference without gradients or autocast only')
        m = _validate_call(q, prefix_k, prefix_v, suffix_k, suffix_v, metadata)
        capacity = m['k_length']
        k, v = _pack_joint_pages(prefix_k, prefix_v, suffix_k, suffix_v, m['k_indices'], capacity)
        # Original metadata helper proves cu_q[0]==0. Its view is the one-entry
        # block table selecting page0, without creating another fill kernel.
        table = m['cu_q'][:1].view(1, 1)
        out = self._kernel(
            q.transpose(1, 2).flatten(0, 1), k, v, m['cu_q'], m['cu_k'], 50, capacity,
            scale=1/16, window_size=(-1, -1), enable_gqa=True,
            seqused_k=m['cu_k'][1:], block_table=table, num_splits=self.splits,
        )
        return out.reshape(1, 50, 8, 256).transpose(1, 2)

    def manifest(self):
        package = Path(__file__).resolve().parents[1]
        paths = (Path(__file__), Path(__file__).with_name('suffix_attention.py'),
                 package / 'models/pi05/modeling_pi05.py')
        api_file = Path(inspect.getfile(self._installed_kernel))
        return dict(version=VERSION, splits=self.splits, supported_splits=list(SUPPORTED_SPLITS), default_splits=8,
                    public_api='torch.nn.attention.varlen.varlen_attn',
                    torch=torch.__version__, torch_api_source_sha256=hashlib.sha256(api_file.read_bytes()).hexdigest(),
                    shape=dict(batch=1,query_heads=8,kv_heads=1,query_tokens=50,head_dim=256),
                    dtype='bfloat16',scale=1/16,causal=False,dropout=0,
                    layout='one page aligned to256; cat then original stable k_indices gather then zero-pad',
                    length='current device cu_k[1] via seqused_k; no host length read',
                    workspace='fresh per-invocation Torch allocations; graph pool owns captured storage; no module KV/mask cache',
                    compile='pure Torch cat/index_select/pad + public custom op; fullgraph must pass CUDA test',
                    numerical_limit='Split/reduction tree differs from unsplit FA2; no bitwise or policy-quality equivalence claimed',
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class SuffixFA2Installation:
    def __init__(self, model, original, candidate):
        self._model = weakref.ref(model)
        self._original, self._candidate = original, candidate
        self.enabled = True

    def manifest(self):
        return dict(self._candidate.manifest(),enabled=self.enabled)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Suffix FA2 installation is no longer active')
        with _mutation_guard(model, 'suffix FA2'):
            inner = model.paligemma_with_expert
            if inner.suffix_attention is not self._candidate:
                raise RuntimeError('Suffix FA2 attention was modified externally')
            inner.suffix_attention = self._original
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_suffix_fa2(model, *, splits=8):
    """Install only on original standaloneFA2; retire all graphs before mutation."""
    _validate_splits(splits)
    with _mutation_guard(model, 'suffix FA2'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Suffix FA2 is already installed')
        inner = model.paligemma_with_expert
        original, expert = inner.suffix_attention, inner.gemma_expert.config
        if ((expert.num_attention_heads, expert.num_key_value_heads, expert.head_dim) != (8, 1, 256)
                or getattr(model, 'action_horizon', None) != 50):
            raise ValueError('Suffix FA2 requires pi05 H8/HKV1/D256/action horizon50')
        if (type(original) is not Pi05FlashAttention or original.scale != 1/16
                or original.backend not in ('torch_fa2', 'flash_attn')
                or 'forward' in original.__dict__ or original._forward_hooks or original._forward_pre_hooks):
            raise ValueError('Suffix FA2 requires original independent suffix attention without another candidate')
        candidate = Pi05FA2SplitSuffixAttention(splits=splits)
        inner.suffix_attention = candidate
        handle = SuffixFA2Installation(model, original, candidate)
        setattr(model, INSTALLATION, handle)
        model.invalidate_inference_caches(weights_changed=False)
        return handle
