"""Declared true-batch camera/FA2 geometry over shared pi05 weights.

The selected B1 Split8 and multi-row Split1 routes remain explicit. Every new
geometry needs real input/oracle/capture/quality qualification; historical B32
support does not certify this module or another physical device.
"""
import hashlib
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from .batch_layout import validate_fixed_batch_size
from .batch_vision import BatchVisionInstallation,_embed_prefix
from .batched_vision import BatchedVisionInstallation,INSTALLATION,_geometry
from .batch_suffix_fa2 import Pi05BatchFA2SplitSuffixAttention
from .attention import Pi05FlashAttention
from .suffix_fa2 import SuffixFA2Installation
from .fused_expert_mlp import _mutation_guard
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

BATCH_SIZES=(1,2,4,8,16,32)


def sources():
    here=Path(__file__).resolve().parent
    return {name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
        ('dynamic_batch_geometry.py','batch_vision.py','batch_suffix_fa2.py','suffix_fa2.py')}


class DynamicVisionInstallation(BatchVisionInstallation):
    def __init__(self,model,geometry):
        super().__init__(model,geometry,batch_size=1)
        self._current_batch=1

    @property
    def batch_size(self):
        if self._batch_size!=1 or self._installed_batch_size!=1:
            raise RuntimeError('Dynamic vision geometry declaration changed')
        return validate_fixed_batch_size(self._current_batch)

    def select_batch(self,batch):
        self._current_batch=validate_fixed_batch_size(batch)

    def manifest(self):
        value=BatchedVisionInstallation.manifest(self)
        value.update(version='source497-dynamic-camera-batches-v1',batch_sizes=list(BATCH_SIZES),
            grouping='fresh camera-major groups, split into exact request order',
            source_sha256={**value['source_sha256'],**sources()},
            inherited_gpu_qualification=False)
        return value


class DynamicSuffixAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.routes=nn.ModuleDict({str(batch):Pi05BatchFA2SplitSuffixAttention(
            splits=8 if batch==1 else 1,batch_size=batch) for batch in BATCH_SIZES})
        self.eval()

    def forward(self,*args,**kwargs):raise RuntimeError('Dynamic suffix requires forward_split')

    def forward_split(self,q,prefix_k,prefix_v,suffix_k,suffix_v,metadata):
        batch=validate_fixed_batch_size(q.shape[0])
        return self.routes[str(batch)].forward_split(q,prefix_k,prefix_v,suffix_k,suffix_v,metadata)

    def manifest(self):
        return dict(version='source497-dynamic-suffix-fa2-v1',batch_sizes=list(BATCH_SIZES),
            routes={str(batch):self.routes[str(batch)].manifest() for batch in BATCH_SIZES},
            source_sha256=sources(),inherited_gpu_qualification=False)


def install_dynamic_geometry(model):
    with _mutation_guard(model,'dynamic pi05 camera/attention geometry'):
        if (getattr(model,INSTALLATION,None) is not None or 'embed_prefix' in model.__dict__
                or getattr(model.embed_prefix,'__func__',None) is not Pi05ForActionPrediction.embed_prefix):
            raise ValueError('Dynamic camera batching requires original unmodified embedding dispatch')
        inner=model.paligemma_with_expert
        original=inner.suffix_attention;expert=inner.gemma_expert.config
        if (getattr(model,'_lite_suffix_fa2_installation',None) is not None
                or (expert.num_attention_heads,expert.num_key_value_heads,expert.head_dim)!=(8,1,256)
                or model.action_horizon!=50 or type(original) is not Pi05FlashAttention
                or original.scale!=1/16 or original.backend not in ('torch_fa2','flash_attn')
                or 'forward' in original.__dict__ or original._forward_hooks or original._forward_pre_hooks):
            raise ValueError('Dynamic suffix requires original standalone H8/HKV1/D256/Q50 BF16 FA2')
        # Construct every child before publishing either model mutation.
        camera=DynamicVisionInstallation(model,_geometry(model))
        attention=DynamicSuffixAttention()
        suffix=SuffixFA2Installation(model,original,attention)
        model.invalidate_inference_caches(weights_changed=False)
        setattr(model,INSTALLATION,camera)
        model.embed_prefix=MethodType(_embed_prefix,model)
        inner.suffix_attention=attention
        setattr(model,'_lite_suffix_fa2_installation',suffix)
        return camera,suffix
