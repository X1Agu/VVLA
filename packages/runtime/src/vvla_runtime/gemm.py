"""Opt-in BF16 projection layouts, tuned only for observed B32 x 608 rows.

Weights are packed once outside capture. Original weights remain the fallback;
versioned changes to either source or packed data fail closed before replay.
"""
import math
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from vvla_runtime.execution.state import tensor_signature

def compile_bf16_function(function, mode):
    """Compile dense BF16 operators while keeping their intermediate dtype conversions."""
    return torch.compile(function, fullgraph=True, dynamic=False, options={
        'triton.cudagraphs':False, 'emulate_precision_casts':True,
        'max_autotune':mode=='tuned', 'max_autotune_gemm_backends':'ATEN,TRITON'})

class PretransposedBF16Linear(nn.Module):
    def __init__(self, original, *, tuned_rows=(19456,)):
        super().__init__()
        if original.bias is not None or original.weight.dtype!=torch.bfloat16:raise ValueError('Bias-free BF16 projection required')
        self.register_parameter('bias',None)
        self.weight=original.weight;self.in_features=original.in_features;self.out_features=original.out_features
        self.tuned_rows=tuple(tuned_rows)
        with torch.no_grad():self.register_buffer('packed_weight',self.weight.t().contiguous())
        self.use_packed=self.weight.is_cuda and torch.cuda.get_device_capability(self.weight.device)==(12,0)
        self._spec=(self.tuned_rows,self.in_features,self.out_features,self.use_packed)
        self._source=tensor_signature(self.weight);self._packed=tensor_signature(self.packed_weight)
        self.eval()
    def assert_fresh(self):
        if self.bias is not None or (self.tuned_rows,self.in_features,self.out_features,self.use_packed)!=self._spec or tensor_signature(self.weight)!=self._source or tensor_signature(self.packed_weight)!=self._packed:
            raise RuntimeError('Packed BF16 weight or layout changed; rebuild projection and retire graphs')
    def forward(self,x):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Inference-only BF16 layout')
        rows=math.prod(x.shape[:-1])
        if not self.use_packed or rows not in self.tuned_rows:return F.linear(x,self.weight)
        return torch.mm(x.reshape(-1,self.in_features),self.packed_weight).reshape(*x.shape[:-1],self.out_features)

class PackedQKV(nn.Module):
    def __init__(self, attention, *, transposed=False,tuned_rows=(19456,)):
        super().__init__()
        self._attention=weakref.ref(attention)
        q,k,v=attention.q_proj,attention.k_proj,attention.v_proj
        if any(m.bias is not None or m.weight.dtype!=torch.bfloat16 or m.in_features!=q.in_features for m in (q,k,v)):raise ValueError('Compatible bias-free BF16 QKV required')
        self.in_features=q.in_features;self.sizes=(q.out_features,k.out_features,v.out_features)
        self.transposed=transposed;self.tuned_rows=tuple(tuned_rows)
        with torch.no_grad():
            packed=torch.cat(self.sources,0)
            self.register_buffer('packed_weight',packed.t().contiguous() if transposed else packed.contiguous())
        self.use_packed=q.weight.is_cuda and (not transposed or torch.cuda.get_device_capability(q.weight.device)==(12,0))
        self._spec=(self.in_features,self.sizes,self.transposed,self.tuned_rows,self.use_packed)
        self._source=tuple(tensor_signature(w) for w in self.sources);self._packed=tensor_signature(self.packed_weight)
        self.eval()
    @property
    def sources(self):
        attention=self._attention()
        if attention is None:raise RuntimeError('QKV source owner was retired')
        modules=tuple(getattr(attention,name) for name in ('q_proj','k_proj','v_proj'))
        if any(type(module) is not nn.Linear or module.bias is not None for module in modules):
            raise RuntimeError('QKV source module semantics changed; rebuild the attention recipe')
        return tuple(module.weight for module in modules)
    def assert_fresh(self):
        if (self.in_features,self.sizes,self.transposed,self.tuned_rows,self.use_packed)!=self._spec or tuple(tensor_signature(w) for w in self.sources)!=self._source or tensor_signature(self.packed_weight)!=self._packed:
            raise RuntimeError('Packed QKV sources or layout changed; rebuild projection and retire graphs')
    def forward(self,x):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Inference-only QKV layout')
        if not self.use_packed or (self.transposed and math.prod(x.shape[:-1]) not in self.tuned_rows):return tuple(F.linear(x,w) for w in self.sources)
        flat=x.reshape(-1,self.in_features)
        result=torch.mm(flat,self.packed_weight) if self.transposed else F.linear(flat,self.packed_weight)
        return result.reshape(*x.shape[:-1],sum(self.sizes)).split(self.sizes,dim=-1)

def install_attention_layout(adapter):
    recipe=adapter.recipe
    adapter._attention_layout=(recipe.qkvo_layout,recipe.fuse_qkv)
    for layer in adapter.model.language_model.model.layers:
        a=layer.self_attn
        if recipe.fuse_qkv:
            a.qkv_proj=PackedQKV(a,transposed=recipe.qkvo_layout=='pretransposed')
            adapter._execution_modules.append(a.qkv_proj)
        if recipe.qkvo_layout=='pretransposed':
            for name in (('o_proj',) if recipe.fuse_qkv else ('q_proj','k_proj','v_proj','o_proj')):
                module=PretransposedBF16Linear(getattr(a,name));setattr(a,name,module);adapter._execution_modules.append(module)

def validate_attention_layout(adapter):
    if (adapter.recipe.qkvo_layout,adapter.recipe.fuse_qkv)!=adapter._attention_layout:
        raise RuntimeError('Attention layout recipe changed; rebuild adapter and retire graphs')
