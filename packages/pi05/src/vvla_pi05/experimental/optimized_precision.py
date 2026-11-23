"""Stage FP8 ablations over the complete selected Lite execution stack.

Original modules, packing and BF16 tactics stay owned. Precision hooks are
explicit instance attributes consumed by the same fusion owners, not a second
engine or a global monkeypatch. Immutable FP8 weights share the graph lifetime.
"""
from copy import deepcopy
import threading
import weakref

import torch
from torch import nn

from vvla_runtime.dynamic_fp8 import UnsupportedFP8, runtime_supported, _epoch
from vvla_runtime.fp8_ops import quantize_tensor, quantize_gelu, scaled_mm
from ..lite.fused_expert_mlp import _mutation_guard
from .precision import STAGES

INSTALLATION='_lite_dynamic_fp8_installation'


class Projection:
    def __init__(self, owner, weights, *, bias=None, name, stage):
        self.owner=owner
        self.weights=tuple(weights)
        self.bias=bias
        self.name,self.stage=name,stage
        if not self.weights or any(w.ndim!=2 or w.dtype!=torch.bfloat16 or not w.is_cuda
                                    for w in self.weights):
            raise UnsupportedFP8('FP8 needs original CUDA BF16 projection weights')
        self.device=self.weights[0].device
        self.width=self.weights[0].shape[1]
        self.output=sum(w.shape[0] for w in self.weights)
        if (not runtime_supported(self.device) or self.width%16 or self.output%16
                or any(w.shape[1]!=self.width or w.device!=self.device for w in self.weights)):
            raise UnsupportedFP8('FP8 packed projection device/K/N is unsupported')
        # Disabling inference mode re-enables gradients; enter no_grad last.
        # Keep versioned buffers without retaining FP32 packing intermediates.
        with torch.inference_mode(False),torch.no_grad():
            value=torch.cat(self.weights,dim=0).float() if len(self.weights)>1 else self.weights[0].float()
            if not torch.isfinite(value).all() or (bias is not None and not torch.isfinite(bias).all()):
                raise ValueError('Cannot pack nonfinite projection weights')
            self.scale=(value.abs().amax().clamp_min(1e-12)/448).reshape(1).clone()
            self.quantized=(value/self.scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous().t()
        self._identity=self.identity()
        self._prepared_compile_token=None

    def identity(self):
        return tuple(_epoch(x) for x in (*self.weights,self.bias,self.scale,self.quantized))

    def validate(self):
        if self.identity()!=self._identity:
            raise RuntimeError('FP8 original/packed weights changed; retire and reconstruct')

    def __call__(self,x,up=None):
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('FP8 compilation requires its prepared graph owner')
        else:
            self.owner.assert_execution_owner()
            self.validate()
        if (torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                or x.dtype!=torch.bfloat16 or x.device!=self.device or x.shape[-1]!=self.width):
            raise ValueError('FP8 projection requires inference BF16 on its original device')
        flat=x.reshape(-1,self.width)
        quantized,scale=(quantize_tensor(flat.contiguous()) if up is None else
                         quantize_gelu(flat,up.reshape(-1,self.width)))
        result=scaled_mm(quantized,self.quantized,scale,self.scale)[:flat.shape[0]]
        if self.bias is not None:result=result+self.bias
        return result.reshape(*x.shape[:-1],self.output)


class PackedMLP:
    def __init__(self, gate_up, down):
        self.gate_up,self.down=gate_up,down

    def __call__(self,x):
        gate,up=self.gate_up(x).chunk(2,dim=-1)
        return self.down(gate,up)


class PrecisionInstallation:
    def __init__(self,model):
        self._model=weakref.ref(model)
        self._runner=None
        self._generation=self._thread=None
        self.enabled=True
        self.sites=[]
        self.bindings=[]
        self.fallback=[]
        self.original=tuple(_epoch(p) for p in (*model.parameters(),*model.buffers()))

    def bind(self,target,attribute,value):
        if attribute=='_projection_precision' and hasattr(target,attribute):
            raise ValueError('Projection precision already installed')
        existed=attribute in target.__dict__
        previous=target.__dict__.get(attribute)
        setattr(target,attribute,value)
        self.bindings.append((target,attribute,value,existed,previous))

    def bind_runner(self,runner):
        if self._runner is not None or runner.model is not self._model() or runner.entries:
            raise RuntimeError('Bind precision once to its fresh native graph owner')
        self._runner=weakref.ref(runner)

    def validate(self):
        model=self._model()
        if not self.enabled or model is None or getattr(model,INSTALLATION,None) is not self:
            raise RuntimeError('FP8 precision owner is inactive')
        if any(getattr(target,name,None) is not value for target,name,value,_,_ in self.bindings):
            raise RuntimeError('FP8 precision binding changed')
        if tuple(_epoch(p) for p in (*model.parameters(),*model.buffers()))!=self.original:
            raise RuntimeError('FP8 source model changed; reconstruct before capture')
        for site in self.sites:site.validate()

    def prepare(self,generation):
        if self._generation is not None and generation!=self._generation:
            raise RuntimeError('FP8 generation changed; retire and reconstruct')
        self.validate()
        self._generation,self._thread=generation,threading.get_ident()

    def assert_execution_owner(self):
        runner=None if self._runner is None else self._runner()
        model=self._model()
        if (not self.enabled or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked() or self._thread!=threading.get_ident()
                or self._generation!=model._inference_cache.generation):
            raise RuntimeError('FP8 requires the current serialized graph owner/generation')

    def compilation_members(self,generation):
        self.assert_execution_owner()
        if generation!=self._generation:raise RuntimeError('Stale FP8 compile generation')
        self.validate()
        return tuple(self.sites)

    def manifest(self):
        self.validate()
        return dict(implementation='optimized-packed-dynamic-fp8-v2',experimental=True,
            activation_calibration=False,activation_scaling='per current packed projection tensor',
            weight_scaling='per packed projection tensor',fallback=deepcopy(self.fallback),
            sites=[dict(name=s.name,stage=s.stage,shape=[s.output,s.width],
                        packed_source_projections=len(s.weights)) for s in self.sites],
            activation_quantizer='tile amax; fused global scale/pad/direct-FP32-E4M3 conversion',
            quantizer_launches=2,original_bf16_weights_retained=True,
            fused_gelu_to_fp8=True,
            additional_weight_bytes=sum(s.quantized.numel()+s.scale.numel()*4 for s in self.sites),
            optimized_bf16_stack_preserved=True,quality_qualified=False)

    def uninstall(self):
        model=self._model()
        with _mutation_guard(model,'dynamic FP8 retirement'):
            runner=None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain graphs before retiring FP8 packed storage')
            if runner is None and self.sites:
                # Installation can fail after asynchronous weight packing but
                # before a graph owner exists. Keep its buffers until drained.
                torch.cuda.synchronize(self.sites[0].device)
            if any(getattr(target,name,None) is not value for target,name,value,_,_ in self.bindings):
                raise RuntimeError('FP8 binding changed before retirement')
            for target,name,value,existed,previous in reversed(self.bindings):
                if existed:setattr(target,name,previous)
                else:delattr(target,name)
            if getattr(model,INSTALLATION,None) is self:delattr(model,INSTALLATION)
            self.enabled=False
            self.bindings.clear();self.sites.clear()
            model.invalidate_inference_caches(weights_changed=False)


def install(model,stages,*,unsupported='error'):
    stages=set(stages)
    if not stages or not stages<=STAGES or unsupported not in ('error','bf16'):
        raise ValueError('Invalid FP8 stages or fallback')
    with _mutation_guard(model,'optimized dynamic FP8 installation'):
        if getattr(model,INSTALLATION,None) is not None or getattr(model,'_lite_prefix_fp8_installation',None) is not None:
            raise ValueError('Choose one explicit FP8 precision recipe')
        owner=PrecisionInstallation(model)
        setattr(model,INSTALLATION,owner)
        skipped=set()
        def projection(weights,**kwargs):
            try:site=Projection(owner,weights,**kwargs)
            except UnsupportedFP8 as error:
                if unsupported=='error':raise
                owner.fallback.append(dict(name=kwargs['name'],stage=kwargs['stage'],reason=str(error)))
                return None
            owner.sites.append(site)
            return site
        try:
            layers=model.paligemma_with_expert.gemma_expert.model.layers
            qkv=getattr(model,'_lite_expert_qkv_installation',None)
            mlp=getattr(model,'_lite_expert_mlp_installation',None)
            lt=getattr(model,'_lite_expert_cublaslt_installation',None)
            for i,layer in enumerate(layers):
                if 'action_attention' in stages and qkv is not None:
                    modules=[getattr(layer.self_attn,name) for name in ('q_proj','k_proj','v_proj')]
                    site=projection([m.weight for m in modules],name=f'expert.{i}.packed_qkv',stage='action_attention')
                    if site is not None:owner.bind(qkv._packers[i],'_projection_precision',site)
                    skipped.update(map(id,modules))
                if 'action_attention' in stages and lt is not None:
                    module=layer.self_attn.o_proj
                    site=projection([module.weight],bias=module.bias,name=f'expert.{i}.o_proj',stage='action_attention')
                    if site is not None:owner.bind(lt.layers[i].sites['attention_out'],'_projection_precision',site)
                    skipped.add(id(module))
                if 'action_mlp' in stages and mlp is not None:
                    modules=[getattr(layer.mlp,name) for name in ('gate_proj','up_proj','down_proj')]
                    gate=projection([m.weight for m in modules[:2]],name=f'expert.{i}.packed_gate_up',stage='action_mlp')
                    down=projection([modules[2].weight],name=f'expert.{i}.down',stage='action_mlp')
                    if (gate is None)!=(down is None):
                        raise UnsupportedFP8('Packed MLP precision needs both gate/up and down; select BF16 explicitly')
                    if gate is not None:owner.bind(mlp._candidates[i],'_projection_precision',PackedMLP(gate,down))
                    skipped.update(map(id,modules))
            vision_lt=getattr(model,'_lite_vision_fc2_cublaslt_installation',None)
            vision_sites={} if vision_lt is None else {id(s.module):s for s in vision_lt.sites}
            for name,module in model.named_modules():
                if type(module) is not nn.Linear or id(module) in skipped:continue
                stage=('vision' if '.vision_tower.' in name else
                       'prefix' if '.paligemma.model.language_model.layers.' in name else
                       'action_attention' if '.gemma_expert.model.layers.' in name and '.self_attn.' in name else
                       'action_mlp' if '.gemma_expert.model.layers.' in name and '.mlp.' in name else None)
                if stage not in stages:continue
                site=projection([module.weight],bias=module.bias,name=name,stage=stage)
                if site is None:continue
                if id(module) in vision_sites:owner.bind(vision_sites[id(module)],'_projection_precision',site)
                else:owner.bind(module,'forward',site)
            if not owner.sites and not owner.fallback:raise ValueError('No FP8 sites selected')
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException as error:
            error.fp8_installation=owner
            try:
                torch.cuda.synchronize(next(model.parameters()).device)
                owner.uninstall()
            except BaseException as cleanup:
                error.add_note('FP8 partial installation is retained for retirement retry: '+repr(cleanup))
            raise
