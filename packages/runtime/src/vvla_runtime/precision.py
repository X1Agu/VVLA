"""Explicit inference precision recipes; original BF16 parameters retained.

The default scope selects Llama MLPs. The opt-in all_linear FP8 scope also
selects attention projections, vision, multimodal/proprio projectors and the
action head. Embeddings, convolutions, normalization and attention stay BF16.
Installation is outside inference_mode and capture.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from vvla_runtime.execution import native
from vvla_runtime.execution.state import tensor_signature

class QuantizedLinear(nn.Module):
    def __init__(self,original,precision,input_amax=None,*,cpu_reference=False):
        super().__init__()
        if torch.is_inference_mode_enabled():raise ValueError('Install quantized weights outside inference_mode')
        if not isinstance(original,nn.Linear) or original.weight.dtype!=torch.bfloat16 or not original.weight.is_cuda:raise ValueError('Expected CUDA BF16 Linear')
        if original.bias is not None and (original.bias.dtype!=original.weight.dtype or original.bias.device!=original.weight.device):raise ValueError('Bias must match BF16 weight device and dtype')
        if precision not in ('fp8','nvfp4'):raise ValueError(precision)
        if type(cpu_reference) is not bool or (cpu_reference and precision!='fp8'):raise ValueError('CPU reference storage is supported for FP8 only')
        align=16 if precision=='fp8' else 32
        if precision=='nvfp4' and (original.bias is not None or original.in_features%align or original.out_features%align):raise ValueError('NVFP4 requires aligned bias-free Linear')
        capability=torch.cuda.get_device_capability(original.weight.device)
        if precision=='nvfp4' and capability!=(12,0):raise ValueError('Current native NVFP4 path validated for SM120 only')
        if precision=='fp8' and capability<(8,9):raise ValueError('FP8 requires SM89 or newer')
        native.load_native()
        self.weight=original.weight
        self.bias=original.bias
        self.in_features=original.in_features;self.out_features=original.out_features;self.precision=precision
        self.padded_in_features=math.ceil(self.in_features/align)*align
        self.padded_out_features=math.ceil(self.out_features/align)*align
        self._geometry=(self.in_features,self.out_features,self.padded_in_features,self.padded_out_features)
        for name in ['packed_weight','weight_scale','input_global_scale','alpha']:self.register_buffer(name,None)
        with torch.no_grad():
            if precision=='fp8':
                weight=F.pad(self.weight.float(),(0,self.padded_in_features-self.in_features,0,self.padded_out_features-self.out_features))
                scale=(weight.abs().amax(1,keepdim=True)/448.).clamp_min(1e-12)
                if not bool(torch.isfinite(scale).all()):raise ValueError('Weight scales must be finite')
                self.packed_weight=(weight/scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous()
                self.weight_scale=scale.contiguous()
            else:
                if input_amax is None or not math.isfinite(float(input_amax)) or input_amax<=0:raise ValueError('NVFP4 needs positive finite activation calibration')
                wg=(2688./self.weight.float().abs().amax().clamp_min(1e-12)).reshape(1)
                if not bool(torch.isfinite(wg).all()):raise ValueError('Weight global scale must be finite')
                self.input_global_scale=torch.tensor([2688./float(input_amax)],dtype=torch.float32,device=self.weight.device)
                self.packed_weight,self.weight_scale=native.quantize_fp4(self.weight,wg)
                self.alpha=(1./(wg*self.input_global_scale)).float()
                if not bool((torch.isfinite(self.alpha)&(self.alpha>0)).all()):raise ValueError('NVFP4 alpha must be finite and positive')
        # This copy is evidence for stale-state checks, never an inference input.
        # FP8 must not keep a second BF16 MLP/model on a 24 GB GPU.
        if cpu_reference:
            self.weight=nn.Parameter(self.weight.detach().to('cpu'),requires_grad=False)
        self._source=self._source_signature();self._packed=self._packed_signature();self.train(original.training)

    def _source_signature(self):
        return tensor_signature(self.weight),None if self.bias is None else tensor_signature(self.bias)

    def _packed_signature(self):
        return (self.precision,*(None if t is None else tensor_signature(t) for t in [self.packed_weight,self.weight_scale,self.input_global_scale,self.alpha]))

    def execution_signature(self):
        return dict(precision=self.precision,in_features=self.in_features,out_features=self.out_features,
                    padded_in_features=self.padded_in_features,padded_out_features=self.padded_out_features,
                    bias=self.bias is not None,reference_weight_device=self.weight.device.type)

    def validate_execution_state(self):
        self.assert_fresh()

    def assert_fresh(self):
        if (self.in_features,self.out_features,self.padded_in_features,self.padded_out_features)!=self._geometry:raise RuntimeError('Quantized geometry changed; reinstall recipe')
        if self._source_signature()!=self._source:raise RuntimeError('Original weight or bias changed: reinstall precision recipe from a fresh model')
        if self._packed_signature()!=self._packed:raise RuntimeError('Packed weight/scale changed: reject stale recipe and graphs')

    def quantize_input(self, x):
        """Per-token FP8 activation, reusable by projections of the same input."""
        if self.precision != 'fp8' or x.dtype != torch.bfloat16 or x.device != self.packed_weight.device:
            raise ValueError('Shared activation requires a matching BF16 input and FP8 projection')
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError('Input geometry does not match quantized Linear')
        flat=x.reshape(-1,self.in_features).contiguous()
        if self.padded_in_features!=self.in_features:
            flat=F.pad(flat,(0,self.padded_in_features-self.in_features))
        return native.quantize_fp8(flat)

    def forward_prequantized(self,q,scale,shape):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Quantized inference requires eval and no_grad')
        rows=math.prod(shape[:-1]);k=self.in_features
        if self.precision=='fp8':
            if (shape[-1]!=k or q.shape!=(rows,self.padded_in_features) or q.dtype!=torch.float8_e4m3fn
                or scale.shape!=(rows,1) or scale.dtype!=torch.float32 or q.device!=self.packed_weight.device
                or scale.device!=q.device or not q.is_contiguous() or not scale.is_contiguous()):
                raise ValueError('Invalid FP8 activation geometry/device/layout')
            result=native.fp8_mm(q,self.packed_weight.t(),scale,self.weight_scale.t())[:,:self.out_features]
            if self.bias is not None:result=result+self.bias
            return result.reshape(*shape[:-1],self.out_features)
        if self.precision!='nvfp4' or shape[-1]!=k or q.shape!=(rows,k//2) or q.dtype!=torch.uint8:raise ValueError('Invalid packed activation geometry')
        expected=(((rows+127)//128)*128,((k+63)//64)*4)
        if scale.shape!=expected or scale.dtype!=torch.float8_e4m3fn or q.device!=self.weight.device or scale.device!=q.device or not q.is_contiguous() or not scale.is_contiguous():raise ValueError('Invalid activation scale layout')
        return native.fp4_mm(q,self.packed_weight,scale,self.weight_scale,self.alpha).reshape(*shape[:-1],self.out_features)

    def forward(self,x=None,*,quantized=None,quant_scale=None,original_shape=None):
        if quantized is not None or quant_scale is not None or original_shape is not None:
            if x is not None or quantized is None or quant_scale is None or original_shape is None:raise ValueError('Provide either BF16 input or a complete prequantized input')
            return self.forward_prequantized(quantized,quant_scale,original_shape)
        if not isinstance(x,torch.Tensor):raise ValueError('BF16 tensor input required')
        if self.training or torch.is_grad_enabled() or x.dtype!=torch.bfloat16:raise RuntimeError('Quantized inference requires BF16 eval/no_grad')
        if x.ndim<1 or x.shape[-1]!=self.in_features or x.device!=self.packed_weight.device:raise ValueError('Input geometry/device does not match quantized Linear')
        shape=x.shape;x=x.reshape(-1,self.in_features).contiguous()
        if self.precision=='fp8':
            if self.padded_in_features!=self.in_features:x=F.pad(x,(0,self.padded_in_features-self.in_features))
            q,scale=native.quantize_fp8(x)
            result=native.fp8_mm(q,self.packed_weight.t(),scale,self.weight_scale.t())
            result=result[:,:self.out_features]
            if self.bias is not None:result=result+self.bias
        else:
            q,scale=native.quantize_fp4(x,self.input_global_scale)
            result=native.fp4_mm(q,self.packed_weight,scale,self.weight_scale,self.alpha)
        return result.reshape(*shape[:-1],self.out_features)

def dense_packed_mlp(x, gate_weight, down_weight):
    gate, up = F.linear(x, gate_weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_weight)

class FusedLlamaMLP(nn.Module):
    def __init__(self,original,down_quant=False,share_quant=False,pack_gate_up=False,compiled_mlp=None):
        super().__init__();self.gate_proj=original.gate_proj;self.up_proj=original.up_proj;self.down_proj=original.down_proj
        self.down_quant=down_quant;self.share_quant=share_quant;self.pack_gate_up=pack_gate_up
        self._compiled_mlp = compiled_mlp
        self._flags=(down_quant,share_quant,pack_gate_up);self.train(original.training)
        if pack_gate_up:
            pair = (self.gate_proj, self.up_proj)
            if (any(type(m) is not nn.Linear or m.bias is not None or m.weight.dtype != torch.bfloat16 for m in pair)
                    or self.gate_proj.weight.shape != self.up_proj.weight.shape):
                raise ValueError("Gate/up packing requires matching bias-free BF16 projections")
            with torch.no_grad():
                self.register_buffer('packed_gate_weight', torch.cat([m.weight for m in pair], dim=0))
            self._packed_gate_source = self._gate_signature()
        if share_quant and (not all(isinstance(m,QuantizedLinear) and m.precision=='fp8' for m in (self.gate_proj,self.up_proj)) or self.gate_proj.in_features!=self.up_proj.in_features):
            raise ValueError('Shared quantization requires matching FP8 gate/up projections')
        if down_quant and (not isinstance(self.down_proj,QuantizedLinear) or self.down_proj.precision!='nvfp4'):raise ValueError('Fused down requires native NVFP4')
    def _gate_signature(self):
        return (type(self.gate_proj), type(self.up_proj), self.gate_proj.bias is None, self.up_proj.bias is None,
                *(tensor_signature(t) for t in (self.gate_proj.weight, self.up_proj.weight, self.packed_gate_weight)))
    def assert_fresh(self):
        if (self.down_quant,self.share_quant,self.pack_gate_up)!=self._flags:raise RuntimeError('MLP fusion flag changed; reinstall recipe')
        if self.pack_gate_up and self._gate_signature() != self._packed_gate_source:
            raise RuntimeError('Packed gate/up weight or source changed; reinstall recipe')
    def forward(self,x):
        if self.pack_gate_up:
            if self._compiled_mlp is not None:
                return self._compiled_mlp(x, self.packed_gate_weight, self.down_proj.weight)
            from vvla_runtime.execution.silu import packed_silu_mul
            return self.down_proj(packed_silu_mul(F.linear(x, self.packed_gate_weight)))
        if self.share_quant:
            q,scale=self.gate_proj.quantize_input(x)
            gate=self.gate_proj(quantized=q,quant_scale=scale,original_shape=x.shape)
            up=self.up_proj(quantized=q,quant_scale=scale,original_shape=x.shape)
        else:
            gate,up=self.gate_proj(x),self.up_proj(x)
        if self.down_quant:
            from vvla_runtime.execution.silu_nvfp4 import silu_mul_nvfp4
            q,scale=silu_mul_nvfp4(gate,up,self.down_proj.input_global_scale)
            return self.down_proj(quantized=q,quant_scale=scale,original_shape=gate.shape)
        from vvla_runtime.execution.silu import silu_mul
        return self.down_proj(silu_mul(gate,up))

def mlp_recipe_signature(recipe):
    return recipe.mlp_precision,recipe.fp8_scope,recipe.nvfp4_layers,recipe.fuse_silu_mul,recipe.fuse_down_quant,recipe.share_mlp_quant,recipe.pack_gate_up,recipe.mlp_compile

def _linear_descendants(root,prefix):
    if root is None:raise ValueError('Missing required FP8 component: '+prefix)
    entries=[]
    for name,module in root.named_modules():
        if isinstance(module,nn.Linear):
            parent_path,_,leaf=name.rpartition('.')
            if not leaf:raise ValueError('FP8 component root must own its Linear children: '+prefix)
            parent=root.get_submodule(parent_path) if parent_path else root
            entries.append((prefix+'.'+name,parent,leaf,'fp8'))
    if not entries:raise ValueError('No Linear modules found for FP8 component: '+prefix)
    return entries

def selected_linears(model,recipe,action_head=None,proprio_projector=None):
    layers=model.language_model.model.layers
    if model.language_model.config.hidden_act!='silu':raise ValueError('Llama fusion only supports official SiLU activation')
    if recipe.mlp_precision=='nvfp4_selective' and any(i>=len(layers) for i in recipe.nvfp4_layers):raise ValueError('Selected layer is absent')
    entries=[]
    if recipe.mlp_precision=='bf16':return entries
    for i,layer in enumerate(layers):
        for name in ['gate_proj','up_proj','down_proj']:
            fp4=recipe.mlp_precision=='nvfp4_full' or (recipe.mlp_precision=='nvfp4_selective' and i in recipe.nvfp4_layers and name!='down_proj')
            entries.append((f'llama.{i}.{name}',layer.mlp,name,'nvfp4' if fp4 else 'fp8'))
        if recipe.fp8_scope in ('llama','llama_vision','all_linear'):
            for name in ('q_proj','k_proj','v_proj','o_proj'):
                entries.append((f'llama.{i}.attention.{name}',layer.self_attn,name,'fp8'))
    if recipe.fp8_scope in ('llama_vision','all_linear'):
        entries.extend(_linear_descendants(model.vision_backbone,'vision'))
    if recipe.fp8_scope=='all_linear':
        for prefix,root in [('multimodal_projector',model.projector),('action_head',action_head),('proprio_projector',proprio_projector)]:
            entries.extend(_linear_descendants(root,prefix))
    ids=[id(getattr(parent,leaf)) for _,parent,leaf,_ in entries]
    if len(set(ids))!=len(ids):raise ValueError('Shared Linear modules require an explicit quantization policy')
    return entries


def _current_quantized_module(adapter, name):
    """Resolve from today's model root, never a retained former parent."""
    parts = name.split('.')
    if parts[0] == 'llama':
        layer = adapter.model.language_model.model.layers[int(parts[1])]
        return getattr(layer.self_attn, parts[3]) if parts[2] == 'attention' else getattr(layer.mlp, parts[2])
    roots = dict(vision=adapter.model.vision_backbone, multimodal_projector=adapter.model.projector,
                 action_head=adapter.action_head, proprio_projector=adapter.proprio_projector)
    return roots[parts[0]].get_submodule('.'.join(parts[1:]))


def _validate_quantized_bindings(adapter):
    for name, installed, precision in adapter._mlp_quantized_bindings:
        try:
            actual = _current_quantized_module(adapter, name)
        except (AttributeError, IndexError, KeyError) as error:
            raise RuntimeError('Quantized model path changed: ' + name) from error
        if actual is not installed or type(actual) is not QuantizedLinear or actual.precision != precision:
            raise RuntimeError('Quantized module installation changed: ' + name)

def install_mlp_recipe(adapter,calibration=None):
    recipe=adapter.recipe
    if adapter.model.training or torch.is_inference_mode_enabled():raise ValueError('Install on eval model outside inference_mode')
    if getattr(adapter.model,'_vla_mlp_installed',False):raise ValueError('Use a fresh checkpoint to change MLP recipes')
    entries=selected_linears(adapter.model,recipe,adapter.action_head,adapter.proprio_projector);calibration=calibration or {};modules=[];manifest=[]
    for key,parent,leaf,precision in entries:
        if not isinstance(getattr(parent,leaf),nn.Linear):raise ValueError('Linear already transformed: '+key)
        if precision=='nvfp4' and key not in calibration:raise ValueError('Missing calibration for '+key)
    if entries or recipe.fuse_silu_mul:
        adapter.model._vla_mlp_installed=True
        try:
            for key,parent,leaf,precision in entries:
                module=QuantizedLinear(getattr(parent,leaf),precision,calibration.get(key),cpu_reference=precision=='fp8')
                setattr(parent,leaf,module);modules.append(module)
                category=('llama_attention' if '.attention.' in key else 'llama_mlp') if key.startswith('llama.') else key.split('.')[0]
                manifest.append(dict(name=key,category=category,shape=list(module.weight.shape),**module.execution_signature()))
            if recipe.fuse_silu_mul:
                compiled_mlp = None
                if recipe.mlp_compile != 'off':
                    from vvla_runtime.gemm import compile_bf16_function
                    compiled_mlp = compile_bf16_function(dense_packed_mlp, recipe.mlp_compile)
                for layer in adapter.model.language_model.model.layers:
                    module=FusedLlamaMLP(layer.mlp,recipe.fuse_down_quant,recipe.share_mlp_quant,recipe.pack_gate_up,compiled_mlp);layer.mlp=module;modules.append(module)
        except BaseException:
            adapter.model._vla_mlp_install_failed=True
            raise
    adapter._execution_modules=modules
    adapter._mlp_recipe_signature=mlp_recipe_signature(recipe)
    adapter.quantization_manifest=manifest
    adapter._mlp_quantized_bindings=tuple((key,_current_quantized_module(adapter,key),precision)
                                         for key,_,_,precision in entries)

def validate_mlp_recipe(adapter):
    if getattr(adapter.model,'_vla_mlp_install_failed',False):raise RuntimeError('Incomplete MLP recipe')
    if mlp_recipe_signature(adapter.recipe)!=adapter._mlp_recipe_signature:raise RuntimeError('MLP recipe changed; rebuild from original checkpoint')
    _validate_quantized_bindings(adapter)
    for module in adapter._execution_modules:module.assert_fresh()

@torch.no_grad()
def calibrate_llama(adapter,batches):
    maxima={};handles=[]
    def hook(key):
        def collect(module,args):
            value=args[0].detach().abs().amax().float()
            maxima[key]=value if key not in maxima else torch.maximum(maxima[key],value)
        return collect
    try:
        for i,layer in enumerate(adapter.model.language_model.model.layers):
            for name in ['gate_proj','up_proj','down_proj']:
                module=getattr(layer.mlp,name)
                if not isinstance(module,nn.Linear):raise ValueError('Calibration requires original BF16 MLP')
                handles.append(module.register_forward_pre_hook(hook(f'llama.{i}.{name}')))
        for inputs in batches:adapter(**inputs)
        return {k:max(float(v),1e-12) for k,v in maxima.items()}
    finally:
        for handle in handles:handle.remove()
