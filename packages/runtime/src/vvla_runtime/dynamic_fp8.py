"""Experimental per-tensor dynamic FP8 Linear, using native Torch scaled GEMM.

No activation calibration dataset. Quantization/cast/padding overhead remains in
the timed forward. This is a mechanism for ablation, not a speed/quality claim.
"""
import torch
from torch import nn


class UnsupportedFP8(ValueError):
    """Cold capability mismatch; safe to select the unchanged BF16 projection."""


def _epoch(value):
    if value is None:return None
    return id(value),value.data_ptr(),value._version,tuple(value.shape),str(value.dtype),str(value.device)


def runtime_supported(device):
    device=torch.device(device)
    return device.type=='cuda' and torch.cuda.get_device_capability(device)>=(8,9) and hasattr(torch,'_scaled_mm')


class DynamicFP8Linear(nn.Module):
    def __init__(self,original):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.weight.dtype!=torch.bfloat16 or not original.weight.is_cuda:
            raise UnsupportedFP8('Dynamic FP8 requires a CUDA BF16 nn.Linear')
        if not runtime_supported(original.weight.device):raise UnsupportedFP8('Native FP8 unavailable for this device/API')
        if original.in_features%16 or original.out_features%16:raise UnsupportedFP8('FP8 requires aligned K/N (multiples of 16)')
        self.weight,self.bias=original.weight,original.bias
        self.in_features,self.out_features=original.in_features,original.out_features
        with torch.inference_mode(False),torch.no_grad():
            w=self.weight.detach().float()
            if not torch.isfinite(w).all() or (self.bias is not None and not torch.isfinite(self.bias).all()):
                raise ValueError('Nonfinite checkpoint parameters cannot be quantized')
            scale=w.abs().amax().clamp_min(1e-12)/448.0
            self.register_buffer('weight_scale',scale.reshape(1).clone())
            self.register_buffer('quantized_weight',(w/scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous().t())
        self._epochs=tuple(_epoch(v) for v in (self.weight,self.bias,self.weight_scale,self.quantized_weight))
        self.eval()

    def assert_fresh(self):
        if tuple(_epoch(v) for v in (self.weight,self.bias,self.weight_scale,self.quantized_weight))!=self._epochs:
            raise RuntimeError('Dynamic FP8 original/packed tensor changed; reconstruct before capture')

    def forward(self,x):
        self.assert_fresh()
        if torch.is_grad_enabled():raise RuntimeError('Dynamic FP8 is inference-only')
        if x.dtype!=torch.bfloat16 or x.device!=self.weight.device or x.shape[-1]!=self.in_features:
            raise ValueError('Dynamic FP8 input must match CUDA BF16 projection geometry')
        flat=x.reshape(-1,self.in_features)
        if flat.shape[0]==0:raise ValueError('Empty projection inputs are unsupported')
        from .fp8_ops import quantize_tensor,scaled_mm
        with torch.profiler.record_function('vvla.fp8.activation_quantize'):
            quantized,scale=quantize_tensor(flat.contiguous())
        with torch.profiler.record_function('vvla.fp8.scaled_mm'):
            out=scaled_mm(quantized,self.quantized_weight,scale,self.weight_scale)
        out=out[:x.numel()//self.in_features]
        if self.bias is not None:out=out+self.bias
        return out.reshape(*x.shape[:-1],self.out_features)


def install_dynamic_fp8(model,select,*,unsupported='error'):
    """Cold transactional replacement. Select returns a stage name or None."""
    if unsupported not in ('error','bf16'):raise ValueError('Unsupported fallback policy')
    selected=[];fallback=[]
    for name,module in model.named_modules():
        stage=select(name,module)
        if not stage or not isinstance(module,nn.Linear):continue
        try:replacement=DynamicFP8Linear(module)
        except UnsupportedFP8 as error:
            if unsupported=='error':raise
            fallback.append(dict(name=name,stage=stage,reason=str(error)))
            continue
        parent_name,_,leaf=name.rpartition('.')
        parent=model.get_submodule(parent_name) if parent_name else model
        selected.append((parent,leaf,module,replacement,stage,name))
    if not selected and not fallback:raise ValueError('No projections matched the FP8 selection')
    installed=[]
    try:
        for parent,leaf,original,replacement,stage,name in selected:
            setattr(parent,leaf,replacement);installed.append((parent,leaf,original))
    except BaseException:
        for parent,leaf,original in reversed(installed):setattr(parent,leaf,original)
        raise
    return dict(implementation='dynamic-per-tensor-native-fp8-v1',experimental=True,
                activation_calibration=False,activation_scaling='per current projection tensor',
                weight_scaling='per projection tensor',fallback=fallback,
                sites=[dict(name=name,stage=stage) for *_,stage,name in selected],
                row_alignment=16,quantize_and_padding_in_timed_forward=True,quality_qualified=False)

