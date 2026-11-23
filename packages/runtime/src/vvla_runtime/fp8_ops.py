"""Opaque graph-compatible dynamic FP8 primitives shared by public backends."""
import torch


def _check_input(value):
    if (value.ndim!=2 or not value.is_cuda or value.dtype!=torch.bfloat16
            or not value.is_contiguous() or value.shape[0]<1 or value.shape[1]%16):
        raise ValueError('FP8 quantization requires nonempty contiguous CUDA BF16 [M,K], K aligned to 16')


@torch.library.custom_op('vvla_runtime::quantize_fp8_tensor_padded', mutates_args=())
def quantize_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _check_input(value)
    from .fp8_quantize_kernel import launch
    rows=(value.shape[0]+15)//16*16
    output=torch.empty((rows,value.shape[1]),device=value.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty((1,),device=value.device,dtype=torch.float32)
    partial=torch.empty(((value.numel()+4095)//4096,),device=value.device,dtype=torch.float32)
    launch(value,output,scale,partial)
    return output,scale


@quantize_tensor.register_fake
def _fake_quantize(value):
    _check_input(value)
    return (torch.empty(((value.shape[0]+15)//16*16,value.shape[1]),device=value.device,dtype=torch.float8_e4m3fn),
            torch.empty((1,),device=value.device,dtype=torch.float32))


def _check_gelu(gate,up):
    if (gate.ndim!=2 or gate.shape!=up.shape or gate.shape[0]<1 or gate.shape[1]%16
            or not gate.is_cuda or gate.device!=up.device or gate.dtype!=torch.bfloat16
            or up.dtype!=gate.dtype or gate.stride(1)!=1 or up.stride(1)!=1
            or gate.stride(0)<gate.shape[1] or up.stride(0)<up.shape[1]):
        raise ValueError('GELU-FP8 requires matching CUDA BF16 matrices with contiguous columns')


@torch.library.custom_op('vvla_runtime::gelu_mul_quantize_fp8', mutates_args=())
def quantize_gelu(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _check_gelu(gate,up)
    from .fp8_quantize_kernel import launch_gelu
    hidden=torch.empty(gate.shape,device=gate.device,dtype=torch.bfloat16)
    quantized=torch.empty(((gate.shape[0]+15)//16*16,gate.shape[1]),device=gate.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty((1,),device=gate.device,dtype=torch.float32)
    partial=torch.empty(((gate.numel()+4095)//4096,),device=gate.device,dtype=torch.float32)
    launch_gelu(gate,up,hidden,quantized,scale,partial)
    return quantized,scale


@quantize_gelu.register_fake
def _fake_gelu(gate,up):
    _check_gelu(gate,up)
    return (torch.empty(((gate.shape[0]+15)//16*16,gate.shape[1]),device=gate.device,dtype=torch.float8_e4m3fn),
            torch.empty((1,),device=gate.device,dtype=torch.float32))


@torch.library.custom_op('vvla_runtime::scaled_fp8_mm', mutates_args=())
def scaled_mm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    return torch._scaled_mm(a,b,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16,use_fast_accum=False)


@scaled_mm.register_fake
def _fake_mm(a,b,sa,sb):
    return torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=torch.bfloat16)
