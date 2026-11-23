"""SiLU x up with the two original BF16 rounding boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _packed_silu_mul(GU, OUT, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(GU + row*2*WIDTH + cols, cols < WIDTH, 0).to(tl.float32)
    up = tl.load(GU + row*2*WIDTH + WIDTH + cols, cols < WIDTH, 0).to(tl.float32)
    act = (gate / (1. + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row*WIDTH + cols, act*up, cols < WIDTH)

def packed_silu_mul(pair):
    if torch.is_grad_enabled() or pair.dtype != torch.bfloat16 or not pair.is_contiguous() or pair.shape[-1] % 2:
        raise ValueError("Packed SiLU requires contiguous BF16 gate/up rows in inference")
    width = pair.shape[-1] // 2
    if not pair.is_cuda:
        gate,up = pair.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    out = pair.new_empty((*pair.shape[:-1], width))
    _packed_silu_mul[(triton.cdiv(width,256), out.numel()//width)](pair,out,width,256,enable_fp_fusion=False)
    return out
@triton.jit
def _silu_mul(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<N,0).to(tl.float32);u=tl.load(U+i,i<N,0).to(tl.float32)
    activation=(g/(1.+libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,activation*u,i<N)
def silu_mul(gate,up):
    if gate.shape!=up.shape or gate.dtype!=torch.bfloat16 or up.dtype!=gate.dtype or gate.device!=up.device or not gate.is_cuda or not gate.is_contiguous() or not up.is_contiguous():raise ValueError('SiLU fusion requires contiguous CUDA BF16 pairs')
    if torch.is_grad_enabled():raise RuntimeError('SiLU fusion is inference only')
    out=torch.empty_like(gate)
    if gate.numel():_silu_mul[(triton.cdiv(gate.numel(),1024),)](gate,up,out,gate.numel(),BLOCK=1024,enable_fp_fusion=False)
    return out
