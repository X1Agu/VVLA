"""BF16 input/output RoPE with original FP32 multiply/add semantics."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X, C, S, Y, B:tl.constexpr, L:tl.constexpr, H:tl.constexpr, D:tl.constexpr,
          X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,X3:tl.constexpr,
          C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,
          S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<B*L*H*D
    d=i%D; h=i//D%H; l=i//(D*H)%L; b=i//(D*H*L)
    off=b*X0+l*X1+h*X2
    x=tl.load(X+off+d*X3,valid,0).to(tl.float32)
    r=tl.load(X+off+((d+D//2)%D)*X3,valid,0).to(tl.float32)
    r=tl.where(d<D//2,-r,r)
    c=tl.load(C+b*C0+l*C1+d*C2,valid,0).to(tl.float32)
    s=tl.load(S+b*S0+l*S1+d*S2,valid,0).to(tl.float32)
    # Do not round products to BF16; do not contract into FMA.
    y=x*c+r*s
    tl.store(Y+i,y,valid)


def fused_rope(q,k,cos,sin):
    if q.ndim!=4 or k.ndim!=4 or q.dtype not in (torch.bfloat16,torch.float16) or k.dtype!=q.dtype:
        raise ValueError('Matching 16-bit B,L,H,D inputs required')
    if q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1] or q.shape[-1]%2:
        raise ValueError('Invalid rotary geometry')
    if cos.shape!=(q.shape[0],q.shape[1],q.shape[-1]) or sin.shape!=cos.shape:
        raise ValueError('Expected per-row B,L,D rotary values')
    outputs=[]
    for x in (q,k):
        y=torch.empty(x.shape,dtype=x.dtype,device=x.device)
        _rope[(triton.cdiv(x.numel(),1024),)](x,cos,sin,y,*x.shape,*x.stride(),*cos.stride(),*sin.stride(),BLOCK=1024,enable_fp_fusion=False)
        outputs.append(y)
    return tuple(outputs)
