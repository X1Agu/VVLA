"""BF16 matmul candidate for actual-shape tuning; FP32 accumulation."""
import torch
import triton
import triton.language as tl
@triton.jit
def _mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SA0:tl.constexpr,SA1:tl.constexpr,SB0:tl.constexpr,SB1:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    accum=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        ks=block*BK+rk
        aa=tl.load(A+rm[:,None]*SA0+ks[None,:]*SA1,(rm[:,None]<M)&(ks[None,:]<K),0)
        bb=tl.load(B+ks[:,None]*SB0+rn[None,:]*SB1,(ks[:,None]<K)&(rn[None,:]<N),0)
        accum=tl.dot(aa,bb,accum)
    tl.store(C+rm[:,None]*N+rn[None,:],accum,(rm[:,None]<M)&(rn[None,:]<N))
def matmul(x,weight,config):
    m,k=x.shape;n=weight.shape[0];b=weight.t();out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    bm,bn,bk,warps,stages=config
    _mm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,b,out,m,n,k,*x.stride(),*b.stride(),bm,bn,bk,num_warps=warps,num_stages=stages)
    return out
