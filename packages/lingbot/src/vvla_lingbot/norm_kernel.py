"""Existing after-mean RMSNorm fusion adapted to official AdaRMSNorm arithmetic."""
import torch
import triton
import triton.language as tl
@triton.jit
def _ada(X,V,W,G,B,Y,H:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);c=tl.arange(0,BLOCK);valid=c<H
 x=tl.load(X+row*H+c,valid,0).to(tl.float32)
 var=tl.load(V+row).to(tl.float32)
 w=tl.load(W+c,valid,0).to(tl.float32)
 g=tl.load(G+c,valid,0).to(tl.float32);b=tl.load(B+c,valid,0).to(tl.float32)
 n=x*tl.rsqrt(var+EPS)
 n=w*n
 n=(1.+g)*n+b
 tl.store(Y+row*H+c,n,valid)
def ada_after_mean(x,var,weight,gamma,beta,eps,*,output_dtype=None):
 assert x.shape[0]==1 and x.is_contiguous()
 if output_dtype is None:
  assert x.dtype==torch.bfloat16
  output_dtype=x.dtype
 else:
  assert x.dtype==torch.float32 and output_dtype in (torch.float16,torch.float32)
 h=x.shape[-1];y=torch.empty_like(x,dtype=output_dtype)
 _ada[(x.numel()//h,)](x,var,weight,gamma,beta,y,h,eps,triton.next_power_of_2(h),enable_fp_fusion=False)
 return y
