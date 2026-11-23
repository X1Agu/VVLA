"""No reduction rewrite and no reassociation across BF16 rounding boundaries."""
import torch
import triton
import triton.language as tl

@triton.jit
def _residual_rmsnorm(R,U,W,Y,Z,WIDTH:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    offset = row * WIDTH + cols
    residual = tl.load(R+offset, valid, 0).to(tl.float32)
    update = tl.load(U+offset, valid, 0).to(tl.float32)
    value = (residual+update).to(tl.bfloat16).to(tl.float32)
    mean_square = tl.sum(value*value, axis=0) / WIDTH
    norm = (value * tl.rsqrt(mean_square+EPS)).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W+cols, valid, 0).to(tl.float32)
    tl.store(Y+offset, value, valid)
    tl.store(Z+offset, norm*weight, valid)

def residual_rmsnorm(residual, update, weight, eps):
    width = residual.shape[-1]
    value, normalized = torch.empty_like(residual), torch.empty_like(residual)
    _residual_rmsnorm[(residual.numel()//width,)](
        residual, update, weight, value, normalized, width, eps,
        triton.next_power_of_2(width), num_warps=8, enable_fp_fusion=False)
    return value, normalized

@triton.jit
def _after_mean(X, V, W, Y, WIDTH:tl.constexpr, EPS:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<WIDTH
    x=tl.load(X+row*WIDTH+col,valid,0).to(tl.float32)
    v=tl.load(V+row).to(tl.float32)
    weight=tl.load(W+col,valid,0).to(tl.float32)
    inv=tl.rsqrt(v+EPS)
    normalized=(x*inv).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row*WIDTH+col,normalized*weight,valid)

@triton.jit
def _residual_square(R,U,Y,S,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    r=tl.load(R+i,valid,0).to(tl.float32);u=tl.load(U+i,valid,0).to(tl.float32)
    value=(r+u).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,value,valid);tl.store(S+i,value*value,valid)

def after_mean(x,variance,weight,eps):
    y=torch.empty_like(x);width=x.shape[-1]
    _after_mean[(x.numel()//width,)](x,variance,weight,y,width,eps,triton.next_power_of_2(width),enable_fp_fusion=False)
    return y

def residual_square(residual,update):
    y=torch.empty_like(residual);squared=torch.empty_like(residual,dtype=torch.float32)
    _residual_square[(triton.cdiv(residual.numel(),1024),)](residual,update,y,squared,residual.numel(),1024,enable_fp_fusion=False)
    return y,squared
