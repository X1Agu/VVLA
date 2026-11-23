"""Pinned native vLLM CUDA operators, independent of its model-loader dependencies.

Uses the official vLLM wheel and registered CUDA operators. The small allocation
wrappers mirror vLLM 0.32.0 _custom_ops; no replacement/fake GEMM implementation.
"""
import importlib,os,sys
from pathlib import Path
import torch

_loaded=False
_flash=None

def load_native():
    global _loaded,_flash
    if _loaded:return
    configured=os.environ.get('VLA_VLLM_NATIVE_PATH')
    if configured:
        root=Path(configured).expanduser().resolve(strict=True)
        if str(root) not in sys.path:sys.path.insert(0,str(root))
    import vllm
    if vllm.__version__!='0.32.0':raise RuntimeError('Native kernel ABI requires pinned vLLM 0.32.0')
    importlib.import_module('vllm._C_stable_libtorch')
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    _flash=flash_attn_varlen_func
    _loaded=True

def flash_attention(q,k,v,*,cu_q,cu_k,max_q,max_k,block_table=None,lengths=None):
    if not _loaded:raise RuntimeError('Initialize native backend outside capture')
    return _flash(q,k,v,max_seqlen_q=max_q,cu_seqlens_q=cu_q,max_seqlen_k=max_k,
                  cu_seqlens_k=cu_k if block_table is None else None,
                  seqused_k=lengths if block_table is not None else None,
                  block_table=block_table,causal=False,softmax_scale=q.shape[-1]**-.5,fa_version=2)

def cache_write(k,v,k_cache,v_cache,slots,scale):
    torch.ops._C_cache_ops.reshape_and_cache_flash(k,v,k_cache,v_cache,slots,'auto',scale,scale)

def quantize_fp8(x):
    if x.ndim!=2 or not x.is_cuda:raise ValueError('Native FP8 requires a CUDA matrix')
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scale=torch.empty((x.shape[0],1),dtype=torch.float32,device=x.device)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(q,x,scale,None)
    return q,scale

def fp8_mm(a,b,sa,sb):
    if b.shape[0]%16 or b.shape[1]%16:raise ValueError('CUTLASS shape alignment required')
    out=torch.empty((a.shape[0],b.shape[1]),dtype=torch.bfloat16,device=a.device)
    torch.ops._C.cutlass_scaled_mm(out,a,b,sa,sb,None)
    return out

def quantize_fp4(x,global_scale):
    if x.ndim!=2 or x.shape[1]%16 or x.dtype!=torch.bfloat16:raise ValueError('NVFP4 requires aligned BF16 rows')
    m,k=x.shape
    q=torch.empty((m,k//2),dtype=torch.uint8,device=x.device)
    scales=torch.empty((((m+127)//128)*128,((k//16+3)//4)),dtype=torch.int32,device=x.device)
    torch.ops._C.scaled_fp4_quant.out(x,global_scale,True,output=q,output_scale=scales)
    return q,scales.view(torch.float8_e4m3fn)

def fp4_mm(a,b,sa,sb,alpha):
    out=torch.empty((a.shape[0],b.shape[0]),dtype=torch.bfloat16,device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out,a,b,sa,sb,alpha)
    return out
