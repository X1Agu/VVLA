"""New true-B1/2/4/8/16/32 primitives preserving source497 FP8 arithmetic.

This is a new backend identity. Historical B1/B32 evidence does not qualify the
intermediate geometries or the current physical GPU. The P0 backend probe must
execute these actual operators before a serving or quality claim is made.
"""
import hashlib
from pathlib import Path

import torch

from .batch_prefix_fp8_native import (ARITHMETIC, COMPILE_OPTIONS, REQUIRED_RUNTIME,
                                    SUPPORTED_GEOMETRIES, validate_runtime)

BATCH_SIZES=(1,2,4,8,16,32)
PER_SLOT_CAPACITIES=(576,640)
SUPPORTED_CAPACITIES=tuple(sorted({b*p for b in BATCH_SIZES for p in PER_SLOT_CAPACITIES}))
VERSION='pi05-source497-dynamic-prefix-fp8-sm120-v1'


def implementation_manifest():
    root=Path(__file__).resolve().parent
    names=('dynamic_prefix_fp8_native.py','batch_prefix_fp8_native.py','batch_prefix_fp8_native_kernel.py')
    return dict(version=VERSION,batch_sizes=list(BATCH_SIZES),per_slot_capacities=list(PER_SLOT_CAPACITIES),
        flattened_rows=list(SUPPORTED_CAPACITIES),geometries=list(SUPPORTED_GEOMETRIES),
        runtime=REQUIRED_RUNTIME,arithmetic=ARITHMETIC,compile_options=COMPILE_OPTIONS,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        inherited_evidence=False,quality_passed=False)


def validate_quantizer_input(x):
    if (x.device.type!='cuda' or x.dtype!=torch.bfloat16 or x.ndim!=2
            or x.shape[0] not in SUPPORTED_CAPACITIES or x.shape[1] not in (2048,16384)
            or x.stride()!=(x.shape[1],1) or x.requires_grad):
        raise ValueError('Dynamic prefix requires its explicit true-batch CUDA BF16 row geometry')


def validate_mm(qx,qw,sx,sw):
    if any(t.device!=qx.device or t.device.type!='cuda' or t.ndim!=2 or t.requires_grad
           for t in (qx,qw,sx,sw)):
        raise ValueError('Dynamic FP8 operands must be same-device inference CUDA matrices')
    m,k=qx.shape;n=qw.shape[0]
    if (m not in SUPPORTED_CAPACITIES or (k,n) not in SUPPORTED_GEOMETRIES
            or qw.shape[1]!=k or qx.dtype!=torch.float8_e4m3fn or qw.dtype!=qx.dtype
            or sx.dtype!=torch.float32 or sw.dtype!=torch.float32
            or qx.stride()!=(k,1) or qw.stride()!=(k,1)
            or sx.shape!=(m,1) or sw.shape!=(n,1) or sx.stride()!=(1,1) or sw.stride()!=(1,1)):
        raise ValueError('Dynamic prefix FP8 geometry/dtype/scale layout is unsupported')


@torch.library.custom_op('vvla_repro_pi05_lite::dynamic_prefix_fp8_quantize_v1',mutates_args=())
def quantize_rows(x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    validate_quantizer_input(x);validate_runtime(x.device)
    from .batch_prefix_fp8_native_kernel import launch
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scale=torch.empty((x.shape[0],1),device=x.device,dtype=torch.float32)
    launch(x,q,scale)
    return q,scale


@quantize_rows.register_fake
def _fake_quantize(x):
    validate_quantizer_input(x)
    return torch.empty_like(x,dtype=torch.float8_e4m3fn),torch.empty((x.shape[0],1),device=x.device,dtype=torch.float32)


@torch.library.custom_op('vvla_repro_pi05_lite::dynamic_prefix_fp8_mm_v1',mutates_args=())
def native_scaled_mm(qx:torch.Tensor,qw:torch.Tensor,sx:torch.Tensor,sw:torch.Tensor)->torch.Tensor:
    validate_mm(qx,qw,sx,sw);validate_runtime(qx.device)
    return torch._scaled_mm(qx,qw.T,sx,sw.T,bias=None,scale_result=None,
        out_dtype=torch.bfloat16,use_fast_accum=False)


@native_scaled_mm.register_fake
def _fake_mm(qx,qw,sx,sw):
    validate_mm(qx,qw,sx,sw)
    return torch.empty((qx.shape[0],qw.shape[0]),device=qx.device,dtype=torch.bfloat16)


def scaled_linear(x,qw,sw):
    qx,sx=quantize_rows(x)
    return native_scaled_mm(qx,qw,sx,sw),qx,sx
