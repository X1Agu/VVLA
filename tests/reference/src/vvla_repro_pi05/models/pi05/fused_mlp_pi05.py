# SPDX-License-Identifier: Apache-2.0
"""Two-input GELU(tanh) × up; preserves the intermediate BF16 rounding."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu_mul(G, U, O, N:tl.constexpr, WIDTH:tl.constexpr, GROW:tl.constexpr, UROW:tl.constexpr, BLOCK:tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, col = offsets//WIDTH, offsets%WIDTH
    g = tl.load(G+row*GROW+col, offsets<N, 0).to(tl.float32)
    u = tl.load(U+row*UROW+col, offsets<N, 0).to(tl.float32)
    arg = 0.7978845608028654*(g + 0.044715*g*g*g)
    activated = (0.5*g*(1.0 + libdevice.tanh(arg))).to(O.dtype.element_ty).to(tl.float32)
    # Written explicitly to retain PyTorch's rounded GELU intermediate.
    tl.store(O+offsets, activated*u, offsets<N)


def gelu_tanh_mul(gate, up):
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("Fused GELU requires matching shapes/dtypes and contiguous columns")
    width = gate.shape[-1]
    # reshape is a view for ordinary contiguous and split gate/up GEMM outputs.
    g, u = gate.reshape(-1,width), up.reshape(-1,width)
    output = torch.empty(gate.shape,device=gate.device,dtype=gate.dtype)
    _gelu_mul[(triton.cdiv(gate.numel(),1024),)](g,u,output,gate.numel(),width,g.stride(0),u.stride(0),1024,enable_fp_fusion=False)
    return output
