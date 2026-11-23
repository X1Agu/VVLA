"""Current-tensor FP8 scaling with two launches and no padded FP32 tensor.

A global scale needs a reduction across CTAs. The first launch writes tile
maxima; the second reduces those maxima while fusing division, padding and FP8
conversion. No global spin barrier or persistent workspace is used.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _direct_e4m3(value):
    bits = tl.inline_asm_elementwise(
        """{ .reg .b16 packed; .reg .b32 extended;
        cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1;
        cvt.u32.u16 extended, packed; and.b32 $0, extended, 255; }""",
        constraints='=r,f', args=[value], dtype=tl.uint32, is_pure=True, pack=1)
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _partial_amax(X, P, N: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X+indices, indices<N, other=0).to(tl.float32)
    tl.store(P+tl.program_id(0), tl.max(tl.abs(values), 0))


@triton.jit
def _gelu_partial(G, U, H, P, N: tl.constexpr, WIDTH: tl.constexpr,
                   GROW: tl.constexpr, UROW: tl.constexpr, BLOCK: tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,col=offsets//WIDTH,offsets%WIDTH
    gate=tl.load(G+row*GROW+col,offsets<N,other=0).to(tl.float32)
    up=tl.load(U+row*UROW+col,offsets<N,other=0).to(tl.float32)
    # Same operation order and both BF16 rounding boundaries as pi05's
    # historical fused_mlp_pi05._gelu_mul; scale sees the rounded product.
    arg=0.7978845608028654*(gate+0.044715*gate*gate*gate)
    activated=(0.5*gate*(1.0+libdevice.tanh(arg))).to(tl.bfloat16).to(tl.float32)
    hidden=(activated*up).to(tl.bfloat16)
    tl.store(H+offsets,hidden,offsets<N)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(hidden.to(tl.float32)),0))


@triton.jit
def _quantize_pad(X, P, Q, S, N: tl.constexpr, PADDED: tl.constexpr,
                  PARTS: tl.constexpr, REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    partial = tl.arange(0, REDUCE)
    maximum = tl.max(tl.load(P+partial, partial<PARTS, other=0), 0)
    # Preserve the per-tensor reference's clamp-before-division boundary.
    scale = tl.maximum(maximum, 1.0e-12) * 0.0022321429569274187
    indices = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X+indices, indices<N, other=0).to(tl.float32)
    value = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q+indices, _direct_e4m3(value), indices<PADDED)
    if tl.program_id(0)==0:
        tl.store(S, scale)


def launch(value, quantized, scale, partial):
    parts = partial.numel()
    _partial_amax[(parts,)](value, partial, value.numel(), BLOCK=4096,
                            num_warps=4, enable_fp_fusion=False)
    _quantize_pad[(triton.cdiv(quantized.numel(), 2048),)](
        value, partial, quantized, scale, value.numel(), quantized.numel(),
        PARTS=parts, REDUCE=triton.next_power_of_2(parts), BLOCK=2048,
        num_warps=4, enable_fp_fusion=False)


def launch_gelu(gate,up,hidden,quantized,scale,partial):
    parts=partial.numel()
    _gelu_partial[(parts,)](gate,up,hidden,partial,gate.numel(),gate.shape[1],
        gate.stride(0),up.stride(0),BLOCK=4096,num_warps=4,enable_fp_fusion=False)
    _quantize_pad[(triton.cdiv(quantized.numel(),2048),)](
        hidden,partial,quantized,scale,hidden.numel(),quantized.numel(),
        PARTS=parts,REDUCE=triton.next_power_of_2(parts),BLOCK=2048,
        num_warps=4,enable_fp_fusion=False)
