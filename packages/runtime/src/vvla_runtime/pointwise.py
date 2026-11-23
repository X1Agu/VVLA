"""Opt-in pointwise fusions with the reference rounding boundaries."""
import torch


def rotary_pos_emb(q, k, cos, sin):
    """BHSD Q/K, BSD cos/sin; return fresh BHSD views with contiguous BSHD storage."""
    values = [k, cos, sin] + ([] if q is None else [q])
    if k.ndim != 4 or k.shape[-1] < 2 or k.shape[-1] % 2:
        raise ValueError("RoPE requires BHSD with an even head dimension")
    b, _, s, d = k.shape
    if cos.shape != (b, s, d) or sin.shape != cos.shape:
        raise ValueError("RoPE cos/sin must have the exact BSD shape")
    if q is not None and (q.ndim != 4 or q.shape[0] != b or q.shape[2:] != (s, d)):
        raise ValueError("RoPE Q/K geometry mismatch")
    if k.dtype not in (torch.bfloat16, torch.float32) or any(t.dtype != k.dtype or t.device != k.device for t in values):
        raise ValueError("RoPE requires matching BF16/FP32 dtype and device")
    if torch.is_grad_enabled():
        raise RuntimeError("Fused RoPE requires no_grad/inference_mode")
    def output(t):
        return torch.empty((b, s, t.shape[1], d), dtype=t.dtype, device=t.device).transpose(1, 2)
    ko, qo = output(k), None if q is None else output(q)
    if not k.is_cuda:
        def rotate(t):
            half = d // 2
            return t * cos[:, None] + torch.cat((-t[..., half:], t[..., :half]), -1) * sin[:, None]
        ko.copy_(rotate(k))
        if q is not None:
            qo.copy_(rotate(q))
        return qo, ko
    import triton
    from vvla_runtime.pointwise_kernels import rope_kernel
    hq = 0 if q is None else q.shape[1]
    qarg = k if q is None else q
    qoarg = ko if qo is None else qo
    nq, nk = b * s * hq * d, k.numel()
    if nq + nk:
        rope_kernel[(triton.cdiv(nq + nk, 1024),)](
            qarg, k, cos, sin, qoarg, ko, nq, nk, s, hq, k.shape[1], d,
            *qarg.stride(), *k.stride(), *cos.stride(), *sin.stride(),
            BLOCK=1024, enable_fp_fusion=False,
        )
    return qo, ko


def residual_square(residual, update):
    """Return reference FP32 residual sum and its FP32 square, without reduction."""
    if (residual.shape != update.shape or residual.dtype != torch.float32
            or update.dtype not in (torch.float32, torch.bfloat16)
            or residual.device != update.device or not residual.is_contiguous()
            or not update.is_contiguous() or residual.ndim < 2):
        raise ValueError("Residual-square requires equal contiguous FP32 residual and BF16/FP32 update")
    if torch.is_grad_enabled():
        raise RuntimeError("Residual-square fusion requires no_grad/inference_mode")
    if not residual.is_cuda:
        result = residual + update
        return result, result.pow(2)
    import triton
    from vvla_runtime.pointwise_kernels import residual_square_kernel
    result, squared = torch.empty_like(residual), torch.empty_like(residual)
    if residual.numel():
        residual_square_kernel[(triton.cdiv(residual.numel(), 1024),)](
            residual, update, result, squared, residual.numel(), BLOCK=1024, enable_fp_fusion=False)
    return result, squared
