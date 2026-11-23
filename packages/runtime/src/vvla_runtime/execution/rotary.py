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
    from vvla_runtime.execution.rotary_kernels import rope_kernel
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

