"""Llama-specific BF16 RMSNorm: retain torch FP32 mean and both roundings."""
import math
import torch
from torch import nn

def residual_rmsnorm_bf16(residual, update, module):
    """Return the rounded residual and Llama RMSNorm output without a squared-state tensor."""
    if (torch.is_grad_enabled() or residual.dtype != torch.bfloat16 or update.dtype != residual.dtype
            or residual.shape != update.shape or residual.device != update.device
            or not residual.is_contiguous() or not update.is_contiguous()
            or module.weight.shape != residual.shape[-1:] or module.weight.dtype != residual.dtype
            or module.weight.device != residual.device or not module.weight.is_contiguous()):
        raise ValueError("Fused residual RMSNorm requires matching contiguous BF16 tensors")
    if not residual.is_cuda:
        value = residual + update
        return value, module(value)
    from .llama_norm_kernels import residual_rmsnorm
    return residual_rmsnorm(residual, update, module.weight, module.variance_epsilon)

class LlamaRMSNormExecution(nn.Module):
    def __init__(self, original, *, fuse_after_mean=False):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon
        self.fuse_after_mean = fuse_after_mean
        self._installed = (fuse_after_mean, tuple(self.weight.shape))
        self.eval()
        self.assert_fresh()

    def assert_fresh(self):
        if (self.fuse_after_mean, tuple(self.weight.shape)) != self._installed:
            raise RuntimeError('Llama norm execution changed; rebuild adapter and retire graphs')
        if type(self.fuse_after_mean) is not bool or len(self.weight.shape) != 1:
            raise ValueError('Invalid Llama norm configuration')
        if not math.isfinite(self.variance_epsilon) or self.variance_epsilon <= 0:
            raise ValueError('Llama epsilon must be finite and positive')

    def forward(self, x, squared=None):
        if torch.is_grad_enabled():raise RuntimeError('Inference-only Llama norm')
        if x.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16:
            raise ValueError('Llama fusion requires BF16 activations and weights')
        if not x.is_contiguous() or not self.weight.is_contiguous() or x.shape[-1] != self.weight.numel() or self.weight.device != x.device:
            raise ValueError('Invalid Llama norm geometry or device')
        if squared is None:
            squared = x.float().pow(2)
        elif squared.shape != x.shape or squared.dtype != torch.float32 or squared.device != x.device or not squared.is_contiguous():
            raise ValueError('Invalid request-local squared residual')
        # Deliberately keep the upstream reduction, its input shape and dtype.
        variance = squared.mean(-1, keepdim=True)
        if not self.fuse_after_mean or not x.is_cuda:
            normalized = x.float() * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * normalized.to(x.dtype)
        from .llama_norm_kernels import after_mean
        return after_mean(x, variance, self.weight, self.variance_epsilon)

def residual_square_bf16(residual, update):
    if torch.is_grad_enabled():raise RuntimeError('Inference-only residual fusion')
    if residual.dtype != torch.bfloat16 or update.dtype != torch.bfloat16 or residual.shape != update.shape or residual.device != update.device or not residual.is_contiguous() or not update.is_contiguous():
        raise ValueError('Residual fusion requires matching contiguous BF16 tensors')
    if not residual.is_cuda:
        value = residual + update
        return value, value.float().pow(2)
    from .llama_norm_kernels import residual_square
    return residual_square(residual, update)
