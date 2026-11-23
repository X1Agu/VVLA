"""The checkpoint's original mixed BF16 layout, independent of Omni loading."""

import torch
from torch import nn


class _Pi05VisionLayerNormFP32(nn.LayerNorm):
    """Retain the shared BF16 fusion preflight's explicit-layout type contract."""
    def forward(self, x):
        out = torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(), self.eps,
        )
        return out.to(dtype=x.dtype)


# Backward-compatible public spelling; instances keep the shared class name.
VisionLayerNormFP32 = _Pi05VisionLayerNormFP32


def _vision_norms_fp32(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            replacement = _Pi05VisionLayerNormFP32(
                child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine,
                bias=child.bias is not None, device=None if child.weight is None else child.weight.device,
                dtype=torch.float32,
            )
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            _vision_norms_fp32(child)


def set_bfloat16_layout(model):
    """Run before loading original weights: conversion back from BF16 is lossy."""
    fp32_names = ("vision_tower", "multi_modal_projector", "input_layernorm", "post_attention_layernorm", "model.norm")
    inner = model.paligemma_with_expert
    for name, param in inner.named_parameters():
        dtype = torch.float32 if any(part in name for part in fp32_names) else torch.bfloat16
        param.data = param.data.to(dtype=dtype)
    for buffer in inner.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.bfloat16)
    for name, param in model.named_parameters():
        if not name.startswith("paligemma_with_expert."):
            param.data = param.data.to(dtype=torch.float32)
    for name, buffer in model.named_buffers():
        if not name.startswith("paligemma_with_expert.") and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.float32)
    inner.vision_output_dtype = None
    model._configured_vision_bf16 = model.config.vision_bf16
    if model.config.vision_bf16:
        vision = inner.paligemma.model.vision_tower
        vision.to(dtype=torch.bfloat16)
        _vision_norms_fp32(vision)
        inner.paligemma.model.multi_modal_projector.to(dtype=torch.bfloat16)
        inner.vision_output_dtype = torch.float32
    model.invalidate_inference_caches()
