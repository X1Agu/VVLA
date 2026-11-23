"""Opt-in SigLIP residual/cast fusion; native FP32 LayerNorm is unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


def residual_bf16_fp32(residual, update):
    if residual.shape != update.shape or residual.dtype != torch.bfloat16 or update.dtype != torch.bfloat16:
        raise ValueError("Vision residual fusion requires equal BF16 shapes")
    if residual.device != update.device or not 1 <= residual.ndim <= 3:
        raise ValueError("Vision residual fusion requires 1D/2D/3D tensors on one device")
    if not residual.is_cuda:
        rounded = (residual + update).contiguous()
        return rounded, rounded.float()
    import triton
    from vvla_repro_pi05.models.pi05.vision_kernels_pi05 import residual_cast_kernel
    # Patch embeddings may have [B, tokens, width] strides [S*D, 1, S].
    # Fuse this layout conversion too; native LayerNorm sees the same row values.
    rounded = torch.empty(residual.shape,device=residual.device,dtype=torch.bfloat16)
    fp32 = torch.empty(residual.shape,device=residual.device,dtype=torch.float32)
    def strides(t):
        return ((0,0,*t.stride()) if t.ndim==1 else (0,*t.stride()) if t.ndim==2 else t.stride())
    if residual.numel():
        residual_cast_kernel[(triton.cdiv(residual.numel(), 1024),)](
            residual, update, rounded, fp32, residual.numel(), residual.shape[-1],
            residual.shape[-2] if residual.ndim>=2 else 1, *strides(residual), *strides(update),
            BLOCK=1024, enable_fp_fusion=False,
        )
    return rounded, fp32


class VisionResidualCastLayer(nn.Module):
    """Matches transformers 5.19 SiglipEncoderLayer, preserving checkpoint paths.

    The BF16 residual must be rounded BEFORE promotion to FP32 for norm2.
    Native layer_norm retains its reduction, affine and output rounding order.
    """
    @staticmethod
    def preflight(original):
        cls = type(original)
        if cls.__name__ != "SiglipEncoderLayer" or cls.__module__ != "transformers.models.siglip.modeling_siglip":
            raise ValueError("Vision fusion supports the validated SiglipEncoderLayer only")
        for name in ("layer_norm1", "layer_norm2"):
            norm = getattr(original, name)
            if type(norm).__name__ != "_Pi05VisionLayerNormFP32":
                raise ValueError("Vision fusion requires the existing explicit FP32 LayerNorm layout")
            if norm.weight is None or norm.bias is None or norm.weight.dtype != torch.float32 or norm.bias.dtype != torch.float32:
                raise ValueError("Vision norm affine parameters must remain FP32")

    def __init__(self, original):
        super().__init__()
        self.preflight(original)
        self.embed_dim = original.embed_dim
        for name in ("layer_norm1", "self_attn", "layer_norm2", "mlp"):
            setattr(self, name, getattr(original, name))
        self._structure = self._token()
        self.train(original.training)

    def _token(self):
        return (tuple(id(getattr(self, name)) for name in ("layer_norm1", "self_attn", "layer_norm2", "mlp")),
                tuple((norm.eps, norm.normalized_shape, norm.weight.dtype, norm.bias.dtype)
                      for norm in (self.layer_norm1, self.layer_norm2)))

    def assert_fresh(self):
        if self.training or self._structure != self._token():
            raise RuntimeError("Vision layer structure changed; rebuild execution recipe and graphs")

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Vision residual fusion requires eval and inference_mode")
        update, _ = self.self_attn(hidden_states=self.layer_norm1(hidden_states), attention_mask=attention_mask, **kwargs)
        residual, fp32 = residual_bf16_fp32(hidden_states, update)
        norm = self.layer_norm2
        normalized = F.layer_norm(fp32, norm.normalized_shape, norm.weight, norm.bias, norm.eps).to(torch.bfloat16)
        return residual + self.mlp(normalized)


def selected_vision_layers(model):
    if not getattr(model.config, "vision_bf16", False):
        raise ValueError("Vision residual fusion requires the explicit vision_bf16 layout")
    tower = model.paligemma_with_expert.paligemma.model.vision_tower
    layers = tower.encoder.layers
    if not len(layers):
        raise ValueError("No SigLIP encoder layers found")
    for layer in layers:
        VisionResidualCastLayer.preflight(layer)
    return layers
