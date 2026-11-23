"""FP16 expert projections on an FP32 residual stream for the explicit mixed recipe."""
import torch

from .norm_kernel import ada_after_mean


def conditioned_input(norm, residual, step):
    shape = residual.shape
    x = residual.reshape(1, -1, shape[-1])
    variance = x.square().mean(dim=-1, keepdim=True)
    scale, bias = norm.modulations[step]
    value = ada_after_mean(x, variance, norm.weight, scale, bias, norm.variance_epsilon,
                           output_dtype=torch.float16)
    return value.reshape(shape)


def action_layers(engine, residual, step):
    if residual.dtype != torch.float32:
        raise ValueError("The FP16 action expert requires a persistent FP32 residual")
    for i, layer in enumerate(engine.action.layers):
        x = conditioned_input(layer.input_layernorm, residual, step)
        attn = layer.self_attn
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q, k = engine._rope(attn.q_proj(x).view(shape), attn.k_proj(x).view(shape), "suffix")
        value = attn.v_proj(x).view(shape)
        attended = engine._attention(q, k, value, i, False)
        residual = residual + attn.o_proj(attended).float()
        x = conditioned_input(layer.post_attention_layernorm, residual, step)
        update, _ = layer.mlp(x)
        residual = residual + update.float()
    norm = engine.action.norm
    variance = residual.square().mean(dim=-1, keepdim=True)
    return norm.weight * (residual * torch.rsqrt(variance + norm.variance_epsilon))
