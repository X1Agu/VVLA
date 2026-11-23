"""FP16 text projections and FP32 residuals for the opt-in LingBot prefix recipe."""
import torch
from torch.nn import functional as F


def normalized(norm, hidden):
    values = hidden.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    return norm.weight * (values * torch.rsqrt(variance + norm.variance_epsilon))


def prefix_layers(engine, hidden, deep):
    residual = hidden.float()
    for i, layer in enumerate(engine.vlm.layers):
        x = normalized(layer.input_layernorm, residual).half()
        attn = layer.self_attn
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = normalized(attn.q_norm, attn.q_proj(x).view(shape)).half()
        k = normalized(attn.k_norm, attn.k_proj(x).view(shape)).half()
        q, k = engine._rope(q, k, "prefix")
        v = attn.v_proj(x).view(shape)
        attended = engine._attention(q, k, v, i, True)
        residual = residual + attn.o_proj(attended).float()
        x = normalized(layer.post_attention_layernorm, residual).half()
        mlp = layer.mlp
        gate = mlp.gate_proj(x).float()
        up = mlp.up_proj(x).float()
        output = mlp.down_proj((F.silu(gate) * up).half())
        residual = residual + output.float()
        if i < len(deep):
            update = residual.index_select(1, engine.visual_indices) + deep[i].float().unsqueeze(0)
            residual = residual.index_copy(1, engine.visual_indices, update)
    return normalized(engine.vlm.norm, residual)
