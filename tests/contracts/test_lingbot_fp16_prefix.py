import types

import torch
from torch import nn

from vvla_lingbot.fp16_prefix import normalized, prefix_layers


def test_text_residual_retains_fp32_value_across_attention_and_deep_visual_update():
    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0123, .9654, 1.0372, .9956]))
            self.variance_epsilon = 1e-5

    def linear(scale):
        layer = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4)*scale)
        return layer

    norm = Norm()
    attn = types.SimpleNamespace(q_proj=linear(.32), k_proj=linear(.18), v_proj=linear(.15), o_proj=linear(.19),
                                 q_norm=Norm(), k_norm=Norm(), head_dim=4)
    mlp=types.SimpleNamespace(gate_proj=linear(.14),up_proj=linear(-.17),down_proj=linear(.23))
    layer=types.SimpleNamespace(input_layernorm=norm, post_attention_layernorm=norm, self_attn=attn, mlp=mlp)
    engine=types.SimpleNamespace(vlm=types.SimpleNamespace(layers=[layer],norm=norm),visual_indices=torch.tensor([1]))
    engine._rope=lambda q,k,label:(q,k)
    def attention(q,k,v,idx,prefix):
        assert q.dtype==k.dtype==v.dtype==torch.float16 and idx==0 and prefix
        return v.flatten(-2)
    engine._attention=attention
    start=torch.tensor([[[.123,.245,.367,.489],[.234,.346,.458,.561]]],dtype=torch.bfloat16)
    deep=torch.tensor([[.00251,.00462,-.00374,-.0258]],dtype=torch.bfloat16)
    x=normalized(norm,start.float()).half()
    expected=start.float()+attn.o_proj(attn.v_proj(x)).float()
    x=normalized(norm,expected).half()
    m=mlp.down_proj((torch.nn.functional.silu(mlp.gate_proj(x).float())*mlp.up_proj(x).float()).half())
    expected=expected+m.float()
    expected[:,1]+=deep.float()
    with torch.no_grad():
        actual=prefix_layers(engine,start,[deep])
    assert actual.dtype==torch.float32 and torch.equal(actual,normalized(norm,expected))
    assert not torch.equal(actual,actual.half().float())
