import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vvla_lingbot.precision_weights import HEADS, restore_action_precision


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(("half_expert", "half_text"), [(False, False), (True, False), (True, True)])
def test_selective_restore_recovers_original_f32_for_parameters_and_buffers(tmp_path, half_expert, half_text, indexed):
    flow = nn.Module()
    for attr in HEADS:
        setattr(flow, attr, nn.Linear(2, 2))
    trunk = flow
    for attr in ("qwenvl_with_expert", "qwen_expert", "model"):
        setattr(trunk, attr, nn.Module())
        trunk = getattr(trunk, attr)
    layer = nn.Module()
    trunk.layers = nn.ModuleList([layer])
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.ones(2))
        norm.gamma, norm.beta = nn.Linear(2, 2), nn.Linear(2, 2)
        setattr(layer, name, norm)
    trunk.norm = nn.Module()
    trunk.norm.weight = nn.Parameter(torch.ones(2))
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(2, 2)
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Linear(2, 2, bias=False)
    layer.mlp.register_buffer("e_score_correction_bias", torch.tensor([0.12345, -0.56789]))
    if half_text:
        text = flow.qwenvl_with_expert
        for field in ('qwenvl','model','language_model'):
            setattr(text,field,nn.Module())
            text=getattr(text,field)
        text_layer=nn.Module()
        text.layers=nn.ModuleList([text_layer])
        text_layer.self_attn, text_layer.mlp = nn.Module(), nn.Module()
        for obj,fields in [(text_layer.self_attn,('q_proj','k_proj','v_proj','o_proj')),
                           (text_layer.mlp,('gate_proj','up_proj','down_proj'))]:
            for field in fields:
                setattr(obj,field,nn.Linear(2,2,bias=False))
        for obj,fields in [(text_layer,('input_layernorm','post_attention_layernorm')),
                           (text_layer.self_attn,('q_norm','k_norm')),(text,('norm',))]:
            for field in fields:
                norm=nn.Module();norm.weight=nn.Parameter(torch.tensor([1.010122,0.909875]))
                setattr(obj,field,norm)
    untouched = flow.trunk_projection = nn.Linear(2, 2)
    original = {"model." + k: v.detach().clone() for k, v in flow.state_dict().items()}
    shard = "model.safetensors"
    save_file(original, str(tmp_path / shard))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: shard for k in original}}))
    flow.bfloat16()
    receipt = restore_action_precision(flow, tmp_path, half_expert=half_expert, half_text=half_text)
    assert receipt["tensors"] == (37 if half_text else 25 if half_expert else 22)
    assert receipt["fp16_tensors"] == (9 if half_text else 2 if half_expert else 0)
    for name, tensor in flow.state_dict().items():
        if ".language_model." in name:
            dtype = torch.float16 if "_proj.weight" in name else torch.float32
            assert tensor.dtype == dtype and torch.equal(tensor, original["model."+name].to(dtype))
        elif ".self_attn." in name and half_expert:
            assert tensor.dtype == torch.float16 and torch.equal(tensor, original["model." + name].half())
        elif name.startswith("trunk_projection") or (not half_expert and (".self_attn." in name or name.endswith("model.norm.weight"))):
            assert tensor.dtype == torch.bfloat16
        else:
            assert tensor.dtype == torch.float32 and torch.equal(tensor, original["model." + name])
    assert flow.trunk_projection is untouched
