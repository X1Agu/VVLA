from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vvla_lingbot.routing import install_moe_forward


@pytest.mark.parametrize('activation',['sigmoid','softmax'])
@pytest.mark.parametrize('normalize',[False,True])
def test_explicit_forward_preserves_router_selection_cast_and_shared_merge(activation,normalize):
    torch.manual_seed(5)
    class MoE(nn.Module):
        def __init__(self):
            super().__init__();self.gate=nn.Linear(8,32,bias=False).to(torch.bfloat16)
            self.shared_expert=nn.Linear(8,8).to(torch.bfloat16)
            self.shared_expert_gate=nn.Linear(8,1).to(torch.bfloat16)
            self.e_score_correction_bias=torch.linspace(-.3,.3,32)
            self.experts=SimpleNamespace(gate_proj=object(),up_proj=object(),down_proj=object(),
                                         _get_robby_moe_workspace=lambda *_:{'owned':True})
            self.top_k=4;self.norm_topk_prob=normalize;self.routed_scaling_factor=1.5
            self._moe_implementation='fused';self._router_activation=activation;self._use_shared_expert_gate=True
            self.eval()
    module=MoE();hidden=torch.randn(2,3,8,dtype=torch.bfloat16);seen={}
    def route(x,weights,selected,gate,up,down,workspace):
        seen.update(weights=weights,selected=selected)
        assert gate is module.experts.gate_proj and workspace=={'owned':True}
        return x.float()*.25
    install_moe_forward(module,routed_forward=route)
    with torch.no_grad():
        result,logits=module(hidden)
        flat=hidden.reshape(-1,8)
        ref_logits=F.linear(flat.float(),module.gate.weight.float())
        scores=ref_logits.sigmoid() if activation=='sigmoid' else ref_logits.softmax(dim=1)
        selected=torch.topk(scores+module.e_score_correction_bias,4,dim=-1).indices
        weights=scores.gather(1,selected)
        if normalize:weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-20)
        weights=(weights*1.5).to(torch.bfloat16)
        expected=((flat.float()*.25).to(torch.bfloat16)+
                  torch.sigmoid(module.shared_expert_gate(flat))*module.shared_expert(flat)).reshape_as(hidden)
    torch.testing.assert_close(logits,ref_logits,rtol=0,atol=0)
    torch.testing.assert_close(seen['weights'],weights,rtol=0,atol=0)
    torch.testing.assert_close(seen['selected'],selected,rtol=0,atol=0)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
