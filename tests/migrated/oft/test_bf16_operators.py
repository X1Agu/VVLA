"""OFT operator transformations preserve BF16 representations and source ownership."""
from types import SimpleNamespace
from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vvla_oft.native_adapter import OFTNativeAdapter, penultimate_vision_patches
from vvla_oft.recipe import OFTExecutionRecipe, recipe_from_config
from vvla_runtime.execution.llama_norm import LlamaRMSNormExecution, residual_rmsnorm_bf16
from vvla_runtime.precision import FusedLlamaMLP


def test_packed_gate_up_uses_current_bf16_checkpoint_and_detects_edits():
    torch.manual_seed(4)
    original = nn.Module()
    for key, dims in (("gate_proj",(64,128)),("up_proj",(64,128)),("down_proj",(128,64))):
        setattr(original,key,nn.Linear(*dims,bias=False,dtype=torch.bfloat16))
    original.eval()
    x = torch.randn(2,56,64,dtype=torch.bfloat16)
    with torch.no_grad():
        packed = FusedLlamaMLP(original,pack_gate_up=True)
        gate, up = F.linear(x,torch.cat([original.gate_proj.weight,original.up_proj.weight])).chunk(2,dim=-1)
        torch.testing.assert_close(packed(x), original.down_proj(F.silu(gate)*up), rtol=0, atol=0)
        packed.assert_fresh()
        original.up_proj.weight.add_(1.)
        with pytest.raises(RuntimeError,match="source changed"):
            packed.assert_fresh()


def test_residual_rmsnorm_rounds_residual_and_norm_before_weighting():
    source = nn.Module()
    source.weight = nn.Parameter(torch.linspace(.3,1.1,64,dtype=torch.bfloat16))
    source.variance_epsilon = 1e-5
    norm = LlamaRMSNormExecution(source,fuse_after_mean=True)
    with torch.no_grad():
        r,u = torch.randn(2,16,64,dtype=torch.bfloat16), torch.randn(2,16,64,dtype=torch.bfloat16)
        value, result = residual_rmsnorm_bf16(r,u,norm)
        torch.testing.assert_close(value,(r+u),rtol=0,atol=0)
        expected=(value.float()*torch.rsqrt(value.float().square().mean(-1,keepdim=True)+source.variance_epsilon)).to(torch.bfloat16)
        torch.testing.assert_close(result,expected*source.weight,rtol=0,atol=0)


def test_penultimate_vision_skips_discarded_last_block_and_prefix_tokens():
    class Discarded(nn.Module):
        def forward(self,x):
            raise AssertionError("called discarded final block")
    tower = SimpleNamespace(patch_embed=nn.Identity(),_pos_embed=nn.Identity(),patch_drop=nn.Identity(),norm_pre=nn.Identity(),
                            blocks=[nn.Identity(),nn.ReLU(),Discarded()],num_prefix_tokens=2)
    x=torch.randn(4,10,32)
    torch.testing.assert_close(penultimate_vision_patches(tower,x), x.relu()[:,2:])


@pytest.mark.parametrize("settings", [
    {"mlp_compile":"tuned"},{"vision_compile":"fused"},{"pack_gate_up":True},{"fuse_residual_rms":True},
    {"skip_unused_vision_block":True},{"packed_single_kv":True}])
def test_opt_in_operator_modes_require_their_input_layout(settings):
    with pytest.raises(ValueError):
        OFTExecutionRecipe(**settings)


def test_compiler_modes_are_propagated_to_pure_bf16_recipe():
    recipe=recipe_from_config(dict(backend="vllm_fa2",batch_camera_views=True,skip_unused_vision_block=True,
                                   vision_compile="fused",pack_gate_up=True,fuse_silu_mul=True,mlp_compile="tuned"))
    assert recipe.mlp_precision == "bf16" and recipe.vision_compile == "fused" and recipe.mlp_compile == "tuned"


def test_private_layout_reuse_reads_current_inputs_and_keys_on_mask_and_readout():
    adapter = object.__new__(OFTNativeAdapter)
    nn.Module.__init__(adapter)
    adapter.recipe = OFTExecutionRecipe(cache_token_layout=True)
    adapter.projected_count, adapter.action_tokens = 2, 2
    adapter._layout_cache = OrderedDict()
    inputs = dict(attention_mask=torch.ones(1,6,dtype=torch.bool), action_starts=torch.tensor([1]),
                  input_ids=torch.arange(6)[None,:], pixel_values=torch.randn(1,3,4,4,dtype=torch.bfloat16))
    def prepare(**changes):
        return adapter._prepare_inputs(dict(inputs,**changes),validate=False,reuse_layout=True)
    first=prepare()
    new_pixels=inputs["pixel_values"]+2
    next_row=prepare(input_ids=inputs["input_ids"]+10,pixel_values=new_pixels)
    assert next_row["head_indices"] is first["head_indices"]
    assert torch.equal(next_row["pixel_values"],new_pixels) and next_row["input_ids"][0,0] == 10
    changed=prepare(action_starts=torch.tensor([2]))
    assert not torch.equal(changed["head_indices"],first["head_indices"])
    masked=inputs["attention_mask"].clone()
    masked[0,-1]=False
    assert not torch.equal(prepare(attention_mask=masked)["packed_valid"],first["packed_valid"])
    assert adapter.prepare_inputs(inputs,validate=False)["head_indices"] is not first["head_indices"]
