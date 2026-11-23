"""CPU planning preserves shifted action readout and row-isolated masks."""
from types import SimpleNamespace
import torch
from vvla_oft.adapter import pack_prompts
from vvla_oft.native_adapter import OFTNativeAdapter
from vvla_oft.recipe import OFTExecutionRecipe


def test_cpu_planning_preserves_packed_positions_and_action_readout():
    adapter=object.__new__(OFTNativeAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.recipe=OFTExecutionRecipe(pack_sequence=True)
    adapter.projected_count=5;adapter.action_tokens=56
    adapter.validate_execution_state=lambda:None
    inputs=pack_prompts([[1,9,29871],[1,8,7,6,29871]],pad_token_id=0,capacity=80)
    inputs.update(pixel_values=torch.zeros(2,3,4,4,dtype=torch.bfloat16),proprio=torch.zeros(2,8))
    actual=adapter.prepare_inputs(inputs,device='cpu')
    head=actual['pack_indices'].gather(1,actual['head_indices'])
    expected=5+inputs['action_starts'][:,None]+torch.arange(56)
    assert torch.equal(head,expected)
    assert actual['packed_valid'].sum(1).tolist()==[65,67]
    assert actual['cu_k'].tolist()==[0,65,132]
    assert all(t.device.type=='cpu' for t in actual.values())
    changed={**inputs,'action_starts':inputs['action_starts']+20}
    try:adapter.prepare_inputs(changed,device='cpu')
    except ValueError:pass
    else:raise AssertionError('masked action readout was accepted')
