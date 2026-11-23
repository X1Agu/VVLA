"""Private down-only FP8 full-model candidate; no GPU qualification inherited."""
import torch,triton
from . import routed_fp8_v1 as full

class PackedRoutedDownFP8(torch.nn.Module):
    def __init__(self,gate,up,down):
        super().__init__()
        if torch.is_inference_mode_enabled():raise RuntimeError('Own packed state outside inference_mode')
        if any(not isinstance(x,torch.nn.Parameter) or not x.is_cuda or x.dtype!=torch.bfloat16 or not x.is_contiguous() for x in (gate,up,down)):raise ValueError('Original contiguous CUDA BF16 Parameters required')
        if gate.shape!=up.shape or gate.shape!=(32,512,768) or down.shape!=(32,768,512) or gate.device!=up.device or gate.device!=down.device or torch.cuda.get_device_capability(gate.device)!=(12,0):raise ValueError('Actual SM120 E32/I512/D768 only')
        self.source_gate,self.source_up,self.source_down=gate,up,down
        self.register_buffer('down_q',None);self.register_buffer('down_scale',None)
        self.down_q,self.down_scale=full.pack_weight(down)
        self._source=tuple(full.tensor_token(x) for x in (gate,up,down));self._packed={n:full.tensor_token(x) for n,x in self.named_buffers()};self.eval()

    def assert_fresh(self):
        if tuple(full.tensor_token(x) for x in (self.source_gate,self.source_up,self.source_down))!=self._source or {n:full.tensor_token(x) for n,x in self.named_buffers()}!=self._packed:raise RuntimeError('Original routed weight or private down Q/scale changed')

    def forward(self,hidden_states,routing_weights,selected_experts,gate_weight,up_weight,down_weight,workspace=None):
        from lingbotvla.ops import robby_moe as original
        self.assert_fresh()
        if gate_weight is not self.source_gate or up_weight is not self.source_up or down_weight is not self.source_down:raise RuntimeError('Different original routed Parameter owner')
        t,d=hidden_states.shape;e,i,k=32,512,4;max_routes=t*k
        if t!=51 or d!=768 or hidden_states.dtype!=torch.bfloat16 or hidden_states.device!=self.source_down.device or not hidden_states.is_contiguous() or routing_weights.shape!=(t,k) or routing_weights.dtype!=torch.bfloat16 or routing_weights.device!=hidden_states.device or selected_experts.shape!=(t,k) or selected_experts.device!=hidden_states.device or selected_experts.dtype not in (torch.int32,torch.int64):raise ValueError('Private full-model B1 H50 real rows and original topk4 only')
        if workspace is None:raise ValueError('Original expert-owned complete workspace required')
        counts,rows,slots,inter,out=(workspace[n] for n in ('counts','rows','slots','inter','out'))
        for value,shape,dtype in ((counts,(e,),torch.int32),(rows,(e,max_routes),torch.int32),(slots,(e,max_routes),torch.int32),(inter,(t,k,i),torch.bfloat16),(out,(t,d),torch.float32)):
            if value.shape!=shape or value.dtype!=dtype or value.device!=hidden_states.device or not value.is_contiguous():raise ValueError('Original sparse workspace geometry/dtype/device changed')
        selected=selected_experts.to(torch.int32).contiguous();route=routing_weights.contiguous()
        # Exact operation order of source-v12 gateup_bf16_down_fp8 label.
        original._zero_i32_kernel[(1,)](counts,e,BLOCK=32,num_warps=1)
        original._moe_pack_selected_kernel[(t,)](selected,route,counts,rows,slots,t,k,max_routes,BLOCK_K=4,num_warps=1)
        original._moe_gate_up_grouped_kernel[(e,triton.cdiv(t,16),triton.cdiv(i,32))](hidden_states,gate_weight,up_weight,counts,rows,slots,route,inter,t,d,e,k,i,max_routes,BLOCK_M=16,BLOCK_I=32,BLOCK_D=64,num_warps=4)
        original._zero_fp32_kernel[(triton.cdiv(out.numel(),1024),)](out,out.numel(),BLOCK=1024,num_warps=4)
        q=torch.empty_like(inter,dtype=torch.float8_e4m3fn);scale=torch.empty(max_routes,device=inter.device,dtype=torch.float32)
        full._quantize_rows[(max_routes,)](inter,q,scale,max_routes,i,triton.next_power_of_2(i),num_warps=4)
        full._down[(e,triton.cdiv(t,16),triton.cdiv(d,64))](q,scale,self.down_q,self.down_scale,counts,rows,slots,out,t,d,e,k,i,max_routes,BLOCK_M=16,BLOCK_D=64,BLOCK_I=64,num_warps=4,enable_fp_fusion=False)
        return out.reshape_as(hidden_states)


def install_overlay(engine, *, posttopk=False, sum_order='pairwise_02_13'):
    """Install explicit per-layer callbacks before any guard or graph is frozen.

    The candidate remains SM120/B1/H50 only. A larger B1 graph cache does not
    change its numerical domain, and is therefore independent of this option.
    """
    from ..routing import install_moe_forward
    if (engine.mode!='framework-bf16' or getattr(engine,'_signature',None) is not None
            or getattr(engine,'_frozen_guard',None) is not None or getattr(engine,'_graph',None) is not None
            or bool(getattr(engine,'_entries',{})) or engine.recipe.get('p2_routed_down_fp8')):
        raise RuntimeError('Install routed-down FP8 on a fresh framework engine before capture')
    setup=None
    if posttopk:
        from .routing_setup_v1 import post_topk_setup, SUM_ORDERS
        if sum_order not in SUM_ORDERS:raise ValueError('Unknown PostTopK sum order')
        def setup(scores,selected,**kwargs):
            return post_topk_setup(scores,selected,sum_order=sum_order,**kwargs)
    # Construct all packed owners first. A failure cannot leave half the
    # forward methods changed; ordinary parameters remain available throughout.
    prepared=[]
    for layer in engine.action.layers:
        mlp=layer.mlp
        if mlp.top_k!=4 or mlp.num_experts!=32:raise ValueError('Down FP8 requires E32/k4')
        owner=PackedRoutedDownFP8(mlp.experts.gate_proj,mlp.experts.up_proj,mlp.experts.down_proj)
        prepared.append((mlp,owner))
    installed=[]
    try:
        for mlp,owner in prepared:
            original=mlp.forward
            installed.append((mlp,original))
            mlp.add_module('_vvla_routed_down_fp8',owner)
            install_moe_forward(mlp,routed_forward=owner.forward,setup=setup)
    except BaseException:
        for mlp,original in reversed(installed):
            mlp.forward=original
            if '_vvla_routed_down_fp8' in mlp._modules:del mlp._modules['_vvla_routed_down_fp8']
        raise
    receipt=dict(name='explicit-routed-down-fp8-v2',qualified=False,quality_evidence=None,
                 compute_capability=[12,0],real_batch=1,action_horizon=50,
                 layers=len(installed),posttopk=posttopk,sum_order=sum_order if posttopk else None,
                 original_parameters_retained=True,activation_calibration=False,
                 weight_scale='per expert/output',activation_scale='per current routed intermediate row')
    engine.recipe['p2_routed_down_fp8']=receipt
    engine.recipe['precision_scope']='dynamic routed-down FP8; BF16 gate/up/shared/VLM; FP32 router and atomic merge'
    engine._write_recipe()
    return receipt

