"""Explicit inference MoE forward, adapted from LingBot's Qwen2 action expert.

The router/correction/top-k/shared-expert math follows the upstream forward.
Selected routing setup and grouped GEMM are declared callbacks, installed before
guard freeze/capture. Runtime source rewriting and global factory patches are
not part of this interface. The fused route fails explicitly on kernel failure.
"""
from types import MethodType
import torch
from torch.nn import functional as F


def action_moe_forward(module, hidden_states, *, routed_forward, setup=None, input_dtype=torch.bfloat16):
    if module.training or torch.is_grad_enabled() or hidden_states.dtype != input_dtype:
        raise RuntimeError("Action MoE requires eval, no_grad and the declared hidden precision")
    if module._moe_implementation != "fused":
        raise RuntimeError("Explicit action MoE requires the fused expert layout")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    with torch.amp.autocast(hidden_flat.device.type, enabled=False):
        router_logits = F.linear(hidden_flat.float(), module.gate.weight.float())
    routing_scores = (router_logits.sigmoid() if module._router_activation == "sigmoid"
                      else F.softmax(router_logits, dim=1, dtype=torch.float))
    scores_for_choice = routing_scores + module.e_score_correction_bias.unsqueeze(0)
    _, selected_experts = torch.topk(scores_for_choice, module.top_k, dim=-1)
    if setup is None:
        routing_weights = routing_scores.gather(1, selected_experts)
        if module.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if module.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * module.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)
    else:
        routing_weights, selected_experts = setup(routing_scores, selected_experts,
            norm_topk_prob=module.norm_topk_prob, routed_scaling_factor=module.routed_scaling_factor)
    final = routed_forward(hidden_flat, routing_weights, selected_experts,
        module.experts.gate_proj, module.experts.up_proj, module.experts.down_proj,
        workspace=module.experts._get_robby_moe_workspace(hidden_flat, selected_experts.shape[1]))
    if final.dtype != hidden_flat.dtype:
        final = final.to(hidden_flat.dtype)
    shared = module.shared_expert(hidden_flat)
    if module._use_shared_expert_gate:
        shared = F.sigmoid(module.shared_expert_gate(hidden_flat)) * shared
    return (final + shared).reshape(batch_size, sequence_length, hidden_dim), router_logits


def install_moe_forward(module, *, routed_forward, setup=None, input_dtype=torch.bfloat16):
    if module.training or module._moe_implementation != "fused":
        raise ValueError("Install explicit MoE only on eval fused experts")
    required = ("gate", "experts", "shared_expert", "_use_shared_expert_gate", "_router_activation",
                "e_score_correction_bias", "top_k", "norm_topk_prob", "routed_scaling_factor")
    if any(not hasattr(module, name) for name in required):
        raise ValueError("Action MoE module does not implement the supported layout")
    if module._router_activation not in ('sigmoid','softmax'):
        raise ValueError('Unknown router activation')
    original = module.forward
    def forward(self, hidden_states):
        return action_moe_forward(self, hidden_states, routed_forward=routed_forward, setup=setup, input_dtype=input_dtype)
    module.forward = MethodType(forward, module)
    return original


def install_post_topk(engine, *, sum_order="pairwise_02_13", routed_forward=None):
    if (getattr(engine, "_signature", None) is not None or getattr(engine, "_frozen_guard", None) is not None
            or getattr(engine, "_graph", None) is not None or bool(getattr(engine, "_entries", {}))):
        raise RuntimeError("Install PostTopK before guard freeze and graph capture")
    if hasattr(engine, "_p2_routing_setup_v1"):
        raise RuntimeError("PostTopK already installed")
    from .experimental.routing_setup_v1 import post_topk_setup, SUM_ORDERS
    if sum_order not in SUM_ORDERS:
        raise ValueError("Unknown explicit sum association")
    if routed_forward is None:
        from .lite_tuning.moe import tight_moe_forward
        routed_forward = tight_moe_forward
    for layer in engine.action.layers:
        if layer.mlp.top_k != 4 or layer.mlp.num_experts != 32:
            raise ValueError("PostTopK kernel supports E32/k4")
    input_dtype = torch.float16 if getattr(engine, "fp16_expert", False) else torch.bfloat16
    def setup(scores, selected, **kwargs):
        return post_topk_setup(scores, selected, sum_order=sum_order, output_dtype=input_dtype, **kwargs)
    installed = []
    try:
        for layer in engine.action.layers:
            old = install_moe_forward(layer.mlp, routed_forward=routed_forward, setup=setup, input_dtype=input_dtype)
            installed.append((layer.mlp, old))
    except BaseException:
        for module, original in reversed(installed):module.forward = original
        raise
    receipt = dict(name="explicit-post-topk-v2", sum_order=sum_order, layers=len(installed),
                   qualified=False, quality_evidence=None, capacity=getattr(engine,"_capacity",1),
                   failure_policy="reject; no silent fused/eager fallback")
    engine._p2_routing_setup_v1 = receipt
    engine.recipe["p2_routing_setup_overlay"] = receipt
    engine._write_recipe()
    return receipt

