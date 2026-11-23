"""Only for distinct top-k expert IDs per token, as produced by torch.topk.
Each expert count <= T. Workspace capacity and MAX_ROUTES stay T*top_k;
only the two launch grids shrink. Arbitrary repeated selected IDs unsupported.
"""
import torch
import triton
from lingbotvla.ops.robby_moe import (_zero_i32_kernel, _zero_fp32_kernel,
    _moe_pack_selected_kernel, _moe_gate_up_grouped_kernel, _moe_down_grouped_kernel)

def tight_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Inference-only grouped MoE path migrated from robbyvla_infer _moe."""
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be 2D, got {tuple(hidden_states.shape)}")
    if selected_experts.ndim != 2 or routing_weights.ndim != 2:
        raise ValueError("selected_experts and routing_weights must be 2D")
    if not hidden_states.is_cuda:
        raise ValueError("robby_moe_forward requires CUDA tensors")

    T, D = hidden_states.shape
    E, I, weight_d = gate_weight.shape
    top_k = selected_experts.shape[1]
    if weight_d != D or up_weight.shape != gate_weight.shape or down_weight.shape != (E, D, I):
        raise ValueError(
            "Unexpected MoE weight shapes: "
            f"hidden={tuple(hidden_states.shape)} gate={tuple(gate_weight.shape)} "
            f"up={tuple(up_weight.shape)} down={tuple(down_weight.shape)}"
        )

    max_routes = T * top_k
    if workspace is None:
        counts = torch.empty((E,), device=hidden_states.device, dtype=torch.int32)
        rows = torch.empty((E, max_routes), device=hidden_states.device, dtype=torch.int32)
        slots = torch.empty((E, max_routes), device=hidden_states.device, dtype=torch.int32)
        inter = torch.empty((T, top_k, I), device=hidden_states.device, dtype=hidden_states.dtype)
        out = torch.empty((T, D), device=hidden_states.device, dtype=torch.float32)
    else:
        counts = workspace["counts"]
        rows = workspace["rows"]
        slots = workspace["slots"]
        inter = workspace["inter"]
        out = workspace["out"]

    selected_i32 = selected_experts.to(torch.int32).contiguous()
    route = routing_weights.contiguous()

    _zero_i32_kernel[(1,)](counts, E, BLOCK=triton.next_power_of_2(E), num_warps=1)
    _moe_pack_selected_kernel[(T,)](
        selected_i32,
        route,
        counts,
        rows,
        slots,
        T,
        top_k,
        max_routes,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    _moe_gate_up_grouped_kernel[
        (E, triton.cdiv(T, 16), triton.cdiv(I, 32))
    ](
        hidden_states,
        gate_weight,
        up_weight,
        counts,
        rows,
        slots,
        route,
        inter,
        T,
        D,
        E,
        top_k,
        I,
        max_routes,
        BLOCK_M=16,
        BLOCK_I=32,
        BLOCK_D=64,
        num_warps=4,
    )
    _zero_fp32_kernel[(triton.cdiv(out.numel(), 1024),)](
        out,
        out.numel(),
        BLOCK=1024,
        num_warps=4,
    )
    _moe_down_grouped_kernel[
        (E, triton.cdiv(T, 16), triton.cdiv(D, 64))
    ](
        inter,
        down_weight,
        counts,
        rows,
        slots,
        out,
        T,
        D,
        E,
        top_k,
        I,
        max_routes,
        BLOCK_M=16,
        BLOCK_D=64,
        BLOCK_I=64,
        num_warps=4,
    )
    return out.reshape_as(hidden_states)
