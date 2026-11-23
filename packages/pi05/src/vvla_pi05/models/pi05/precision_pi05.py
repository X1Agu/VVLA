# SPDX-License-Identifier: Apache-2.0
"""Explicit, opt-in inference recipes. Original checkpoint parameters are retained.

Packing is done outside CUDA Graph capture. The caller must reinstall after a
weight change; stale quantized/concatenated weights fail closed. This prototype
targets compute latency, not compressed checkpoint or resident-weight memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ExecutionRecipe:
    prefix_mlp: str = "none"
    vision_mlp: str = "none"
    expert_mlp: str = "none"
    fuse_prefix_mlp: bool = False
    fuse_prefix_down_quant: bool = False
    fuse_gate_up: bool = False
    fuse_prefix_rmsnorm: bool = False
    prefix_rmsnorm_reduction: str = "triton"
    nvfp4_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(2, 15)))
    nvfp4_backend: str = "vllm_cutlass"
    fuse_vision_residual_cast: bool = False
    fuse_rope: bool = False
    fuse_prefix_residual_square: bool = False

    def __post_init__(self):
        if type(self.fuse_rope) is not bool or type(self.fuse_prefix_residual_square) is not bool:
            raise ValueError("Pointwise fusion switches must be booleans")
        if self.fuse_prefix_residual_square and (not self.fuse_prefix_rmsnorm or self.prefix_rmsnorm_reduction != "torch"):
            raise ValueError("Residual-square fusion requires prefix RMSNorm with original torch mean")
        if type(self.fuse_vision_residual_cast) is not bool:
            raise ValueError("fuse_vision_residual_cast must be a boolean")
        if type(self.fuse_prefix_down_quant) is not bool:
            raise ValueError("fuse_prefix_down_quant must be a boolean")
        if self.fuse_prefix_down_quant and (self.prefix_mlp != "nvfp4_full" or not self.fuse_prefix_mlp
                                           or self.nvfp4_backend != "vllm_cutlass"):
            raise ValueError("Fused down quantization requires full prefix NVFP4, MLP fusion and vllm_cutlass")
        if self.prefix_rmsnorm_reduction not in ("triton", "torch"):
            raise ValueError("Prefix RMSNorm reduction must be triton or torch")
        if self.prefix_mlp not in ("none", "fp8", "nvfp4_selective", "nvfp4_full"):
            raise ValueError("Unsupported prefix MLP recipe")
        if self.vision_mlp not in ("none", "fp8") or self.expert_mlp not in ("none", "fp8"):
            raise ValueError("Vision/expert initially support selective FP8 only")
        if self.nvfp4_backend not in ("vllm_cutlass", "flashinfer_cutlass", "flashinfer_cudnn"):
            raise ValueError("Unknown NVFP4 backend")
        if self.fuse_gate_up and (not self.fuse_prefix_mlp or self.prefix_mlp != "none"):
            raise ValueError("Gate/up concatenation is a separate BF16 fusion experiment")
        if any(type(i) is not int or i < 0 for i in self.nvfp4_layers) or len(set(self.nvfp4_layers)) != len(self.nvfp4_layers):
            raise ValueError("NVFP4 layer indices must be distinct nonnegative integers")
        object.__setattr__(self, "nvfp4_layers", tuple(self.nvfp4_layers))


def _identity(t):
    if t is None:
        return None
    try:
        version = t._version
    except RuntimeError:
        version = None
    return id(t), t.data_ptr(), version, t.dtype, t.device, tuple(t.shape)


def quantize_fp8_weight(weight):
    """Per-output-channel E4M3; all-zero channels use a positive scale."""
    scale = (weight.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    quantized = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.contiguous()


class QuantizedLinear(nn.Module):
    @property
    def input_dtype(self):
        # A packed module has no retained dense weight from which to infer dtype.
        return torch.bfloat16

    @classmethod
    def from_packed(cls, original, precision, tensors, backend="vllm_cutlass"):
        """Restore checked packed tensors without re-quantizing an original weight."""
        if not isinstance(original, nn.Linear) or original.weight.dtype != torch.bfloat16:
            raise ValueError("Packed restore requires a fresh BF16 Linear")
        k, n = original.in_features, original.out_features
        alignment = 32 if precision == "nvfp4" else 16
        if k <= 0 or n <= 0 or k % alignment or n % alignment:
            raise ValueError("Packed linear geometry violates backend alignment")
        spec = {"packed_weight": ((n, k), torch.float8_e4m3fn), "weight_scale": ((n, 1), torch.float32)}
        if precision == "nvfp4":
            spec = {"packed_weight": ((n, k // 2), torch.uint8),
                    "weight_scale": ((((n + 127) // 128) * 128, ((k + 63) // 64) * 4), torch.float8_e4m3fn),
                    "input_global_scale": ((1,), torch.float32), "alpha": ((1,), torch.float32)}
        elif precision != "fp8":
            raise ValueError("Unsupported packed precision")
        if set(tensors) != set(spec):
            raise ValueError("Packed tensor keys do not match precision")
        for name, (shape, dtype) in spec.items():
            t = tensors[name]
            if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                raise ValueError(f"Invalid packed tensor geometry/dtype: {name}")
            if name != "packed_weight" and dtype == torch.float32 and (not torch.isfinite(t).all() or not (t > 0).all()):
                raise ValueError(f"Nonpositive or nonfinite packed scale: {name}")
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = k, n
        self.precision, self.backend = precision, backend
        for name in ("packed_weight", "weight_scale", "input_global_scale", "alpha"):
            t = tensors.get(name)
            self.register_buffer(name, None if t is None else t.to(original.weight.device).clone(), persistent=False)
        self._source_identity = (_identity(self.weight), _identity(self.bias))
        self._packed_identity = self._packing_identity()
        self.train(original.training)
        return self

    def __init__(self, original, precision, *, input_amax=None, backend="vllm_cutlass"):
        super().__init__()
        if not isinstance(original, nn.Linear) or original.weight.dtype != torch.bfloat16:
            raise ValueError("Quantized linear requires an explicitly BF16 nn.Linear")
        if precision not in ("fp8", "nvfp4"):
            raise ValueError(precision)
        alignment = 16 if precision == "fp8" else 32
        if original.in_features % alignment or original.out_features % alignment:
            raise ValueError(f"{precision} kernels require K and N divisible by {alignment}")
        self.weight = original.weight
        self.bias = original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.precision, self.backend = precision, backend
        self.register_buffer("packed_weight", None, persistent=False)
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("input_global_scale", None, persistent=False)
        self.register_buffer("alpha", None, persistent=False)
        with torch.no_grad():
            if precision == "fp8":
                self.packed_weight, self.weight_scale = quantize_fp8_weight(self.weight)
            else:
                if not self.weight.is_cuda:
                    raise ValueError("Native NVFP4 requires CUDA; no fake-quantized fallback")
                if input_amax is None or not math.isfinite(float(input_amax)) or input_amax <= 0:
                    raise ValueError("NVFP4 requires a positive finite calibrated activation amax")
                from vllm import _custom_ops as ops
                weight_global = (2688.0 / self.weight.float().abs().amax().clamp_min(1e-12)).reshape(1)
                self.input_global_scale = torch.tensor([2688.0 / float(input_amax)], device=self.weight.device, dtype=torch.float32)
                self.packed_weight, self.weight_scale = ops.scaled_fp4_quant(self.weight, weight_global, backend="cutlass")
                self.alpha = (1.0 / (weight_global * self.input_global_scale)).float()
        self._source_identity = (_identity(self.weight), _identity(self.bias))
        self._packed_identity = self._packing_identity()
        self.train(original.training)

    def _packing_identity(self):
        return (self.precision, self.backend, *(_identity(x) for x in
                (self.packed_weight, self.weight_scale, self.input_global_scale, self.alpha)))

    def assert_fresh(self):
        if self._source_identity != (_identity(self.weight), _identity(self.bias)):
            raise RuntimeError("Original linear weights changed; reinstall execution recipe before inference")
        if self._packed_identity != self._packing_identity():
            raise RuntimeError("Packed weights/scales/backend changed; rebuild execution recipe before inference")

    def forward_prequantized(self, qx, sx, input_shape):
        """Consume the same CUTLASS NVFP4 representation produced by scaled_fp4_quant."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Low-precision recipes require eval and no_grad/inference_mode")
        if self.precision != "nvfp4" or self.backend != "vllm_cutlass":
            raise ValueError("Prequantized input currently requires vllm_cutlass NVFP4")
        rows = math.prod(input_shape[:-1])
        if (not input_shape or input_shape[-1] != self.in_features
                or qx.shape != (rows, self.in_features // 2) or qx.dtype != torch.uint8
                or sx.shape != (((rows + 127) // 128) * 128, ((self.in_features + 63) // 64) * 4)
                or sx.dtype != torch.float8_e4m3fn or qx.device != self.packed_weight.device
                or sx.device != qx.device or not qx.is_cuda or not qx.is_contiguous() or not sx.is_contiguous()):
            raise ValueError("Invalid prequantized NVFP4 input geometry, dtype or device")
        from vllm import _custom_ops as ops
        result = ops.cutlass_scaled_fp4_mm(qx, self.packed_weight, sx, self.weight_scale, self.alpha, torch.bfloat16)
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*input_shape[:-1], self.out_features)

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Low-precision recipes require eval and no_grad/inference_mode")
        if x.dtype != torch.bfloat16:
            raise ValueError("Low-precision input must be BF16")
        shape = (*x.shape[:-1], self.out_features)
        flat = x.reshape(-1, self.in_features).contiguous()
        if not flat.is_cuda:
            # CPU numerical oracle for unit checks, not a performance backend.
            if self.precision != "fp8":
                raise ValueError("NVFP4 has no CPU execution path")
            sx = (flat.float().abs().amax(1, keepdim=True) / 448.0).clamp_min(1e-12)
            qx = (flat.float()/sx).clamp(-448,448).to(torch.float8_e4m3fn).float()
            result = ((qx*sx) @ (self.packed_weight.float()*self.weight_scale).t()).to(x.dtype)
            if self.bias is not None:
                result = result + self.bias
        else:
            from vllm import _custom_ops as ops
            if self.precision == "fp8":
                qx, sx = ops.scaled_fp8_quant(flat, use_per_token_if_dynamic=True)
                result = ops.cutlass_scaled_mm(qx, self.packed_weight.t(), sx, self.weight_scale.t(), x.dtype, self.bias)
            else:
                qx, sx = ops.scaled_fp4_quant(flat, self.input_global_scale, backend="cutlass")
                if self.backend == "vllm_cutlass":
                    result = ops.cutlass_scaled_fp4_mm(qx, self.packed_weight, sx, self.weight_scale, self.alpha, x.dtype)
                else:
                    from flashinfer import mm_fp4
                    result = mm_fp4(qx, self.packed_weight.t(), sx, self.weight_scale.t(), self.alpha,
                                    out_dtype=x.dtype, backend=self.backend.removeprefix("flashinfer_"))
                if self.bias is not None:
                    result = result + self.bias
        return result.reshape(shape)


class FusedPrefixMLP(nn.Module):
    def __init__(self, original, concatenate=False, fuse_down_quant=False):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = original.gate_proj, original.up_proj, original.down_proj
        self.act_fn = original.act_fn
        self.concatenate = concatenate
        self.fuse_down_quant = fuse_down_quant
        if fuse_down_quant and (concatenate or not isinstance(self.down_proj, QuantizedLinear)
                                or self.down_proj.precision != "nvfp4" or self.down_proj.backend != "vllm_cutlass"):
            raise ValueError("Fused GELU/down quantization requires a native NVFP4 down projection")
        self._fusion_flags = (concatenate, fuse_down_quant)
        self.register_buffer("gate_up_weight", None, persistent=False)
        if concatenate:
            if any(m.bias is not None for m in (self.gate_proj, self.up_proj)):
                raise ValueError("Concatenated gate/up currently requires bias-free projections")
            self.gate_up_weight = torch.cat([self.gate_proj.weight.detach(), self.up_proj.weight.detach()], 0)
        self._source_identity = (_identity(self.gate_proj.weight), _identity(self.up_proj.weight))
        self.train(original.training)

    def assert_fresh(self):
        if self._fusion_flags != (self.concatenate, self.fuse_down_quant):
            raise RuntimeError("MLP fusion changed; rebuild model and CUDA Graphs")
        if self.concatenate and self._source_identity != (_identity(self.gate_proj.weight), _identity(self.up_proj.weight)):
            raise RuntimeError("Concatenated GEMM weights changed; reinstall execution recipe")

    def forward(self, x):
        if self.concatenate:
            gate, up = F.linear(x, self.gate_up_weight).chunk(2, -1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_down_quant:
            from vvla_pi05.models.pi05.fused_gelu_nvfp4_pi05 import gelu_mul_nvfp4
            qx, sx = gelu_mul_nvfp4(gate, up, self.down_proj.input_global_scale)
            return self.down_proj.forward_prequantized(qx, sx, gate.shape)
        if x.is_cuda:
            from vvla_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
            hidden = gelu_tanh_mul(gate, up)
        else:
            hidden = F.gelu(gate, approximate="tanh") * up
        return self.down_proj(hidden)


class PrefixRMSNormCast(nn.Module):
    """Gemma RMSNorm with a BF16 output, leaving the FP32 residual untouched.

    Only replaces prefix norms immediately consumed by BF16 projections. No
    copied weights or persistent output buffer; Graph owns its captured output.
    Reduction order can differ from eager PyTorch, so this is opt-in and tested
    numerically as well as through paired closed-loop evaluation.
    """
    @staticmethod
    def preflight(original, projection):
        cls = type(original)
        if cls.__name__ != "GemmaRMSNorm" or cls.__module__ != "transformers.models.gemma.modeling_gemma":
            raise ValueError("Prefix fusion requires the supported GemmaRMSNorm (1 + weight) convention")
        if original.weight.dtype != torch.float32 or original.weight.ndim != 1 or not original.weight.is_contiguous():
            raise ValueError("Prefix RMSNorm requires contiguous FP32 weights")
        if original.weight.numel() == 0 or original.weight.numel() > 8192:
            raise ValueError("Unsupported prefix RMSNorm width")
        if not math.isfinite(float(original.eps)) or original.eps <= 0:
            raise ValueError("Prefix RMSNorm epsilon must be positive and finite")
        if projection.weight.dtype != torch.bfloat16 or projection.weight.device != original.weight.device:
            raise ValueError("Prefix RMSNorm output must feed a BF16 projection on the same device")
        if projection.weight.shape[1] != original.weight.numel():
            raise ValueError("Prefix RMSNorm/projection width mismatch")

    def __init__(self, original, projection, reduction="triton"):
        super().__init__()
        self.preflight(original, projection)
        if reduction not in ("triton", "torch"):
            raise ValueError("Unsupported prefix RMSNorm reduction")
        self.reduction = reduction
        self.weight = original.weight
        self.eps = float(original.eps)
        self._source_identity = (_identity(self.weight), self.eps, self.reduction)
        self.train(original.training)

    def assert_fresh(self):
        if self._source_identity != (_identity(self.weight), self.eps, self.reduction):
            raise RuntimeError("Prefix RMSNorm weights/epsilon/reduction changed; reinstall execution recipe")
        if self.training:
            raise RuntimeError("Prefix RMSNorm fusion requires eval mode")

    def forward(self, x, squared=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Prefix RMSNorm fusion requires eval and no_grad/inference_mode")
        if x.dtype != torch.float32 or x.ndim < 2 or x.shape[-1] != self.weight.numel():
            raise ValueError("Prefix RMSNorm fusion expects FP32 residual rows with matching width")
        if x.device != self.weight.device or not x.is_contiguous():
            raise ValueError("Prefix RMSNorm fusion requires contiguous input on the weight device")
        if squared is not None and (self.reduction != "torch" or squared.shape != x.shape
                or squared.dtype != torch.float32 or squared.device != x.device or not squared.is_contiguous()):
            raise ValueError("Invalid request-local square for torch mean")
        if x.is_cuda:
            from vvla_pi05.models.pi05.fused_rmsnorm_pi05 import gemma_rmsnorm_bf16
            return gemma_rmsnorm_bf16(x, self.weight, self.eps, self.reduction, squared=squared)
        # CPU oracle only; local development does not run GPU experiments.
        normalized = x * torch.rsqrt((x.pow(2) if squared is None else squared).mean(-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight)).to(torch.bfloat16)


def selected_prefix_norms(model, recipe):
    if not recipe.fuse_prefix_rmsnorm:
        return []
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    active_mlp = len(layers) - int(getattr(model.config, "prefix_kv_only", False))
    entries = []
    for i, layer in enumerate(layers):
        entries.append((f"prefix.{i}.input_layernorm", layer, "input_layernorm", layer.self_attn.q_proj))
        if i < active_mlp:
            entries.append((f"prefix.{i}.post_attention_layernorm", layer, "post_attention_layernorm", layer.mlp.up_proj))
    return entries


def selected_linears(model, recipe):
    """Only explicitly selected MLP projections; never norms/heads/QKV."""
    prefix = model.paligemma_with_expert.paligemma.model.language_model.layers
    active = len(prefix) - int(getattr(model.config, "prefix_kv_only", False))
    if recipe.prefix_mlp == "nvfp4_selective" and any(i >= active for i in recipe.nvfp4_layers):
        raise ValueError("NVFP4 selection includes an absent or KV-only MLP")
    entries = []
    for i, layer in enumerate(prefix[:active]):
        if recipe.prefix_mlp != "none":
            for name in ("gate_proj", "up_proj", "down_proj"):
                use_nvfp4 = recipe.prefix_mlp == "nvfp4_full" or (
                    recipe.prefix_mlp == "nvfp4_selective" and i in recipe.nvfp4_layers and name != "down_proj"
                )
                precision = "nvfp4" if use_nvfp4 else "fp8"
                entries.append((f"prefix.{i}.{name}", layer.mlp, name, precision))
    if recipe.vision_mlp != "none":
        if not getattr(model.config, "vision_bf16", False):
            raise ValueError("Vision FP8 must be compared with an explicit vision_bf16 control")
        vision = model.paligemma_with_expert.paligemma.model.vision_tower
        before = len(entries)
        for name, module in vision.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("mlp.fc1", "mlp.fc2")):
                parent_name, leaf = name.rsplit(".", 1)
                entries.append((f"vision.{name}", vision.get_submodule(parent_name), leaf, "fp8"))
        if len(entries) == before:
            raise ValueError("No supported vision MLP linears found; refusing a no-op quantization recipe")
    if recipe.expert_mlp != "none":
        for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                entries.append((f"expert.{i}.{name}", layer.mlp, name, "fp8"))
    return entries


@torch.no_grad()
def install_execution_recipe(model, recipe, calibration=None, *, packed=None):
    if torch.is_inference_mode_enabled():
        raise ValueError("Install recipes outside inference_mode so scale mutations retain version tracking")
    if model.training:
        raise ValueError("Set model.eval() before installing an inference recipe")
    if getattr(model, "_execution_state", None) is not None:
        raise ValueError("Use a fresh checkpoint/model to switch recipes; no implicit lossy restoration")
    recipe = recipe if isinstance(recipe, ExecutionRecipe) else ExecutionRecipe(**recipe)
    entries = selected_linears(model, recipe)
    vision_layers = None
    if recipe.fuse_vision_residual_cast:
        from vvla_pi05.models.pi05.fused_vision_pi05 import selected_vision_layers, VisionResidualCastLayer
        vision_layers = selected_vision_layers(model)
    if recipe.fuse_prefix_down_quant:
        for _, parent, leaf, _ in entries:
            device = getattr(parent, leaf).weight.device
            if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
                raise ValueError("Fused GELU/down quantization currently requires validated SM120 hardware")
    norms = selected_prefix_norms(model, recipe)
    for name, parent, leaf, projection in norms:
        PrefixRMSNormCast.preflight(getattr(parent, leaf), projection)
    calibration = calibration or {}
    for name, parent, leaf, precision in entries:
        if packed is not None and name not in packed:
            raise ValueError(f"Missing packed module: {name}")
        if packed is None and precision == "nvfp4" and name not in calibration:
            raise ValueError(f"Missing activation calibration for {name}")
        if not isinstance(getattr(parent, leaf), nn.Linear):
            raise ValueError(f"Already transformed or unsupported module: {name}")
    state = {"status": "preparing", "recipe": recipe, "modules": [], "manifest": [], "norm_manifest": []}
    model._execution_state = state
    try:
        for name, parent, leaf, precision in entries:
            converted = (QuantizedLinear(getattr(parent, leaf), precision, input_amax=calibration.get(name), backend=recipe.nvfp4_backend)
                         if packed is None else QuantizedLinear.from_packed(getattr(parent, leaf), precision, packed[name], recipe.nvfp4_backend))
            setattr(parent, leaf, converted)
            state["modules"].append(converted)
            state["manifest"].append({"name": name, "precision": precision, "shape": list(converted.weight.shape)})
        if recipe.fuse_prefix_mlp:
            layers = model.paligemma_with_expert.paligemma.model.language_model.layers
            active = len(layers) - int(getattr(model.config, "prefix_kv_only", False))
            for layer in layers[:active]:
                layer.mlp = FusedPrefixMLP(layer.mlp, recipe.fuse_gate_up, recipe.fuse_prefix_down_quant)
                state["modules"].append(layer.mlp)
        for name, parent, leaf, projection in norms:
            converted = PrefixRMSNormCast(getattr(parent, leaf), projection, recipe.prefix_rmsnorm_reduction)
            setattr(parent, leaf, converted)
            state["modules"].append(converted)
            state["norm_manifest"].append({"name": name, "width": converted.weight.numel(),
                                           "eps": converted.eps, "reduction": converted.reduction, "output_dtype": "bfloat16"})
        state["flags"] = tuple(model.optimization_flags())
        if vision_layers is not None:
            for i, layer in enumerate(vision_layers):
                converted = VisionResidualCastLayer(layer)
                vision_layers[i] = converted
                state["modules"].append(converted)
        model.paligemma_with_expert._pointwise_fusion = (recipe.fuse_rope, recipe.fuse_prefix_residual_square)
        state["config_recipe"] = repr(getattr(model.config, "execution_recipe", None))
        state["status"] = "ready"
        model.invalidate_inference_caches(weights_changed=False)
        return state["manifest"]
    except BaseException:
        state["status"] = "failed"
        model.invalidate_inference_caches(weights_changed=False)
        raise


def validate_execution_recipe(model):
    state = getattr(model, "_execution_state", None)
    if state is None:
        return
    if state["status"] != "ready":
        raise RuntimeError("Execution recipe is incomplete")
    if tuple(model.optimization_flags()) != state["flags"]:
        raise RuntimeError("Model flags changed after recipe installation; rebuild model")
    if repr(getattr(model.config, "execution_recipe", None)) != state["config_recipe"]:
        raise RuntimeError("Execution recipe configuration changed; rebuild model")
    if getattr(model.paligemma_with_expert, "_pointwise_fusion", (False, False)) != (state["recipe"].fuse_rope, state["recipe"].fuse_prefix_residual_square):
        raise RuntimeError("Pointwise fusion flags changed; rebuild execution recipe and graphs")
    for module in state["modules"]:
        module.assert_fresh()


def set_pointwise_fusions(model, *, fuse_rope=False, fuse_prefix_residual_square=False):
    """Select weight-preserving fusions after packed load and before Graph capture."""
    from contextlib import ExitStack
    from dataclasses import asdict, replace
    with ExitStack() as locks:
        for runner in tuple(getattr(model, "_graph_runners", ())):
            lock = getattr(runner, "_call_lock", None)
            if lock is not None:
                if not lock.acquire(blocking=False):
                    raise RuntimeError("Cannot change pointwise fusions during a Graph request")
                locks.callback(lock.release)
            if runner.entries:
                raise ValueError("Clear all live Graph runners before changing pointwise fusions")
        validate_execution_recipe(model)
        state = getattr(model, "_execution_state", None)
        if state is None:
            raise ValueError("Install the base execution recipe first")
        if torch.is_inference_mode_enabled() or model.training:
            raise ValueError("Select pointwise fusions in eval mode outside inference_mode")
        recipe = replace(state["recipe"], fuse_rope=fuse_rope, fuse_prefix_residual_square=fuse_prefix_residual_square)
        state["recipe"] = recipe
        model.config.execution_recipe = asdict(recipe)
        state["config_recipe"] = repr(model.config.execution_recipe)
        model.paligemma_with_expert._pointwise_fusion = (recipe.fuse_rope, recipe.fuse_prefix_residual_square)
        model.invalidate_inference_caches(weights_changed=False)
        return recipe
