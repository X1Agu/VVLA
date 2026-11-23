"""Opt-in layer-local FP32 gated residual; no global function replacement."""

import hashlib
from pathlib import Path
from types import MethodType
import weakref

import torch

from vvla_repro_pi05.lite.fused_expert_mlp import _mutation_guard

INSTALLATION = "_lite_gated_residual_installation"
DISPATCH = "_lite_gated_residual"
VERSION = "expert-fp32-gated-residual-no-fma-v1"


def _validate_inputs(residual, out, gate):
    if (residual.dtype != torch.float32 or out.dtype not in (torch.bfloat16, torch.float32)
            or gate.dtype != torch.float32 or residual.device != out.device or residual.device != gate.device
            or residual.shape != out.shape or residual.ndim not in (2, 3)
            or not residual.is_contiguous() or not out.is_contiguous()
            or any(size < 1 for size in residual.shape)):
        raise ValueError("Gated residual requires contiguous FP32 residual, BF16/FP32 projection, and FP32 gate")
    if (gate.ndim != residual.ndim or gate.shape[0] != residual.shape[0]
            or gate.shape[-1] != residual.shape[-1] or gate.stride(-1) != 1
            or (residual.ndim == 3 and gate.shape[1] not in (1, residual.shape[1]))):
        raise ValueError("Gated residual gate must match [B,D], [B,1,D], or [B,S,D] with contiguous columns")


def fused_gated_residual(residual, out, gate):
    from vvla_repro_pi05.models.pi05.modeling_pi05 import _gated_residual

    # None already needs one add kernel; preserve Torch's general fallback.
    if (gate is None or not residual.is_cuda or torch.is_grad_enabled()
            or torch.is_autocast_enabled(residual.device.type)):
        return _gated_residual(residual, out, gate)
    _validate_inputs(residual, out, gate)
    output = torch.empty_like(residual)
    import triton
    from vvla_repro_pi05.lite.fused_gated_residual_kernel import gated_residual_kernel

    tokens = residual.shape[1] if residual.ndim == 3 else 1
    gate_token_stride = gate.stride(1) if gate.ndim == 3 and gate.shape[1] != 1 else 0
    gated_residual_kernel[(triton.cdiv(residual.numel(), 1024),)](
        residual, out, gate, output, N=residual.numel(), WIDTH=residual.shape[-1],
        TOKENS=tokens, GB=gate.stride(0), GS=gate_token_stride,
        BLOCK=1024, enable_fp_fusion=False,
    )
    return output


class GatedResidualInstallation:
    def __init__(self, model, patches):
        self._model = weakref.ref(model)
        self._patches = patches
        self.enabled = True

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / "models/pi05/modeling_pi05.py"
        paths = (Path(__file__), Path(__file__).with_name("fused_gated_residual_kernel.py"), core)
        return {"version": VERSION, "enabled": self.enabled, "layers": len(self._patches),
                "residual_dtype": "float32", "gate_dtype": "float32",
                "projection_dtype": ["bfloat16", "float32"], "output_dtype": "float32",
                "operation_order": "FP32 multiply then FP32 add", "enable_fp_fusion": False,
                "numerical_limit": "same arithmetic boundaries; CUDA bitwise validation still required",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Gated residual installation is no longer active")
        with _mutation_guard(model, "gated residual"):
            if any(getattr(layer, DISPATCH, None) is not callback for layer, callback in self._patches):
                raise RuntimeError("Gated residual dispatch was modified externally")
            for layer, _ in self._patches:
                delattr(layer, DISPATCH)
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_gated_residual(model):
    """Explicit installation only; applies to both residuals of each expert layer."""
    with _mutation_guard(model, "gated residual"):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Gated residual candidate is already installed")
        layers = list(model.paligemma_with_expert.gemma_expert.model.layers)
        if not layers or len({id(layer) for layer in layers}) != len(layers):
            raise ValueError("Gated residual requires distinct nonempty expert layers")
        if any(layer.training or hasattr(layer, DISPATCH) for layer in layers):
            raise ValueError("Gated residual requires unchanged eval layer dispatch slots")
        patches = []
        for layer in layers:
            def residual_call(layer, residual, out, gate):
                if layer.training:
                    from vvla_repro_pi05.models.pi05.modeling_pi05 import _gated_residual
                    return _gated_residual(residual, out, gate)
                return fused_gated_residual(residual, out, gate)
            callback = MethodType(residual_call, layer)
            patches.append((layer, callback))
        for layer, callback in patches:
            setattr(layer, DISPATCH, callback)
        installation = GatedResidualInstallation(model, patches)
        setattr(model, INSTALLATION, installation)
        model.invalidate_inference_caches(weights_changed=False)
        return installation
