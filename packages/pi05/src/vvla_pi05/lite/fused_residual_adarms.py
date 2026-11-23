"""Default-off fusion of only attention-output residual and its next AdaRMS.

The exact reference is the already-enabled gated-residual and full-reduction
AdaRMS pair. This helper adds no parameter, activation, modulation or weight
cache; all checkpoint modules and existing installations stay in place.
"""
import hashlib
from pathlib import Path
from types import MethodType
import weakref

import torch

from vvla_pi05.lite.fused_adarms import _validate_norm, _projection_contract
from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
from vvla_pi05.lite.fused_gated_residual import _validate_inputs

INSTALLATION = '_lite_residual_adarms_installation'
DISPATCH = '_lite_residual_adarms'
VERSION = 'attention-residual-post-adarms-fp32-no-fma-v1'


def fused_residual_adarms(norm, residual, output, gate, cond=None, modulation=None):
    """Return (updated FP32 residual, BF16 postnorm, unchanged FP32 next gate)."""
    from vvla_pi05.models.pi05.modeling_pi05 import _gated_residual
    if (not residual.is_cuda or norm.training or torch.is_grad_enabled()
            or torch.is_autocast_enabled(residual.device.type) or gate is None):
        updated = _gated_residual(residual, output, gate)
        normalized, next_gate = norm(updated, cond, modulation=modulation)
        return updated, normalized.to(torch.bfloat16), next_gate
    _validate_inputs(residual, output, gate)
    device = _validate_norm(norm)
    if (norm.dense is None or residual.shape[-1] != norm.dim or residual.device != device
            or residual.dtype != torch.float32):
        raise ValueError('Residual-AdaRMS requires conditioned postnorm on matching FP32 residual rows')
    if modulation is None:
        if (cond is None or cond.shape != (residual.shape[0], norm.cond_dim) or cond.device != device):
            raise ValueError('Residual-AdaRMS requires cond [B,cond_dim] or precomputed modulation')
        modulation = norm.dense(cond.to(norm.dense.weight.dtype))
    if (modulation.shape != (residual.shape[0], 3 * norm.dim) or modulation.device != device
            or modulation.dtype not in (torch.float32, torch.bfloat16) or not modulation.is_contiguous()):
        raise ValueError('Residual-AdaRMS modulation must be contiguous FP32/BF16 [B,3D]')
    expanded = modulation.unsqueeze(1) if residual.ndim == 3 else modulation
    next_gate = expanded.chunk(3, dim=-1)[2].to(residual.dtype)
    updated = torch.empty_like(residual)
    normalized = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)
    tokens = residual.shape[1] if residual.ndim == 3 else 1
    gate_token_stride = gate.stride(1) if gate.ndim == 3 and gate.shape[1] != 1 else 0
    import triton
    from vvla_pi05.lite.fused_residual_adarms_kernel import residual_adarms_kernel
    residual_adarms_kernel[(residual.numel() // norm.dim,)](
        residual, output, gate, modulation, updated, normalized,
        WIDTH=norm.dim, TOKENS=tokens, EPS=float(norm.eps),
        GB=gate.stride(0), GS=gate_token_stride,
        BLOCK=triton.next_power_of_2(norm.dim), num_warps=4, enable_fp_fusion=False,
    )
    return updated, normalized, next_gate


class _Patch:
    def __init__(self, layer, adarms_patch):
        self.layer, self.norm, self.norm_forward = layer, layer.post_attention_layernorm, adarms_patch.forward
        self.projections = (layer.mlp.up_proj, layer.mlp.gate_proj)
        self.residual_dispatch = layer._lite_gated_residual
        self.device = _validate_norm(self.norm)
        self.dim, self.cond_dim, self.eps = self.norm.dim, self.norm.cond_dim, float(self.norm.eps)
        def forward(layer, residual, output, gate, cond=None, modulation=None):
            if layer.training:
                updated = self.residual_dispatch(residual, output, gate)
                normalized, next_gate = self.norm(updated, cond, modulation=modulation)
                return updated, normalized.to(torch.bfloat16), next_gate
            return fused_residual_adarms(self.norm, residual, output, gate, cond, modulation)
        self.forward = MethodType(forward, layer)

    def validate(self):
        layer = self.layer
        if (getattr(layer, DISPATCH, None) is not self.forward or layer.post_attention_layernorm is not self.norm
                or self.norm.__dict__.get('forward') is not self.norm_forward
                or getattr(layer, '_lite_gated_residual', None) is not self.residual_dispatch
                or layer.mlp.up_proj is not self.projections[0] or layer.mlp.gate_proj is not self.projections[1]):
            raise RuntimeError('Residual-AdaRMS layer/norm/consumer dispatch changed; clear graphs and reinstall')
        if (_validate_norm(self.norm) != self.device or self.norm.dense is None
                or (self.norm.dim, self.norm.cond_dim, float(self.norm.eps)) != (self.dim, self.cond_dim, self.eps)):
            raise ValueError('Residual-AdaRMS norm geometry/device/epsilon changed; clear graphs and reinstall')
        _projection_contract(self.projections, self.dim, torch.bfloat16, self.device)


class ResidualAdaRMSInstallation:
    def __init__(self, model, patches, adarms, residual):
        self._model = weakref.ref(model)
        self._patches, self._adarms, self._residual = patches, adarms, residual
        self.enabled = True

    def prepare(self, generation):
        model = self._model()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or getattr(model, '_lite_fused_adarms_installation', None) is not self._adarms
                or getattr(model, '_lite_gated_residual_installation', None) is not self._residual
                or not self._adarms.enabled or not self._adarms.fuse_reduction or not self._residual.enabled):
            raise RuntimeError('Residual-AdaRMS prerequisites changed or installation is inactive')
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers) != len(self._patches) or any(layer is not patch.layer for layer, patch in zip(layers, self._patches, strict=True)):
            raise RuntimeError('Residual-AdaRMS layer topology changed')
        if model.training:
            raise ValueError('Residual-AdaRMS graph preparation requires eval mode')
        for patch in self._patches:
            patch.validate()

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / 'models/pi05/modeling_pi05.py'
        paths = (Path(__file__), Path(__file__).with_name('fused_residual_adarms_kernel.py'), core)
        return dict(version=VERSION, enabled=self.enabled, pairs_per_layer=1, layers=len(self._patches),
                    scope='attention output gated residual -> post_attention_layernorm only',
                    residual_dtype='float32', normalized_dtype='bfloat16', next_gate_dtype='float32',
                    output_ownership='fresh residual and normalized tensors; original next-gate view semantics',
                    operation_order='FP32 multiply; FP32 add; FP32 square/tree sum/rsqrt; FP32 modulation; BF16 norm store',
                    enable_fp_fusion=False, reduction='same block and num_warps=4 as existing adarms_with_reduction',
                    parameter_cache=False, activation_cache=False,
                    numerical_limit='must pass bitwise against existing two CUDA kernels and native capture contracts',
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Residual-AdaRMS installation is inactive')
        with _mutation_guard(model, 'residual AdaRMS'):
            if any(getattr(p.layer, DISPATCH, None) is not p.forward for p in self._patches):
                raise RuntimeError('Residual-AdaRMS dispatch was modified externally')
            for patch in self._patches:
                delattr(patch.layer, DISPATCH)
            delattr(model, INSTALLATION)
            self.enabled = False
            model.invalidate_inference_caches(weights_changed=False)


def install_residual_adarms(model):
    """Install after full-reduction AdaRMS and gated residual; explicit opt-in."""
    with _mutation_guard(model, 'residual AdaRMS'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Residual-AdaRMS is already installed')
        adarms = getattr(model, '_lite_fused_adarms_installation', None)
        residual = getattr(model, '_lite_gated_residual_installation', None)
        if (adarms is None or residual is None or not adarms.enabled or not residual.enabled
                or not adarms.fuse_reduction):
            raise ValueError('Residual-AdaRMS requires existing full-reduction AdaRMS and gated residual')
        layers = tuple(model.paligemma_with_expert.gemma_expert.model.layers)
        if not layers or len({id(layer) for layer in layers}) != len(layers):
            raise ValueError('Residual-AdaRMS requires distinct nonempty expert layers')
        norm_patches = {id(p.norm): p for p in adarms._patches}
        patches = []
        for layer in layers:
            norm = layer.post_attention_layernorm
            patch = norm_patches.get(id(norm))
            if (layer.training or hasattr(layer, DISPATCH) or patch is None
                    or patch.output_dtype != torch.bfloat16 or norm.dense is None
                    or norm.__dict__.get('forward') is not patch.forward
                    or not callable(getattr(layer, '_lite_gated_residual', None))):
                raise ValueError('Residual-AdaRMS requires original installed conditioned BF16 postnorm dispatch')
            _projection_contract((layer.mlp.up_proj, layer.mlp.gate_proj), norm.dim, torch.bfloat16, _validate_norm(norm))
            patches.append(_Patch(layer, patch))
        installed = []
        handle = ResidualAdaRMSInstallation(model, patches, adarms, residual)
        try:
            for patch in patches:
                setattr(patch.layer, DISPATCH, patch.forward)
                installed.append(patch)
            setattr(model, INSTALLATION, handle)
            model.invalidate_inference_caches(weights_changed=False)
            return handle
        except BaseException:
            for patch in installed:
                delattr(patch.layer, DISPATCH)
            if getattr(model, INSTALLATION, None) is handle:
                delattr(model, INSTALLATION)
            handle.enabled = False
            raise
