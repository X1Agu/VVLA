"""Opt-in, reversible expert AdaRMS tail fusion; disabled unless installed.

Install on an eval model after clearing every live graph runner, outside
inference_mode. The module tree and all parameter objects/names remain intact.
The next request must recapture. Uninstall uses the same lifecycle contract.

By default, the original Torch square/mean remains separate work on CUDA. One
Triton kernel handles eps/rsqrt, normalization, modulation and consumer cast.
The independent fuse_reduction option moves FP32 square/mean into that kernel;
its reduction tree can differ from Torch, and it is disabled by default.
CPU, training, grad-enabled and autocast calls use the original Torch forward
plus the consumer cast. CUDA inputs require contiguous FP32/BF16 [B,D]/[B,S,D]
and matching contiguous [B,3D] modulation. No silent shape approximation occurs.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MethodType
import weakref

import torch
from torch import nn

VERSION = 'torch-mean-adarms-tail-v1'
REDUCTION_VERSION = 'triton-mean-adarms-v1'
_INSTALLATION = '_lite_fused_adarms_installation'


def _validate_norm(norm):
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05AdaRMSNorm
    if type(norm) is not Pi05AdaRMSNorm:
        raise TypeError('Fusion requires an unchanged Pi05AdaRMSNorm module')
    if type(norm.dim) is not int or not 1 <= norm.dim <= 8192:
        raise ValueError('AdaRMS fusion requires width 1..8192')
    if not math.isfinite(float(norm.eps)) or norm.eps <= 0:
        raise ValueError('AdaRMS epsilon must be finite and positive')
    dense = norm.dense
    if dense is None:
        weight = norm.weight
        if weight is None or weight.shape != (norm.dim,):
            raise ValueError('Unconditioned AdaRMS weight has incompatible width')
        values = (weight,)
    else:
        if type(dense) is not nn.Linear:
            raise ValueError('Conditioned AdaRMS requires the original dense projection')
        weight = dense.weight
        if weight.shape != (3 * norm.dim, norm.cond_dim):
            raise ValueError('Conditioned AdaRMS requires the original dense projection')
        bias = dense.bias
        if bias is None or bias.shape != (3 * norm.dim,):
            raise ValueError('Conditioned AdaRMS requires its original bias')
        values = (weight, bias)
    if any(value.dtype not in (torch.float32, torch.bfloat16) for value in values):
        raise ValueError('AdaRMS parameters must be FP32 or BF16')
    if any(value.device != values[0].device for value in values):
        raise ValueError('AdaRMS parameters must share one device')
    return values[0].device


def fused_adarms(norm, x, cond=None, modulation=None, *, output_dtype, fuse_reduction=False):
    """Return norm + original-dtype gate, with the declared next-projection cast."""
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05AdaRMSNorm
    if type(fuse_reduction) is not bool:
        raise ValueError('AdaRMS fuse_reduction must be a boolean')
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('AdaRMS output must target a BF16 or FP32 projection')
    device = _validate_norm(norm)
    if x.dtype not in (torch.float32, torch.bfloat16) or x.ndim not in (2, 3) or x.shape[-1] != norm.dim:
        raise ValueError('AdaRMS expects FP32/BF16 [B,D] or [B,S,D] with matching width')
    if x.device != device or any(size < 1 for size in x.shape):
        raise ValueError('AdaRMS inputs must be nonempty and on the parameter device')
    # Keep autograd and autocast semantics for unsupported execution contexts.
    # This still performs the same cast already present at the consumer boundary.
    if not x.is_cuda or norm.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type):
        result, gate = Pi05AdaRMSNorm.forward(norm, x, cond, modulation)
        return result.to(output_dtype), gate
    if not x.is_contiguous():
        raise ValueError('CUDA AdaRMS fusion requires contiguous residual rows')
    conditioned = norm.dense is not None
    gate = None
    if conditioned:
        if cond is None and modulation is None:
            raise ValueError('Conditioned AdaRMS requires cond or precomputed modulation')
        if modulation is None:
            if cond.shape != (x.shape[0], norm.cond_dim) or cond.device != device:
                raise ValueError('AdaRMS cond must be [B,cond_dim] on the residual device')
            modulation = norm.dense(cond.to(norm.dense.weight.dtype))
        if (modulation.shape != (x.shape[0], 3 * norm.dim) or modulation.device != device
                or modulation.dtype not in (torch.float32, torch.bfloat16) or not modulation.is_contiguous()):
            raise ValueError('AdaRMS modulation must be contiguous FP32/BF16 [B,3D] on the residual device')
        expanded = modulation.unsqueeze(1) if x.ndim == 3 else modulation
        gate = expanded.chunk(3, dim=-1)[2].to(x.dtype)
    import triton
    if fuse_reduction:
        output = torch.empty(x.shape, device=x.device, dtype=output_dtype)
        from vvla_repro_pi05.lite.fused_adarms_kernel import adarms_with_reduction
        adarms_with_reduction[(x.numel() // norm.dim,)](
            x, modulation if conditioned else x, x if conditioned else norm.weight, output,
            WIDTH=norm.dim, TOKENS=x.shape[1] if x.ndim == 3 else 1, EPS=float(norm.eps),
            CONDITIONED=conditioned, INPUT_BF16=x.dtype == torch.bfloat16,
            BLOCK=triton.next_power_of_2(norm.dim), num_warps=4, enable_fp_fusion=False,
        )
        return output, gate
    # Keep this exact expression and its reduction implementation/layout for
    # the original control path. No Torch variance is computed in the new path.
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    output = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    from vvla_repro_pi05.lite.fused_adarms_kernel import adarms_after_mean
    adarms_after_mean[(x.numel() // norm.dim,)](
        x, variance, modulation if conditioned else x,
        x if conditioned else norm.weight, output,
        WIDTH=norm.dim, TOKENS=x.shape[1] if x.ndim == 3 else 1, EPS=float(norm.eps),
        CONDITIONED=conditioned, INPUT_BF16=x.dtype == torch.bfloat16,
        BLOCK=triton.next_power_of_2(norm.dim), num_warps=4, enable_fp_fusion=False,
    )
    return output, gate


@contextmanager
def _mutation_guard(model):
    if model.training or torch.is_inference_mode_enabled():
        raise ValueError('Install/uninstall AdaRMS in eval mode outside inference_mode')
    with ExitStack() as locks:
        for runner in tuple(getattr(model, '_graph_runners', ())):
            lock = getattr(runner, '_call_lock', None)
            if lock is None or not lock.acquire(blocking=False):
                raise RuntimeError('Cannot mutate AdaRMS during an active/unknown graph request')
            locks.callback(lock.release)
            if runner.entries:
                raise ValueError('Clear all live graph runners before changing AdaRMS fusion')
        if not callable(getattr(model, 'invalidate_inference_caches', None)):
            raise TypeError('Model must provide inference cache invalidation')
        yield


def _projection_contract(projections, width, dtype, device):
    for projection in projections:
        if type(projection) is not nn.Linear:
            raise ValueError('AdaRMS consumer must be an unchanged Linear with matching width/device/dtype')
        weight = projection.weight
        if (weight.ndim != 2 or weight.shape[1] != width or weight.dtype != dtype
                or weight.device != device):
            raise ValueError('AdaRMS consumer must be an unchanged Linear with matching width/device/dtype')


def _forward_for(projections, dtype, fuse_reduction):
    def forward(norm, x, cond=None, modulation=None):
        device = norm.dense.weight.device if norm.dense is not None else norm.weight.device
        _projection_contract(projections, norm.dim, dtype, device)
        return fused_adarms(norm, x, cond, modulation, output_dtype=dtype, fuse_reduction=fuse_reduction)
    return forward


@dataclass
class _Patch:
    name: str
    norm: nn.Module
    forward: object
    output_dtype: torch.dtype


class AdaRMSInstallation:
    def __init__(self, model, patches, fuse_reduction):
        self._model = weakref.ref(model)
        self._patches = patches
        self._fuse_reduction = fuse_reduction
        self.enabled = True

    @property
    def fuse_reduction(self):
        return self._fuse_reduction

    def manifest(self):
        paths = (Path(__file__), Path(__file__).with_name('fused_adarms_kernel.py'))
        return dict(version=REDUCTION_VERSION if self.fuse_reduction else VERSION, enabled=self.enabled,
                    fuse_reduction=self.fuse_reduction,
                    reduction=('tl.sum(x.float32 * x.float32, axis=0) / WIDTH' if self.fuse_reduction
                               else 'torch.mean(torch.square(x.float()), dim=-1, keepdim=True)'),
                    reduction_tree=('Triton per-row tree; may differ from Torch' if self.fuse_reduction
                                    else 'unchanged Torch square/mean'),
                    square_dtype='float32', reduction_dtype='float32', rsqrt_dtype='float32',
                    enable_fp_fusion=False, gate_dtype='input dtype',
                    norms=[dict(name=p.name, output_dtype=str(p.output_dtype)) for p in self._patches],
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, _INSTALLATION, None) is not self:
            raise RuntimeError('AdaRMS installation is no longer active')
        with _mutation_guard(model):
            if any(p.norm.__dict__.get('forward') is not p.forward for p in self._patches):
                raise RuntimeError('An installed AdaRMS forward was modified externally')
            for patch in self._patches:
                delattr(patch.norm, 'forward')
            delattr(model, _INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_fused_adarms(model, *, fuse_reduction=False):
    """Install on all expert input/post norms + FP32 final norm; no defaults change."""
    if type(fuse_reduction) is not bool:
        raise ValueError('AdaRMS fuse_reduction must be a boolean')
    with _mutation_guard(model):
        if getattr(model, _INSTALLATION, None) is not None:
            raise ValueError('AdaRMS fusion is already installed')
        expert = model.paligemma_with_expert.gemma_expert.model
        selections = []
        for index, layer in enumerate(expert.layers):
            selections.append((f'expert.{index}.input_layernorm', layer.input_layernorm,
                               (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj), torch.bfloat16))
            selections.append((f'expert.{index}.post_attention_layernorm', layer.post_attention_layernorm,
                               (layer.mlp.up_proj, layer.mlp.gate_proj), torch.bfloat16))
        if not selections:
            raise ValueError('Expert must have at least one layer')
        selections.append(('expert.norm', expert.norm, (model.action_out_proj,), torch.float32))
        if len({id(norm) for _, norm, _, _ in selections}) != len(selections):
            raise ValueError('Shared expert norms are not supported by this candidate')
        patches = []
        # Validate every module before making any mutation.
        for name, norm, projections, dtype in selections:
            device = _validate_norm(norm)
            if norm.training or 'forward' in norm.__dict__:
                raise ValueError('AdaRMS candidate requires eval norms with original forwards')
            _projection_contract(projections, norm.dim, dtype, device)
            patches.append(_Patch(name, norm, MethodType(_forward_for(projections, dtype, fuse_reduction), norm), dtype))
        for patch in patches:
            patch.norm.forward = patch.forward
        installation = AdaRMSInstallation(model, patches, fuse_reduction)
        setattr(model, _INSTALLATION, installation)
        model.invalidate_inference_caches(weights_changed=False)
        return installation
