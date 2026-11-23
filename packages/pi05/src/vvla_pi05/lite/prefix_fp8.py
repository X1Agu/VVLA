"""Historical prefix FP8 owner, retaining packed storage and graph lifecycle.

The historical SM120 projection/quantizer arithmetic is unchanged. Admission
uses runtime capability and geometry; audit files and author identity are absent.
The final KV-only prefix layer is excluded at every supported model depth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import MethodType
import weakref

import torch
from torch import nn

from . import prefix_fp8_native as native
from .fused_expert_mlp import _mutation_guard
from ..models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_prefix_fp8_installation'
VERSION = 'pi05-prefix-fp8-live-owner-v1'
PRECISION = 'mixed_bf16_fp8_prefix'
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
LIVE_CHECKS = ['module_forward_bindings', 'original_weight_identity', 'quantized_weight_identity',
    'weight_scale_identity', 'backend_identity', 'unique_runner_generation',
    'no_extra_quantized_sites', 'unchanged_nonprefix_precision', 'supported_capacity_dispatch']
PRECISION_BOUNDARIES = dict(vision='unchanged_selected_bf16_control',
    prefix_attention='unchanged_selected_bf16_control', expert_projections='unchanged_selected_bf16_control',
    normalization='unchanged_selected_control_FP32_boundaries', residual_euler='unchanged_selected_control_FP32_boundaries',
    action_time_projections='unchanged_selected_control_FP32_boundaries')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _tensor_record(value):
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype).removeprefix('torch.'),
        content_sha256=hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())


def _math_settings():
    return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            str(torch.backends.cuda.preferred_blas_library()))


def _hardware(device):
    return dict(name=torch.cuda.get_device_name(device),
                capability=list(torch.cuda.get_device_capability(device)))


def _runtime_descriptor(device):
    native.validate_runtime(device)
    core = native.implementation_manifest()
    return dict(operator='torch._scaled_mm', torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, hardware=_hardware(device),
                core=core, quantizer_files_sha256=core['quantizer_files_sha256'],
                qualification='not inferred from runtime admission')


def _layers(model):
    return model.paligemma_with_expert.paligemma.model.language_model.layers


def _gelu_extension(model):
    if getattr(model, '_lite_prefix_fp8_gelu_installation', None) is not None:
        raise RuntimeError('This historical prefix recipe does not include the separate GELU experiment')
    return None


def _vision_extension(model):
    if getattr(model, '_lite_vision_fp8_installation', None) is not None:
        raise RuntimeError('This historical prefix recipe preserves BF16 vision')
    return None


def _validate_module(module, device, shape, *, installing=False):
    if (type(module) is not nn.Linear or module.training or module.bias is not None
            or module._forward_hooks or module._forward_pre_hooks
            or (installing and 'forward' in module.__dict__)
            or tuple(module.weight.shape) != shape or module.weight.dtype != torch.bfloat16
            or module.weight.device != device or device.type != 'cuda' or not module.weight.is_contiguous()
            or (module.in_features, module.out_features) != (shape[1], shape[0])):
        raise ValueError('Prefix FP8 requires original contiguous CUDA BF16 bias-free Linear projections')


def _selected_modules(model, *, inventory_proof=None):
    from ..models.pi05.precision_pi05 import FusedPrefixMLP, QuantizedLinear
    layers = _layers(model)
    if len(layers) < 2 or model.optimization_flags()[2] is not True:
        raise ValueError('Prefix FP8 requires prefix layers with the final layer KV-only')
    modules, fusion = [], _gelu_extension(model)
    for layer in layers[:-1]:
        mlp = layer.mlp
        if (type(mlp) is not FusedPrefixMLP or mlp.training or mlp.concatenate or mlp.fuse_down_quant
                or ('forward' in mlp.__dict__ and (fusion is None or not fusion.accepts_forward(mlp)))
                or mlp._forward_hooks or mlp._forward_pre_hooks):
            raise ValueError('Prefix FP8 preserves the independent original FusedPrefixMLP/GELU boundary')
        modules.extend(getattr(mlp, name) for name in PROJECTIONS)
    if inventory_proof is not None:
        raise RuntimeError('Historical prefix owner does not accept an unrelated inventory proof')
    if len({id(module) for module in modules}) != 3 * (len(layers) - 1) or any(
            isinstance(m, QuantizedLinear) for m in model.modules()):
        raise ValueError('Require one unique gate/up/down triple per live prefix MLP and no pre-existing quantized modules')
    return tuple(modules)


@torch.no_grad()
def _pack_weight(weight, keepalive):
    # Match the backend's cold weight path exactly. This is not a captured op.
    if torch.is_inference_mode_enabled():
        raise ValueError('Pack FP8 weights outside inference_mode so live version counters exist')
    fp32 = weight.detach().float()
    keepalive.append(fp32)
    scale = (fp32.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    keepalive.append(scale)
    quantized = (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    keepalive.append(quantized)
    if not bool(torch.isfinite(scale).all() & (scale > 0).all() & torch.isfinite(quantized.float()).all()):
        raise ValueError('Nonfinite prefix weights/scales cannot establish an FP8 epoch')
    return quantized, scale.contiguous()


class _Site:
    def __init__(self, owner, index, module):
        self.owner, self.index, self.module, self.weight = owner, index, module, module.weight
        self.device, self.shape = self.weight.device, tuple(self.weight.shape)
        self.quantized = self.scale = None
        self._packed = self._identities = None
        self._prepared_compile_token = None
        def forward(module, x, _site=self):
            return _site.call(module, x)
        self.forward = MethodType(forward, module)
        self.record = None

    def initialize(self):
        self.quantized, self.scale = _pack_weight(self.weight, self.owner._pending)
        self._packed = (self.quantized, self.scale)
        self._identities = tuple(tensor_identity(t) for t in (self.weight, *self._packed))
        layer, projection = self.index // 3, PROJECTIONS[self.index % 3]
        self.record = dict(name=f'prefix.{layer}.{projection}',
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None,
            original_weight=_tensor_record(self.weight), quantized_weight=_tensor_record(self.quantized),
            weight_scale=_tensor_record(self.scale))
        self.owner._pending.clear()  # Hash copies/finite gates retired cold work.

    def _validate_binding_geometry(self):
        if (self.module.__dict__.get('forward') is not self.forward or self.module.weight is not self.weight
                or self.quantized is not self._packed[0] or self.scale is not self._packed[1]):
            raise RuntimeError('Prefix FP8 source/forward/packed ownership changed; retire and reinstall')
        _validate_module(self.module, self.device, self.shape)

    def validate(self):
        self._validate_binding_geometry()
        if tuple(tensor_identity(t) for t in (self.weight, self.quantized, self.scale)) != self._identities:
            raise RuntimeError('Prefix FP8 immutable packing epoch changed; retire and reinstall')

    def call(self, module, x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type)
                or x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous()
                or x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or (x.shape[0] * x.shape[1]) % 16
                or x.shape[2] != self.shape[1]):
            raise ValueError('Prefix FP8 only supports the aligned batched inference geometry; no BF16 fallback')
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('Prefix FP8 compilation requires the native prepared owner scope')
        else:
            self.owner.assert_execution_owner()
            self.validate()
        y, _, _ = native.scaled_linear(x.view(-1, self.shape[1]), self.quantized, self.scale)
        return y.view(x.shape[0], x.shape[1], self.shape[0])

    def restore(self):
        if self.module.__dict__.get('forward') is self.forward:
            delattr(self.module, 'forward')
        self._prepared_compile_token = None


class PrefixFP8Installation:
    def __init__(self, model, backend):
        self._model = weakref.ref(model)
        self.backend = deepcopy(backend)
        self.sites, self.enabled, self._runner = (), True, None
        self._installed_sites, self._pending = (), []
        self._generation = self._thread = self._signature = None
        self._native_epoch_validation = self._native_epoch_binding = None
        self._bindings = (native.quantize_rows, native.native_scaled_mm, native.scaled_linear, torch._scaled_mm)
        self._math_settings = _math_settings()
        target_ids = {id(module) for module in _selected_modules(model)}
        self._nonprefix_forwards = tuple((m, m.__dict__.get('forward')) for m in model.modules() if id(m) not in target_ids)
        self._epoch_tensors = tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers()))
        self._precision = tuple((name, type(m), tuple((n, type(t), t.dtype, t.device)
            for n, t in (*m._parameters.items(), *m._buffers.items()) if t is not None))
            for name, m in model.named_modules())
        self._manifest = None

    def bind_runner(self, runner):
        from .graph import Pi05NativeGraphRunner
        model = self._model()
        with _mutation_guard(model, 'prefix FP8 runner binding'):
            if (not self.enabled or self._runner is not None or type(runner) is not Pi05NativeGraphRunner
                    or runner.model is not model or not runner.compile_sample_actions
                    or tuple(r for r in model._graph_runners if not r._closed) != (runner,)):
                raise ValueError('Prefix FP8 requires exactly one compiled native graph runner')
            self._runner = weakref.ref(runner)

    def _owner(self):
        model, runner = self._model(), None if self._runner is None else self._runner()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked()
                or tuple(r for r in model._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix FP8 requires its unique locked native runner')
        return model

    def _validate(self, *, full=False, inventory_proof=None, native_epoch=False):
        model = self._model()
        if not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix FP8 installation is inactive')
        if self._bindings != (native.quantize_rows, native.native_scaled_mm, native.scaled_linear, torch._scaled_mm):
            raise RuntimeError('Prefix FP8 backend callable binding changed')
        from .compilation import INDUCTOR_OPTIONS
        if _canonical(INDUCTOR_OPTIONS) != _canonical(native.COMPILE_OPTIONS) or _math_settings() != self._math_settings:
            raise RuntimeError('Selected compile/math settings changed inside the FP8 epoch')
        fusion = _gelu_extension(model)
        vision = _vision_extension(model)
        if any(m.__dict__.get('forward') is not forward and
               (fusion is None or not fusion.accepts_original_forward(m, forward)) and
               (vision is None or not vision.accepts_original_forward(m, forward))
               for m, forward in self._nonprefix_forwards):
            raise RuntimeError('Nonprefix forward binding changed; precision scope is no longer proven')
        modules = _selected_modules(model, inventory_proof=inventory_proof)
        if (len(self.sites) != len(modules) or len(self._installed_sites) != len(modules)
                or any(site is not original for site, original in zip(self.sites, self._installed_sites, strict=True))
                or any(site.module is not module for site, module in zip(self.sites, modules, strict=True))):
            raise RuntimeError('Prefix FP8 exact ordered site topology changed')
        if native_epoch:
            from .fp8_tensor_epochs import validate_owner
            for site in self.sites:
                site._validate_binding_geometry()
            validate_owner(self, tuple(t for site in self.sites for t in
                (site.weight, site.quantized, site.scale)))
        else:
            for site in self.sites:
                site.validate()
        if fusion is not None:
            fusion.validate_bindings()
        if vision is not None:
            vision.validate_bindings()
        if full:
            current = tuple((name, type(m), tuple((n, type(t), t.dtype, t.device)
                for n, t in (*m._parameters.items(), *m._buffers.items()) if t is not None))
                for name, m in model.named_modules())
            if current != self._precision:
                raise RuntimeError('Nonprefix precision/module scope changed')
            if tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers())) != self._epoch_tensors:
                raise RuntimeError('Original model tensors changed; establish a new immutable FP8 identity')

    def prepare(self, generation, *, inventory_proof=None):
        model = self._owner()
        if type(generation) is not int or generation != model._inference_cache.generation:
            raise RuntimeError('Prefix FP8 requires the complete current shared lifecycle generation')
        if self._generation is not None and generation != self._generation:
            # Explicit invalidation can signal a .data write whose Tensor
            # version did not change. Equal signatures cannot prove equal
            # contents; a published packing epoch must never silently accept it.
            raise RuntimeError('Prefix FP8 epoch invalidated; retire and reinstall with new content identity')
        if self._native_epoch_validation is not self._native_epoch_binding:
            raise RuntimeError('Native FP8 epoch owner binding changed')
        self._validate(full=self._signature is None, inventory_proof=inventory_proof,
            native_epoch=self._native_epoch_validation is not None)
        signature = model._inference_cache.signature
        if signature is None:
            raise RuntimeError('Prefix FP8 cannot skip the full model lifecycle snapshot')
        if self._signature is not None and signature != self._signature:
            raise RuntimeError('FP8 source model/precision epoch changed; clear and reinstall before new preflight')
        self._signature = signature
        self._generation, self._thread = generation, threading.get_ident()

    def assert_execution_owner(self):
        model = self._owner()
        if (self._thread != threading.get_ident() or self._generation is None
                or self._generation != model._inference_cache.generation):
            raise RuntimeError('Prefix FP8 requires the current prepared generation/thread')

    def compilation_members(self, generation):
        self.assert_execution_owner()
        if generation != self._generation:
            raise RuntimeError('Prefix FP8 compiled generation is stale')
        self._validate()
        return self.sites

    def manifest(self):
        self._validate(full=True)
        return self._manifest_with_extension()

    def _manifest_with_extension(self):
        return deepcopy(self._manifest)

    def validate_precision_manifest(self):
        self._validate(full=True)
        actual = _runtime_descriptor(self.sites[0].device)
        if _canonical(actual) != _canonical(self.backend):
            raise RuntimeError('Prefix FP8 backend changed since cold packing')
        runner = None if self._runner is None else self._runner()
        if (runner is None or runner._closed or runner.model is not self._model()
                or tuple(r for r in self._model()._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix FP8 live manifest requires its active native runner')
        return self._manifest_with_extension()

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix FP8 installation is inactive')
        with _mutation_guard(model, 'prefix FP8 uninstall'):
            if _vision_extension(model) is not None:
                raise RuntimeError('Uninstall vision FP8 before its base prefix owner')
            if _gelu_extension(model) is not None:
                raise RuntimeError('Uninstall prefix GELU extension before its FP8 storage owner')
            if getattr(model, '_lite_prefix_fp8_inventory_installation', None) is not None:
                raise RuntimeError('Uninstall prefix FP8 inventory before its storage owner')
            runner = None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain native graph work before freeing prefix FP8 packed storage')
            if self._pending or runner is None:
                torch.cuda.synchronize(next(model.parameters()).device)
            for site in reversed(self._installed_sites):
                site.restore()
            delattr(model, INSTALLATION)
            self.enabled = False
            self.sites = ()
            self._installed_sites = ()
            self._native_epoch_validation = self._native_epoch_binding = None
            self._pending.clear()
            self._runner = None
            model.invalidate_inference_caches(weights_changed=False)


def install_prefix_fp8(model):
    """Install after the BF16 recipe, before native runner creation; no fallback."""
    from ..models.pi05 import fused_mlp_pi05
    with _mutation_guard(model, 'prefix FP8 installation'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Prefix FP8 is already installed')
        modules = _selected_modules(model)
        device = modules[0].weight.device
        for index, module in enumerate(modules):
            shape = (2048, 16384) if index % 3 == 2 else (16384, 2048)
            _validate_module(module, device, shape, installing=True)
        backend = _runtime_descriptor(device)
        owner = PrefixFP8Installation(model, backend)
        sites, installed = [], []
        try:
            for index, module in enumerate(modules):
                site = _Site(owner, index, module)
                sites.append(site)
                owner._installed_sites = tuple(sites)
                site.initialize()
            owner.sites = tuple(sites)
            for site in sites:
                site.module.forward = site.forward
                installed.append(site)
            setattr(model, INSTALLATION, owner)
            owner._manifest = dict(schema=1, precision=PRECISION, implementation='pi05-prefix-rowwise-fp8-v1',
                scope=f'prefix_mlp_layers_0_{len(_layers(model))-2}', historical_full_depth_sites=51,
                precision_boundaries=deepcopy(PRECISION_BOUNDARIES), backend=deepcopy(backend),
                quantizer=dict(source_sha256=_digest(backend['quantizer_files_sha256']),
                    nonlinear_source_sha256=_sha(fused_mlp_pi05.__file__), recipe=dict(variant=native.VARIANT, arithmetic=deepcopy(native.ARITHMETIC)),
                    input_dtype='bfloat16', output_dtype='bfloat16', accumulator_dtype='float32',
                    weight_layout='contiguous_NK_then_transpose',
                    scale_layout='activation[M,1];weight[N,1]_then_transpose', scale_result=None,
                    gate_up='independent_quantization',
                    nonlinear_boundary='BF16 gate/up -> gelu_tanh_mul -> BF16 down input'),
                lifecycle=dict(version=VERSION, source_sha256=_sha(__file__), enabled=True,
                    owner='Pi05NativeGraphRunner', original_parameters_preserved=True,
                    content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
                    live_checks=list(LIVE_CHECKS), fallback_permitted=False),
                sites=[deepcopy(site.record) for site in sites])
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException as error:
            # All cold tensor hashing/finite gates synchronize before publication.
            # Still explicitly drain before dropping partial packing allocations.
            try:
                torch.cuda.synchronize(device)
            except BaseException as cleanup:
                # Keep ownership reachable for caller cleanup after a failed drain.
                owner.sites = tuple(sites)
                setattr(model, INSTALLATION, owner)
                error.add_note('Partial FP8 packing drain failed; retained model owner: ' + str(cleanup))
                raise error
            for site in reversed(installed):
                site.restore()
            if getattr(model, INSTALLATION, None) is owner:
                delattr(model, INSTALLATION)
            owner.enabled = False
            owner.sites = owner._installed_sites = ()
            owner._pending.clear()
            sites.clear()
            raise
