"""Explicit SM89 17-down FP8 owner; all 34 gate/up projections remain BF16.

One immutable cold packing epoch owns its evidence and tensors. Changed source
weights, packed tensors, precision or topology fail closed. Retire graphs and
reinstall to establish new content hashes/recipe/preflight; warm prepare never
replaces storage before the native runner has retired its old graphs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from types import MethodType
import weakref

import torch
from torch import nn

from . import prefix_fp8_down_sm89 as native
from .fused_expert_mlp import _mutation_guard
from ..models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_prefix_fp8_installation'
VERSION = 'pi05-prefix-down-fp8-live-owner-v1'
PRECISION = 'mixed_bf16_fp8_prefix'
CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
PROJECTIONS = ('down_proj',)
LIVE_CHECKS = ['module_forward_bindings', 'original_weight_identity', 'quantized_weight_identity',
    'weight_scale_identity', 'backend_identity', 'unique_runner_generation',
    'no_extra_quantized_sites', 'unchanged_nonprefix_precision', 'supported_capacity_dispatch',
    'scalar_one_identity', 'unchanged_gate_up_precision']
PRECISION_BOUNDARIES = dict(vision='unchanged_selected_bf16_control',
    prefix_attention='unchanged_selected_bf16_control', expert_projections='unchanged_selected_bf16_control',
    normalization='unchanged_selected_control_FP32_boundaries', residual_euler='unchanged_selected_control_FP32_boundaries',
    action_time_projections='unchanged_selected_control_FP32_boundaries',
    prefix_gate_up='unchanged_selected_bf16_control')


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
    # The experiment queue selects CUDA_VISIBLE_DEVICES by physical UUID. Do
    # not accidentally query host GPU0 when this process sees a remapped GPU0.
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, 'uuid', ''))
    if not uuid.startswith('GPU-'):
        index = torch.cuda.current_device() if device.index is None else device.index
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        uuid = visible[index].strip() if index < len(visible) else ''
    if not uuid.startswith('GPU-'):
        raise RuntimeError('FP8 evidence requires a resolved physical CUDA GPU UUID')
    fields = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    name, actual_uuid, driver, memory = [value.strip() for value in fields]
    if actual_uuid != uuid or name != props.name:
        raise RuntimeError('CUDA and nvidia-smi physical GPU identity disagree')
    return dict(name=name, uuid=uuid, driver_version=driver, memory_mib=int(memory), hostname=socket.gethostname())


def _load_evidence(config, device):
    from .prefix_fp8_evidence import (BINARY_FILES, DECLARATIONS, recipe_for_variant,
                                     validate_backend_payload, validate_evidence_config)
    validate_evidence_config(config)
    raw = Path(config['backend_report']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['backend_report_sha256']:
        raise ValueError('Prefix FP8 backend original report bytes differ from the requested SHA')
    payload = json.loads(raw)
    if type(payload) is not dict or payload.get('quantizer_variant') != native.VARIANT:
        raise ValueError('Prefix down owner requires its explicit SM89 backend report variant')
    native.validate_runtime(device)
    root, hardware = Path(torch.__file__).resolve().parent, _hardware(device)
    core = native.implementation_manifest()
    backend = dict(operator='torch._scaled_mm', dispatch='ATEN', torch_version=str(torch.__version__),
        torch_git_version=torch.version.git_version, cuda_version=torch.version.cuda,
        hardware=hardware, capability=list(torch.cuda.get_device_capability(device)),
        compile_options=dict(native.COMPILE_OPTIONS),
        installed_declarations={name: _sha(root / name) for name in DECLARATIONS},
        installed_binary_sha256={name: _sha(root / name) for name in BINARY_FILES},
        probe_source_sha256=payload.get('source', {}).get('script_sha256'),
        shared_helper_files_sha256=payload.get('source', {}).get('shared_helper_files_sha256'),
        probe_recipe_sha256=_digest(recipe_for_variant(native.VARIANT)),
        report_sha256=config['backend_report_sha256'], quantizer_variant=native.VARIANT,
        quantizer_files_sha256=core['core_files_sha256'], supported_capacities=list(native.SUPPORTED_CAPACITIES))
    validate_backend_payload(payload, backend, hardware)
    if _canonical(payload.get('core_implementation_manifest')) != _canonical(core):
        raise ValueError('Executed backend did not use this actual production core')
    return raw, backend


def _layers(model):
    return model.paligemma_with_expert.paligemma.model.language_model.layers


def _validate_module(module, device, shape, *, installing=False):
    if (type(module) is not nn.Linear or module.training or module.bias is not None
            or module._forward_hooks or module._forward_pre_hooks
            or (installing and 'forward' in module.__dict__)
            or tuple(module.weight.shape) != shape or module.weight.dtype != torch.bfloat16
            or module.weight.device != device or device.type != 'cuda' or not module.weight.is_contiguous()
            or (module.in_features, module.out_features) != (shape[1], shape[0])):
        raise ValueError('Prefix FP8 requires original contiguous CUDA BF16 bias-free Linear projections')


def _selected_modules(model):
    from ..models.pi05.precision_pi05 import FusedPrefixMLP, QuantizedLinear
    layers = _layers(model)
    if len(layers) != 18 or model.optimization_flags()[2] is not True:
        raise ValueError('Prefix FP8 requires 18 prefix layers with the final layer KV-only')
    modules, gate_up = [], []
    for layer in layers[:17]:
        mlp = layer.mlp
        if (type(mlp) is not FusedPrefixMLP or mlp.training or mlp.concatenate or mlp.fuse_down_quant
                or 'forward' in mlp.__dict__ or mlp._forward_hooks or mlp._forward_pre_hooks):
            raise ValueError('Prefix FP8 preserves the independent original FusedPrefixMLP/GELU boundary')
        modules.append(mlp.down_proj)
        for name in ('gate_proj', 'up_proj'):
            module = getattr(mlp, name)
            _validate_module(module, mlp.down_proj.weight.device, (16384, 2048), installing=True)
            gate_up.append(module)
    if (len({id(module) for module in (*modules, *gate_up)}) != 51
            or any(isinstance(m, QuantizedLinear) for m in model.modules())):
        raise ValueError('Require exactly 17 distinct down sites and no pre-existing quantized modules')
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


class _UnitScale:
    """One ordinary versioned tensor shared by the 17 scalar GEMMs.

    Exact content is checked once, before publication. Every prepare checks all
    six identity fields; the same member is enrolled in the prepared compile
    scope, so no graph can substitute an unowned scalar constant.
    """
    def __init__(self, owner, device):
        self.owner, self.device = owner, device
        self.value = self._original = None
        self._identity = self.record = None
        self._prepared_compile_token = None

    def initialize(self):
        # Publish ownership before the cold content read can synchronize/fail;
        # this allocation remains reachable for explicit drain/retry cleanup.
        with torch.inference_mode(False):
            self.value = torch.ones((1,), device=self.device, dtype=torch.float32)
        self.owner._pending.append(self.value)
        proof = native.validate_unit_scale(self.value)
        self._original = self.value
        self._identity = tensor_identity(self.value)
        self.record = _tensor_record(self.value)
        if (proof.get('passed') is not True or proof.get('raw_sha256') != self.record['content_sha256']
                or self.record['content_sha256'] != hashlib.sha256(b'\x00\x00\x80\x3f').hexdigest()):
            raise ValueError('Prefix down scalar one lacks exact cold content proof')
        self.validate()
        self.owner._pending.clear()

    def validate(self):
        value = self.value
        if (value is not self._original or value is None or torch.is_inference(value)
                or value.requires_grad or value.device != self.device or value.dtype != torch.float32
                or tuple(value.shape) != (1,) or value.stride() != (1,)
                or tensor_identity(value) != self._identity):
            raise RuntimeError('Prefix down scalar-one immutable identity changed; retire and reinstall')

    def retire(self):
        self._prepared_compile_token = None
        self.value = self._original = None


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
        layer, projection = self.index, 'down_proj'
        self.record = dict(name=f'prefix.{layer}.{projection}',
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None,
            original_weight=_tensor_record(self.weight), quantized_weight=_tensor_record(self.quantized),
            weight_scale=_tensor_record(self.scale))
        self.owner._pending.clear()  # Hash copies/finite gates retired cold work.

    def validate(self):
        if (self.module.__dict__.get('forward') is not self.forward or self.module.weight is not self.weight
                or self.quantized is not self._packed[0] or self.scale is not self._packed[1]):
            raise RuntimeError('Prefix FP8 source/forward/packed ownership changed; retire and reinstall')
        _validate_module(self.module, self.device, self.shape)
        if tuple(tensor_identity(t) for t in (self.weight, self.quantized, self.scale)) != self._identities:
            raise RuntimeError('Prefix FP8 immutable packing epoch changed; retire and reinstall with new evidence')

    def call(self, module, x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type)
                or x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous()
                or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in native.SUPPORTED_CAPACITIES
                or x.shape[2] != self.shape[1]):
            raise ValueError('Prefix FP8 only supports the explicit B1 inference geometry; no BF16 fallback')
        if torch.compiler.is_compiling():
            if (self._prepared_compile_token is None
                    or self.owner.unit._prepared_compile_token is not self._prepared_compile_token):
                raise RuntimeError('Prefix FP8 compilation requires the native prepared owner scope')
        else:
            self.owner.assert_execution_owner()
            self.validate()
            self.owner.unit.validate()
        y, _, _ = native.scaled_linear(x.view(-1, self.shape[1]), self.quantized, self.scale,
                                      self.owner.unit.value)
        return y.view(1, x.shape[1], self.shape[0])

    def restore(self):
        if self.module.__dict__.get('forward') is self.forward:
            delattr(self.module, 'forward')
        self._prepared_compile_token = None


class PrefixDownFP8Installation:
    def __init__(self, model, evidence, raw, backend):
        self._model = weakref.ref(model)
        self.evidence, self.raw_report, self.backend = deepcopy(evidence), raw, deepcopy(backend)
        self.sites, self.enabled, self._runner = (), True, None
        self._installed_sites, self._pending = (), []
        self.unit = self._installed_unit = None
        self._generation = self._thread = self._signature = None
        self._bindings = _backend_bindings()
        self._math_settings = _math_settings()
        target_ids = {id(module) for module in _selected_modules(model)}
        self._gate_up = tuple((getattr(layer.mlp, name), getattr(layer.mlp, name).weight)
            for layer in _layers(model)[:17] for name in ('gate_proj', 'up_proj'))
        self._nonprefix_forwards = tuple((m, m.__dict__.get('forward')) for m in model.modules() if id(m) not in target_ids)
        self._epoch_tensors = tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers()))
        self._precision = tuple((name, id(m), type(m), tuple((n, id(child)) for n, child in m._modules.items()),
            tuple((n, type(t), t.dtype, t.device)
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
                or runner is None or runner._closed or runner.model is not model or not runner.compile_sample_actions
                or not runner._call_lock.locked()
                or tuple(r for r in model._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix FP8 requires its unique locked native runner')
        return model

    def _validate(self, *, full=False):
        model = self._model()
        if not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix FP8 installation is inactive')
        if self._bindings != _backend_bindings():
            raise RuntimeError('Prefix FP8 backend callable binding changed')
        from .compilation import INDUCTOR_OPTIONS
        if _canonical(INDUCTOR_OPTIONS) != _canonical(native.COMPILE_OPTIONS) or _math_settings() != self._math_settings:
            raise RuntimeError('Selected compile/math settings changed inside the FP8 epoch')
        if any(m.__dict__.get('forward') is not forward for m, forward in self._nonprefix_forwards):
            raise RuntimeError('Nonprefix forward binding changed; precision scope is no longer proven')
        modules = _selected_modules(model)
        current_gate_up = tuple(getattr(layer.mlp, name) for layer in _layers(model)[:17]
                                for name in ('gate_proj', 'up_proj'))
        if any(current is not original or current.weight is not weight
               for current, (original, weight) in zip(current_gate_up, self._gate_up, strict=True)):
            raise RuntimeError('Prefix gate/up original BF16 module/Parameter binding changed')
        if (len(self.sites) != 17 or len(self._installed_sites) != 17
                or any(site is not original for site, original in zip(self.sites, self._installed_sites, strict=True))
                or any(site.module is not module for site, module in zip(self.sites, modules, strict=True))):
            raise RuntimeError('Prefix FP8 exact ordered site topology changed')
        for site in self.sites:
            site.validate()
        if self.unit is not self._installed_unit or self.unit is None:
            raise RuntimeError('Prefix down scalar-one owner binding changed')
        self.unit.validate()
        if full:
            current = tuple((name, id(m), type(m), tuple((n, id(child)) for n, child in m._modules.items()),
                tuple((n, type(t), t.dtype, t.device)
                for n, t in (*m._parameters.items(), *m._buffers.items()) if t is not None))
                for name, m in model.named_modules())
            if current != self._precision:
                raise RuntimeError('Nonprefix precision/module scope changed')
            if tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers())) != self._epoch_tensors:
                raise RuntimeError('Original model tensors changed; establish a new immutable FP8 identity')

    def prepare(self, generation):
        model = self._owner()
        if type(generation) is not int or generation != model._inference_cache.generation:
            raise RuntimeError('Prefix FP8 requires the complete current shared lifecycle generation')
        if self._generation is not None and generation != self._generation:
            # Explicit invalidation can signal a .data write whose Tensor
            # version did not change. Equal signatures cannot prove equal
            # contents; a published packing epoch must never silently accept it.
            raise RuntimeError('Prefix FP8 epoch invalidated; retire and reinstall with new content identity')
        self._validate(full=self._signature is None)
        model_signature = model._inference_cache.signature
        if model_signature is None:
            raise RuntimeError('Prefix FP8 cannot skip the full model lifecycle snapshot')
        signature = (model_signature, tensor_identity(self.unit.value))
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
        return (*self.sites, self.unit)

    def manifest(self):
        self._validate(full=True)
        return deepcopy(self._manifest)

    def validate_precision_manifest(self):
        self._validate(full=True)
        if (self._generation is not None
                and self._generation != self._model()._inference_cache.generation):
            raise RuntimeError('Prefix down live manifest generation changed; retire and reinstall')
        raw, backend = _load_evidence(self.evidence, self.sites[0].device)
        if raw != self.raw_report or _canonical(backend) != _canonical(self.backend):
            raise RuntimeError('Actual backend evidence/runtime changed since packing')
        runner = None if self._runner is None else self._runner()
        if (runner is None or runner._closed or runner.model is not self._model() or not runner.compile_sample_actions
                or tuple(r for r in self._model()._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix FP8 live manifest requires its active native runner')
        return deepcopy(self._manifest)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix FP8 installation is inactive')
        with _mutation_guard(model, 'prefix FP8 uninstall'):
            runner = None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain native graph work before freeing prefix FP8 packed storage')
            if self._pending or runner is None:
                torch.cuda.synchronize(next(model.parameters()).device)
            for site in reversed(self._installed_sites):
                site.restore()
            if self._installed_unit is not None:
                self._installed_unit.retire()
            delattr(model, INSTALLATION)
            self.enabled = False
            self.sites = ()
            self._installed_sites = ()
            self.unit = self._installed_unit = None
            self._pending.clear()
            self._runner = None
            model.invalidate_inference_caches(weights_changed=False)


def _backend_bindings():
    return (native.quantize_rows, native.native_scaled_mm, native.scaled_linear,
            native.scalar_mm_fp32, native.rowwise_postscale, native.validate_unit_scale,
            native.create_unit_scale, torch._scaled_mm)


def install_prefix_fp8_down(model, evidence):
    """Install exactly 17 down sites after the BF16 recipe; no hardware fallback."""
    from .prefix_fp8_evidence import recipe_for_variant
    from ..models.pi05 import fused_mlp_pi05
    with _mutation_guard(model, 'prefix FP8 installation'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Prefix FP8 is already installed')
        modules = _selected_modules(model)
        device = modules[0].weight.device
        for module in modules:
            _validate_module(module, device, (2048, 16384), installing=True)
        raw, backend = _load_evidence(evidence, device)
        owner = PrefixDownFP8Installation(model, evidence, raw, backend)
        sites, installed = [], []
        try:
            owner.unit = owner._installed_unit = _UnitScale(owner, device)
            owner.unit.initialize()
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
            owner._manifest = dict(schema=1, precision=PRECISION, implementation='pi05-prefix-down-scalar-fp8-sm89-v1',
                checkpoint_sha256=CHECKPOINT_SHA256, scope='prefix_mlp_down_layers_0_16',
                scalar_one=deepcopy(owner.unit.record),
                precision_boundaries=deepcopy(PRECISION_BOUNDARIES), backend=deepcopy(backend),
                quantizer=dict(source_sha256=_digest(backend['quantizer_files_sha256']),
                    nonlinear_source_sha256=_sha(fused_mlp_pi05.__file__), recipe=recipe_for_variant(native.VARIANT),
                    input_dtype='bfloat16', output_dtype='bfloat16', accumulator_dtype='float32',
                    weight_layout='contiguous_NK_then_transpose',
                    scale_layout='activation[M,1];weight[N,1]', scale_result=None,
                    intermediate_dtype='float32',
                    postscale='two ordered mul.rn.f32 sx[row] then sw[column]; sole BF16 output cast',
                    gate_up='unchanged_BF16',
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
            if owner._installed_unit is not None:
                owner._installed_unit.retire()
            owner.unit = owner._installed_unit = None
            owner._pending.clear()
            sites.clear()
            raise
