"""Opt-in 51-site prefix NVFP4 owner, preserving every original BF16 Parameter.

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

from . import prefix_nvfp4_native as native
from .fused_expert_mlp import _mutation_guard
from ..models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_prefix_nvfp4_installation'
VERSION = 'pi05-prefix-nvfp4-live-owner-v1'
PRECISION = 'mixed_bf16_nvfp4_prefix'
CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
LIVE_CHECKS = ['module_forward_bindings', 'original_weight_identity', 'quantized_weight_identity',
    'weight_scale_identity', 'backend_identity', 'unique_runner_generation',
    'no_extra_quantized_sites', 'unchanged_nonprefix_precision', 'supported_capacity_dispatch',
    'input_global_scale_identity', 'weight_global_scale_identity', 'alpha_identity', 'calibration_byte_identity']
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
    # The experiment queue selects CUDA_VISIBLE_DEVICES by physical UUID. Do
    # not accidentally query host GPU0 when this process sees a remapped GPU0.
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, 'uuid', ''))
    if not uuid.startswith('GPU-'):
        index = torch.cuda.current_device() if device.index is None else device.index
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        uuid = visible[index].strip() if index < len(visible) else ''
    if not uuid.startswith('GPU-'):
        raise RuntimeError('NVFP4 evidence requires a resolved physical CUDA GPU UUID')
    fields = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    name, actual_uuid, driver, memory = [value.strip() for value in fields]
    if actual_uuid != uuid or name != props.name:
        raise RuntimeError('CUDA and nvidia-smi physical GPU identity disagree')
    return dict(name=name, uuid=uuid, driver_version=driver, memory_mib=int(memory), hostname=socket.gethostname())


def _load_evidence(config, device, native_path):
    from .prefix_nvfp4_evidence import validate_evidence_config, validate_calibration, validate_backend_report
    validate_evidence_config(config)
    raw = Path(config['backend_report']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['backend_report_sha256']:
        raise ValueError('Prefix NVFP4 backend original report bytes differ from the requested SHA')
    raw_calibration = Path(config['calibration']).read_bytes()
    if hashlib.sha256(raw_calibration).hexdigest() != config['calibration_sha256']:
        raise ValueError('Prefix NVFP4 calibration original bytes differ from the requested SHA')
    payload = json.loads(raw)
    calibration = validate_calibration(json.loads(raw_calibration))
    runtime = native.load_native_backend(native_path, device)
    actual, hardware = runtime.manifest(), _hardware(device)
    validate_backend_report(payload, actual, hardware)
    if _canonical(calibration['hardware']) != _canonical(hardware):
        raise ValueError('NVFP4 calibration does not refer to this actual physical GPU')
    if _canonical(config['bf16_control']) != _canonical(calibration['bf16_control']):
        raise ValueError('NVFP4 calibration/control anchors disagree')
    backend = dict(variant=native.VARIANT, report_sha256=config['backend_report_sha256'],
        report_source_sha256=payload['source_sha256'], hardware=hardware, native_runtime=actual,
        support_scope='direct native operators; policy quality and actual wrapper contracts are separate',
        implementation=native.implementation_manifest())
    return raw, raw_calibration, calibration, backend, runtime


def _layers(model):
    return model.paligemma_with_expert.paligemma.model.language_model.layers


def _validate_module(module, device, shape, *, installing=False):
    if (type(module) is not nn.Linear or module.training or module.bias is not None
            or module._forward_hooks or module._forward_pre_hooks
            or (installing and 'forward' in module.__dict__)
            or tuple(module.weight.shape) != shape or module.weight.dtype != torch.bfloat16
            or module.weight.device != device or device.type != 'cuda' or not module.weight.is_contiguous()
            or (module.in_features, module.out_features) != (shape[1], shape[0])):
        raise ValueError('Prefix NVFP4 requires original contiguous CUDA BF16 bias-free Linear projections')


def _selected_modules(model):
    from ..models.pi05.precision_pi05 import FusedPrefixMLP, QuantizedLinear
    layers = _layers(model)
    if len(layers) != 18 or model.optimization_flags()[2] is not True:
        raise ValueError('Prefix NVFP4 requires 18 prefix layers with the final layer KV-only')
    modules = []
    for layer in layers[:17]:
        mlp = layer.mlp
        if (type(mlp) is not FusedPrefixMLP or mlp.training or mlp.concatenate or mlp.fuse_down_quant
                or 'forward' in mlp.__dict__ or mlp._forward_hooks or mlp._forward_pre_hooks):
            raise ValueError('Prefix NVFP4 preserves the independent original FusedPrefixMLP/GELU boundary')
        modules.extend(getattr(mlp, name) for name in PROJECTIONS)
    if len({id(module) for module in modules}) != 51 or any(isinstance(m, QuantizedLinear) for m in model.modules()):
        raise ValueError('Require exactly 51 distinct sites and no pre-existing quantized modules')
    return tuple(modules)


def _scalar_value(value):
    # Cold only: record the actual scalar arithmetic, never a Python estimate.
    cpu = value.detach().cpu().contiguous()
    return dict(value=float(cpu.item()), fp32_bits='0x' + format(int(cpu.view(torch.int32).item()) & 0xffffffff, '08x'))


@torch.no_grad()
def _pack_weight(weight, calibration, keepalive):
    if torch.is_inference_mode_enabled():
        raise ValueError('Pack NVFP4 outside inference_mode so persistent scales retain version counters')
    device = weight.device
    numerator = torch.full((1,), 2688., device=device, dtype=torch.float32)
    denominator = torch.tensor([max(calibration['input_amax'], 1e-12)], device=device, dtype=torch.float32)
    keepalive.extend((numerator, denominator))
    input_global_scale = torch.div(numerator, denominator)
    keepalive.append(input_global_scale)
    native.validate_cold_scale(input_global_scale)
    if _canonical(_scalar_value(input_global_scale)) != _canonical(calibration['input_global_scale']):
        raise ValueError('Actual CUDA FP32 input global scale differs from frozen calibration bits')
    fp32 = weight.detach().float()
    keepalive.append(fp32)
    amax = fp32.abs().amax().clamp_min(1e-12).reshape(1)
    keepalive.append(amax)
    weight_global_scale = torch.div(numerator, amax)
    keepalive.append(weight_global_scale)
    one = torch.ones((1,), dtype=torch.float32, device=device)
    product = input_global_scale * weight_global_scale
    keepalive.extend((one, product))
    alpha = torch.div(one, product)
    keepalive.append(alpha)
    native.validate_cold_scale(weight_global_scale)
    native.validate_cold_scale(alpha)
    quantized, scale = native.quantize_weight(weight, weight_global_scale)
    keepalive.extend((quantized, scale))
    if not bool(torch.isfinite(scale.float()).all() & (scale.float() >= 0).all()):
        raise ValueError('Native NVFP4 produced nonfinite/negative block scales')
    return quantized, scale, input_global_scale, weight_global_scale, alpha


class _Site:
    def __init__(self, owner, index, module):
        self.owner, self.index, self.module, self.weight = owner, index, module, module.weight
        self.device, self.shape = self.weight.device, tuple(self.weight.shape)
        self.quantized = self.scale = self.input_global_scale = self.weight_global_scale = self.alpha = None
        self._packed = self._identities = None
        self._prepared_compile_token = None
        def forward(module, x, _site=self):
            return _site.call(module, x)
        self.forward = MethodType(forward, module)
        self.record = None

    def initialize(self):
        layer, projection = self.index // 3, PROJECTIONS[self.index % 3]
        name = f'prefix.{layer}.{projection}'
        calibration = self.owner.calibration['sites'][name]
        original = _tensor_record(self.weight)
        if original['content_sha256'] != calibration['weight_sha256']:
            raise ValueError('Loaded original NVFP4 site weight differs from calibration checkpoint identity: ' + name)
        values = _pack_weight(self.weight, calibration, self.owner._pending)
        self.quantized, self.scale, self.input_global_scale, self.weight_global_scale, self.alpha = values
        self._packed = values
        self._identities = tuple(tensor_identity(t) for t in (self.weight, *values))
        if any(t.is_inference() for t in values):
            raise RuntimeError('Persistent NVFP4 tensors require live version counters')
        self.record = dict(name=name,
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None,
            original_weight=original, quantized_weight=_tensor_record(self.quantized),
            weight_scale=_tensor_record(self.scale), input_global_scale=_tensor_record(self.input_global_scale),
            weight_global_scale=_tensor_record(self.weight_global_scale), alpha=_tensor_record(self.alpha),
            scalar_values=dict(input_global_scale=_scalar_value(self.input_global_scale),
                               weight_global_scale=_scalar_value(self.weight_global_scale), alpha=_scalar_value(self.alpha)),
            calibration=deepcopy(calibration))
        self.owner._pending.clear()  # Hash copies/finite gates retired cold work.

    def validate(self):
        if (self.module.__dict__.get('forward') is not self.forward or self.module.weight is not self.weight
                or any(actual is not expected for actual, expected in zip(
                    (self.quantized, self.scale, self.input_global_scale, self.weight_global_scale, self.alpha),
                    self._packed, strict=True))):
            raise RuntimeError('Prefix NVFP4 source/forward/packed ownership changed; retire and reinstall')
        _validate_module(self.module, self.device, self.shape)
        if tuple(tensor_identity(t) for t in (self.weight, *self._packed)) != self._identities:
            raise RuntimeError('Prefix NVFP4 immutable packing epoch changed; retire and reinstall with new evidence')

    def call(self, module, x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type)
                or x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous()
                or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in native.SUPPORTED_CAPACITIES
                or x.shape[2] != self.shape[1]):
            raise ValueError('Prefix NVFP4 only supports the explicit B1 inference geometry; no BF16 fallback')
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('Prefix NVFP4 compilation requires the native prepared owner scope')
        else:
            self.owner.assert_execution_owner()
            self.validate()
        y, _, _ = native.scaled_linear(x.view(-1, self.shape[1]), self.quantized, self.scale,
                                      self.input_global_scale, self.alpha)
        return y.view(1, x.shape[1], self.shape[0])

    def restore(self):
        if self.module.__dict__.get('forward') is self.forward:
            delattr(self.module, 'forward')
        self._prepared_compile_token = None


class PrefixNVFP4Installation:
    def __init__(self, model, evidence, raw, raw_calibration, calibration, backend, runtime, native_path):
        self._model = weakref.ref(model)
        self.evidence, self.raw_report, self.backend = deepcopy(evidence), raw, deepcopy(backend)
        self.raw_calibration, self.calibration = raw_calibration, deepcopy(calibration)
        self.runtime, self.native_path = runtime, str(Path(native_path).resolve())
        self._raw_binding, self._evidence_binding = (raw, raw_calibration), _canonical(self.evidence)
        self._runtime_binding = runtime
        self._cold_binding = (_canonical(self.backend), _canonical(self.calibration))
        self.sites, self.enabled, self._runner = (), True, None
        self._installed_sites, self._pending = (), []
        self._generation = self._thread = self._signature = None
        self._bindings = (native.quantize_rows, native.native_scaled_mm, native.scaled_linear, native.quantize_weight)
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
        with _mutation_guard(model, 'prefix NVFP4 runner binding'):
            if (not self.enabled or self._runner is not None or type(runner) is not Pi05NativeGraphRunner
                    or runner.model is not model or not runner.compile_sample_actions
                    or tuple(r for r in model._graph_runners if not r._closed) != (runner,)):
                raise ValueError('Prefix NVFP4 requires exactly one compiled native graph runner')
            self._runner = weakref.ref(runner)

    def _owner(self):
        model, runner = self._model(), None if self._runner is None else self._runner()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked()
                or tuple(r for r in model._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix NVFP4 requires its unique locked native runner')
        return model

    def _validate(self, *, full=False):
        model = self._model()
        if not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix NVFP4 installation is inactive')
        if self._bindings != (native.quantize_rows, native.native_scaled_mm, native.scaled_linear, native.quantize_weight):
            raise RuntimeError('Prefix NVFP4 backend callable binding changed')
        if self.runtime is not self._runtime_binding:
            raise RuntimeError('Prefix NVFP4 native backend owner was replaced')
        self.runtime.validate(self.sites[0].device)
        if (self.raw_report is not self._raw_binding[0] or self.raw_calibration is not self._raw_binding[1]
                or _canonical(self.evidence) != self._evidence_binding):
            raise RuntimeError('Immutable NVFP4 evidence byte binding changed')
        from .compilation import INDUCTOR_OPTIONS
        if _canonical(INDUCTOR_OPTIONS) != _canonical(native.COMPILE_OPTIONS) or _math_settings() != self._math_settings:
            raise RuntimeError('Selected compile/math settings changed inside the NVFP4 epoch')
        if any(m.__dict__.get('forward') is not forward for m, forward in self._nonprefix_forwards):
            raise RuntimeError('Nonprefix forward binding changed; precision scope is no longer proven')
        modules = _selected_modules(model)
        if (len(self.sites) != 51 or len(self._installed_sites) != 51
                or any(site is not original for site, original in zip(self.sites, self._installed_sites, strict=True))
                or any(site.module is not module for site, module in zip(self.sites, modules, strict=True))):
            raise RuntimeError('Prefix NVFP4 exact ordered site topology changed')
        for site in self.sites:
            site.validate()
        if full:
            if ((_canonical(self.backend), _canonical(self.calibration)) != self._cold_binding
                    or (self._manifest is not None and _digest(self._manifest) != self._manifest_digest)):
                raise RuntimeError('Immutable NVFP4 cold provenance or manifest changed')
            current = tuple((name, type(m), tuple((n, type(t), t.dtype, t.device)
                for n, t in (*m._parameters.items(), *m._buffers.items()) if t is not None))
                for name, m in model.named_modules())
            if current != self._precision:
                raise RuntimeError('Nonprefix precision/module scope changed')
            if tuple(tensor_identity(t) for t in (*model.parameters(), *model.buffers())) != self._epoch_tensors:
                raise RuntimeError('Original model tensors changed; establish a new immutable NVFP4 identity')

    def prepare(self, generation):
        model = self._owner()
        if type(generation) is not int or generation != model._inference_cache.generation:
            raise RuntimeError('Prefix NVFP4 requires the complete current shared lifecycle generation')
        if self._generation is not None and generation != self._generation:
            # Explicit invalidation can signal a .data write whose Tensor
            # version did not change. Equal signatures cannot prove equal
            # contents; a published packing epoch must never silently accept it.
            raise RuntimeError('Prefix NVFP4 epoch invalidated; retire and reinstall with new content identity')
        self._validate(full=self._signature is None)
        signature = model._inference_cache.signature
        if signature is None:
            raise RuntimeError('Prefix NVFP4 cannot skip the full model lifecycle snapshot')
        if self._signature is not None and signature != self._signature:
            raise RuntimeError('NVFP4 source model/precision epoch changed; clear and reinstall before new preflight')
        self._signature = signature
        self._generation, self._thread = generation, threading.get_ident()

    def assert_execution_owner(self):
        model = self._owner()
        if (self._thread != threading.get_ident() or self._generation is None
                or self._generation != model._inference_cache.generation):
            raise RuntimeError('Prefix NVFP4 requires the current prepared generation/thread')

    def compilation_members(self, generation):
        self.assert_execution_owner()
        if generation != self._generation:
            raise RuntimeError('Prefix NVFP4 compiled generation is stale')
        self._validate()
        return self.sites

    def manifest(self):
        self._validate(full=True)
        return deepcopy(self._manifest)

    def validate_precision_manifest(self):
        self._validate(full=True)
        raw, raw_calibration, calibration, backend, runtime = _load_evidence(
            self.evidence, self.sites[0].device, self.native_path)
        if (raw != self.raw_report or raw_calibration != self.raw_calibration or runtime is not self.runtime
                or _canonical(calibration) != _canonical(self.calibration)
                or _canonical(backend) != _canonical(self.backend)):
            raise RuntimeError('Actual backend evidence/runtime changed since packing')
        runner = None if self._runner is None else self._runner()
        if (runner is None or runner._closed or runner.model is not self._model()
                or tuple(r for r in self._model()._graph_runners if not r._closed) != (runner,)):
            raise RuntimeError('Prefix NVFP4 live manifest requires its active native runner')
        return deepcopy(self._manifest)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix NVFP4 installation is inactive')
        with _mutation_guard(model, 'prefix NVFP4 uninstall'):
            runner = None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain native graph work before freeing prefix NVFP4 packed storage')
            if self._pending or runner is None:
                torch.cuda.synchronize(next(model.parameters()).device)
            for site in reversed(self._installed_sites):
                site.restore()
            delattr(model, INSTALLATION)
            self.enabled = False
            self.sites = ()
            self._installed_sites = ()
            self._pending.clear()
            self._runner = None
            model.invalidate_inference_caches(weights_changed=False)


def install_prefix_nvfp4(model, evidence, native_path):
    """Install after the BF16 recipe, before native runner creation; no fallback."""
    from .prefix_nvfp4_evidence import (CALIBRATION_FORMULA, WEIGHT_FORMULA, ALPHA_FORMULA,
                                      validate_calibration_binding)
    from ..models.pi05 import fused_mlp_pi05
    with _mutation_guard(model, 'prefix NVFP4 installation'):
        if (getattr(model, INSTALLATION, None) is not None
                or getattr(model, '_lite_prefix_fp8_installation', None) is not None):
            raise ValueError('Prefix NVFP4 and prefix FP8 are separate mutually exclusive installations')
        modules = _selected_modules(model)
        device = modules[0].weight.device
        for index, module in enumerate(modules):
            shape = (2048, 16384) if index % 3 == 2 else (16384, 2048)
            _validate_module(module, device, shape, installing=True)
        raw, raw_calibration, calibration, backend, runtime = _load_evidence(evidence, device, native_path)
        owner = PrefixNVFP4Installation(model, evidence, raw, raw_calibration, calibration, backend, runtime, native_path)
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
            owner._manifest = dict(schema=1, precision=PRECISION, implementation='pi05-prefix-nvfp4-cutlass-v1',
                checkpoint_sha256=CHECKPOINT_SHA256, scope='prefix_mlp_layers_0_16',
                precision_boundaries=deepcopy(PRECISION_BOUNDARIES), backend=deepcopy(backend),
                calibration=dict(sha256=evidence['calibration_sha256'], version=calibration['version'],
                    activation_report_sha256=calibration['activation_report_sha256'],
                    scope=calibration['scope'], single_fixture=True, capacity=calibration['capacity'],
                    quality_validated=False, formula=CALIBRATION_FORMULA,
                    calibration_source_scope='one BF16 fixture, not formal calibration',
                    bf16_control=deepcopy(evidence['bf16_control'])),
                quantizer=dict(source_sha256=_digest(backend['implementation']['source_files_sha256']),
                    nonlinear_source_sha256=_sha(fused_mlp_pi05.__file__),
                    variant=native.VARIANT, format='NVFP4 E2M1 values plus E4M3 block scales and FP32 global scales',
                    input_dtype='bfloat16', output_dtype='bfloat16', accumulator_dtype='float32',
                    weight_layout='contiguous_NK_packed_K/2', block_size=16,
                    scale_layout='CUTLASS swizzle128x4 padded per native ABI',
                    input_global_scale_formula=CALIBRATION_FORMULA, weight_global_scale_formula=WEIGHT_FORMULA,
                    alpha_formula=ALPHA_FORMULA, online_conversion=True, weight_packing='cold_only',
                    gate_up='independent_quantization',
                    nonlinear_boundary='BF16 gate/up -> gelu_tanh_mul -> BF16 down input'),
                lifecycle=dict(version=VERSION, source_sha256=_sha(__file__), enabled=True,
                    owner='Pi05NativeGraphRunner', original_parameters_preserved=True,
                    content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
                    live_checks=list(LIVE_CHECKS), fallback_permitted=False),
                sites=[deepcopy(site.record) for site in sites])
            validate_calibration_binding(calibration, owner._manifest, backend['hardware'])
            owner._manifest_digest = _digest(owner._manifest)
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
                error.add_note('Partial NVFP4 packing drain failed; retained model owner: ' + str(cleanup))
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
