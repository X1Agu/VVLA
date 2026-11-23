"""Default-off all27 biased vision fc2 candidate with native graph ownership.

Install after model.eval()/BF16 loading, before runner construction, then bind
exactly one runner. Retire native graphs before uninstall. Original Linear,
Parameter and state_dict identities are preserved; no parameter is packed.
"""
import hashlib
from pathlib import Path
import threading
from types import MethodType
import weakref

import torch
from torch import nn

from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
from vvla_pi05.lite.vision_fc2_cublaslt_runtime import (
    INPUT_SHAPE, K, N, VisionFC2LtRuntime, vision_fc2_cublaslt,
)
from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_vision_fc2_cublaslt_installation'
VERSION = 'all27-vision-fc2-biased-lt-lifecycle-v1'


def _layers(model):
    return model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers


def _geometry(module, device, *, installing=False):
    if (type(module) is not nn.Linear or module.training or module.in_features != K or module.out_features != N
            or module.bias is None or module._forward_hooks or module._forward_pre_hooks
            or (installing and 'forward' in module.__dict__)
            or tuple(module.weight.shape) != (N, K) or tuple(module.bias.shape) != (N,)
            or any(t.dtype != torch.bfloat16 or t.device != device or not t.is_contiguous()
                   for t in (module.weight, module.bias))):
        raise ValueError('Vision fc2 requires original biased 4304->1152 contiguous BF16 Linear without hooks')


class _Site:
    def __init__(self, installation, layer, plan):
        self.installation, self.layer, self.module, self.plan = installation, layer, layer.mlp.fc2, plan
        self.weight, self.bias = self.module.weight, self.module.bias
        self.device = self.weight.device
        self._allocation = (self.weight.data_ptr(), self.bias.data_ptr())
        self._source_identity = None
        self._prepared_compile_token = None
        def forward(module, x, _site=self):
            return _site.call(module, x)
        self.forward = MethodType(forward, self.module)

    def validate(self):
        module = self.module
        if (self.layer.mlp.fc2 is not module or module.__dict__.get('forward') is not self.forward
                or module.weight is not self.weight or module.bias is not self.bias):
            raise RuntimeError('Vision fc2 module/parameter/forward was replaced; retire and reinstall')
        _geometry(module, self.device)
        if (self.weight.data_ptr(), self.bias.data_ptr()) != self._allocation:
            raise RuntimeError('Vision fc2 weight/bias storage changed; retire and reinstall')
        return tensor_identity(self.weight), tensor_identity(self.bias)

    def assert_ready(self):
        identity = self.validate()
        if self._source_identity is None or identity != self._source_identity:
            raise RuntimeError('Vision fc2 sources changed or unprepared; prepare model lifecycle')

    def call(self, module, x):
        precision = getattr(self, '_projection_precision', None)
        if precision is not None and not (module.training or torch.is_grad_enabled()
                                          or torch.is_autocast_enabled(x.device.type)):
            return precision(x)
        # Empty camera and non-target shapes retain native arithmetic, including
        # its original errors for incompatible dtypes/devices. No implicit cast.
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                or tuple(x.shape) != INPUT_SHAPE or x.dtype != torch.bfloat16
                or x.device != self.device or not x.is_contiguous()):
            return nn.Linear.forward(module, x)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('Vision fc2 compilation requires the native prepared scope')
        else:
            self.installation.assert_execution_owner()
            self.assert_ready()
        # Explicit bias is a graph input, not just an invisible descriptor ref.
        return vision_fc2_cublaslt(x, self.weight, self.bias, self.plan.key)

    def restore(self):
        if self.module.__dict__.get('forward') is self.forward:
            delattr(self.module, 'forward')
        self._prepared_compile_token = None


class VisionFC2CublasLtInstallation:
    def __init__(self, model, runtime):
        self._model, self.runtime = weakref.ref(model), runtime
        self.sites = []
        self._installed_sites = ()
        self.enabled = True
        self._runner = None
        self._generation = self._request_thread = None
        self.refreshes = 0

    def bind_runner(self, runner):
        from vvla_pi05.lite.graph import Pi05NativeGraphRunner
        model = self._model()
        with _mutation_guard(model, 'vision fc2 cuBLASLt owner binding'):
            if self._runner is not None or not self.enabled or type(runner) is not Pi05NativeGraphRunner:
                raise ValueError('Bind vision fc2 exactly once to a native runner')
            owners = tuple(r for r in getattr(model, '_graph_runners', ()) if not r._closed)
            if owners != (runner,) or runner.model is not model:
                raise ValueError('Vision fc2 requires exactly one native graph owner')
            self.runtime.bind_owner(self)
            self._runner = weakref.ref(runner)

    def _check_owner(self, *, enumerate_owners):
        model = self._model()
        runner = None if self._runner is None else self._runner()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked()):
            raise RuntimeError('Vision fc2 requires its bound native runner and locked request')
        if enumerate_owners:
            owners = tuple(r for r in getattr(model, '_graph_runners', ()) if not r._closed)
            if owners != (runner,):
                raise RuntimeError('Vision fc2 does not permit multiple native graph owners')
        return model

    def assert_execution_owner(self):
        model = self._check_owner(enumerate_owners=False)
        if (self._request_thread != threading.get_ident() or self._generation is None
                or model._inference_cache.generation != self._generation):
            raise RuntimeError('Vision fc2 requires the current prepared owner generation/thread')

    def _validate(self):
        self.runtime.validate()
        layers = _layers(self._model())
        if (len(layers) != 27 or len(self.sites) != 27
                or len(self._installed_sites) != 27
                or any(site is not original for site, original in zip(self.sites, self._installed_sites, strict=True))
                or any(layer is not site.layer for layer, site in zip(layers, self.sites, strict=True))
                or any(site.plan is not plan for site, plan in zip(self.sites, self.runtime.plans, strict=True))):
            raise RuntimeError('Vision fc2 layer topology changed; retire and reinstall')
        return tuple(site.validate() for site in self.sites)

    def prepare(self, generation):
        model = self._check_owner(enumerate_owners=True)
        if type(generation) is not int or generation != model._inference_cache.generation:
            raise RuntimeError('Vision fc2 prepare requires the shared current model generation')
        identities = self._validate()
        if generation == self._generation:
            if any(site._source_identity != identity for site, identity in zip(self.sites, identities, strict=True)):
                raise RuntimeError('Vision fc2 sources changed without a model generation refresh')
        else:
            # No copy: all descriptors retain original weight and bias addresses.
            # Live versions are refreshed only after the shared lifecycle check.
            for site, identity in zip(self.sites, identities, strict=True):
                site._source_identity = identity
            self._generation = generation
            self.refreshes += 1
        self._request_thread = threading.get_ident()

    def compilation_members(self, generation):
        self._check_owner(enumerate_owners=True)
        self.assert_execution_owner()
        if generation != self._generation:
            raise RuntimeError('Vision fc2 compilation requires its current prepared generation')
        identities = self._validate()
        if any(site._source_identity != identity for site, identity in zip(self.sites, identities, strict=True)):
            raise RuntimeError('Vision fc2 compilation sources changed')
        return tuple(self.sites)

    def manifest(self):
        files = (Path(__file__), Path(__file__).with_name('vision_fc2_cublaslt_runtime.py'))
        return dict(version=VERSION, enabled=self.enabled, layers=len(self.sites), selected_layers='all27',
                    input_shape=list(INPUT_SHAPE), fallback='native Linear for all other shapes, training, grad, autocast',
                    parameter_ownership='original modules/parameters/state_dict; no transpose or packed copies',
                    refresh='all live W/bias identities and topology each prepare; in-place updates require model generation',
                    bias_lifetime='explicit custom-op input; fixed per-site descriptor; retained until graphs drained',
                    ownership='one bound native runner; prepared generation/thread; runner orders streams and output ownership',
                    numerical_limit='screened BF16 algorithm, not native-bitwise; full-model action/closed-loop quality required',
                    runtime=self.runtime.manifest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Vision fc2 installation is inactive')
        with _mutation_guard(model, 'vision fc2 cuBLASLt'):
            runner = None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Retire native runner work before freeing vision fc2 descriptors')
            # Cleanup remains possible after an externally replaced parameter or
            # module was correctly rejected. Do not overwrite external forwards.
            for site in reversed(self._installed_sites):
                site.restore()
            delattr(model, INSTALLATION)
            self.enabled = False
            try:
                self.runtime.close()
            finally:
                self.sites.clear()
                self._installed_sites = ()
                self._runner = None
                model.invalidate_inference_caches(weights_changed=False)


def install_vision_fc2_cublaslt(model, *, _runtime_factory=None):
    with _mutation_guard(model, 'vision fc2 cuBLASLt'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Vision fc2 cuBLASLt is already installed')
        layers = _layers(model)
        if len(layers) != 27:
            raise ValueError('Vision fc2 candidate requires all27 original SigLIP layers')
        modules = tuple(layer.mlp.fc2 for layer in layers)
        if len({id(module) for module in modules}) != 27:
            raise ValueError('Vision fc2 candidate requires 27 distinct Linear modules')
        device = modules[0].weight.device
        for module in modules:
            _geometry(module, device, installing=True)
        factory = VisionFC2LtRuntime if _runtime_factory is None else _runtime_factory
        runtime = factory(device, tuple((module.weight, module.bias) for module in modules))
        handle = VisionFC2CublasLtInstallation(model, runtime)
        installed = []
        try:
            if len(runtime.plans) != 27:
                raise ValueError('Vision fc2 runtime must own exactly 27 descriptor sets')
            handle.sites = [_Site(handle, layer, plan) for layer, plan in zip(layers, runtime.plans, strict=True)]
            handle._installed_sites = tuple(handle.sites)
            for site in handle.sites:
                site.module.forward = site.forward
                installed.append(site)
            setattr(model, INSTALLATION, handle)
            model.invalidate_inference_caches(weights_changed=False)
            return handle
        except BaseException as error:
            for site in reversed(installed):
                site.restore()
            if getattr(model, INSTALLATION, None) is handle:
                delattr(model, INSTALLATION)
            handle.enabled = False
            try:
                runtime.close()
            except BaseException as cleanup:
                error.add_note('Partial vision fc2 installation cleanup: ' + str(cleanup))
            raise
