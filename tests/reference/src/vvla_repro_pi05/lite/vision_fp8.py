"""Default-off 54-site vision FP8 extension over the selected prefix FP8 owner.

All original modules, Parameters, biases and SigLIP activations are preserved.
Packed storage is one immutable epoch. The only BF16 dispatch is the existing
cold masked all-minus-one camera cache builder, under an explicit local scope.
"""
from copy import deepcopy
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import threading
from types import MethodType
import weakref

import torch
from torch import nn

from . import prefix_fp8 as base_module
from . import vision_fp8_native as primitive
from . import vision_fp8_native_kernel as kernel
from .fused_expert_mlp import _mutation_guard
from .vision_fp8_evidence import (VERSION, VISION_BOUNDARY, LIFECYCLE, canonical, digest,
    validate_evidence_config, validate_backend_bytes, validate_vision_manifest)
from ..models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_vision_fp8_installation'


def _layers(model):
    return model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers


def _geometry(module, device, shape, *, installing=False):
    if (type(module) is not nn.Linear or module.training or module._forward_hooks or module._forward_pre_hooks
            or (installing and 'forward' in module.__dict__) or module.bias is None
            or (module.out_features,module.in_features) != shape or tuple(module.weight.shape) != shape
            or tuple(module.bias.shape) != (shape[0],) or device.type != 'cuda'
            or any(type(value) is not nn.Parameter or value.dtype != torch.bfloat16
                   or value.device != device or not value.is_contiguous() for value in (module.weight,module.bias))):
        raise ValueError('Vision FP8 requires original contiguous biased BF16 SigLIP Linear projections')


def _activation_state(mlp):
    activation = mlp.activation_fn
    if (not isinstance(activation, nn.Module) or activation.training or activation._forward_hooks
            or activation._forward_pre_hooks or 'forward' in activation.__dict__
            or mlp.training or mlp._forward_hooks or mlp._forward_pre_hooks or 'forward' in mlp.__dict__):
        raise ValueError('Vision FP8 preserves the original unpatched SigLIP activation and MLP')
    # Preserve exact object/callable identity and ordinary activation scalar
    # attributes; no substitute GELU approximation or additional cast is added.
    scalar = tuple((k,v) for k,v in activation.__dict__.items() if type(v) in (str,bool,int,float,type(None)))
    return (activation, type(activation), type(activation).forward, type(mlp).forward, scalar)


def _load_evidence(evidence, device, backend):
    validate_evidence_config(evidence)
    path = Path(evidence['backend_report'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4*1024**2:
        raise ValueError('Unsafe/missing vision primitive report')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence['backend_report_sha256']:
        raise ValueError('Vision primitive report original bytes changed')
    report = json.loads(raw)
    primitive.validate_runtime(device)
    hardware = base_module._hardware(device)
    root = Path(torch.__file__).resolve().parent
    record = dict(report_sha256=evidence['backend_report_sha256'],
        source_sha256=report.get('source',{}).get('script_sha256'),
        implementation=primitive.implementation_manifest(),
        runtime=dict(torch=str(torch.__version__),torch_git=torch.version.git_version,cuda=torch.version.cuda,
                     capability=list(torch.cuda.get_device_capability(device)),triton=package_version('triton')),
        installed_binary_sha256={'lib/libtorch_cuda.so':base_module._sha(root/'lib/libtorch_cuda.so')},
        hardware=hardware,contract_math=report.get('contract_math'),
        scope='synthetic_primitive_support_only; not quality or performance')
    validate_backend_bytes(raw,record,hardware)
    if (canonical(backend['hardware']) != canonical(hardware)
            or backend['installed_binary_sha256']['lib/libtorch_cuda.so'] != record['installed_binary_sha256']['lib/libtorch_cuda.so']):
        raise ValueError('Vision primitive and prefix owner use different actual runtime/GPU')
    # The original-byte report records its executed probe hash. Production
    # consumes only the actual core pair/binary/runtime and this explicit raw
    # evidence; it does not import or require the benchmark directory.
    return raw, record


class _Site:
    def __init__(self,owner,index,layer,module):
        self.owner,self.index,self.layer,self.module = owner,index,layer,module
        self.weight,self.bias = module.weight,module.bias
        self.shape,self.device = tuple(self.weight.shape),self.weight.device
        self.quantized=self.scale=self.execution_bias=self._packed=self._identities=self.record=None
        self._prepared_compile_token=None
        self.original_forward=module.__dict__.get('forward')
        def forward(module,x,_site=self): return _site.call(module,x)
        self.forward=MethodType(forward,module)

    def initialize(self):
        self.quantized,self.scale=base_module._pack_weight(self.weight,self.owner._pending)
        if torch.is_inference_mode_enabled():
            raise ValueError('Vision execution bias must be packed outside inference_mode')
        with torch.no_grad():
            # AOTAutograd may erase detach-only aliases and lift the original
            # grad-bearing Parameter back into an opaque operator. This cold
            # exact copy has independent storage and an ordinary live version.
            self.execution_bias=self.bias.detach().clone(memory_format=torch.contiguous_format)
        self.owner._pending.append(self.execution_bias)
        self._packed=(self.quantized,self.scale,self.execution_bias)
        self._validate_execution_bias()
        self._identities=tuple(tensor_identity(t) for t in (self.weight,self.bias,*self._packed))
        layer,name=self.index//2,('fc1','fc2')[self.index%2]
        self.record=dict(name=f'vision.{layer}.{name}',
            module_path=f'paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.{layer}.mlp.{name}',
            module_type='torch.nn.modules.linear.Linear',original_weight=base_module._tensor_record(self.weight),
            original_bias=base_module._tensor_record(self.bias),execution_bias=base_module._tensor_record(self.execution_bias),
            quantized_weight=base_module._tensor_record(self.quantized),
            weight_scale=base_module._tensor_record(self.scale))
        if canonical(self.record['execution_bias']) != canonical(self.record['original_bias']):
            raise RuntimeError('Cold execution bias must preserve the original BF16 bias bytes exactly')
        self.owner._pending.clear() # Actual CPU hashes retired cold copies.

    def _validate_execution_bias(self):
        value=self.execution_bias
        if (type(value) is not torch.Tensor or value.requires_grad or value.is_inference()
                or value.dtype != torch.bfloat16 or value.device != self.device
                or tuple(value.shape) != (self.shape[0],) or value.stride() != (1,)
                or value.untyped_storage()._cdata == self.bias.untyped_storage()._cdata):
            raise RuntimeError('Vision FP8 requires its independent versioned nongrad BF16 execution bias')

    def _validate_binding_geometry(self):
        if (self.module.__dict__.get('forward') is not self.forward or self.module.weight is not self.weight
                or self.module.bias is not self.bias or self.quantized is not self._packed[0]
                or self.scale is not self._packed[1] or self.execution_bias is not self._packed[2]):
            raise RuntimeError('Vision FP8 original/packed/bias/forward ownership changed; retire and reinstall')
        _geometry(self.module,self.device,self.shape)
        self._validate_execution_bias()

    def validate(self):
        self._validate_binding_geometry()
        if tuple(tensor_identity(t) for t in (self.weight,self.bias,*self._packed)) != self._identities:
            raise RuntimeError('Vision FP8 immutable packing epoch changed; retire and reinstall')

    def call(self,module,x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type) or x.device != self.device
                or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.ndim != 3
                or x.shape[1:] != (256,self.shape[1]) or x.shape[0] not in (1,2)):
            raise ValueError('Vision FP8 requires explicit B1-camera inference geometry; no generic fallback')
        if x.shape[0] == 1:
            # This is not a shape-only fallback: only the validated blank-camera
            # cache builder may authorize it, outside Dynamo and CUDA capture.
            if torch.compiler.is_compiling(): raise RuntimeError('M256 vision BF16 is cold-empty only')
            self.owner.assert_empty_camera_scope()
            self.validate()
            return nn.Linear.forward(module,x)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('Vision FP8 requires its native prepared compilation scope')
        else:
            self.owner.assert_execution_owner()
            self.validate()
        # Original Parameter and independent cold bias are both guarded outside
        # compilation. No per-request copy or detach alias enters AOTAutograd.
        y,_,_=primitive.scaled_linear(x.view(512,self.shape[1]),self.quantized,self.scale,self.execution_bias)
        return y.view(2,256,self.shape[0])

    def restore(self):
        if self.module.__dict__.get('forward') is self.forward: delattr(self.module,'forward')
        self._prepared_compile_token=None


class VisionFP8Installation:
    def __init__(self,model,base,evidence,raw,contracts):
        self._model,self.base=weakref.ref(model),base
        self.evidence,self.raw_report,self.contracts=deepcopy(evidence),raw,deepcopy(contracts)
        self.sites=self._installed_sites=()
        self._lookup,self._pending={},[]
        self.enabled=True
        self._runner=self._native=self._generation=self._thread=self._empty_thread=None
        self._native_epoch_validation=self._native_epoch_binding=None
        self._native_manifest=None
        self._collector=None
        self._batching=getattr(model,'_lite_batched_vision_installation',None)
        self._batching_geometry=self._batching.geometry
        self._embed_prefix=model.__dict__.get('embed_prefix')
        self._layers=tuple(_layers(model))
        self._activations=tuple(_activation_state(layer.mlp) for layer in self._layers)
        self._bindings=(primitive.quantize_rows,primitive.native_scaled_mm,primitive.scaled_linear,
                        kernel.launch,nn.Linear.forward,torch._scaled_mm,base_module._pack_weight)
        self._math=base_module._math_settings()
        self._owner_sha=base_module._sha(__file__)
        self._validator_sha=base_module._sha(Path(__file__).with_name('vision_fp8_evidence.py'))

    def accepts_original_forward(self,module,original):
        site=self._lookup.get(id(module))
        return (self.enabled and site is not None and site.module is module and original is None
                and site.original_forward is None and module.__dict__.get('forward') is site.forward)

    def validate_bindings(self):
        model=self._model()
        if (not self.enabled or model is None or getattr(model,INSTALLATION,None) is not self
                or type(self.base) is not base_module.PrefixFP8Installation or not self.base.enabled
                or getattr(model,base_module.INSTALLATION,None) is not self.base
                or getattr(model,'_lite_vision_fc2_cublaslt_installation',None) is not None
                or getattr(model,'_lite_batched_vision_installation',None) is not self._batching
                or not self._batching.enabled or model.__dict__.get('embed_prefix') is not self._embed_prefix
                or self._batching.geometry != self._batching_geometry
                or self._native is not None and getattr(model,'_lite_native_lifecycle_signature_installation',None) is not self._native
                or self._bindings != (primitive.quantize_rows,primitive.native_scaled_mm,primitive.scaled_linear,
                    kernel.launch,nn.Linear.forward,torch._scaled_mm,base_module._pack_weight)
                or self._math != base_module._math_settings()
                or len(self.sites) != 54 or len(self._installed_sites) != 54 or len(self._lookup) != 54
                or any(a is not b for a,b in zip(self.sites,self._installed_sites,strict=True))):
            raise RuntimeError('Vision FP8 installation/backend/site binding changed')
        layers=_layers(model)
        if len(layers) != 27 or any(a is not b for a,b in zip(layers,self._layers,strict=True)):
            raise RuntimeError('Vision FP8 ordered layer topology changed')
        for index,layer in enumerate(layers):
            if _activation_state(layer.mlp) != self._activations[index]:
                raise RuntimeError('Original vision activation object/behavior changed')
            for offset,name in enumerate(('fc1','fc2')):
                site=self.sites[2*index+offset]
                if (site.module is not getattr(layer.mlp,name) or site.layer is not layer
                        or self._lookup.get(id(site.module)) is not site
                        or not self.accepts_original_forward(site.module,None)):
                    raise RuntimeError('Vision FP8 exact Linear topology/forward changed')

    def bind_runner(self,runner):
        from .graph import Pi05NativeGraphRunner
        from .native_lifecycle import NativeLifecycleSignatureInstallation
        model=self._model()
        with _mutation_guard(model,'vision FP8 runner binding'):
            native=getattr(model,'_lite_native_lifecycle_signature_installation',None)
            if (self._runner is not None or type(runner) is not Pi05NativeGraphRunner or not runner.compile_sample_actions
                    or runner.model is not model or self.base._runner is None or self.base._runner() is not runner
                    or tuple(r for r in getattr(model,'_graph_runners',()) if not r._closed) != (runner,)
                    or type(native) is not NativeLifecycleSignatureInstallation or not native.enabled
                    or native.manifest().get('collector_variant') != 'v3'):
                raise ValueError('Vision FP8 requires selected prefix51 and exactly one compiled native-v3 owner')
            self.validate_bindings()
            self._native,self._native_manifest=native,native.manifest()
            self._collector=native._collector
            self._runner=weakref.ref(runner)

    def assert_execution_owner(self):
        self.base.assert_execution_owner()
        model=self._model(); runner=None if self._runner is None else self._runner()
        if (runner is None or self.base._runner() is not runner or self._native is None
                or getattr(model,'_lite_native_lifecycle_signature_installation',None) is not self._native
                or self._native._collector is not self._collector
                or self._native._active_model() is not model
                or not self._native.enabled or self._thread != threading.get_ident()
                or self._generation is None or self._generation != self.base._generation
                or self._generation != model._inference_cache.generation):
            raise RuntimeError('Vision FP8 requires its current native-v3 owner generation/thread')

    def prepare(self,generation):
        self.base.assert_execution_owner()
        if self._generation is not None and generation != self._generation:
            raise RuntimeError('Vision FP8 epoch invalidated; retire/reinstall with new content identity')
        if generation != self.base._generation: raise RuntimeError('Vision FP8 stale base generation')
        self.validate_bindings()
        if self._native_epoch_validation is not self._native_epoch_binding:
            raise RuntimeError('Native vision FP8 epoch owner binding changed')
        if self._native_epoch_validation is None:
            for site in self.sites: site.validate()
        else:
            from .fp8_tensor_epochs import validate_owner
            for site in self.sites: site._validate_binding_geometry()
            validate_owner(self,tuple(t for site in self.sites for t in (site.weight,site.bias,*site._packed)))
        self._generation,self._thread=generation,threading.get_ident()
        self.assert_execution_owner()

    def encode_empty_camera(self,image,mask):
        self.assert_execution_owner()
        if (self._empty_thread is not None or torch.cuda.is_current_stream_capturing()
                or tuple(image.shape) != (1,3,224,224) or tuple(mask.shape) != (1,)
                or mask.dtype != torch.bool or mask.device != image.device
                or bool(mask.any()) or not bool((image == -1).all())):
            raise ValueError('BF16 vision fallback requires the actual cold masked all-minus-one camera')
        self._empty_thread=threading.get_ident()
        try:
            return self._model().paligemma_with_expert.embed_image(image).detach()
        finally:
            self._empty_thread=None

    def assert_empty_camera_scope(self):
        self.assert_execution_owner()
        if self._empty_thread != threading.get_ident() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('M256 BF16 vision is only authorized inside cold empty-camera construction')

    def compilation_members(self,generation):
        self.assert_execution_owner()
        if generation != self._generation: raise RuntimeError('Vision FP8 compiled generation is stale')
        self.validate_bindings()
        for site in self.sites: site.validate()
        return self.sites

    def manifest(self):
        self.validate_bindings()
        for site in self.sites: site.validate()
        if (self._native is None or canonical(self._native.manifest()) != canonical(self._native_manifest)
                or canonical(primitive.implementation_manifest()) != canonical(self.contracts['implementation'])
                or base_module._sha(__file__) != self._owner_sha
                or base_module._sha(Path(__file__).with_name('vision_fp8_evidence.py')) != self._validator_sha):
            raise RuntimeError('Vision FP8 manifest requires its unchanged actual bound native-v3/source epoch')
        value=dict(schema=1,version=VERSION,enabled=True,scope='vision_mlp_fc1_fc2_layers_0_26',
            contracts=deepcopy(self.contracts),sites=[deepcopy(site.record) for site in self.sites],
            lifecycle=deepcopy(LIFECYCLE),owner_source_sha256=self._owner_sha,
            evidence_validator_sha256=self._validator_sha,native_lifecycle_manifest_sha256=digest(self._native_manifest),
            base_owner_source_sha256=self.base._manifest['lifecycle']['source_sha256'],
            activation='original activation_fn object and original BF16 input/output boundary',dispatch=VISION_BOUNDARY,
            quality_validated=False,performance_validated=False)
        return validate_vision_manifest(value,self.base.backend)

    def validate_live_evidence(self):
        raw,contracts=_load_evidence(self.evidence,self.sites[0].device,self.base.backend)
        if raw != self.raw_report or canonical(contracts) != canonical(self.contracts):
            raise RuntimeError('Actual vision backend evidence/runtime changed')
        return self.manifest()

    def uninstall(self):
        model=self._model()
        if not self.enabled or model is None or getattr(model,INSTALLATION,None) is not self:
            raise RuntimeError('Vision FP8 installation is inactive')
        with _mutation_guard(model,'vision FP8 uninstall'):
            runner=None if self._runner is None else self._runner()
            if self._empty_thread is not None or runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain native work before retiring vision FP8 storage')
            if self._pending or runner is None: torch.cuda.synchronize(self._installed_sites[0].device)
            if any(site.module.__dict__.get('forward') is not site.forward for site in self._installed_sites
                   if site.module.__dict__.get('forward') is not None):
                raise RuntimeError('Vision FP8 foreign forward cannot be overwritten during cleanup')
            # Failure here keeps every forward, packed tensor and handle intact.
            model.invalidate_inference_caches(weights_changed=False)
            for site in reversed(self._installed_sites): site.restore()
            delattr(model,INSTALLATION)
            self.enabled=False
            self.sites=self._installed_sites=()
            self._native_epoch_validation=self._native_epoch_binding=None
            self._lookup.clear(); self._pending.clear(); self._runner=None


def install_vision_fp8(model,evidence):
    from .batched_vision import BatchedVisionInstallation, _batched_embed_prefix
    with _mutation_guard(model,'vision FP8 installation'):
        base=getattr(model,base_module.INSTALLATION,None)
        batching=getattr(model,'_lite_batched_vision_installation',None)
        if (getattr(model,INSTALLATION,None) is not None or type(base) is not base_module.PrefixFP8Installation
                or not base.enabled or base._runner is not None or base._generation is not None
                or tuple(getattr(model,'_graph_runners',())) or type(batching) is not BatchedVisionInstallation or not batching.enabled
                or getattr(model.embed_prefix,'__func__',None) is not _batched_embed_prefix
                or getattr(model,'_lite_vision_fc2_cublaslt_installation',None) is not None
                or base.backend.get('quantizer_variant') != base_module.native.VARIANT):
            raise ValueError('Vision FP8 requires unbound selected prefix51/batched vision and no old vision fc2 Lt')
        base._validate(full=True)
        layers=tuple(_layers(model))
        if len(layers) != 27: raise ValueError('Vision FP8 requires all27 original SigLIP MLPs')
        modules=tuple(getattr(layer.mlp,name) for layer in layers for name in ('fc1','fc2'))
        if len({id(module) for module in modules}) != 54: raise ValueError('Vision FP8 requires54 distinct projections')
        device=modules[0].weight.device
        for index,module in enumerate(modules):
            _geometry(module,device,(4304,1152) if index%2 == 0 else (1152,4304),installing=True)
        raw,contracts=_load_evidence(evidence,device,base.backend)
        owner=VisionFP8Installation(model,base,evidence,raw,contracts)
        sites,installed=[],[]
        try:
            for index,module in enumerate(modules):
                site=_Site(owner,index,layers[index//2],module)
                sites.append(site); owner._installed_sites=tuple(sites)
                site.initialize()
            owner.sites=tuple(sites); owner._lookup={id(site.module):site for site in sites}
            for site in sites:
                site.module.forward=site.forward; installed.append(site)
            setattr(model,INSTALLATION,owner)
            owner.validate_bindings()
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException as error:
            try: torch.cuda.synchronize(device)
            except BaseException as cleanup:
                owner.sites=tuple(sites); setattr(model,INSTALLATION,owner)
                error.add_note('Partial vision FP8 drain failed; retained model owner: '+str(cleanup))
                raise error
            for site in reversed(installed): site.restore()
            if getattr(model,INSTALLATION,None) is owner: delattr(model,INSTALLATION)
            owner.enabled=False; owner.sites=owner._installed_sites=()
            owner._pending.clear(); sites.clear()
            raise
