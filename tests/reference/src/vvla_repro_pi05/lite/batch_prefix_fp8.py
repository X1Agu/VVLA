"""Independent true-B32 prefix51 owner extending the winning B1 recipe.

Only the geometry-specific primitive/forward and evidence schema change. Cold
packing and the original owner lifecycle remain shared, SHA-bound code. Original
BF16 Parameters are preserved; invalidation requires graph retirement/reinstall,
never an implicit repack. Install after all other nonprefix forward changes.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from . import batch_prefix_fp8_native as native
from . import prefix_fp8 as base
from .batch_prefix_fp8_evidence import (BINARIES, DECLARATIONS, ORIGIN_B1, RUNTIME,
    canonical, digest, exact, file_sha, source_hashes, validate_config, validate_manifest, validate_report)
from ..models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = base.INSTALLATION
VERSION = 'pi05-b32-prefix-fp8-live-owner-v1'
PRECISION = base.PRECISION
_UNSUPPORTED = ('_lite_prefix_fp8_inventory_installation', '_lite_prefix_fp8_gelu_installation',
    '_lite_vision_fp8_installation', '_lite_prefix_nvfp4_installation')


def _reject_extensions(model):
    if any(getattr(model, name, None) is not None for name in _UNSUPPORTED):
        raise ValueError('B32 prefix51 does not accept B1 inventory/GELU/vision-FP8/NVFP4 extensions')


def _load_evidence(config, device):
    validate_config(config)
    raw = Path(config['backend_report']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['backend_report_sha256']:
        raise ValueError('B32 prefix backend report original bytes changed')
    payload = json.loads(raw)
    native.validate_runtime(device)
    torch_root = Path(torch.__file__).resolve().parent
    hardware, sources = base._hardware(device), source_hashes()
    torch_files = {name:file_sha(torch_root/name) for name in (*DECLARATIONS,*BINARIES)}
    manifest = native.implementation_manifest()
    validate_report(payload, actual_hardware=hardware, actual_sources=sources,
        actual_torch_files=torch_files, actual_native_manifest=manifest)
    report_root=Path(config['backend_report']).resolve(strict=True).parent
    for row in payload['rows']:
        reference=row['sample_arrays']; sample=report_root/reference['path']
        if sample.is_symlink() or sample.resolve(strict=True).parent!=report_root or file_sha(sample)!=reference['sha256']:
            raise ValueError('B32 backend actual sample-array bytes or path changed')
    backend = dict(report_sha256=config['backend_report_sha256'], hardware=hardware,
        runtime=deepcopy(RUNTIME), source_files_sha256=sources, torch_files_sha256=torch_files,
        native_manifest=manifest, executed_rows_sha256=digest(payload['rows']))
    return raw, backend


class _Site(base._Site):
    def initialize(self):
        super().initialize()
        self._identity_object = self._identities

    def validate(self):
        if self._identities is not self._identity_object:
            raise RuntimeError('B32 prefix cold tensor-identity tuple was replaced')
        super().validate()

    def call(self, module, x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type) or x.dtype != torch.bfloat16
                or x.device != self.device or not x.is_contiguous() or x.ndim != 3
                or x.shape[0] != native.BATCH_SIZE or x.shape[1] not in native.PER_SLOT_CAPACITIES
                or x.shape[2] != self.shape[1]):
            raise ValueError('B32 prefix FP8 requires exact contiguous inference [32,576|640,K]; no B1 fallback')
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('B32 prefix compilation requires its current prepared owner scope')
        else:
            self.owner.assert_execution_owner()
            self.validate()
        output, _, _ = native.scaled_linear(x.view(-1,self.shape[1]), self.quantized, self.scale)
        return output.view(native.BATCH_SIZE,x.shape[1],self.shape[0])


class BatchPrefixFP8Installation(base.PrefixFP8Installation):
    """Original mutable-owner API, independently validated for B32 only.

The separate private serving engine may move static checks to cold/explicit
audits. This owner never independently authorizes such a move or caches a
successful hot audit. Its ordinary prepare validates all current live fields.
"""
    def __init__(self, model, evidence, raw, backend):
        _reject_extensions(model)
        super().__init__(model,evidence,raw,backend)
        self._bindings=(native.quantize_rows,native.native_scaled_mm,native.scaled_linear,torch._scaled_mm)
        self._source_hashes=source_hashes()
        self._manifest_bytes=None

    def _validate(self, *, full=False, inventory_proof=None, native_epoch=False):
        model=self._model()
        if not self.enabled or model is None or getattr(model,INSTALLATION,None) is not self:
            raise RuntimeError('B32 prefix installation is inactive')
        _reject_extensions(model)
        if inventory_proof is not None or native_epoch or self._native_epoch_validation is not None or self._native_epoch_binding is not None:
            raise RuntimeError('B32 prefix has no B1 inventory/native-host-epoch extension')
        if self._bindings!=(native.quantize_rows,native.native_scaled_mm,native.scaled_linear,torch._scaled_mm):
            raise RuntimeError('B32 prefix backend callable binding changed')
        from .compilation import INDUCTOR_OPTIONS
        if canonical(INDUCTOR_OPTIONS)!=canonical(native.COMPILE_OPTIONS) or base._math_settings()!=self._math_settings:
            raise RuntimeError('B32 prefix compile/math settings changed')
        if any(module.__dict__.get('forward') is not forward for module,forward in self._nonprefix_forwards):
            raise RuntimeError('B32 prefix nonprefix forward binding changed')
        modules=base._selected_modules(model)
        if (len(self.sites)!=51 or len(self._installed_sites)!=51
                or any(site is not original for site,original in zip(self.sites,self._installed_sites,strict=True))
                or any(site.module is not module for site,module in zip(self.sites,modules,strict=True))):
            raise RuntimeError('B32 prefix ordered site topology changed')
        for site in self.sites: site.validate()
        if full:
            current=tuple((name,type(module),tuple((name,type(tensor),tensor.dtype,tensor.device)
                for name,tensor in (*module._parameters.items(),*module._buffers.items()) if tensor is not None))
                for name,module in model.named_modules())
            if current!=self._precision or tuple(tensor_identity(t) for t in (*model.parameters(),*model.buffers()))!=self._epoch_tensors:
                raise RuntimeError('B32 prefix original model precision/tensor epoch changed')
            exact(source_hashes(),self._source_hashes,'cold owner source bytes')

    def _manifest_with_extension(self):
        _reject_extensions(self._model())
        if self._manifest_bytes is None or canonical(self._manifest)!=self._manifest_bytes:
            raise RuntimeError('B32 prefix cold manifest declaration changed')
        validate_manifest(self._manifest)
        return deepcopy(self._manifest)

    def validate_precision_manifest(self):
        self._validate(full=True)
        raw,backend=_load_evidence(self.evidence,self.sites[0].device)
        if raw!=self.raw_report or canonical(backend)!=canonical(self.backend):
            raise RuntimeError('B32 prefix current backend evidence/runtime changed')
        runner=None if self._runner is None else self._runner()
        if (runner is None or runner._closed or runner.model is not self._model()
                or tuple(r for r in self._model()._graph_runners if not r._closed)!=(runner,)):
            raise RuntimeError('B32 prefix full audit requires its actual unique native runner')
        return self._manifest_with_extension()


def install_batch_prefix_fp8(model,evidence):
    """Install last, before bind_runner/prepare/seal; no policy or BF16 fallback."""
    with base._mutation_guard(model,'B32 prefix FP8 installation'):
        if getattr(model,INSTALLATION,None) is not None:
            raise ValueError('A prefix FP8 owner is already installed')
        _reject_extensions(model)
        modules=base._selected_modules(model); device=modules[0].weight.device
        for index,module in enumerate(modules):
            shape=(2048,16384) if index%3==2 else (16384,2048)
            base._validate_module(module,device,shape,installing=True)
        raw,backend=_load_evidence(evidence,device)
        owner=BatchPrefixFP8Installation(model,evidence,raw,backend)
        sites,installed=[],[]
        try:
            for index,module in enumerate(modules):
                site=_Site(owner,index,module); sites.append(site); owner._installed_sites=tuple(sites)
                site.initialize()
            owner.sites=tuple(sites)
            for site in sites:
                site.module.forward=site.forward; installed.append(site)
            setattr(model,INSTALLATION,owner)
            owner._manifest=dict(schema=1,precision=PRECISION,implementation='pi05-b32-prefix-rowwise-fp8-v1',
                checkpoint_sha256=base.CHECKPOINT_SHA256,scope='prefix_mlp_layers_0_16',batch=32,
                per_slot_capacities=list(native.PER_SLOT_CAPACITIES),flattened_rows=list(native.SUPPORTED_CAPACITIES),
                origin_b1=deepcopy(ORIGIN_B1),backend=deepcopy(backend),
                precision_boundaries=deepcopy(base.PRECISION_BOUNDARIES),
                nonlinear_boundary='independent BF16 gate/up -> gelu_tanh_mul -> BF16 down input',
                lifecycle=dict(version=VERSION,source_sha256=file_sha(__file__),enabled=True,
                    owner='Pi05NativeGraphRunner',original_parameters_preserved=True,
                    content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
                    live_checks=list(base.LIVE_CHECKS),fallback_permitted=False,
                    serving_scope='ordinary_prepare_full_live_checks;private_sealed_engine_cold_and_explicit_full_audit'),
                sites=[deepcopy(site.record) for site in sites])
            owner._manifest_bytes=canonical(owner._manifest)
            validate_manifest(owner._manifest)
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException as error:
            try: torch.cuda.synchronize(device)
            except BaseException as cleanup:
                owner.sites=tuple(sites); setattr(model,INSTALLATION,owner)
                error.add_note('Partial B32 FP8 packing drain failed; retained owner: '+str(cleanup))
                raise error
            for site in reversed(installed): site.restore()
            if getattr(model,INSTALLATION,None) is owner: delattr(model,INSTALLATION)
            owner.enabled=False; owner.sites=owner._installed_sites=()
            owner._pending.clear(); sites.clear()
            raise
