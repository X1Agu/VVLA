"""New evidence-gated shared prefix51 owner for all declared true batches."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from . import prefix_fp8 as base
from . import dynamic_prefix_fp8_native as native
from .fused_expert_mlp import _mutation_guard

VERSION='source497-dynamic-prefix51-owner-v1'


def load_evidence(evidence,device):
    if type(evidence) is not dict or set(evidence)!={'backend_report','backend_report_sha256'}:
        raise ValueError('Dynamic prefix requires its own same-card backend path and SHA')
    path=Path(evidence['backend_report'])
    if path.is_symlink() or not path.is_file():raise ValueError('Backend report must be a regular local file')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=evidence['backend_report_sha256']:
        raise ValueError('Dynamic prefix backend report changed')
    report=json.loads(raw)
    native.validate_runtime(device)
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if (report.get('status')!='complete' or report.get('gpu_uuid')!=uuid
            or base._canonical(report.get('manifest'))!=base._canonical(native.implementation_manifest())
            or report.get('policy_quality_passed') is not False):
        raise ValueError('Dynamic prefix evidence does not bind this GPU, source, runtime and scope')
    expected={(b,p,k,n) for b in native.BATCH_SIZES for p in native.PER_SLOT_CAPACITIES
              for k,n in native.SUPPORTED_GEOMETRIES}
    actual=set()
    for row in report.get('rows',[]):
        key=tuple(row.get(name) for name in ('batch','prefix_capacity','k','n'))
        if key in actual or row.get('m')!=key[0]*key[1]:raise ValueError('Duplicate/invalid dynamic FP8 geometry evidence')
        actual.add(key)
        if not all(row.get(name) is True for name in ('quantized_bytes_exact','scales_exact',
                'aten_gemm_exact','fullgraph_exact','capture_replay_exact')):
            raise ValueError('Dynamic prefix requires complete converter/GEMM/compile/capture proof')
    if actual!=expected:raise ValueError('Missing actual dynamic prefix backend geometries')
    return raw,dict(report_sha256=evidence['backend_report_sha256'],gpu_uuid=uuid,
        implementation=native.implementation_manifest(),policy_quality_passed=False)


class _Site(base._Site):
    def call(self,module,x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type) or x.dtype!=torch.bfloat16
                or x.device!=self.device or not x.is_contiguous() or x.ndim!=3
                or x.shape[0] not in native.BATCH_SIZES or x.shape[1] not in native.PER_SLOT_CAPACITIES
                or x.shape[2]!=self.shape[1]):
            raise ValueError('Dynamic prefix requires exact supported contiguous inference [B,P,K]')
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:raise RuntimeError('Dynamic prefix requires prepared compile ownership')
        else:self.owner.assert_execution_owner();self.validate()
        output,_,_=native.scaled_linear(x.view(-1,self.shape[1]),self.quantized,self.scale)
        return output.view(x.shape[0],x.shape[1],self.shape[0])


class DynamicPrefixFP8Installation(base.PrefixFP8Installation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._dynamic_bindings=(native.quantize_rows,native.native_scaled_mm,native.scaled_linear)

    def _validate(self,**kwargs):
        if self._dynamic_bindings!=(native.quantize_rows,native.native_scaled_mm,native.scaled_linear):
            raise RuntimeError('Dynamic prefix operator bindings changed')
        return super()._validate(**kwargs)

    def validate_precision_manifest(self):
        self._validate(full=True)
        raw,backend=load_evidence(self.evidence,self.sites[0].device)
        if raw!=self.raw_report or backend!=self.backend:raise RuntimeError('Dynamic backend evidence changed after packing')
        runner=None if self._runner is None else self._runner()
        if runner is None or runner._closed or runner.model is not self._model():
            raise RuntimeError('Dynamic precision manifest requires its active native owner')
        return self._manifest_with_extension()


def install_dynamic_prefix_fp8(model,evidence):
    with _mutation_guard(model,'dynamic prefix51 FP8'):
        if getattr(model,base.INSTALLATION,None) is not None:raise ValueError('Prefix owner already installed')
        modules=base._selected_modules(model);device=modules[0].weight.device
        for index,module in enumerate(modules):
            base._validate_module(module,device,(2048,16384) if index%3==2 else (16384,2048),installing=True)
        raw,backend=load_evidence(evidence,device)
        owner=DynamicPrefixFP8Installation(model,evidence,raw,backend)
        sites=[];installed=[]
        try:
            for index,module in enumerate(modules):
                site=_Site(owner,index,module);sites.append(site);owner._installed_sites=tuple(sites);site.initialize()
            owner.sites=tuple(sites)
            for site in sites:site.module.forward=site.forward;installed.append(site)
            setattr(model,base.INSTALLATION,owner)
            owner._manifest=dict(schema=1,precision=base.PRECISION,implementation=VERSION,
                checkpoint_sha256=base.CHECKPOINT_SHA256,scope='prefix_mlp_layers_0_16',
                batch_sizes=list(native.BATCH_SIZES),per_slot_capacities=list(native.PER_SLOT_CAPACITIES),
                precision_boundaries=deepcopy(base.PRECISION_BOUNDARIES),backend=deepcopy(backend),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sites=[deepcopy(site.record) for site in sites],
                quality_claimed=False,original_parameters_preserved=True)
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException as error:
            try:torch.cuda.synchronize(device)
            except BaseException as cleanup:
                owner.sites=tuple(sites);setattr(model,base.INSTALLATION,owner)
                error.add_note('Retained partial dynamic prefix owner after failed drain: '+str(cleanup));raise error
            for site in reversed(installed):site.restore()
            if getattr(model,base.INSTALLATION,None) is owner:delattr(model,base.INSTALLATION)
            owner.enabled=False;owner.sites=owner._installed_sites=();owner._pending.clear()
            raise
