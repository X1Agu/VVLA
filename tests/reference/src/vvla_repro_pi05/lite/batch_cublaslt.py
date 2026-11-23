"""B32 extensions of the selected BF16 expert and biased vision Lt owners.

Algorithms come only from a new, same-device B32 support/timing report. The B1
M50/M512 specifications and implementation files are not modified or reused as
batch qualification. Original parameters and existing graph retirement remain.
"""
import ctypes as C
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import MethodType

import torch

from . import cublaslt_runtime as ffi
from . import expert_cublaslt as expert
from . import vision_fc2_cublaslt as vision
from . import vision_fc2_cublaslt_runtime as biased
from .fused_expert_mlp import _mutation_guard
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity

VERSION = 'pi05-b32-selected-bf16-lt-v1'
GEOMETRIES = {
    'qkv': dict(m=1600, k=1024, n=2560, layout='original_nk'),
    'gate_up_packed': dict(m=1600, k=1024, n=8192, layout='contiguous_kn'),
    'attention_out': dict(m=1600, k=2048, n=1024, layout='original_nk'),
    'vision_fc2': dict(m=16384, k=4304, n=1152, layout='original_nk'),
}
VISION_SHAPE = (64, 256, 4304)


def sources():
    here = Path(__file__).resolve().parent
    names = ('batch_cublaslt.py', 'cublaslt_runtime.py', 'expert_cublaslt.py',
             'vision_fc2_cublaslt.py', 'vision_fc2_cublaslt_runtime.py')
    return {name: ffi.file_sha(here/name) for name in names}


def load_evidence(value):
    if type(value) is not dict or set(value) != {'backend_report', 'backend_report_sha256'}:
        raise ValueError('B32 Lt evidence requires an exact new report path and SHA')
    path = Path(value['backend_report'])
    if path.is_symlink() or not path.is_file() or ffi.file_sha(path) != value['backend_report_sha256']:
        raise ValueError('B32 Lt backend report changed')
    report = json.loads(path.read_text())
    if (report.get('status') != 'complete' or report.get('implementation') != VERSION
            or report.get('source_sha256') != sources() or report.get('policy_quality_passed') is not False
            or report.get('library_sha256') != ffi.LIBRARY_SHA256
            or report.get('checkpoint_sha256') != '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
            or report.get('batch') != 32 or set(report.get('cases', {})) != set(GEOMETRIES)):
        raise ValueError('B32 Lt report does not qualify these exact source/shape/library boundaries')
    for label, shape in GEOMETRIES.items():
        case = report['cases'][label]
        spec = case['selected_spec']
        if ({key:spec.get(key) for key in shape} != shape or case.get('support_passed') is not True
                or case.get('fullgraph_passed') is not True or case.get('cuda_graph_passed') is not True
                or case.get('oracle_passed') is not True or case.get('timing_samples', 0) < 40):
            raise ValueError('Incomplete B32 Lt numerical/capture/timing qualification: '+label)
        raw = path.parent / case['raw_path']
        if (Path(case['raw_path']).name != case['raw_path'] or raw.is_symlink()
                or not raw.is_file() or ffi.file_sha(raw) != case['raw_sha256']):
            raise ValueError('B32 Lt raw evidence changed: '+label)
    return report


def load_api(device):
    device = torch.device(device)
    if (device.type != 'cuda' or torch.__version__ != '2.13.0+cu130'
            or torch.version.git_version != 'cf30153c4c131c8164ee7798e5022d810682e2cb'
            or torch.version.cuda != '13.0' or torch.cuda.is_current_stream_capturing()):
        raise ValueError('B32 Lt requires the pinned Torch/cu130 runtime before capture')
    if (torch.cuda.get_device_capability(device) != (12,0)
            or 'RTX PRO 6000 Blackwell' not in torch.cuda.get_device_name(device)):
        raise ValueError('B32 Lt evidence is specific to PRO6000 SM120')
    if torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('B32 Lt requires selected TF32-off/library-default BF16 reduction math')
    with torch.cuda.device(device):
        torch.empty((1,1),device=device).mm(torch.empty((1,1),device=device))
        library, include = ffi.installed_library()
        headers = biased.verify_bias_headers(include)
        if (headers['header_sha256'] != ffi.HEADER_REFERENCE_SHA256
                or headers['header_version'] != 130101 or ffi.file_sha(library) != ffi.LIBRARY_SHA256):
            raise RuntimeError('B32 Lt library/header ABI identity changed')
        api = ffi.LtAPI(C.CDLL(str(library)))
        if int(api.lib.cublasLtGetVersion()) != 130101:
            raise RuntimeError('B32 Lt runtime version changed')
        uuid=subprocess.check_output(['nvidia-smi','--id='+str(device.index if device.index is not None else torch.cuda.current_device()),
            '--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
        return api, dict(library_sha256=ffi.file_sha(library), runtime_version=130101,gpu_uuid=uuid,
                        header_sha256=headers['header_sha256'],bias_enums=headers['bias_enums'],
                        device_name=torch.cuda.get_device_name(device),capability=[12,0])


class BatchBiasedPlan(biased.BiasedPlan):
    """Same audited column-major bias semantics, new measured M16384 geometry."""
    def __init__(self, api, bias):
        m,k,n = (GEOMETRIES['vision_fc2'][key] for key in ('m','k','n'))
        if (tuple(bias.shape) != (n,) or bias.dtype != torch.bfloat16
                or not bias.is_contiguous() or bias.data_ptr()%256):
            raise ValueError('B32 vision bias requires original aligned contiguous BF16[N]')
        self.api,self.resources=api,[]
        self.m,self.k,self.n,self.layout=m,k,n,'original_nk'
        self.bias,self.bias_pointer=bias,bias.data_ptr()
        self.alpha,self.beta=C.c_float(1),C.c_float(0)
        try:
            self.handle=self.create('cublasLtCreate','cublasLtDestroy')
            self.desc=self.create('cublasLtMatmulDescCreate','cublasLtMatmulDescDestroy',68,0)
            for attr,value in ((2,0),(3,1),(4,0)):
                api.attribute('cublasLtMatmulDescSetAttribute',self.desc,attr,C.c_int32(value))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,7,C.c_uint32(4))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,8,C.c_void_p(self.bias_pointer))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,26,C.c_int32(14))
            self.a,self.b,self.d=self.matrix(k,n,k),self.matrix(k,m,k),self.matrix(n,m,n)
            self.pref=self.create('cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceDestroy')
            api.attribute('cublasLtMatmulPreferenceSetAttribute',self.pref,1,C.c_uint64(ffi.WORKSPACE_BYTES))
            api.attribute('cublasLtMatmulPreferenceSetAttribute',self.pref,3,C.c_uint32(7))
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:error.add_note('B32 biased descriptor cleanup: '+str(cleanup))
            raise


class BatchBiasedBoundPlan(biased.BoundPlan):
    def __init__(self,api,weight,bias,spec):
        self.key,self.closed,self.owner=next(biased._IDS),False,None
        self.plan=self.workspace=None
        self.weight,self.bias,self.device=weight,bias,weight.device
        self.weight_pointer,self.bias_pointer=weight.data_ptr(),bias.data_ptr()
        self.spec=deepcopy(spec);self._installed_spec=deepcopy(spec)
        try:
            self.plan=BatchBiasedPlan(api,bias)
            self.rank,self.result,info=ffi.select_algorithm(self.plan,self.spec)
            self.info={key:value for key,value in info.items() if key!='opaque_algo_hex'}
            self.workspace=torch.empty(max(256,spec['workspace_bytes']),dtype=torch.uint8,device=self.device)
            self._workspace_identity=tensor_identity(self.workspace)
            biased._PLANS[self.key]=self
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:error.add_note('B32 biased plan cleanup: '+str(cleanup))
            raise

    def execute(self,x,weight,bias):
        owner=None if self.owner is None else self.owner()
        if self.closed or owner is None:raise RuntimeError('B32 vision plan is unbound/retired')
        owner.assert_execution_owner()
        if (tuple(x.shape)!=VISION_SHAPE or tuple(weight.shape)!=(1152,4304) or tuple(bias.shape)!=(1152,)
                or any(t.dtype!=torch.bfloat16 or t.device!=self.device or not t.is_contiguous() for t in (x,weight,bias))
                or weight.data_ptr()!=self.weight_pointer or bias.data_ptr()!=self.bias_pointer
                or x.data_ptr()%16 or weight.data_ptr()%16 or bias.data_ptr()%256 or self.workspace.data_ptr()%256):
            raise ValueError('B32 vision execution requires its fixed original BF16 sources and [64,256,4304]')
        output=x.new_empty((64,256,1152))
        self.plan.launch(x,weight,output,self.result.algo,self.workspace,torch.cuda.current_stream(x.device))
        return output


class BatchExpertRuntime(ffi.ExpertLtRuntime):
    def __init__(self,device,report):
        self.plans={};self.profile='g4-b32-measured-three'
        self.specs={key:deepcopy(report['cases'][key]['selected_spec']) for key in GEOMETRIES if key!='vision_fc2'}
        self.device=torch.device(device);self.api,self.evidence=load_api(device)
        if report.get('gpu_uuid')!=self.evidence['gpu_uuid']:raise ValueError('B32 expert Lt support was measured on another card')
        self.evidence['batch_report_sha256']=report['_report_sha256']
        try:
            for label,spec in self.specs.items():self.plans[label]=ffi.BoundPlan(self.api,spec,self.device)
        except BaseException:
            self.close();raise


class BatchVisionRuntime(biased.VisionFC2LtRuntime):
    def __init__(self,device,sources_,report):
        self.plans=[];self.device=torch.device(device);self.api,self.evidence=load_api(device)
        if report.get('gpu_uuid')!=self.evidence['gpu_uuid']:raise ValueError('B32 vision Lt support was measured on another card')
        self.evidence['batch_report_sha256']=report['_report_sha256']
        spec=report['cases']['vision_fc2']['selected_spec']
        try:
            for weight,bias in sources_:self.plans.append(BatchBiasedBoundPlan(self.api,weight,bias,spec))
            self._installed_plans=tuple(self.plans)
        except BaseException:
            self.close();raise

    def manifest(self):
        return dict(version=VERSION,batch=32,evidence=deepcopy(self.evidence),
            math=dict(input='BF16',weight='BF16',bias='BF16',output='BF16',main_accumulator='FP32',
                      tf32=False,epilogue='BIAS before BF16 output',reduction_policy='library-default'),
            policy_quality_passed=False,
            plans=[dict(spec=deepcopy(p.spec),selected_rank=p.rank,algorithm=dict(p.info),workspace_bytes=p.workspace.numel())for p in self.plans])


class _ExpertSite(expert._Site):
    def __call__(self,x):
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:raise RuntimeError('B32 Lt requires native prepared compile scope')
        else:self.installation.assert_execution_owner();self.assert_ready()
        if (x.dtype!=torch.bfloat16 or x.device!=self.device or tuple(x.shape)!=(32,50,self.spec['k'])
                or not x.is_contiguous()):raise ValueError('B32 expert Lt requires exact contiguous [32,50,K] BF16')
        return ffi.expert_cublaslt(x,self._weight,self.plan.key,self.spec['n'])


class _LayerCalls(expert._LayerCalls):
    def __init__(self,installation,layer,packer,candidate):
        self.installation,self.layer,self.packer,self.candidate=installation,layer,packer,candidate
        self.original_qkv,self.original_mlp,self.original_out=packer.project,candidate.forward,layer.self_attn.o_proj.forward
        self.sites=dict(qkv=_ExpertSite(installation,'qkv',lambda:packer._packed),
            gate_up_packed=_ExpertSite(installation,'gate_up_packed',lambda:candidate._packed),
            attention_out=_ExpertSite(installation,'attention_out',lambda:layer.self_attn.o_proj.weight))
        self.qkv_forward,self.mlp_forward=self.project,self.mlp
        def forward(module,x,_calls=self):return _calls.out(module,x)
        self.out_forward=MethodType(forward,layer.self_attn.o_proj)


class BatchExpertInstallation(expert.ExpertCublasLtInstallation):
    def manifest(self):
        value=super().manifest();value.update(version=VERSION,batch=32,source_sha256=sources())
        return value


class _VisionSite(vision._Site):
    def call(self,module,x):
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                or tuple(x.shape)!=VISION_SHAPE or x.dtype!=torch.bfloat16 or x.device!=self.device or not x.is_contiguous()):
            return torch.nn.Linear.forward(module,x)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:raise RuntimeError('B32 vision Lt requires native compile scope')
        else:self.installation.assert_execution_owner();self.assert_ready()
        return biased.vision_fc2_cublaslt(x,self.weight,self.bias,self.plan.key)


class BatchVisionInstallation(vision.VisionFC2CublasLtInstallation):
    def manifest(self):
        value=super().manifest();value.update(version=VERSION,batch=32,input_shape=list(VISION_SHAPE),source_sha256=sources())
        return value


def install_batch_cublaslt(model,evidence):
    """Install both B32 owners after generic fusions, before prefix packing."""
    report=load_evidence(evidence);report['_report_sha256']=evidence['backend_report_sha256']
    with _mutation_guard(model,'B32 selected Lt installation'):
        if getattr(model,expert.INSTALLATION,None) is not None or getattr(model,vision.INSTALLATION,None) is not None:
            raise ValueError('B32 Lt requires unmodified original expert/Lt dispatch')
        qkv=getattr(model,'_lite_expert_qkv_installation',None);mlp=getattr(model,'_lite_expert_mlp_installation',None)
        if qkv is None or mlp is None or not qkv.enabled or not mlp.enabled or not mlp.concatenate:
            raise ValueError('B32 Lt requires selected packed QKV and gate/up prerequisites')
        layers=model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers)!=18 or len(qkv._packers)!=18 or len(mlp._candidates)!=18:
            raise ValueError('B32 Lt requires all18 original expert layers')
        for layer,packer,candidate in zip(layers,qkv._packers,mlp._candidates,strict=True):
            packer.validate_geometry();candidate.validate()
            if (packer.width!=1024 or packer.sizes!=(2048,256,256) or candidate.width!=1024
                    or candidate.intermediate!=4096 or 'project'in packer.__dict__ or 'forward'in candidate.__dict__
                    or 'forward'in layer.self_attn.o_proj.__dict__):raise ValueError('B32 Lt prerequisite geometry/dispatch changed')
        vision_layers=vision._layers(model)
        if len(vision_layers)!=27 or len({id(layer.mlp.fc2)for layer in vision_layers})!=27:
            raise ValueError('B32 vision Lt requires all27 distinct original layers')
        device=qkv._packers[0].device
        for layer in vision_layers:vision._geometry(layer.mlp.fc2,device,installing=True)
        expert_owner=vision_owner=None;expert_runtime=vision_runtime=None
        installed_expert=[];installed_vision=[]
        try:
            expert_runtime=BatchExpertRuntime(device,report)
            expert_owner=BatchExpertInstallation(model,expert_runtime,qkv,mlp,'g4-b32-measured-three',False)
            for layer,packer,candidate in zip(layers,qkv._packers,mlp._candidates,strict=True):
                expert_owner.layers.append(_LayerCalls(expert_owner,layer,packer,candidate))
            for calls in expert_owner.layers:calls.install();installed_expert.append(calls)
            setattr(model,expert.INSTALLATION,expert_owner)
            vision_runtime=BatchVisionRuntime(device,tuple((layer.mlp.fc2.weight,layer.mlp.fc2.bias)for layer in vision_layers),report)
            vision_owner=BatchVisionInstallation(model,vision_runtime)
            vision_owner.sites=[_VisionSite(vision_owner,layer,plan)for layer,plan in zip(vision_layers,vision_runtime.plans,strict=True)]
            vision_owner._installed_sites=tuple(vision_owner.sites)
            for site in vision_owner.sites:site.module.forward=site.forward;installed_vision.append(site)
            setattr(model,vision.INSTALLATION,vision_owner)
            model.invalidate_inference_caches(weights_changed=False)
            return expert_owner,vision_owner
        except BaseException as error:
            for site in reversed(installed_vision):site.restore()
            for calls in reversed(installed_expert):calls.restore()
            for name,owner,runtime in ((vision.INSTALLATION,vision_owner,vision_runtime),(expert.INSTALLATION,expert_owner,expert_runtime)):
                if getattr(model,name,None)is owner and owner is not None:delattr(model,name)
                if owner is not None:owner.enabled=False
                if runtime is not None:
                    try:runtime.close()
                    except BaseException as cleanup:error.add_note('B32 Lt partial cleanup: '+str(cleanup))
            raise
