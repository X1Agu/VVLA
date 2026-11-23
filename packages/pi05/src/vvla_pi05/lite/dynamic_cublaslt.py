"""Historical per-geometry Lt implementation for shared-weight true batching.

Explicit semantic tactics are supplied as an asset. No author identity gate,
autotuning, silent fallback, or fabricated algorithm selection is permitted.
"""
import ctypes as C
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch

from . import cublaslt_runtime as ffi
from . import vision_fc2_cublaslt_runtime as biased
from . import expert_cublaslt as expert
from . import vision_fc2_cublaslt as vision
from .batch_layout import validate_fixed_batch_size
from .fused_expert_mlp import _mutation_guard
from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

VERSION='source497-dynamic-bf16-lt-v1'
BATCH_SIZES=(1,2,4,8,16,32)
_UNRETIRED=[]
BASE_SHAPES={'qkv':(1024,2560,'original_nk'),
    'gate_up_packed':(1024,8192,'contiguous_kn'),
    'attention_out':(2048,1024,'original_nk'),
    'vision_fc2':(4304,1152,'original_nk')}


def geometries(batch):
    validate_fixed_batch_size(batch)
    return {label:dict(m=batch*(512 if label=='vision_fc2' else 50),k=k,n=n,layout=layout)
        for label,(k,n,layout) in BASE_SHAPES.items()}


def sources():
    here=Path(__file__).resolve().parent
    return {name:ffi.file_sha(here/name) for name in ('dynamic_cublaslt.py','cublaslt_runtime.py',
        'vision_fc2_cublaslt_runtime.py','expert_cublaslt.py','vision_fc2_cublaslt.py')}


def load_tactics(value):
    """Load explicit historical semantic tactics; never select new parameters.

    Accept the original report or an extracted spec-only asset. Identity and
    timing evidence remain provenance, not GPU UUID or weight admission locks.
    """
    if not isinstance(value,(str,Path)):
        raise ValueError('Provide assets.lt_tactics pointing to the preserved historical per-batch algorithm table')
    path=Path(value).expanduser().resolve(strict=True)
    raw=path.read_bytes();report=json.loads(raw)
    if report.get('implementation')!=VERSION or set(report.get('batches',{}))!={str(b) for b in BATCH_SIZES}:
        raise ValueError('Expected the complete historical dynamic Lt table for B=1,2,4,8,16,32')
    algorithm_keys={'id','tile','stages','swizzle','split_k','reduction_scheme','custom_option','numerical_impl_flags'}
    for batch in BATCH_SIZES:
        cases=report['batches'][str(batch)]
        if set(cases)!=set(BASE_SHAPES):raise ValueError('Incomplete dynamic Lt projection table')
        for label,shape in geometries(batch).items():
            spec=cases[label]['selected_spec']
            if (set(spec)!={*shape,'workspace_bytes','algorithm'} or
                    {key:spec.get(key) for key in shape}!=shape or
                    type(spec['workspace_bytes']) is not int or not 0<=spec['workspace_bytes']<=ffi.WORKSPACE_BYTES or
                    set(spec['algorithm'])!=algorithm_keys or
                    any(type(v) is not int or v<0 for v in spec['algorithm'].values()) or
                    spec['algorithm']['numerical_impl_flags']&0xff0000!=0x020000):
                raise ValueError('Invalid historical Lt geometry, workspace or BF16 algorithm specification')
    report['_report_sha256']=hashlib.sha256(raw).hexdigest()
    return report


def load_api(device):
    device=torch.device(device)
    if (device.type!='cuda' or not str(torch.__version__).startswith('2.13.') or
            torch.version.cuda is None or torch.version.cuda.split('.')[0]!='13' or
            torch.cuda.get_device_capability(device)!=(12,0)):
        raise ValueError('Historical dynamic Lt uses the Torch 2.13/CUDA 13 SM120 backend')
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Build Lt plans before capture')
    if torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('Preserve TF32-off and library-default BF16 reduction')
    with torch.cuda.device(device):
        torch.empty((1,1),device=device).mm(torch.empty((1,1),device=device))
        library,include=ffi.installed_library();headers=biased.verify_bias_headers(include)
        api=ffi.LtAPI(C.CDLL(str(library)));version=int(api.lib.cublasLtGetVersion())
        if version//10000!=13:raise RuntimeError('cuBLASLt runtime is outside the checked ABI')
    return api,dict(library_sha256=ffi.file_sha(library),runtime_version=version,
        header_sha256=headers['header_sha256'],capability=[12,0],
        admission='checked ABI and exact semantic tactic validation; no autotuning',quality_qualified=False)


class TacticUnavailable(RuntimeError):
    """Validated input describes a tactic unsupported by this actual device."""


def preflight_tactics(report,device):
    """Check all supplied algorithms before loading weights; no GEMM tuning.

    Only a tiny bias tensor and Lt descriptors are needed. Allocation, ABI and
    unexpected runtime failures propagate instead of becoming portable fallback.
    """
    api,identity=load_api(device)
    checked=0
    with torch.cuda.device(device):
        bias=torch.empty((1152,),device=device,dtype=torch.bfloat16)
        for batch in BATCH_SIZES:
            for label,shape in geometries(batch).items():
                plan=(DynamicBiasedPlan(api,bias,batch) if label=='vision_fc2' else
                      ffi.Plan(api,shape['m'],shape['k'],shape['n'],shape['layout'],'library-default'))
                try:
                    ffi.select_algorithm(plan,report['batches'][str(batch)][label]['selected_spec'])
                    checked+=1
                except ffi.AlgorithmUnavailable as error:
                    raise TacticUnavailable(f'B={batch} {label}: {error}') from error
                except ffi.LtError as error:
                    if error.status in (7,8,15) and error.operation in (
                        'cublasLtMatmulAlgoInit','cublasLtMatmulAlgoConfigSetAttribute','cublasLtMatmulAlgoCheck'):
                        raise TacticUnavailable(f'B={batch} {label}: {error}') from error
                    raise
                finally:plan.close()
        properties=torch.cuda.get_device_properties(device)
    return dict(geometries_checked=checked,device_name=properties.name,
                multiprocessor_count=properties.multi_processor_count,runtime=identity,
                selection='same supplied semantic tactics; no timing or autotuning',performance_qualified=False)



class DynamicBiasedPlan(biased.BiasedPlan):
    def __init__(self,api,bias,batch):
        self.api,self.resources=api,[]
        self.m,self.k,self.n,self.layout=512*validate_fixed_batch_size(batch),4304,1152,'original_nk'
        if tuple(bias.shape)!=(1152,) or bias.dtype!=torch.bfloat16 or not bias.is_contiguous() or bias.data_ptr()%256:
            raise ValueError('Dynamic vision bias must retain aligned original BF16[N]')
        self.bias,self.bias_pointer=bias,bias.data_ptr();self.alpha,self.beta=C.c_float(1),C.c_float(0)
        try:
            self.handle=self.create('cublasLtCreate','cublasLtDestroy')
            self.desc=self.create('cublasLtMatmulDescCreate','cublasLtMatmulDescDestroy',68,0)
            for attr,value in ((2,0),(3,1),(4,0)):api.attribute('cublasLtMatmulDescSetAttribute',self.desc,attr,C.c_int32(value))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,7,C.c_uint32(4))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,8,C.c_void_p(self.bias_pointer))
            api.attribute('cublasLtMatmulDescSetAttribute',self.desc,26,C.c_int32(14))
            self.a,self.b,self.d=self.matrix(self.k,self.n,self.k),self.matrix(self.k,self.m,self.k),self.matrix(self.n,self.m,self.n)
            self.pref=self.create('cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceDestroy')
            api.attribute('cublasLtMatmulPreferenceSetAttribute',self.pref,1,C.c_uint64(ffi.WORKSPACE_BYTES))
            api.attribute('cublasLtMatmulPreferenceSetAttribute',self.pref,3,C.c_uint32(7))
        except BaseException:
            self.close();raise


class DynamicBiasedBoundPlan(biased.BoundPlan):
    def __init__(self,api,weight,bias,spec,batch):
        self.key,self.closed,self.owner=next(biased._IDS),False,None
        self.plan=self.workspace=None
        self.weight,self.bias,self.device=weight,bias,weight.device
        self.weight_pointer,self.bias_pointer=weight.data_ptr(),bias.data_ptr()
        self.spec=deepcopy(spec);self._installed_spec=deepcopy(spec);self.batch=validate_fixed_batch_size(batch)
        if (weight.device.type!='cuda' or bias.device!=weight.device or weight.dtype!=torch.bfloat16
                or tuple(weight.shape)!=(1152,4304) or not weight.is_contiguous()
                or {key:spec.get(key) for key in geometries(batch)['vision_fc2']}!=geometries(batch)['vision_fc2']):
            raise ValueError('Dynamic biased plan requires actual qualified CUDA BF16 sources and geometry')
        try:
            self.plan=DynamicBiasedPlan(api,bias,batch)
            self.rank,self.result,info=ffi.select_algorithm(self.plan,self.spec)
            self.info={k:v for k,v in info.items() if k!='opaque_algo_hex'}
            if self.info['numerical_impl_flags']&0xff0000!=0x020000:raise RuntimeError('Dynamic vision Lt must use exclusively BF16 input math')
            self.workspace=torch.empty(max(256,spec['workspace_bytes']),device=self.device,dtype=torch.uint8)
            self._workspace_identity=tensor_identity(self.workspace);biased._PLANS[self.key]=self
        except BaseException:
            self.close();raise

    def execute(self,x,weight,bias):
        owner=None if self.owner is None else self.owner()
        if self.closed or owner is None:raise RuntimeError('Dynamic biased plan closed or detached')
        owner.assert_execution_owner()
        if (self.spec['m']!=512*self.batch or tuple(x.shape)!=(2*self.batch,256,4304) or tuple(weight.shape)!=(1152,4304) or tuple(bias.shape)!=(1152,)
                or any(t.dtype!=torch.bfloat16 or t.device!=self.device or not t.is_contiguous() for t in (x,weight,bias))
                or weight.data_ptr()!=self.weight_pointer or bias.data_ptr()!=self.bias_pointer
                or x.data_ptr()%16 or weight.data_ptr()%16 or bias.data_ptr()%256 or self.workspace.data_ptr()%256):
            raise ValueError('Dynamic vision Lt requires its fixed qualified true-B camera geometry')
        output=x.new_empty((2*self.batch,256,1152))
        self.plan.launch(x,weight,output,self.result.algo,self.workspace,torch.cuda.current_stream(x.device))
        return output


class _ExpertRuntime(ffi.ExpertLtRuntime):
    def __init__(self,device,report):
        self.device=torch.device(device);self.api,self.evidence=load_api(device)
        self.evidence['backend_report_sha256']=report['_report_sha256']
        self.profile=VERSION;self.batch_plans={};self.plans={};self.specs={}
        try:
            for batch in BATCH_SIZES:
                self.batch_plans[batch]={}
                for label in ('qkv','gate_up_packed','attention_out'):
                    spec=deepcopy(report['batches'][str(batch)][label]['selected_spec'])
                    self.batch_plans[batch][label]=ffi.BoundPlan(self.api,spec,self.device)
            self.plans=self.batch_plans[1]
            self.specs={label:plan.spec for label,plan in self.plans.items()}
        except BaseException:self.close();raise
    def bind_owner(self,owner):
        for plans in self.batch_plans.values():
            for plan in plans.values():plan.bind_owner(owner)
    def validate(self):
        if set(self.batch_plans)!=set(BATCH_SIZES):raise RuntimeError('Dynamic Lt plan geometry set changed')
        for plans in self.batch_plans.values():
            if set(plans)!={'qkv','gate_up_packed','attention_out'}:raise RuntimeError('Dynamic Lt projection set changed')
            for plan in plans.values():plan.validate()
    def manifest(self):
        return dict(version=VERSION,evidence=deepcopy(self.evidence),
            plans={str(b):{label:dict(spec=deepcopy(p.spec),algorithm=p.info) for label,p in plans.items()} for b,plans in self.batch_plans.items()})
    def close(self):
        failures=[]
        for plans in reversed(tuple(self.batch_plans.values())):
            for plan in plans.values():
                try:plan.close()
                except BaseException as error:failures.append(repr(error))
        if failures:raise RuntimeError('Dynamic Lt cleanup: '+'; '.join(failures))
        self.batch_plans.clear();self.plans={}


class _ExpertSite(expert._Site):
    def __call__(self,x):
        precision=getattr(self,"_projection_precision",None)
        if precision is not None:return precision(x)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:raise RuntimeError('Dynamic Lt requires prepared compile scope')
        else:self.installation.assert_execution_owner();self.assert_ready()
        batch=validate_fixed_batch_size(x.shape[0])
        if x.dtype!=torch.bfloat16 or x.device!=self.device or tuple(x.shape)!=(batch,50,self.spec['k']) or not x.is_contiguous():
            raise ValueError('Dynamic expert Lt requires current [B,50,K]')
        plan=self.installation.runtime.batch_plans[batch][self.label]
        return ffi.expert_cublaslt(x,self._weight,plan.key,self.spec['n'])


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


class _VisionRuntime(biased.VisionFC2LtRuntime):
    def __init__(self,device,sources_,report):
        self.device=torch.device(device);self.api,self.evidence=load_api(device)
        self.evidence['backend_report_sha256']=report['_report_sha256']
        self.plans=[];self.batch_plans={}
        try:
            for batch in BATCH_SIZES:
                self.batch_plans[batch]=[]
                spec=report['batches'][str(batch)]['vision_fc2']['selected_spec']
                for weight,bias in sources_:
                    plan=DynamicBiasedBoundPlan(self.api,weight,bias,spec,batch)
                    self.plans.append(plan);self.batch_plans[batch].append(plan)
            self._installed_plans=tuple(self.plans)
        except BaseException:self.close();raise
    def manifest(self):
        return dict(version=VERSION,evidence=deepcopy(self.evidence),
            plans={str(batch):[dict(spec=deepcopy(p.spec),algorithm=p.info) for p in plans]
                   for batch,plans in self.batch_plans.items()})
    def close(self):
        super().close()
        self.batch_plans.clear()


class _VisionSite(vision._Site):
    def __init__(self,installation,layer,index):
        super().__init__(installation,layer,installation.runtime.batch_plans[1][index])
        self.index=index
    def call(self,module,x):
        if module is not self.module:raise RuntimeError('Dynamic vision source module changed')
        precision=getattr(self,'_projection_precision',None)
        if precision is not None:return precision(x)
        model=self.installation._model()
        if model is None:raise RuntimeError('Dynamic vision model detached')
        camera=getattr(model,'_lite_batched_vision_installation')
        batch=camera.batch_size
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                or tuple(x.shape)!=(2*batch,256,4304) or x.dtype!=torch.bfloat16
                or x.device!=self.device or not x.is_contiguous()):
            # Preserve source497's original fallback for initialization of the
            # single masked camera, training and other noncandidate geometry.
            return torch.nn.Linear.forward(module,x)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:raise RuntimeError('Dynamic vision requires prepared compile scope')
        else:self.installation.assert_execution_owner();self.assert_ready()
        plan=self.installation.runtime.batch_plans[batch][self.index]
        return biased.vision_fc2_cublaslt(x,self.weight,self.bias,plan.key)


class DynamicExpertInstallation(expert.ExpertCublasLtInstallation):
    def manifest(self):
        value=super().manifest();value.update(version=VERSION,batch_sizes=list(BATCH_SIZES),source_sha256=sources())
        return value


class DynamicVisionInstallation(vision.VisionFC2CublasLtInstallation):
    def _validate(self):
        self.runtime.validate()
        layers=vision._layers(self._model())
        if (set(self.runtime.batch_plans)!=set(BATCH_SIZES)
                or any(len(plans)!=len(layers) for plans in self.runtime.batch_plans.values())
                or tuple(self.runtime.plans)!=tuple(p for batch in BATCH_SIZES for p in self.runtime.batch_plans[batch])
                or not layers or len(self.sites)!=len(layers) or len(self._installed_sites)!=len(layers)
                or any(a is not b for a,b in zip(self.sites,self._installed_sites,strict=True))
                or any(layer is not site.layer for layer,site in zip(layers,self.sites,strict=True))
                or any(site.plan is not plan for site,plan in zip(self.sites,self.runtime.batch_plans[1],strict=True))):
            raise RuntimeError('Dynamic vision layer/geometry/plan ownership changed')
        return tuple(site.validate() for site in self.sites)
    def manifest(self):
        value=super().manifest();value.update(version=VERSION,batch_sizes=list(BATCH_SIZES),source_sha256=sources())
        value['input_shape']=['2B',256,4304]
        return value


def install_dynamic_cublaslt(model,tactics):
    report=load_tactics(tactics)
    with _mutation_guard(model,'source497 dynamic Lt installation'):
        if getattr(model,expert.INSTALLATION,None) is not None or getattr(model,vision.INSTALLATION,None) is not None:
            raise ValueError('Dynamic Lt requires original unmodified dispatch')
        qkv=getattr(model,'_lite_expert_qkv_installation',None)
        mlp=getattr(model,'_lite_expert_mlp_installation',None)
        if qkv is None or mlp is None or not qkv.enabled or not mlp.enabled or not mlp.concatenate:
            raise ValueError('Dynamic Lt requires selected original packed QKV and gate/up')
        layers=model.paligemma_with_expert.gemma_expert.model.layers
        if not layers or len(qkv._packers)!=len(layers) or len(mlp._candidates)!=len(layers):
            raise ValueError('Dynamic Lt requires every original expert layer')
        for layer,packer,candidate in zip(layers,qkv._packers,mlp._candidates,strict=True):
            packer.validate_geometry();candidate.validate()
            if (packer.width!=1024 or packer.sizes!=(2048,256,256) or candidate.width!=1024 or candidate.intermediate!=4096
                    or 'project' in packer.__dict__ or 'forward' in candidate.__dict__ or 'forward' in layer.self_attn.o_proj.__dict__):
                raise ValueError('Dynamic Lt source geometry/dispatch changed')
        vision_layers=vision._layers(model)
        if not vision_layers:raise ValueError('Dynamic vision requires all declared layers')
        for layer in vision_layers:vision._geometry(layer.mlp.fc2,layers[0].self_attn.o_proj.weight.device,installing=True)
        er=vr=eh=vh=None
        try:
            er=_ExpertRuntime(layers[0].self_attn.o_proj.weight.device,report)
            eh=DynamicExpertInstallation(model,er,qkv,mlp,VERSION,False)
            for layer,packer,candidate in zip(layers,qkv._packers,mlp._candidates,strict=True):
                eh.layers.append(_LayerCalls(eh,layer,packer,candidate))
            vr=_VisionRuntime(er.device,[(l.mlp.fc2.weight,l.mlp.fc2.bias) for l in vision_layers],report)
            vh=DynamicVisionInstallation(model,vr)
            vh.sites=[_VisionSite(vh,layer,index) for index,layer in enumerate(vision_layers)]
            vh._installed_sites=tuple(vh.sites)
            for calls in eh.layers:calls.install()
            for site in vh.sites:site.module.forward=site.forward
            setattr(model,expert.INSTALLATION,eh);setattr(model,vision.INSTALLATION,vh)
            model.invalidate_inference_caches(weights_changed=False)
            return eh,vh
        except BaseException as error:
            try:torch.cuda.synchronize()
            except BaseException as cleanup:
                _UNRETIRED.append((model,er,vr,eh,vh))
                error.add_note('Retained partial dynamic Lt owners after failed drain: '+repr(cleanup))
                raise error
            if eh is not None:
                for calls in eh.layers:calls.restore()
            if vh is not None:
                for site in vh.sites:site.restore()
            for name,owner in ((expert.INSTALLATION,eh),(vision.INSTALLATION,vh)):
                if owner is not None and getattr(model,name,None) is owner:delattr(model,name)
            if vr is not None:vr.close()
            if er is not None:er.close()
            raise
