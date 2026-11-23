"""Private shared-weight source497 adaptation for independent B1 admission.

The executor accepts only real power-of-two batches; schedulers decompose other
queue sizes without dummy lanes. New same-card primitive evidence is mandatory.
Model quality and service results are reported by the separate P0 campaign.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .batch_pi05 import _batch_options,_digest,_sources,ORIGIN_RESOLVED_CONFIG,ORIGIN_RECIPE
from .batch_layout import validate_fixed_batch_size
from .batch_processor import validate_observations
from .dynamic_batch_prepare import DynamicCoalescedProcessor,DynamicHostLayoutCache,implementation_manifest

BATCH_SIZES=(1,2,4,8,16,32)
VERSION='source497-private-dynamic-batches-v1'


def _suffix_evidence(value,device):
    from .dynamic_batch_geometry import sources
    from .prefix_fp8 import _hardware
    if type(value) is not dict or set(value)!={'backend_report','backend_report_sha256'}:
        raise ValueError('Dynamic suffix requires its own primitive report')
    path=Path(value['backend_report']);raw=path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest()!=value['backend_report_sha256']:
        raise ValueError('Dynamic suffix evidence changed')
    report=json.loads(raw)
    expected={(b,p) for b in BATCH_SIZES for p in (576,640)}
    if (report.get('status')!='complete' or report.get('source_sha256')!=sources()
            or report.get('gpu_uuid')!=_hardware(device)['uuid']
            or report.get('policy_quality_passed') is not False
            or {(r['batch'],r['prefix_capacity']) for r in report['rows']}!=expected
            or any(r['status']!='complete' or r['oracle']['passed'] is not True for r in report['rows'])):
        raise ValueError('Dynamic suffix report lacks all exact same-card geometries')
    for row in report['rows']:
        p=path.parent/row['raw_path']
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(path.parent.resolve()):
            raise ValueError('Unsafe suffix evidence path')
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=row['raw_sha256']:raise ValueError('Suffix raw evidence changed')


class DynamicPi05Engine:
    __slots__=('__policy','__processors','__state','__sources','__manifest','__last_noise','__calls')
    def __init__(self,*args,**kwargs):raise TypeError('Use the private checkpoint factory')
    def __setattr__(self,name,value):raise AttributeError('Immutable checkpoint executor; reconstruct')
    def __reduce_ex__(self,protocol):raise TypeError('Private executor cannot be serialized')

    @classmethod
    def from_pretrained(cls,checkpoint,*,tokenizer,prefix_evidence,lt_evidence,suffix_evidence,
                        device='cuda:0',max_graphs=16):
        if cls is not DynamicPi05Engine:raise TypeError('Private factory requires its exact engine type')
        from .pi05 import Pi05LitePolicy
        from .dynamic_batch_geometry import install_dynamic_geometry
        from .dynamic_cublaslt import install_dynamic_cublaslt
        from .dynamic_prefix_fp8 import install_dynamic_prefix_fp8
        from .batch_checkpoint_engine import BatchSealedModelState,_CAPABILITY
        selected,loader=_batch_options({})
        policy=Pi05LitePolicy.from_pretrained(checkpoint,tokenizer=tokenizer,device=device,
            profile='full-bf16',attention_backend='torch_fa2',cuda_graph=True,max_graphs=max_graphs,
            precision='bf16',config_overrides=loader)
        state=None
        try:
            if (policy._provenance['checkpoint_sha256']!='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
                    or _digest(policy._provenance['resolved_config'])!=ORIGIN_RESOLVED_CONFIG
                    or (policy.config.chunk_size,policy.config.num_inference_steps,policy.config.max_action_dim,policy.config.action_dim)!=(50,10,32,7)):
                raise ValueError('Dynamic execution requires the exact source497 checkpoint/config/action recipe')
            _suffix_evidence(suffix_evidence,policy.device)
            runner=policy.graph_runner
            runner._host_layouts=DynamicHostLayoutCache(capacity=max_graphs*2)
            policy._vision_installation,policy._suffix_fa2_installation=install_dynamic_geometry(policy.model)
            policy._expert_cublaslt_installation,policy._vision_fc2_cublaslt_installation=install_dynamic_cublaslt(policy.model,lt_evidence)
            for owner in (policy._expert_cublaslt_installation,policy._vision_fc2_cublaslt_installation):owner.bind_runner(runner)
            try:policy._prefix_fp8_installation=install_dynamic_prefix_fp8(policy.model,prefix_evidence)
            except BaseException:
                policy._prefix_fp8_installation=getattr(policy.model,'_lite_prefix_fp8_installation',None);raise
            policy._prefix_fp8_installation.bind_runner(runner)
            options=policy._lite_optimizations
            for key,value in selected.items():
                if key.startswith('lite_'):options[key[5:]]=deepcopy(value)
            options.update(expert_cublaslt=dict(profile='same-card-per-B-measured',include_down=False),
                suffix_fa2=dict(splits={str(b):8 if b==1 else 1 for b in BATCH_SIZES}),
                dynamic_input_preparation=implementation_manifest(),fixed_checkpoint_engine=True,
                dynamic_evidence=deepcopy(dict(prefix=prefix_evidence,lt=lt_evidence,suffix=suffix_evidence)))
            for key,owner in (('batched_vision_manifest',policy._vision_installation),('suffix_fa2_manifest',policy._suffix_fa2_installation),
                    ('expert_cublaslt_manifest',policy._expert_cublaslt_installation),('vision_fc2_cublaslt_manifest',policy._vision_fc2_cublaslt_installation)):
                options[key]=owner.manifest()
            engine=object.__new__(cls)
            processors={b:DynamicCoalescedProcessor(policy.processor,b,output_processor=policy._action_norm_cache) for b in BATCH_SIZES}
            with policy._lock,runner._call_lock:
                state=BatchSealedModelState(policy,_CAPABILITY);state.audit()
                precision=policy._prefix_fp8_installation.validate_precision_manifest()
                manifest=dict(implementation=VERSION,batch_sizes=list(BATCH_SIZES),precision='mixed_bf16_fp8_prefix',
                    checkpoint_sha256=policy._provenance['checkpoint_sha256'],resolved_config_sha256=ORIGIN_RESOLVED_CONFIG,
                    origin_recipe_sha256=ORIGIN_RECIPE,options=deepcopy(options),quantization=precision,
                    source_sha256=_sources(),input_preparation=implementation_manifest(),
                    dummy_lanes=False,per_request_noise='independent seeded CUDA FP32[1,50,32], concatenated',
                    sampler_steps=10,action_shape_per_request=[50,7],action_space='physical_libero',
                    policy_quality_passed=False,qualification_scope='primitive evidence and cold owner seal; external model/service campaign required',
                    compile_libdevice_cache=False)
            for name,value in dict(policy=policy,processors=processors,state=state,sources=_sources(),manifest=manifest,last_noise=None,calls=0).items():
                object.__setattr__(engine,'_DynamicPi05Engine__'+name,value)
            return engine
        except BaseException as error:
            try:
                if state is not None:
                    with policy._lock,policy.graph_runner._call_lock:state.detach(policy,policy.graph_runner)
                policy.close()
            except BaseException as cleanup:error.add_note('Dynamic factory cleanup: '+repr(cleanup))
            raise

    @torch.inference_mode()
    def infer_batch(self,observations,seeds):
        batch=validate_fixed_batch_size(len(observations));validate_observations(observations,batch)
        if (type(seeds) is not list or len(seeds)!=batch or
                any(type(s) is not int or not 0<=s<2**63 for s in seeds)):
            raise ValueError('Each current observation needs an independent uint63 seed')
        policy=self.__policy;started=time.perf_counter_ns()
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Serialize access to the shared GPU executor')
        try:
            self.__state._assert_live()
            policy._vision_installation.select_batch(batch)
            processor=self.__processors[batch]
            images,image_masks,tokens,masks,layout=processor.build_model_inputs(observations,return_host_layout=True)
            noise=torch.cat([torch.randn((1,50,32),device=policy.device,dtype=torch.float32,
                generator=torch.Generator(device=policy.device).manual_seed(seed)) for seed in seeds],0)
            prepared=time.perf_counter_ns()
            value=policy.graph_runner(images=images,image_masks=image_masks,lang_tokens=tokens,lang_masks=masks,
                noise=noise,num_steps=10,host_layout=layout)
            actions=processor.build_model_outputs(value,observations)
            object.__setattr__(self,'_DynamicPi05Engine__last_noise',noise)
            object.__setattr__(self,'_DynamicPi05Engine__calls',self.__calls+1)
            info=dict(policy.graph_runner.last_call)
            return dict(actions=actions,cold=bool(info['cold']),diagnostics=dict(info,batch_size=batch,
                model_calls=1,request_seeds=seeds,total_ms=(time.perf_counter_ns()-started)/1e6,
                preparation_ms=(prepared-started)/1e6,graph_stats=dict(policy.graph_runner.stats)))
        finally:policy._lock.release()

    def infer_requests(self,requests):
        result=self.infer_batch([r.observation for r in requests],[r.seed for r in requests])
        return [dict(actions=np.array(a,copy=True),cold=result['cold'],diagnostics=deepcopy(result['diagnostics'])) for a in result['actions']]

    def last_noise(self):
        with self.__policy._lock:
            self.__state._assert_live()
            if self.__last_noise is None:raise RuntimeError('No completed inference')
            return self.__last_noise.detach().cpu().numpy().copy()

    def metadata(self):
        policy=self.__policy;runner=policy.graph_runner
        with policy._lock,runner._call_lock:
            self.__state.audit()
            if _sources()!=self.__sources:raise RuntimeError('Dynamic source changed')
            if policy._prefix_fp8_installation.validate_precision_manifest()!=self.__manifest['quantization']:
                raise RuntimeError('Dynamic precision seal changed')
            if policy._lite_optimizations!=self.__manifest['options']:raise RuntimeError('Dynamic options changed')
            return deepcopy(self.__manifest)

    def close(self):
        policy=self.__policy
        if policy._closed:return
        with policy._lock,policy.graph_runner._call_lock:self.__state.detach(policy,policy.graph_runner)
        policy.close()
        object.__setattr__(self,'_DynamicPi05Engine__last_noise',None)


def create_adapter(config):
    values=dict(config);checkpoint=values.pop('checkpoint');return DynamicPi05Engine.from_pretrained(checkpoint,**values)
