"""Private fixed-batch L4 BF16 deployment descended from L4 formal-v4.

The old B1 identity is provenance, never a batch denominator or qualification.
The initial adaptation deliberately excludes B1 Lt plans and paged Split8.
One native BF16 graph call consumes all current rows and their independent
FP32 draws. Full model state is sealed cold and audited explicitly.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .batch_layout import BatchHostLayoutCache,validate_fixed_batch_size
from .batch_processor import Pi05BatchProcessor,validate_observations

BATCH_SIZE=32
PRECISION='bf16'
IMPLEMENTATION='pi05-l4-native-fixed-batch-bf16-v1'
ORIGIN_IDENTITY_SHA256='aedddeb6f3683b00048740356270de164e00285c0816e2923798f3d1b866eb71'
ORIGIN_RECIPE='c4aa50f2a0e914456c4edb916afb547fc320d880685fde4700465ec74e185ca9'
ORIGIN_RESOLVED_CONFIG='6199d7cf23b3a127ab75b72de10e2293229dd27bd5e336e0fbce3b9a11c028b7'
CHECKPOINT_SHA256='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
_CAPABILITY=object()
_SELECTED=dict.fromkeys((
    'lite_host_layout_cache','lite_batched_vision','lite_gpu_uint8_prepare',
    'lite_fused_expert_qkv','lite_fused_expert_gelu','lite_action_norm_cache',
    'lite_fused_adarms','lite_fused_adarms_reduction','lite_fused_gated_residual',
    'lite_skip_weight_init','lite_native_lifecycle','lite_compile_sample_actions',
    'lite_compile_preserve_action_linears','lite_elide_identity_gathers'),True)
_SELECTED.update(dict.fromkeys((
    'lite_fixed_checkpoint_engine','lite_prefix_fp8','lite_prefix_nvfp4',
    'lite_prefix_fp8_gelu','lite_prefix_fp8_inventory','lite_vision_fp8',
    'lite_fp8_host_epochs','lite_request_identity_lookup','lite_single_pass_lifecycle',
    'lite_fused_vision_qkv','lite_vision_fc2_cublaslt','lite_fused_expert_gate_up',
    'lite_fused_residual_adarms','lite_packed_input_metadata'),False))
_SELECTED.update(lite_native_lifecycle_variant='v2',lite_expert_cublaslt=None,
                 lite_suffix_fa2=None,lite_suffix_attention=None)

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _sources():
    here=Path(__file__).resolve().parent
    paths=[*here.glob('*.py'),*(here.parent/'models/pi05').glob('*.py')]
    return {str(path.relative_to(here.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}

def _origin(path=None):
    if path is None:
        path=Path(__file__).resolve().parents[4]/'benchmarks/pi05-b32-20261110/l4-preparation-20261110/L4_FORMAL_V4_IDENTITY.json'
    path=Path(path)
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=ORIGIN_IDENTITY_SHA256:
        raise ValueError('L4 batch requires the exact original L4 BF16 formal-v4 identity bytes')
    metadata=json.loads(path.read_bytes())['metadata']
    if (metadata['recipe_sha256']!=ORIGIN_RECIPE or metadata['precision']!='bf16'
            or metadata['checkpoint_sha256']!=CHECKPOINT_SHA256
            or _digest(metadata['resolved_config'])!=ORIGIN_RESOLVED_CONFIG):
        raise ValueError('Original L4 identity does not bind its audited BF16 recipe/config/checkpoint')
    return dict(identity_path=str(path.resolve()),identity_sha256=ORIGIN_IDENTITY_SHA256,
        recipe_sha256=ORIGIN_RECIPE,resolved_config_sha256=ORIGIN_RESOLVED_CONFIG,
        scope='B1 L4 mathematical lineage only; no inherited batch source inventory, quality, timing or qualification')

def _batch_options(overrides):
    values=deepcopy(dict(overrides or {}))
    if set(values)-set(_SELECTED):
        raise ValueError('L4 batch accepts only its fixed BF16 adaptation options')
    for key,expected in _SELECTED.items():
        if key in values and (type(values[key]) is not type(expected) or values[key]!=expected):
            raise ValueError('L4 batch requires its fixed BF16 option: '+key)
        values[key]=deepcopy(expected)
    loader=deepcopy(values)
    loader['lite_batched_vision']=False  # installed with the explicit batch owner before sealing
    return values,loader

def _seeds(values,batch_size):
    if type(values) is not list or len(values)!=batch_size:
        raise ValueError(f'Exactly {batch_size} independent request seeds are required')
    if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
           or not 0<=int(value)<2**63 for value in values):
        raise ValueError('Every slot seed must be an integer in [0,2**63)')
    return [int(value) for value in values]

def _hardware(device):
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('The L4 deployment requires its actual CUDA device')
    props=torch.cuda.get_device_properties(device)
    if (props.name!='NVIDIA L4' or (props.major,props.minor)!=(8,9)
            or str(torch.__version__).split('+')[0]!='2.13.0' or torch.version.cuda!='13.0'
            or torch.version.git_version!='cf30153c4c131c8164ee7798e5022d810682e2cb'):
        raise ValueError('L4 BF16 batch requires the restored SM89/Torch2.13 CUDA13 runtime')
    return dict(name=props.name,capability=[props.major,props.minor],uuid=str(props.uuid),
                torch=str(torch.__version__),torch_git=torch.version.git_version,cuda=torch.version.cuda)

def _bf16_manifest(policy):
    if any(getattr(policy,name,None) is not None for name in (
            '_prefix_fp8_installation','_prefix_nvfp4_installation',
            '_prefix_fp8_inventory_installation','_prefix_fp8_gelu_installation','_vision_fp8_installation')):
        raise RuntimeError('L4 BF16 batch has an unexpected quantization owner')
    recipe=policy.config.execution_recipe or {}
    if any(recipe.get(key,'none')!='none' for key in ('prefix_mlp','vision_mlp','expert_mlp')):
        raise RuntimeError('L4 BF16 batch precision recipe changed')
    inventory={}
    for name,tensor in policy.model.named_parameters():
        if tensor.dtype not in (torch.bfloat16,torch.float32):
            raise RuntimeError('Unexpected L4 BF16 parameter dtype: '+name)
        row=inventory.setdefault(str(tensor.dtype),dict(tensors=0,elements=0))
        row['tensors']+=1;row['elements']+=tensor.numel()
    return dict(precision='bf16',sampler_dtype='float32',quantized_owners=[],
        parameter_dtype_inventory=inventory,execution_recipe=deepcopy(recipe),
        scope='Actual current registered parameters and absence of packed FP8/NVFP4 owners; checkpoint load and cold full collector bind their provenance')

class Pi05L4BatchEngine:
    __slots__=('__policy','__processor','__batch','__sources','__calls','__episodes',
               '__last_noise','__state','__manifest','__origin','__hardware','__compile_owner')

    def __init__(self,*args,**kwargs):
        raise TypeError('Use Pi05L4BatchEngine.from_pretrained; external models are unsupported')
    def __setattr__(self,name,value):raise AttributeError('L4 fixed batch settings are immutable; reconstruct')
    def __reduce_ex__(self,protocol):raise TypeError('Private L4 batch engines cannot be serialized')

    @property
    def batch_size(self):return self.__batch

    @classmethod
    def from_pretrained(cls,checkpoint,*,batch_size=32,precision='bf16',origin_identity=None,
                        compile_libdevice_cache=False,batch_l4_compile_libdevice_evidence=None,**kwargs):
        if cls is not Pi05L4BatchEngine:raise TypeError('Use the exact private L4 checkpoint factory')
        batch_size=validate_fixed_batch_size(batch_size)
        if precision!='bf16':raise ValueError('L4 initial batch deployment is explicitly BF16')
        if type(compile_libdevice_cache) is not bool:
            raise ValueError('L4 compile_libdevice_cache must be an explicit immutable boolean')
        if not compile_libdevice_cache and batch_l4_compile_libdevice_evidence is not None:
            raise ValueError('Disabled L4 compiler initialization cannot accept evidence')
        if compile_libdevice_cache:
            if batch_size!=16:raise ValueError('L4 compiler initialization is only qualified for B16')
            from .batch_l4_compile_libdevice import load_qualification
            load_qualification(batch_l4_compile_libdevice_evidence)
        allowed={'device','tokenizer','revision','config_overrides','attention_backend',
                 'cuda_graph','max_graphs','profile','pointwise_fusions'}
        if set(kwargs)-allowed:
            raise ValueError('Unknown or foreign evidence/model input to the private L4 checkpoint factory')
        if (kwargs.get('cuda_graph',True) is not True or kwargs.get('profile','full-bf16')!='full-bf16'
                or kwargs.get('attention_backend','torch_fa2')!='torch_fa2' or kwargs.get('pointwise_fusions') is not None):
            raise ValueError('L4 batch requires native graphs/full-bf16/Torch FA2 and its fixed pointwise recipe')
        selected,loader=_batch_options(kwargs.get('config_overrides'))
        origin=_origin(origin_identity);hardware=_hardware(kwargs.get('device','cuda'))
        kwargs=deepcopy(kwargs);kwargs['config_overrides']=loader
        from .pi05 import Pi05LitePolicy
        policy=Pi05LitePolicy.from_pretrained(checkpoint,precision='bf16',**kwargs)
        compile_owner=None
        try:
            if (policy.graph_runner is None or policy.config.chunk_size!=50 or policy.config.num_inference_steps!=10
                    or policy.config.max_action_dim!=32 or policy.config.action_dim!=7
                    or policy._provenance['checkpoint_sha256']!=CHECKPOINT_SHA256
                    or _digest(policy._provenance['resolved_config'])!=ORIGIN_RESOLVED_CONFIG):
                raise ValueError('L4 batch requires the original checkpoint, resolved config, H50/ten steps/32-to7 actions')
            _install_batch_recipe(policy,selected,batch_size)
            if compile_libdevice_cache:
                from .batch_l4_compile_libdevice import L4BatchCompileLibdeviceOwner
                proof=deepcopy(batch_l4_compile_libdevice_evidence)
                compile_owner=L4BatchCompileLibdeviceOwner(policy,proof,batch_size=batch_size)
                policy._lite_optimizations['batch_l4_compile_libdevice_cache']=dict(
                    qualification=proof,manifest=compile_owner.manifest())
                return _seal_policy(policy,batch_size,origin,hardware,compile_owner=compile_owner)
            return _seal_policy(policy,batch_size,origin,hardware)
        except BaseException as error:
            try:policy.close()
            except BaseException as cleanup:
                # Keep the actual still-unretired owners reachable for retry.
                # Compiler owner.close() is deliberately not in a finally.
                error.l4_batch_policy_owner=policy
                error.l4_batch_compile_owner=compile_owner
                error.add_note('L4 batch factory cleanup failed: '+str(cleanup))
            else:
                if compile_owner is not None:compile_owner.close()
            raise

    def _policy(self):
        policy=object.__getattribute__(self,'_Pi05L4BatchEngine__policy')
        if policy._closed:raise RuntimeError('L4 batch engine is closed')
        return policy

    def reset_batch(self,episode_ids):
        if (type(episode_ids) is not list or len(episode_ids)!=self.__batch
                or any(type(value) is not str for value in episode_ids) or len(set(episode_ids))!=self.__batch):
            raise ValueError('reset_batch requires distinct ordered episode IDs for the fixed batch')
        policy=self._policy()
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot reset an active L4 batch')
        try:object.__setattr__(self,'_Pi05L4BatchEngine__episodes',tuple(episode_ids))
        finally:policy._lock.release()

    def _noise(self,policy,seeds):
        return torch.cat([torch.randn((1,50,32),dtype=torch.float32,device=policy.device,
            generator=torch.Generator(device=policy.device).manual_seed(seed)) for seed in seeds],dim=0)

    @torch.inference_mode()
    def infer_batch(self,observations,seeds):
        started=time.perf_counter_ns();validate_observations(observations,self.__batch)
        seeds=_seeds(seeds,self.__batch);policy=self._policy()
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Serialize requests on one L4 batch engine')
        try:
            if policy._closed:raise RuntimeError('L4 batch engine is closed')
            object.__setattr__(self,'_Pi05L4BatchEngine__last_noise',None)
            if self.__state is not None:self.__state._assert_live()
            values=self.__processor.build_model_inputs(observations,return_host_layout=self.__state is not None)
            images,cameras,tokens,masks=values[:4];noise=self._noise(policy,seeds)
            prepared=time.perf_counter_ns()
            args=dict(images=images,image_masks=cameras,lang_tokens=tokens,lang_masks=masks,noise=noise,num_steps=10)
            if self.__state is not None:args['host_layout']=values[4]
            if self.__compile_owner is not None:self.__compile_owner.ensure_initialized(policy)
            output=policy.graph_runner(**args);submitted=time.perf_counter_ns()
            actions=self.__processor.build_model_outputs(output,observations)
            object.__setattr__(self,'_Pi05L4BatchEngine__last_noise',noise)
            object.__setattr__(self,'_Pi05L4BatchEngine__calls',self.__calls+1)
            completed=time.perf_counter_ns();info=dict(policy.graph_runner.last_call)
            if self.__compile_owner is not None:info['compile_libdevice_cache']=self.__compile_owner.diagnostics()
            return dict(actions=actions,cold=bool(info['cold']),diagnostics=dict(**info,
                batch_size=self.__batch,model_calls_per_request=1,seeds=seeds,batch_request_index=self.__calls-1,
                episode_ids=list(self.__episodes),total_ms=(completed-started)/1e6,
                host_prepare_ms=(prepared-started)/1e6,host_submit_ms=(submitted-prepared)/1e6,
                postprocess_and_wait_ms=(completed-submitted)/1e6,graph_stats=dict(policy.graph_runner.stats)))
        finally:policy._lock.release()

    def last_noise(self):
        policy=self._policy()
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot inspect noise during an active L4 request')
        try:
            if self.__last_noise is None:raise RuntimeError('No successful L4 batch noise is available')
            return self.__last_noise.detach().cpu().numpy().copy(order='C')
        finally:policy._lock.release()

    @torch.inference_mode()
    def prepared_fixture_batch(self,observations,seeds):
        policy=self._policy();seeds=_seeds(seeds,self.__batch)
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot inspect an active L4 request')
        try:
            if self.__state is not None:self.__state._assert_live()
            images,cameras,tokens,masks=self.__processor.build_model_inputs(observations)
            noise=self._noise(policy,seeds)
            result=dict(tokens=tokens.cpu().numpy().copy(),masks=masks.cpu().numpy().copy(),
                noise_fp32=noise.cpu().numpy().copy(),noise_bf16_bits=noise.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy())
            for index,image in enumerate(images):
                result[f'image_{index}_bf16_bits']=image.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy()
                result[f'image_mask_{index}']=cameras[index].cpu().numpy().copy()
            return result
        finally:policy._lock.release()

    def metadata(self):
        policy=self._policy()
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot audit an active L4 request')
        runner=policy.graph_runner
        if not runner._call_lock.acquire(blocking=False):
            policy._lock.release();raise RuntimeError('Cannot audit an active L4 native runner')
        try:
            if self.__state is None:raise RuntimeError('CPU mechanism scaffold has no formal L4 deployment metadata')
            if _sources()!=self.__sources or _origin(self.__origin['identity_path'])!=self.__origin:
                raise RuntimeError('L4 execution source or origin identity changed')
            if _hardware(policy.device)!=self.__hardware:raise RuntimeError('L4 device/runtime changed')
            self.__state.audit()
            if self.__compile_owner is not None:
                self.__compile_owner.audit(policy)
                if self.__compile_owner.manifest()!=policy._lite_optimizations['batch_l4_compile_libdevice_cache']['manifest']:
                    raise RuntimeError('Actual L4 compile owner manifest changed')
            precision=_bf16_manifest(policy);options=deepcopy(policy._lite_optimizations)
            actual=_deployment(self.__batch,self.__sources,self.__origin,self.__hardware,precision,options)
            if actual!=self.__manifest:raise RuntimeError('L4 cold deployment precision/options/source seal changed')
            result=policy.metadata()
            components=dict(base_loader_recipe_sha256=result['recipe_sha256'],origin_b1_identity_sha256=ORIGIN_IDENTITY_SHA256,
                origin_b1_recipe_sha256=ORIGIN_RECIPE,quantization_manifest_sha256=_digest(precision),
                lite_optimizations_sha256=_digest(options),batch_engine=actual)
            result.update(framework='vvla-lite-batched',precision='bf16',batch=self.__batch,batch_size=self.__batch,
                true_batch=True,sampler_dtype='float32',action_shape=[self.__batch,50,7],
                origin_b1_identity_sha256=ORIGIN_IDENTITY_SHA256,origin_b1_recipe_sha256=ORIGIN_RECIPE,
                origin_b1_resolved_config_sha256=ORIGIN_RESOLVED_CONFIG,
                quantization_manifest=precision,lite_optimizations=options,batch_engine=actual,
                noise=dict(algorithm='torch.randn',generator_device='cuda',dtype='float32',shape=[self.__batch,50,32],per_slot_independent=True),
                latency_boundary=f'raw_observations{self.__batch}_to_owned_cpu_physical_actions{self.__batch}',
                recipe_components=components,recipe_sha256=_digest(components))
            return result
        finally:runner._call_lock.release();policy._lock.release()

    def close(self):
        policy=object.__getattribute__(self,'_Pi05L4BatchEngine__policy')
        if policy._closed:
            if self.__compile_owner is not None:self.__compile_owner.close()
            return
        if self.__state is not None:
            if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot close an active L4 request')
            runner=policy.graph_runner
            if not runner._call_lock.acquire(blocking=False):
                policy._lock.release();raise RuntimeError('Cannot close an active L4 native runner')
            try:self.__state.detach(policy,runner)
            finally:runner._call_lock.release();policy._lock.release()
        policy.close()
        # A policy/state retirement error above retains this owner and its
        # anchors. Only normal retirement permits discarding those bindings.
        if self.__compile_owner is not None:self.__compile_owner.close()
        object.__setattr__(self,'_Pi05L4BatchEngine__last_noise',None)

def _wrap_policy(policy,capability,*,batch_size,origin=None,hardware=None,compile_owner=None):
    if capability is not _CAPABILITY:raise TypeError('Only the private L4 checkpoint factory can construct this engine')
    batch_size=validate_fixed_batch_size(batch_size);engine=object.__new__(Pi05L4BatchEngine)
    for name,value in dict(policy=policy,processor=Pi05BatchProcessor(policy.processor,batch_size,
            output_processor=getattr(policy,'_action_norm_cache',None)),batch=batch_size,sources=_sources(),
            calls=0,episodes=(),last_noise=None,state=None,manifest=None,origin=deepcopy(origin),hardware=deepcopy(hardware),
            compile_owner=compile_owner).items():
        object.__setattr__(engine,'_Pi05L4BatchEngine__'+name,value)
    return engine

def _install_batch_recipe(policy,selected,batch_size):
    from .batch_vision import install_batch_vision
    runner=policy.graph_runner
    if runner.entries or policy._calls:raise RuntimeError('L4 batch owners require a fresh private loader')
    if policy._expert_cublaslt_installation is not None or policy._suffix_fa2_installation is not None:
        raise RuntimeError('B1 Lt/SplitKV tactics must not enter the L4 batch deployment')
    runner._host_layouts=BatchHostLayoutCache(capacity=runner.max_graphs*2,batch_size=batch_size)
    policy._vision_installation=install_batch_vision(policy.model,batch_size=batch_size)
    for key,value in selected.items():policy._lite_optimizations[key[5:]]=deepcopy(value)
    policy._lite_optimizations.update(batched_vision_manifest=policy._vision_installation.manifest(),
        batch_fixed_checkpoint_deployment='private-l4-bf16-fixed-batch-v1',
        batch_tactic_adaptation=dict(origin_B1_expert_cublaslt=dict(profile='l4_exact_three',include_down=True),
            actual_expert_cublaslt=None,origin_B1_suffix_splits=8,actual_suffix='original native BF16 Torch FA2',
            reason='B1 and G4 tactics are not qualified for this L4 batch geometry; future Split1/Lt require separate actual-card evidence'))

def _deployment(batch,sources,origin,hardware,precision,options):
    return dict(schema=1,implementation=IMPLEMENTATION,deployment_contract='private-l4-bf16-fixed-batch-v1',
        batch=batch,model_calls_per_request=1,true_batch=True,precision='bf16',sampler_dtype='float32',
        origin=deepcopy(origin),hardware=deepcopy(hardware),source_sha256=deepcopy(sources),
        precision_manifest_sha256=_digest(precision),options_sha256=_digest(options),
        cold_audit_checks_complete=True,live_checks_complete=False,
        cold_audit_checks=['actual full shared model collector and installed owner prepare','original weight/module epochs and complete dtype inventory','actual origin identity and current source bytes'],
        hot_request_checks=['serialized policy/native-runner locks','unique attached cold generation/signature','process math/autocast guards','current raw inputs, masks/packing/positions and graph key','independent CUDA FP32 seed draws','one current batch graph execution and owned CPU physical actions'],
        unsupported_mutations=['external model/processor injection','private reference access/monkeypatching','weight/precision/installation/options/training changes; close and reconstruct'],
        public_api=['from_pretrained','infer_batch','reset_batch','last_noise','prepared_fixture_batch','metadata','close'])

def _seal_policy(policy,batch_size,origin,hardware,*,compile_owner=None):
    from .batch_checkpoint_engine import BatchSealedModelState,_CAPABILITY as state_capability
    engine=_wrap_policy(policy,_CAPABILITY,batch_size=batch_size,origin=origin,hardware=hardware,compile_owner=compile_owner)
    if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot seal an active L4 policy')
    runner=policy.graph_runner
    if not runner._call_lock.acquire(blocking=False):
        policy._lock.release();raise RuntimeError('Cannot seal an active L4 native runner')
    state=None
    try:
        state=BatchSealedModelState(policy,state_capability);state.audit()
        deployment=_deployment(batch_size,engine._Pi05L4BatchEngine__sources,origin,hardware,_bf16_manifest(policy),policy._lite_optimizations)
        object.__setattr__(engine,'_Pi05L4BatchEngine__state',state)
        object.__setattr__(engine,'_Pi05L4BatchEngine__manifest',deepcopy(deployment))
        return engine
    except BaseException:
        if state is not None:state.detach(policy,runner)
        raise
    finally:runner._call_lock.release();policy._lock.release()

def create_adapter(config):
    values=deepcopy(dict(config));checkpoint=values.pop('checkpoint')
    values.pop('factory',None);values.pop('name',None)
    return Pi05L4BatchEngine.from_pretrained(checkpoint,**values)
