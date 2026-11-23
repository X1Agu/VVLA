"""Private true-B32 extension of the winning prefix51 fixed-checkpoint API.

One native graph runner executes all32 current observations. Original BF16
parameters load privately before B32-qualified prefix FP8, selected fusions,
camera batching, suffix FA2 and new Lt tactics are installed and cold-sealed.
The frozen B1 API and its evidence are unchanged.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .batch_processor import Pi05BatchProcessor, validate_observations

BATCH_SIZE = 32
IMPLEMENTATION = "pi05-native-b32-prefix51-fixed-checkpoint-v1"
PRECISION = "mixed_bf16_fp8_prefix"
ORIGIN_RECIPE = "71dc611b5230da387b05713727fb6368b434ec0818d4856d18e0480f5e581046"
ORIGIN_SOURCE = "7d3886dbbbe4def09aadf4e66f3b8e445298a83aced58cae1487f5df6da4a9c1"
ORIGIN_RESOLVED_CONFIG = '6199d7cf23b3a127ab75b72de10e2293229dd27bd5e336e0fbce3b9a11c028b7'
INPUT_MODES = ('per_row', 'coalesced_fresh_v1')
_CAPABILITY = object()
_EXCLUDED = (
    "lite_prefix_nvfp4",
    "lite_prefix_fp8_gelu", "lite_prefix_fp8_inventory", "lite_vision_fp8", "lite_fp8_host_epochs",
    "lite_request_identity_lookup", "lite_single_pass_lifecycle", "lite_fused_vision_qkv",
)
_SELECTED = dict.fromkeys((
    "lite_fixed_checkpoint_engine", "lite_prefix_fp8", "lite_host_layout_cache", "lite_batched_vision",
    "lite_vision_fc2_cublaslt", "lite_gpu_uint8_prepare", "lite_packed_input_metadata",
    "lite_fused_expert_qkv", "lite_fused_expert_gate_up", "lite_fused_expert_gelu",
    "lite_fused_residual_adarms", "lite_action_norm_cache", "lite_fused_adarms", "lite_fused_adarms_reduction",
    "lite_fused_gated_residual", "lite_skip_weight_init", "lite_native_lifecycle",
    "lite_compile_sample_actions", "lite_compile_preserve_action_linears", "lite_elide_identity_gathers",
), True)
_SELECTED.update(lite_native_lifecycle_variant="v3",lite_suffix_fa2={"splits":8},
                 lite_expert_cublaslt={"profile":"g4_best_three","include_down":False})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources():
    here = Path(__file__).resolve().parent
    paths=[*here.glob("*.py"),*((here.parent/"models/pi05").glob("*.py"))]
    return {str(path.relative_to(here.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _batch_options(overrides):
    values = deepcopy(dict(overrides or {}))
    for key in _EXCLUDED:
        if key in values and values[key] is not False:
            raise ValueError(f"{key} is outside the selected B32 prefix51 recipe")
    if values.get('lite_suffix_attention') is not None:
        raise ValueError('Selected B32 uses the new paged suffix FA2 route')
    for key,expected in _SELECTED.items():
        if key in values and (type(values[key]) is not type(expected) or values[key]!=expected):
            raise ValueError('B32 requires the latest selected option: '+key)
        values[key]=deepcopy(expected)
    # These flags are installed with new B32 owners after the unchanged generic
    # BF16 parameter loader returns, before any graph or model call exists.
    loader=deepcopy(values)
    for key in ('lite_fixed_checkpoint_engine','lite_prefix_fp8','lite_batched_vision','lite_vision_fc2_cublaslt'):
        loader[key]=False
    loader['lite_suffix_fa2']=loader['lite_expert_cublaslt']=None
    return values,loader


def _seeds(values):
    if type(values) is not list or len(values) != BATCH_SIZE:
        raise ValueError("Exactly 32 independent request seeds are required")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           or not 0 <= int(value) < 2**63 for value in values):
        raise ValueError("Each slot seed must be an integer in [0,2**63)")
    return [int(value) for value in values]


def _input_mode(value):
    if type(value) is not str or value not in INPUT_MODES:
        raise ValueError('B32 input_preparation must select a supported immutable preparation method')
    return value


def _processor_type(mode):
    if _input_mode(mode) == 'coalesced_fresh_v1':
        from .batch_coalesced_processor import CoalescedBatchProcessor
        return CoalescedBatchProcessor
    return Pi05BatchProcessor


class Pi05BatchEngine:
    """Private checkpoint factory with exact ordered B32 input/output slots."""
    __slots__ = ("__policy", "__processor", "__sources", "__calls", "__episodes", "__last_noise", "__state", "__manifest", "__input_mode", "__processor_binding", "__compile_owner")

    def __init__(self, *args, **kwargs):
        raise TypeError("Use Pi05BatchEngine.from_pretrained; external models are unsupported")

    def __setattr__(self,name,value):
        raise AttributeError('B32 checkpoint settings are immutable; close and reconstruct')

    def __reduce_ex__(self,protocol):
        raise TypeError('Private B32 checkpoint engines cannot be serialized')

    @classmethod
    def from_pretrained(cls, checkpoint, *, batch_size=BATCH_SIZE, precision=PRECISION,
                        batch_prefix_fp8_evidence=None,batch_cublaslt_evidence=None,batch_suffix_evidence=None,
                        input_preparation='per_row',batch_input_preparation_evidence=None,
                        compile_libdevice_cache=False,batch_compile_libdevice_evidence=None,**kwargs):
        if cls is not Pi05BatchEngine or type(batch_size) is not int or batch_size != BATCH_SIZE:
            raise ValueError("This qualified batch API requires exactly B32")
        if precision != PRECISION:
            raise ValueError("Formal B32 requires the latest prefix51 mixed recipe, not the BF16 scaffolding")
        input_preparation = _input_mode(input_preparation)
        if type(compile_libdevice_cache) is not bool:
            raise ValueError('compile_libdevice_cache must be an explicit immutable boolean')
        if not compile_libdevice_cache and batch_compile_libdevice_evidence is not None:
            raise ValueError('Libdevice evidence requires the explicit compile cache method')
        if compile_libdevice_cache:
            from .batch_compile_libdevice import load_qualification as load_compile_evidence
            load_compile_evidence(batch_compile_libdevice_evidence)
        if input_preparation == 'coalesced_fresh_v1':
            if not isinstance(kwargs.get('tokenizer'),(str,Path)):
                raise ValueError('Coalesced inputs require the actual qualified local tokenizer path')
            from .batch_coalesced_evidence import load_evidence
            load_evidence(batch_input_preparation_evidence)
        elif batch_input_preparation_evidence is not None:
            raise ValueError('Preparation evidence requires the explicit coalesced input method')
        if kwargs.get("cuda_graph", True) is not True:
            raise ValueError("B32 deployment requires its native CUDA graph runner")
        if (kwargs.get('profile','full-bf16')!='full-bf16' or kwargs.get('attention_backend','torch_fa2')!='torch_fa2'
                or kwargs.get('pointwise_fusions') is not None):
            raise ValueError('B32 extends the selected full-bf16/Torch FA2 pointwise recipe')
        from .pi05 import Pi05LitePolicy
        from .batch_prefix_fp8_evidence import validate_config
        from .batch_cublaslt import load_evidence
        from .batch_suffix_evidence import read_support
        validate_config(batch_prefix_fp8_evidence);load_evidence(batch_cublaslt_evidence);read_support(batch_suffix_evidence)
        selected,loader = _batch_options(kwargs.get("config_overrides"))
        kwargs["config_overrides"] = loader
        kwargs=deepcopy(kwargs)
        policy = Pi05LitePolicy.from_pretrained(checkpoint, precision="bf16", **kwargs)
        compile_owner=None
        try:
            if (policy.graph_runner is None or policy.config.chunk_size != 50
                    or policy.config.num_inference_steps != 10 or policy.config.max_action_dim != 32
                    or policy.config.action_dim != 7):
                raise ValueError("B32 recipe requires native graphs, H50, ten steps and the original 32/7 action boundaries")
            if policy._provenance['checkpoint_sha256']!='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae':
                raise ValueError('B32 requires the exact original winning checkpoint')
            if _digest(policy._provenance['resolved_config'])!=ORIGIN_RESOLVED_CONFIG:
                raise ValueError('B32 requires the actual winning resolved model config; batch geometry lives in its new owners')
            _install_batch_recipe(policy,selected,deepcopy(batch_prefix_fp8_evidence),deepcopy(batch_cublaslt_evidence),deepcopy(batch_suffix_evidence))
            if input_preparation == 'coalesced_fresh_v1':
                from .batch_coalesced_evidence import validate_runtime
                proof=deepcopy(batch_input_preparation_evidence)
                qualified=validate_runtime(proof,policy.device,tokenizer=kwargs.get('tokenizer'))
                policy._lite_optimizations['batch_input_preparation']=dict(
                    mode=input_preparation,qualification=proof,qualification_manifest=qualified)
            if compile_libdevice_cache:
                from .batch_compile_libdevice import BatchCompileLibdeviceOwner
                proof=deepcopy(batch_compile_libdevice_evidence)
                compile_owner=BatchCompileLibdeviceOwner(policy,proof)
                policy._lite_optimizations['batch_compile_libdevice_cache']=dict(
                    qualification=proof,manifest=compile_owner.manifest())
            return _seal_policy(policy,input_preparation=input_preparation,compile_owner=compile_owner)
        except BaseException as error:
            try:
                try:policy.close()
                finally:
                    if compile_owner is not None:compile_owner.close()
            except BaseException as cleanup:
                error.add_note("B32 factory cleanup failed: " + str(cleanup))
            raise

    def _policy(self):
        policy = object.__getattribute__(self, "_Pi05BatchEngine__policy")
        if policy._closed:
            raise RuntimeError("B32 engine is closed")
        return policy

    def _processor(self, policy):
        processor = self.__processor
        expected = _processor_type(self.__input_mode)
        output = getattr(policy,'_action_norm_cache',None)
        if output is None:output=policy.processor
        actual = (type(processor),getattr(processor.build_model_inputs,'__func__',None),
            getattr(processor.build_model_outputs,'__func__',None),processor._single,processor._output)
        if (any(current is not sealed for current,sealed in zip(actual,self.__processor_binding,strict=True))
                or type(processor) is not expected or processor._single is not policy.processor
                or type(processor.batch_size) is not int or processor.batch_size != BATCH_SIZE
                or processor._output is not output
                or getattr(processor.build_model_inputs,'__func__',None) is not expected.build_model_inputs
                or getattr(processor.build_model_outputs,'__func__',None) is not Pi05BatchProcessor.build_model_outputs):
            raise RuntimeError('Private B32 processor ownership or preparation method changed')
        return processor

    def reset_batch(self, episode_ids):
        if (type(episode_ids) is not list or len(episode_ids) != BATCH_SIZE
                or any(type(value) is not str for value in episode_ids) or len(set(episode_ids)) != BATCH_SIZE):
            raise ValueError("reset_batch requires 32 distinct ordered episode IDs")
        policy = self._policy()
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError("Cannot reset during an active B32 request")
        try:
            object.__setattr__(self,'_Pi05BatchEngine__episodes',tuple(episode_ids))
        finally:
            policy._lock.release()

    def _noise(self, policy, seeds):
        # Independent B1-compatible CUDA FP32 draws, then a true batch tensor.
        return torch.cat([torch.randn((1, 50, 32), dtype=torch.float32, device=policy.device,
                                      generator=torch.Generator(device=policy.device).manual_seed(seed))
                          for seed in seeds], dim=0)

    @torch.inference_mode()
    def infer_batch(self, observations, seeds):
        started = time.perf_counter_ns()
        validate_observations(observations, BATCH_SIZE)
        seeds = _seeds(seeds)
        policy = self._policy()
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError("One B32 engine requires serialized batch requests")
        try:
            if policy._closed:
                raise RuntimeError("B32 engine is closed")
            object.__setattr__(self,'_Pi05BatchEngine__last_noise',None)
            state=self.__state
            if state is not None:state._assert_live()
            processor=self._processor(policy)
            prepared_inputs=processor.build_model_inputs(observations,return_host_layout=state is not None)
            images,image_masks,tokens,masks=prepared_inputs[:4]
            noise = self._noise(policy, seeds)
            prepared = time.perf_counter_ns()
            arguments=dict(images=images,image_masks=image_masks,lang_tokens=tokens,lang_masks=masks,noise=noise,num_steps=10)
            if state is not None:arguments['host_layout']=prepared_inputs[4]
            if self.__compile_owner is not None:self.__compile_owner.ensure_initialized(policy)
            output = policy.graph_runner(**arguments)
            submitted = time.perf_counter_ns()
            actions = processor.build_model_outputs(output, observations)
            object.__setattr__(self,'_Pi05BatchEngine__last_noise',noise)
            completed = time.perf_counter_ns()
            info = dict(policy.graph_runner.last_call)
            object.__setattr__(self,'_Pi05BatchEngine__calls',self.__calls+1)
            diagnostics=dict(
                **info, batch_size=BATCH_SIZE, model_calls_per_request=1, seeds=seeds,
                batch_request_index=self.__calls - 1, episode_ids=list(self.__episodes),
                total_ms=(completed-started)/1e6, host_prepare_ms=(prepared-started)/1e6,
                host_submit_ms=(submitted-prepared)/1e6, postprocess_and_wait_ms=(completed-submitted)/1e6,
                graph_stats=dict(policy.graph_runner.stats))
            if self.__compile_owner is not None:
                diagnostics['compile_libdevice_cache']=self.__compile_owner.diagnostics()
            return dict(actions=actions,cold=bool(info['cold']),diagnostics=diagnostics)
        finally:
            policy._lock.release()

    def last_noise(self):
        """Untimed owned snapshot of the last successful call's actual draws."""
        policy = self._policy()
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError("Cannot inspect noise during an active B32 request")
        try:
            if self.__last_noise is None:
                raise RuntimeError("No successful B32 inference noise is available")
            return self.__last_noise.detach().cpu().numpy().copy(order="C")
        finally:
            policy._lock.release()

    @torch.inference_mode()
    def prepared_fixture_batch(self, observations, seeds):
        policy = self._policy()
        seeds = _seeds(seeds)
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError("Cannot inspect an active B32 request")
        try:
            images, image_masks, tokens, masks = self._processor(policy).build_model_inputs(observations)
            noise = self._noise(policy, seeds)
            values = dict(tokens=tokens.cpu().numpy().copy(), masks=masks.cpu().numpy().copy(),
                          noise_fp32=noise.cpu().numpy().copy(),
                          noise_bf16_bits=noise.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy())
            for index, image in enumerate(images):
                values[f"image_{index}_bf16_bits"] = image.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy()
                values[f"image_mask_{index}"] = image_masks[index].cpu().numpy().copy()
            return values
        finally:
            policy._lock.release()

    def metadata(self):
        policy = self._policy()
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError("Cannot audit during an active B32 request")
        runner = policy.graph_runner
        if not runner._call_lock.acquire(blocking=False):
            policy._lock.release()
            raise RuntimeError("Cannot audit an active native graph runner")
        try:
            if _sources() != self.__sources:
                raise RuntimeError("B32 execution source changed; reconstruct from the frozen source")
            if self.__state is None:
                raise RuntimeError('CPU mechanism scaffold has no formal latest-recipe metadata')
            self.__state.audit()
            self._processor(policy)
            if self.__compile_owner is not None:
                self.__compile_owner.audit(policy)
                if self.__compile_owner.manifest()!=policy._lite_optimizations['batch_compile_libdevice_cache']['manifest']:
                    raise RuntimeError('Actual compile runtime manifest changed')
            if self.__input_mode == 'coalesced_fresh_v1':
                from .batch_coalesced_evidence import validate_runtime
                preparation=policy._lite_optimizations['batch_input_preparation']
                if (preparation['mode'] != self.__input_mode or
                        validate_runtime(preparation['qualification'],policy.device) != preparation['qualification_manifest']):
                    raise RuntimeError('Actual coalesced preparation qualification changed')
            from .batch_suffix_evidence import validate_runtime
            validate_runtime(policy._lite_optimizations['batch_suffix_evidence'],policy.device)
            quantization=policy._prefix_fp8_installation.validate_precision_manifest()
            options=deepcopy(policy._lite_optimizations)
            actual=_deployment(self.__sources,quantization,options,self.__input_mode)
            if actual!=self.__manifest:raise RuntimeError('B32 cold deployment precision/source/options seal changed')
            quantization=deepcopy(quantization);quantization['fixed_checkpoint_engine']=deepcopy(actual)
            result=deepcopy(policy._provenance)
            deployment = dict(implementation=IMPLEMENTATION, batch_size=BATCH_SIZE,
                model_calls_per_request=1, true_batch=True, sampler_dtype="float32",
                input=("ordered independent raw observations; original checkpoint per-row transforms then axis0 concatenation"
                    if self.__input_mode == 'per_row' else
                    "ordered independent raw observations; fresh camera-major uint8 upload and typed metadata, original per-row state/prompt/tokenizer transforms"),
                input_preparation=self.__input_mode,
                noise="32 independent CUDA FP32[1,50,32] draws by per-slot seed; concatenated unchanged",
                output="batched original checkpoint inverse normalization and per-row absolute-state shift; one owned CPU float32 copy",
                request_cache="no inference reuse of observations, RNG, actions, prefix KV or whole context; last noise retained only for untimed audit",
                validation="private fixed checkpoint; complete cold seal and explicit full audit; hot dynamic input/context and cheap owner/math guards",
                source_sha256=self.__sources)
            components = dict(base_loader_recipe_sha256=result["recipe_sha256"],
                base_loader_scope='actual BF16 parameter loader and selected generic fusions before B32 owners',
                origin_b1_recipe_sha256=ORIGIN_RECIPE,origin_b1_source_inventory_sha256=ORIGIN_SOURCE,
                quantization_manifest_sha256=_digest(quantization),lite_optimizations_sha256=_digest(options),batch_engine=deployment)
            result.update(framework="vvla-lite-batched", batch=BATCH_SIZE,batch_size=BATCH_SIZE,true_batch=True,
                precision=PRECISION,sampler_dtype='float32',origin_b1_recipe_sha256=ORIGIN_RECIPE,
                origin_b1_source_inventory_sha256=ORIGIN_SOURCE,action_shape=[BATCH_SIZE,50,7],
                origin_b1_resolved_config_sha256=ORIGIN_RESOLVED_CONFIG,
                horizon=50,denoise_steps=10,action_space='physical_libero',attention_backend=policy.config.attention_backend,
                cuda_graph=True,runtime='in_process',latency_boundary='raw_observations32_to_owned_cpu_physical_actions32',
                quantization_manifest=quantization,lite_optimizations=options,
                noise=dict(algorithm="torch.randn",generator_device=policy.device.type,dtype="float32",shape=[BATCH_SIZE,50,32],
                           per_slot_independent=True), batch_engine=deployment,
                recipe_components=components, recipe_sha256=_digest(components))
            return result
        finally:
            runner._call_lock.release()
            policy._lock.release()

    def close(self):
        policy=object.__getattribute__(self, "_Pi05BatchEngine__policy")
        owner=self.__compile_owner
        if policy._closed:
            if owner is not None:owner.close()
            return
        state=self.__state
        if state is not None:
            if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot close an active B32 request')
            runner=policy.graph_runner
            if not runner._call_lock.acquire(blocking=False):
                policy._lock.release();raise RuntimeError('Cannot close an active B32 native runner')
            try:state.detach(policy,runner)
            finally:runner._call_lock.release();policy._lock.release()
        try:policy.close()
        finally:
            if owner is not None:owner.close()
        object.__setattr__(self,'_Pi05BatchEngine__last_noise',None)


def _wrap_policy(policy, capability, *, input_preparation='per_row', compile_owner=None):
    if capability is not _CAPABILITY:
        raise TypeError("Only the private checkpoint factory can construct a B32 engine")
    input_preparation=_input_mode(input_preparation)
    engine = object.__new__(Pi05BatchEngine)
    for name,value in dict(policy=policy,processor=_processor_type(input_preparation)(policy.processor,BATCH_SIZE,
            output_processor=getattr(policy,'_action_norm_cache',None)),sources=_sources(),calls=0,
            episodes=(),last_noise=None,state=None,manifest=None,input_mode=input_preparation,
            compile_owner=compile_owner).items():
        object.__setattr__(engine,'_Pi05BatchEngine__'+name,value)
    processor=engine._Pi05BatchEngine__processor
    object.__setattr__(engine,'_Pi05BatchEngine__processor_binding',(type(processor),
        processor.build_model_inputs.__func__,processor.build_model_outputs.__func__,
        processor._single,processor._output))
    return engine


def _install_batch_recipe(policy,selected,prefix_evidence,lt_evidence,suffix_evidence):
    from .batch_layout import BatchHostLayoutCache
    from .batch_vision import install_batch_vision
    from .batch_suffix_fa2 import install_batch_suffix_fa2
    from .batch_cublaslt import install_batch_cublaslt
    from .batch_prefix_fp8 import install_batch_prefix_fp8
    from .batch_suffix_evidence import validate_runtime
    model,runner=policy.model,policy.graph_runner
    if runner.entries or policy._calls:raise RuntimeError('B32 owners require a fresh private loader')
    runner._host_layouts=BatchHostLayoutCache(capacity=runner.max_graphs*2)
    policy._vision_installation=install_batch_vision(model)
    validate_runtime(suffix_evidence,policy.device)
    policy._suffix_fa2_installation=install_batch_suffix_fa2(model,splits=1)
    policy._expert_cublaslt_installation,policy._vision_fc2_cublaslt_installation=install_batch_cublaslt(model,lt_evidence)
    policy._expert_cublaslt_installation.bind_runner(runner)
    policy._vision_fc2_cublaslt_installation.bind_runner(runner)
    # The prefix owner snapshots all nonprefix bindings, so it must be last.
    try:policy._prefix_fp8_installation=install_batch_prefix_fp8(model,prefix_evidence)
    except BaseException:
        policy._prefix_fp8_installation=getattr(model,'_lite_prefix_fp8_installation',None);raise
    policy._prefix_fp8_installation.bind_runner(runner)
    options=policy._lite_optimizations
    for key,value in selected.items():
        if key.startswith('lite_'):options[key[5:]]=deepcopy(value)
    options['expert_cublaslt']={'profile':'g4-b32-measured-three','include_down':False}
    options['suffix_fa2']={'splits':1}
    options['batch_suffix_adaptation']=dict(origin_B1_splits=8,actual_B32_splits=1,
        reason='executed B32 varlen paged Split8 overwrites lane0 and leaves other outputs unwritten; Split1 passed all32/oracle/repeat/capture/isolation',
        backend_report_sha256=suffix_evidence['backend_report_sha256'])
    options['fixed_checkpoint_engine']=True
    options['batch_fixed_checkpoint_deployment']='private-b32-prefix51-immutable-v1'
    for key,owner in (('batched_vision_manifest',policy._vision_installation),
                      ('suffix_fa2_manifest',policy._suffix_fa2_installation),
                      ('expert_cublaslt_manifest',policy._expert_cublaslt_installation),
                      ('vision_fc2_cublaslt_manifest',policy._vision_fc2_cublaslt_installation)):
        options[key]=owner.manifest()
    options['batch_prefix_fp8_evidence']=deepcopy(prefix_evidence)
    options['batch_cublaslt_evidence']=deepcopy(lt_evidence)
    options['batch_suffix_evidence']=deepcopy(suffix_evidence)


def _deployment(sources,quantization,options,input_preparation='per_row'):
    return dict(schema=1,implementation=IMPLEMENTATION,deployment_contract='private-b32-prefix51-immutable-v1',
        batch=32,input_preparation=_input_mode(input_preparation),origin_b1_recipe_sha256=ORIGIN_RECIPE,origin_b1_source_inventory_sha256=ORIGIN_SOURCE,
        origin_b1_resolved_config_sha256=ORIGIN_RESOLVED_CONFIG,
        origin_scope='provenance only; no B32 denominator or quality inheritance',
        source_sha256=deepcopy(sources),base_quantization_manifest_sha256=_digest(quantization),
        native_lifecycle_manifest_sha256=_digest(options['native_lifecycle_manifest']),
        options_sha256=_digest(options),cold_audit_checks_complete=True,live_checks_complete=False,
        cold_audit_checks=['actual full shared model collector','all installed packed-owner prepare and complete precision manifest',
                           'current original/packed weight epochs and module/forward topology','source and backend evidence bytes'],
        hot_request_checks=['both serialized owner locks','closed/attached unique model/runner ownership','cold generation/signature',
                            'process math/autocast guards','current raw32 preprocessing and per-slot FP32 RNG',
                            'current masks/packing/positions/metadata and graph key','fresh graph execution and owned CPU actions'],
        unsupported_mutations=['external model or processor injection','private reference access/monkeypatching',
                               'weight/precision/options/training/installation changes; close and reconstruct'],
        public_api=['from_pretrained','infer_batch','reset_batch','last_noise','prepared_fixture_batch','metadata','close'])


def _seal_policy(policy, *, input_preparation='per_row', compile_owner=None):
    from .batch_checkpoint_engine import BatchSealedModelState,_CAPABILITY as state_capability
    engine=_wrap_policy(policy,_CAPABILITY,input_preparation=input_preparation,compile_owner=compile_owner)
    if not policy._lock.acquire(blocking=False):raise RuntimeError('Cannot seal an active B32 policy')
    runner=policy.graph_runner
    if not runner._call_lock.acquire(blocking=False):
        policy._lock.release();raise RuntimeError('Cannot seal an active B32 runner')
    state=None
    try:
        state=BatchSealedModelState(policy,state_capability)
        state.audit()
        if compile_owner is not None:compile_owner.audit(policy)
        manifest=policy._prefix_fp8_installation.validate_precision_manifest()
        deployment=_deployment(engine._Pi05BatchEngine__sources,manifest,policy._lite_optimizations,input_preparation)
        object.__setattr__(engine,'_Pi05BatchEngine__state',state)
        object.__setattr__(engine,'_Pi05BatchEngine__manifest',deepcopy(deployment))
        return engine
    except BaseException:
        if state is not None:state.detach(policy,runner)
        raise
    finally:runner._call_lock.release();policy._lock.release()


def create_adapter(config):
    values = dict(config)
    checkpoint = values.pop("checkpoint")
    values.pop('factory',None);values.pop('name',None)
    return Pi05BatchEngine.from_pretrained(checkpoint, **values)
