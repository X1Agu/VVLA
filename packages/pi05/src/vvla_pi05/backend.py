"""Public Pi0.5 facade. Shared mathematics remains in Pi05LitePolicy."""
from copy import deepcopy
import numpy as np

from vvla.adapters import image, state, validate_batch
from vvla.assets import resolve_assets
from vvla.dispatch import prepare_config
from vvla.errors import InputError, UnsupportedConfiguration


class Pi05Engine:
    def __init__(self, policy, *, sealed=True, max_batch=1):
        if type(max_batch) is not int or not 1<=max_batch<=32:
            raise InputError('max_batch must be an integer in [1,32]')
        if (max_batch>1 and policy._lite_optimizations.get('host_layout_cache')
                and not policy._lite_optimizations.get('batch_geometry')):
            raise UnsupportedConfiguration('Portable BF16 batching uses per-batch metadata; B1 host layout cache cannot be reused')
        self.__policy = policy
        self.__max_batch = max_batch
        self.__batches=tuple(b for b in (1,2,4,8,16,32) if b<=max_batch)
        self.__batched_geometry=policy._lite_optimizations.get('batch_geometry')
        self.__processors = {}
        self.__state = None
        self._last_width = None
        if sealed and policy.graph_runner is not None:
            from .lite.batch_checkpoint_engine import BatchSealedModelState, _CAPABILITY
            with policy._lock, policy.graph_runner._call_lock:
                self.__state = BatchSealedModelState(policy, _CAPABILITY)
                self.__state.audit()

    def infer_batch(self, observations, seeds):
        if self.__policy._closed:raise RuntimeError('Pi0.5 engine is closed')
        validate_batch(observations,seeds,sizes=self.__batches)
        config=self.__policy.config
        for obs in observations:
            state(obs.get('state'),config.state_dim)
            if type(obs.get('prompt','')) is not str:
                raise InputError('Pi0.5 prompt must be a string')
            images=obs.get('images')
            if type(images) is not dict or not images:
                raise InputError('Pi0.5 SDK observations require an images dictionary with checkpoint camera names')
            names=set(config.image_feature_keys or ())
            if names and set(images)-names:raise InputError('Unknown checkpoint camera name')
            if not any(v is not None for v in images.values()):raise InputError('At least one camera must be present')
            if config.input_contract=='openpi-libero-pi05':
                if any(images.get(k) is None for k in ('base_0_rgb','left_wrist_0_rgb')) or images.get('right_wrist_0_rgb') is not None:
                    raise InputError('OpenPI LIBERO requires base/left wrist and a masked right wrist camera')
            for name,value in images.items():
                if value is not None:image(value,name=name,allow_float=True)
        if self.__state is not None:self.__state._assert_live()
        if len(observations)==1 and not self.__batched_geometry:
            self._last_width = 1
            return [np.array(self.__policy.infer(observations[0],seeds[0])['actions'],copy=True)]
        import torch
        from .lite.batch_processor import Pi05BatchProcessor
        batch=len(observations);policy=self.__policy
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Serialize access to the model owner')
        try:
            if batch not in self.__processors:
                processor_type=Pi05BatchProcessor
                if (self.__batched_geometry and self.__batched_geometry!='sm89'
                        and policy.config.input_contract!='openpi-libero-pi05'):
                    from .lite.dynamic_batch_prepare import DynamicCoalescedProcessor
                    processor_type=DynamicCoalescedProcessor
                self.__processors[batch]=processor_type(policy.processor,batch,output_processor=policy._action_norm_cache)
            processor=self.__processors[batch]
            with torch.inference_mode():
                host_layout=None
                if self.__batched_geometry:
                    policy._vision_installation.select_batch(batch)
                    images,image_masks,tokens,masks,host_layout=processor.build_model_inputs(
                        observations,return_host_layout=True)
                else:
                    images,image_masks,tokens,masks=processor.build_model_inputs(observations)
                self._last_width = tokens.shape[0]
                noise=torch.cat([torch.randn((1,config.chunk_size,config.max_action_dim),device=policy.device,
                    dtype=torch.float32,generator=torch.Generator(device=policy.device).manual_seed(seed)) for seed in seeds])
                runner=policy.graph_runner or policy.model.sample_actions
                extra={} if host_layout is None else dict(host_layout=host_layout)
                value=runner(images=images,image_masks=image_masks,lang_tokens=tokens,lang_masks=masks,
                             noise=noise,num_steps=config.num_inference_steps,**extra)
                values=processor.build_model_outputs(value,observations)
                policy._calls+=1
                return [np.array(row,copy=True) for row in values]
        finally:policy._lock.release()

    def capabilities(self):
        config=self.__policy.config
        return dict(model='pi05',batch_sizes=list(self.__batches),seed_bits=63,
                    batch_independent_text_fields=['prompt'],
                    batch_execution='one true model call; '+(self.__batched_geometry or 'native')+' geometry',dummy_lanes=False,
                    precision=('mixed_bf16_fp8_prefix' if getattr(self.__policy,'_prefix_fp8_installation',None) is not None
                               else 'dynamic-fp8-experimental' if self.__policy._provenance.get('experimental_precision') else 'bf16'),
                    observation_adapter=config.input_contract,action_shape=[config.chunk_size,config.action_dim],
                    action_space=('libero_delta_no_additional_state_shift' if config.input_contract=='openpi-libero-pi05'
                                  else 'checkpoint_inverse_normalization_then_absolute'),
                    full_weight_release_qualified=False)

    def execution_receipt(self):
        runner = self.__policy.graph_runner
        call = runner.last_call if runner else {}
        return dict(actual_tensor_batch=call.get("actual_tensor_batch", self._last_width), graph_build=call.get("graph") == "capture",
                    graph_rebuild=bool(call.get("rebuild")), graph_replay=call.get("graph") == "replay")

    def metadata(self):
        if self.__state is not None:
            with self.__policy._lock,self.__policy.graph_runner._call_lock:self.__state.audit()
        return dict(self.__policy.metadata(),public_backend='vvla-pi05',capabilities=self.capabilities())

    def reset(self,episode_id=None):return self.__policy.reset(episode_id)

    def close(self):
        if self.__policy._closed:return
        if self.__state is not None:
            with self.__policy._lock,self.__policy.graph_runner._call_lock:
                self.__state.detach(self.__policy,self.__policy.graph_runner)
        self.__policy.close()
        self.__processors.clear()


def load_backend(config):
    from .recipes import RECIPE_NAMES, resolve_recipe, merge_options
    if config['recipe'] not in (*RECIPE_NAMES, 'dynamic-fp8'):
        raise UnsupportedConfiguration('Unknown Pi0.5 execution recipe')
    assets=resolve_assets(config['checkpoint'],config.get('assets',{}),offline=config.get('offline',False),
                          optional_missing=('lt_tactics',) if config.get('auto_tactics_fallback') else ())
    options=deepcopy(config.get('options',{}))
    allowed={'cuda_graph','max_graphs','attention_backend','config_overrides','sealed',
             'precision_stages','precision_fallback','max_batch','base_recipe','allow_recipe_overrides'}
    if options.keys()-allowed:raise InputError(f'Unknown Pi0.5 options: {sorted(options.keys()-allowed)}')
    sealed=options.pop('sealed',True)
    if type(sealed) is not bool:raise InputError('sealed must be boolean')
    max_batch=options.pop('max_batch',1)
    if type(max_batch) is not int or not 1<=max_batch<=32:
        raise InputError('max_batch must be an integer in [1,32]')
    overrides=options.get('config_overrides',{})
    if any('evidence' in key or ('fp8' in key and key!='lite_prefix_fp8') or 'nvfp4' in key or 'fixed_checkpoint_engine' in key for key in overrides):
        raise UnsupportedConfiguration('Historical qualification options are outside the public BF16 recipe')
    stages=options.pop('precision_stages',None)
    fallback=options.pop('precision_fallback','error')
    base_recipe=options.pop('base_recipe',None)
    if config['recipe']=='dynamic-fp8':
        if not stages:raise InputError('dynamic-fp8 requires explicit precision_stages for ablation')
        options['experimental_precision']=dict(stages=stages,unsupported=fallback)
    elif stages is not None:
        raise InputError('precision_stages requires recipe=dynamic-fp8')
    elif base_recipe is not None:
        raise InputError('base_recipe is only used by a precision ablation')
    import torch
    from .lite.pi05 import Pi05LitePolicy, load_lite_config
    model_config,_=load_lite_config(config['checkpoint'])
    selected=resolve_recipe(base_recipe or ('bf16-optimized' if config['recipe']=='dynamic-fp8' else config['recipe']),
                            capability=torch.cuda.get_device_capability(config['device']),
                            horizon=overrides.get('chunk_size',model_config.chunk_size),max_batch=max_batch,
                            input_contract=overrides.get('input_contract',model_config.input_contract),
                            lt_tactics_available=bool(assets.get('lt_tactics')))
    # Match the historical reference arithmetic before checking tactics,
    # packing weights or graph capture.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.backends.cuda.preferred_blas_library('cublas')
    if 'lt_tactics' in selected['required_assets'] and not assets.get('lt_tactics') and config.get('auto_tactics_fallback'):
        selected = resolve_recipe('bf16-portable', capability=selected['capability'], horizon=selected['horizon'],
                                  max_batch=max_batch, input_contract=selected['input_contract'])
        selected['fallback'] = dict(reason='missing_lt_tactics')
    if 'lt_tactics' in selected['required_assets']:
        if not assets.get('lt_tactics'):
            raise InputError('The preserved Blackwell batch recipe requires assets.lt_tactics: '
                             'the historical per-batch selected_spec table; no slow fallback is selected')
        from .lite.dynamic_cublaslt import load_tactics,preflight_tactics,TacticUnavailable
        receipt = {}
        try:
            receipt = load_tactics(assets['lt_tactics'])
            preflight = preflight_tactics(receipt, config['device'])
        except (TacticUnavailable, OSError, ValueError) as error:
            if not isinstance(error, TacticUnavailable) and not config.get('auto_tactics_fallback'):
                raise
            if not config.get('auto_tactics_fallback') and selected['requested'] not in ('bf16','bf16-optimized'):
                raise UnsupportedConfiguration(str(error)) from error
            selected=resolve_recipe('bf16-portable' if config.get('auto_tactics_fallback') else selected['requested'],capability=selected['capability'],
                horizon=selected['horizon'],max_batch=max_batch,input_contract=selected['input_contract'])
            selected['fallback']=dict(reason='incompatible_lt_tactics',detail=str(error),
                                        rejected_asset_sha256=receipt.get('_report_sha256'))
        else:
            selected['lt_tactics_sha256']=receipt['_report_sha256']
            selected['tactics_preflight']=preflight
            selected['loader']['config_overrides']['lite_dynamic_cublaslt']=assets['lt_tactics']
    allow_recipe_overrides=options.get('allow_recipe_overrides',False)
    options=merge_options(selected,options)
    if not sealed and selected['historical_configuration'] and not allow_recipe_overrides:
        raise InputError('Disabling the historical sealed owner requires allow_recipe_overrides=True')
    selected['execution']=dict(sealed=sealed)
    selected['customized'] = selected['customized'] or not sealed
    policy=Pi05LitePolicy.from_pretrained(config['checkpoint'],tokenizer=assets.get('tokenizer'),
        device=config['device'],**options)
    policy._provenance['public_recipe']=deepcopy(selected)
    try:return Pi05Engine(policy,sealed=sealed,max_batch=max_batch)
    except BaseException:
        policy.close();raise

