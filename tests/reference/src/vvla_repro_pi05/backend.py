"""Public Pi0.5 facade. Shared mathematics remains in Pi05LitePolicy."""
from copy import deepcopy
import numpy as np

from vvla.adapters import image, state, validate_batch
from vvla.assets import resolve_assets
from vvla.errors import InputError, UnsupportedConfiguration


class Pi05Engine:
    def __init__(self, policy, *, sealed=True, max_batch=1):
        if type(max_batch) is not int or not 1<=max_batch<=32:
            raise InputError('max_batch must be an integer in [1,32]')
        if max_batch>1 and policy._lite_optimizations.get('host_layout_cache'):
            raise UnsupportedConfiguration('Portable BF16 batching uses per-batch metadata; B1 host layout cache cannot be reused')
        self.__policy = policy
        self.__max_batch = max_batch
        self.__processors = {}
        self.__state = None
        if sealed and policy.graph_runner is not None:
            from .lite.batch_checkpoint_engine import BatchSealedModelState, _CAPABILITY
            with policy._lock, policy.graph_runner._call_lock:
                self.__state = BatchSealedModelState(policy, _CAPABILITY)
                self.__state.audit()

    def infer_batch(self, observations, seeds):
        validate_batch(observations,seeds,sizes=range(1,self.__max_batch+1))
        config=self.__policy.config
        for obs in observations:
            state(obs.get('state'),config.state_dim)
            if type(obs.get('prompt','')) is not str:
                raise InputError('Pi0.5 prompt must be a string')
            images=obs.get('images')
            if type(images) is not dict or not images:
                raise InputError('Pi0.5 SDK observations require an images dictionary with checkpoint camera names')
            for name,value in images.items():
                if value is not None:image(value,name=name,allow_float=True)
        if self.__state is not None:self.__state._assert_live()
        if len(observations)==1:
            return [np.array(self.__policy.infer(observations[0],seeds[0])['actions'],copy=True)]
        import torch
        from .lite.batch_processor import Pi05BatchProcessor
        batch=len(observations);policy=self.__policy
        if not policy._lock.acquire(blocking=False):raise RuntimeError('Serialize access to the model owner')
        try:
            if batch not in self.__processors:
                self.__processors[batch]=Pi05BatchProcessor(policy.processor,batch,output_processor=policy._action_norm_cache)
            processor=self.__processors[batch]
            with torch.inference_mode():
                images,image_masks,tokens,masks=processor.build_model_inputs(observations)
                noise=torch.cat([torch.randn((1,config.chunk_size,config.max_action_dim),device=policy.device,
                    dtype=torch.float32,generator=torch.Generator(device=policy.device).manual_seed(seed)) for seed in seeds])
                runner=policy.graph_runner or policy.model.sample_actions
                value=runner(images=images,image_masks=image_masks,lang_tokens=tokens,lang_masks=masks,
                             noise=noise,num_steps=config.num_inference_steps)
                values=processor.build_model_outputs(value,observations)
                policy._calls+=1
                return [np.array(row,copy=True) for row in values]
        finally:policy._lock.release()

    def capabilities(self):
        config=self.__policy.config
        return dict(model='pi05',batch_sizes=list(range(1,self.__max_batch+1)),seed_bits=63,
                    batch_execution='one true model call; portable BF16 geometry',dummy_lanes=False,
                    precision='dynamic-fp8-experimental' if self.__policy._provenance.get('experimental_precision') else 'bf16',
                    observation_adapter=config.input_contract,action_shape=[config.chunk_size,config.action_dim],
                    action_space=('libero_delta_no_additional_state_shift' if config.input_contract=='openpi-libero-pi05'
                                  else 'checkpoint_inverse_normalization_then_absolute'),
                    full_weight_release_qualified=False)

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
    if config['recipe'] not in ('bf16','bf16-optimized','dynamic-fp8'):
        raise UnsupportedConfiguration('Pi0.5 release recipes: bf16, bf16-optimized; low-bit candidates require separate qualification')
    assets=resolve_assets(config['checkpoint'],config.get('assets',{}))
    options=deepcopy(config.get('options',{}))
    allowed={'cuda_graph','max_graphs','attention_backend','config_overrides','sealed','precision_stages','precision_fallback','max_batch'}
    if options.keys()-allowed:raise InputError(f'Unknown Pi0.5 options: {sorted(options.keys()-allowed)}')
    sealed=options.pop('sealed',True)
    max_batch=options.pop('max_batch',1)
    overrides=options.get('config_overrides',{})
    if any('evidence' in key or 'fp8' in key or 'nvfp4' in key or 'fixed_checkpoint_engine' in key for key in overrides):
        raise UnsupportedConfiguration('Historical qualification options are outside the public BF16 recipe')
    stages=options.pop('precision_stages',None)
    fallback=options.pop('precision_fallback','error')
    if config['recipe']=='dynamic-fp8':
        if not stages:raise InputError('dynamic-fp8 requires explicit precision_stages for ablation')
        options['experimental_precision']=dict(stages=stages,unsupported=fallback)
    elif stages is not None:
        raise InputError('precision_stages requires recipe=dynamic-fp8')
    from .lite.pi05 import Pi05LitePolicy
    policy=Pi05LitePolicy.from_pretrained(config['checkpoint'],tokenizer=assets.get('tokenizer'),
        device=config['device'],profile='full-bf16' if config['recipe']=='bf16-optimized' else 'bf16',**options)
    try:return Pi05Engine(policy,sealed=sealed,max_batch=max_batch)
    except BaseException:
        policy.close();raise

