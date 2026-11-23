"""LingBot adapter with checkpoint-relative assets and explicit recipe options."""
import numpy as np
from vvla.adapters import validate_batch, image, state
from vvla.assets import resolve_assets
from vvla.dispatch import prepare_config
from vvla.errors import InputError, UnsupportedConfiguration


class LingBotEngine:
    def __init__(self,policy,recipe=None):
        self.__policy = policy
        self.__recipe = recipe
        self._last_execution = {}
    def infer_batch(self,observations,seeds):
        from .lite import CAMERAS
        validate_batch(observations,seeds,sizes=(1,),seed_bits=32)
        observation=observations[0]
        if set(observation)!={'task','observation.state',*CAMERAS}:
            raise InputError('LingBot requires original task, observation.state and three named RoboTwin cameras')
        if type(observation['task']) is not str or not observation['task']:raise InputError('task must be nonempty')
        state(observation['observation.state'],14)
        for key in CAMERAS:image(observation[key],name=key)
        result = self.__policy.infer(observation,seeds[0])
        self._last_execution = dict(result.get('diagnostics', {}), actual_tensor_batch=1,
                                    execution_path=self.__recipe['resolved'] if self.__recipe else 'single')
        return [np.array(result['actions'],copy=True)]
    def capabilities(self):
        return dict(model='lingbot',batch_sizes=[1],seed_bits=32,observation_adapter='robotwin-three-camera',
                    action_shape=[50,14],action_space='physical_robotwin',full_weight_release_qualified=False)
    def metadata(self):
        from copy import deepcopy
        return dict(self.__policy.metadata(),public_backend='vvla-lingbot',denoise_steps=10,
                    public_recipe=deepcopy(self.__recipe),last_execution=deepcopy(self._last_execution),
                    capabilities=self.capabilities())
    def execution_receipt(self):
        return dict(self._last_execution)
    def reset(self,episode_id=None):return self.__policy.reset(episode_id)
    def close(self):return self.__policy.close()


class TrueBatchLingBotEngine:
    """Exact supported batch widths with independent per-request seeds."""
    def __init__(self,bundle,recipe=None,max_batch=None):
        from .experimental.true_batch_v2 import RawTrueBatchPolicy
        self._bundle=bundle
        self._recipe=recipe
        self._widths=tuple(b for b in bundle.engine.allowed_batches if max_batch is None or b<=max_batch)
        self._policy=RawTrueBatchPolicy(bundle.policy,bundle.engine,runtime_bundle=bundle)
    def infer_batch(self,observations,seeds):
        from .lite import CAMERAS
        validate_batch(observations,seeds,sizes=self._widths,seed_bits=32)
        for observation in observations:
            if set(observation)!={'task','observation.state',*CAMERAS}:
                raise InputError('LingBot requires task, state and three RoboTwin cameras')
            if type(observation['task']) is not str or not observation['task']:raise InputError('task must be nonempty')
            state(observation['observation.state'],14)
            for key in CAMERAS:image(observation[key],name=key)
        identities=[str(i) for i in range(len(observations))]
        result=self._policy.infer_seeded(observations,identities,seeds)['action']
        return [np.array(row,copy=True) for row in result]
    def capabilities(self):
        return dict(model='lingbot',batch_sizes=list(self._widths),seed_bits=32,dummy_lanes=False,
                    batch_independent_text_fields=['task'],
                    observation_adapter='robotwin-three-camera',action_shape=[50,14],
                    action_space='physical_robotwin',experimental=True,full_weight_release_qualified=False)
    def metadata(self):
        from copy import deepcopy
        return dict(public_backend='vvla-lingbot',attention_backend='vllm_fa2',denoise_steps=10,config=self._bundle.config.to_mapping(),
                    sources=self._bundle.sources,runtime=self._bundle.actual_runtime,
                    recipe=self._bundle.engine.recipe,public_recipe=deepcopy(self._recipe),
                    last_execution=dict(self._bundle.engine.last_call),capabilities=self.capabilities())
    def execution_receipt(self):
        return dict(self._bundle.engine.last_call)
    def reset(self,episode_id=None):
        self._policy.reset_all()
    def close(self):self._policy.close()


def load_backend(config):
    from .presets import resolve_recipe
    selected=resolve_recipe(config.get('recipe','recommended'),config.get('options',{}))
    assets=resolve_assets(config['checkpoint'],config.get('assets',{}),offline=config.get('offline',False),required=('source','qwen','norm_stats'))
    options=dict(selected['options'])
    sealed=options.pop('sealed_guard')
    params=dict(checkpoint=config['checkpoint'],official_root=assets['source'],qwen=assets['qwen'],
                norm_stats=assets['norm_stats'],mode=selected['mode'],
                device=config['device'],posttopk=selected['posttopk'],
                routed_down_fp8=selected['routed_down_fp8'],true_batch=selected['true_batch'],**options)
    if assets.get('native'):params['native_path']=assets['native']
    for name in ('training_config','robot_config','calibration','packed_weights'):
        if assets.get(name):params[name]=assets[name]
    if selected['true_batch']:
        from .runtime import build_managed_runtime
        bundle=build_managed_runtime(params)
        try:return TrueBatchLingBotEngine(bundle,selected,config.get('max_batch'))
        except BaseException:
            bundle.close();raise
    if selected['resolved'] in ('official-bf16','vlm-fp8','vlm-nvfp4'):
        from .lite import LingBotLitePolicy
        policy=LingBotLitePolicy(params)
    else:
        from .private_lite import factory
        policy=factory(params,sealed_guard=sealed)
    return LingBotEngine(policy,selected)
