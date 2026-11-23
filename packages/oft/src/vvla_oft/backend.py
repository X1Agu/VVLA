"""OpenVLA-OFT adapter behind the unified isolated-worker protocol."""
import os
import numpy as np
from vvla.adapters import validate_batch
from vvla.assets import resolve_assets
from vvla.dispatch import prepare_config
from vvla.errors import InputError, UnsupportedConfiguration


class OftEngine:
    def __init__(self,policy,batch_size,recipe=None):
        self.__policy=policy;self.__batch_size=batch_size;self.__recipe=recipe
    def infer_batch(self,observations,seeds):
        validate_batch(observations,seeds,sizes=range(1,self.__batch_size+1))
        try:
            for observation in observations:self.__policy._observation(observation)
        except (ValueError,TypeError,KeyError) as error:raise InputError(str(error)) from error
        # L1 OFT has no sampling RNG; explicit seeds are accepted and recorded as
        # unused. It does not reseed global CUDA or pretend to be stochastic.
        return [np.array(row,copy=True) for row in self.__policy.predict_batch(observations)]
    def capabilities(self):
        return dict(model='oft',batch_sizes=list(range(1,self.__batch_size+1)),seed_bits=63,seed_semantics='deterministic L1; unused',
            batch_independent_text_fields=['prompt'],
            observation_adapter='oft-libero-two-camera',action_shape=[8,7],
            action_space='checkpoint_unnormalized_before_libero_gripper_convention',full_weight_release_qualified=False)
    def execution_receipt(self):
        call = self.__policy._last_call
        return dict(actual_tensor_batch=call['actual_tensor_batch'],
                    graph_rebuild=bool(call['graph_delta']['state_invalidations'] and call['graph_delta']['captures']),
                    graph_fallback=bool(call['graph_delta']['capacity_fallbacks']), graph_build=bool(call['graph_delta']['captures']),
                    graph_replay=bool(call['graph_delta'].get('replays', 0)))

    def metadata(self):
        from copy import deepcopy
        return dict(self.__policy.metadata(),public_backend='vvla-oft',
                    public_recipe=deepcopy(self.__recipe),capabilities=self.capabilities())
    def reset(self,episode_id=None):return self.__policy.reset(episode_id)
    def close(self):return self.__policy.close()


def load_backend(config):
    from .presets import RECIPE_NAMES, resolve_recipe, merge_options
    if config['recipe'] not in RECIPE_NAMES:
        raise UnsupportedConfiguration('Unknown OFT execution recipe')
    assets=resolve_assets(config['checkpoint'],config.get('assets',{}),offline=config.get('offline',False),required=('source','transformers_source'))
    os.environ['VVLA_OFT_TRANSFORMERS_SOURCE']=assets['transformers_source']
    options=dict(config.get('options',{}))
    if options.keys()-{'batch_size','norm_key','cuda_graph','native_options','native_state_comparator','versioned_state_checks','paired_image_preprocessing','gpu_image_geometry','native_jpeg','allow_recipe_overrides','blas_library'}:
        raise InputError('Unknown OFT options')
    import torch
    torch.cuda.set_device(config['device'])
    selected=resolve_recipe(config['recipe'],capability=torch.cuda.get_device_capability(config['device']),
                            batch_size=options.get('batch_size',1))
    blas = options.pop('blas_library', selected['blas_library'])
    if blas not in ('cublas', 'cublaslt'):
        raise InputError('Invalid OFT BLAS backend')
    customized_blas = blas != selected['blas_library']
    if customized_blas and selected['historical_configuration'] and options.get('allow_recipe_overrides') is not True:
        raise InputError('Changing the historical BLAS backend requires allow_recipe_overrides=True')
    options=merge_options(selected,options)
    selected['blas_library'] = blas
    selected['customized'] |= customized_blas
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.backends.cuda.preferred_blas_library(selected['blas_library'])
    from .lite import OFTLitePolicy
    needs_native=options.get('native_options',{}).get('backend','sdpa')!='sdpa'
    policy=OFTLitePolicy(assets['source'],config['checkpoint'],
        native_path=assets.get('native') if needs_native else None,**options)
    return OftEngine(policy,options.get('batch_size',1),selected)

