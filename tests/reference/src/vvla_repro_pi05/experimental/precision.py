from vvla_runtime.dynamic_fp8 import install_dynamic_fp8

STAGES={'vision','prefix','action_attention','action_mlp'}


def install(model,stages,*,unsupported='error'):
    stages=set(stages)
    if not stages or not stages<=STAGES:raise ValueError(f'Select stages from {sorted(STAGES)}')
    def select(name,module):
        stage=None
        if '.vision_tower.' in name:stage='vision'
        elif '.paligemma.model.language_model.layers.' in name:stage='prefix'
        elif '.gemma_expert.model.layers.' in name:
            if '.self_attn.' in name:stage='action_attention'
            elif '.mlp.' in name:stage='action_mlp'
        return stage if stage in stages else None
    return install_dynamic_fp8(model,select,unsupported=unsupported)

