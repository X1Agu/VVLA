"""Named LingBot compositions; all implementations live in the installed package."""
from copy import deepcopy
from vvla.errors import InputError, UnsupportedConfiguration

RECOMMENDED_RECIPE = 'cache32-posttopk-fp16-prefix-expert'
SPLIT_STAGE_RECIPE = 'split-stage-posttopk-fp16-prefix-expert'

MIXED_RECIPES = {
    SPLIT_STAGE_RECIPE: 'fp16-prefix-expert',
    'cache32-posttopk-fp16-prefix-expert': 'fp16-prefix-expert',
    'true-batch-posttopk-fp16-prefix-expert': 'fp16-prefix-expert',
    'cache32-posttopk-fp16-expert': 'fp16-expert',
    'true-batch-posttopk-fp16-expert': 'fp16-expert',
    'cache32-posttopk-fp32-heads': 'fp32-heads',
    'true-batch-posttopk-fp32-heads': 'fp32-heads',
    'cache32-posttopk-fp32-clock': 'fp32-clock',
    'cache32-posttopk-fp32-clock-accum': 'fp32-clock-accum',
    'true-batch-posttopk-fp32-clock': 'fp32-clock',
    'true-batch-posttopk-fp32-clock-accum': 'fp32-clock-accum',
}

RECIPE_NAMES = ('recommended', 'bf16', 'bf16-optimized', 'bf16-reference', 'official-bf16',
                'posttopk-bf16', 'cache32-posttopk-bf16', 'pooled-bf16',
                'routed-down-fp8', 'posttopk-down-fp8', 'cache32-posttopk-down-fp8',
                'true-batch-bf16','vlm-fp8','vlm-nvfp4', *MIXED_RECIPES)


def resolve_recipe(name, options=None):
    requested = name
    if name not in RECIPE_NAMES:
        raise UnsupportedConfiguration('Unknown LingBot recipe: ' + name)
    if name == 'recommended':
        name = RECOMMENDED_RECIPE
    if name in ('bf16', 'bf16-optimized', 'cache32-posttopk-bf16'):
        name = 'posttopk-bf16'
    allowed = {'graph_cache_capacity', 'graph_memory_budget_bytes', 'capture_reserve_bytes',
               'sealed_guard', 'posttopk_sum_order', 'asset_integrity'}
    options = deepcopy(options or {})
    if options.keys() - allowed:
        raise InputError('Unknown LingBot options: ' + ', '.join(sorted(options.keys() - allowed)))
    pooled = name in ('posttopk-bf16', 'pooled-bf16', 'cache32-posttopk-down-fp8')
    mixed = name in MIXED_RECIPES
    true_batch = name.startswith('true-batch-') or name == SPLIT_STAGE_RECIPE
    pooled = pooled or (mixed and not true_batch)
    defaults = dict(graph_cache_capacity=32 if pooled else 1,
                    sealed_guard=not true_batch and name not in ('official-bf16', 'bf16-reference', 'true-batch-bf16','vlm-fp8','vlm-nvfp4'),
                    posttopk_sum_order='pairwise_02_13')
    customized = any(k in defaults and defaults[k] != v for k, v in options.items())
    defaults.update(options)
    if name == SPLIT_STAGE_RECIPE:
        defaults['split_stage'] = True
    if mixed:
        defaults['denoise_precision'] = MIXED_RECIPES[name]
    if name.startswith('vlm-') and defaults['sealed_guard']:
        raise InputError('Historical VLM precision preserves the original full per-request guard')
    return dict(requested=requested, resolved=name, options=defaults, customized=customized,
                mode=('framework-'+name.removeprefix('vlm-')) if name.startswith('vlm-') else
                    ('official-bf16' if name == 'official-bf16' else 'framework-bf16'),
                posttopk=mixed or name in ('posttopk-bf16', 'posttopk-down-fp8', 'cache32-posttopk-down-fp8'),
                routed_down_fp8=name in ('routed-down-fp8', 'posttopk-down-fp8', 'cache32-posttopk-down-fp8'),
                true_batch=true_batch, full_weight_release_qualified=False,
                composition=(f"cap{defaults['graph_cache_capacity']}/private-owner/PostTopK({defaults['posttopk_sum_order']})/RoutedDownFP8"
                             if name=='cache32-posttopk-down-fp8' else
                             'cap32/private-owner/PostTopK(pairwise_02_13)' if pooled and name=='posttopk-bf16' else name))
