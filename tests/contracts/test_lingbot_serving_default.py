from vvla import LoadConfig
from vvla_lingbot.presets import RECOMMENDED_RECIPE, SPLIT_STAGE_RECIPE, resolve_recipe


def test_omitted_recipe_selects_model_default():
    assert LoadConfig('lingbot', '/weights').recipe == 'auto'
    assert resolve_recipe('recommended')['resolved'] == RECOMMENDED_RECIPE == 'cache32-posttopk-fp16-prefix-expert'
    for name in ('oft', 'pi05'):
        assert LoadConfig(name, '/weights').recipe == 'auto'


def test_explicit_precision_and_legacy_batching():
    bf16 = resolve_recipe(LoadConfig('lingbot', '/weights', recipe='bf16').recipe)
    assert bf16['resolved'] == 'posttopk-bf16'
    assert 'denoise_precision' not in bf16['options']
    legacy = resolve_recipe('true-batch-posttopk-fp16-prefix-expert')
    split = resolve_recipe(SPLIT_STAGE_RECIPE)
    assert split['options']['split_stage'] and 'split_stage' not in legacy['options']
    assert split['true_batch'] and legacy['true_batch']
    assert split['options']['denoise_precision'] == legacy['options']['denoise_precision'] == 'fp16-prefix-expert'
