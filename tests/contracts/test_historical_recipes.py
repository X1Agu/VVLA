"""Independent historical identity fixtures must reach the real backend loader."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vvla.errors import InputError, UnsupportedConfiguration
from vvla_pi05.recipes import resolve_recipe, merge_options

FIXTURES = json.loads((Path(__file__).parents[1] / 'fixtures/historical-recipes.json').read_text())


@pytest.mark.parametrize('sm', [(8, 9), (12, 0)])
def test_default_and_explicit_pi05_preserve_entire_historical_loader_config(sm):
    name = f'bf16-sm{sm[0]}{sm[1]}'
    expected = FIXTURES['pi05-' + name]['config']
    for requested in ('bf16', 'bf16-optimized', name):
        selected = resolve_recipe(requested, capability=sm, horizon=50)
        assert selected['resolved'] == name
        assert selected['loader'] == expected
        assert selected['historical_configuration']
        assert not selected['full_weight_release_qualified']
    # No mutable preset state may leak from one load to another.
    selected['loader']['config_overrides'].clear()
    assert resolve_recipe(name, capability=sm, horizon=50)['loader'] == expected


def test_explicit_presets_reject_wrong_arch_or_geometry_and_label_customization():
    with pytest.raises(UnsupportedConfiguration):
        resolve_recipe('bf16-sm120', capability=(8, 9), horizon=50)
    with pytest.raises(UnsupportedConfiguration):
        resolve_recipe('bf16-sm89', capability=(8, 9), horizon=10)
    selected = resolve_recipe('bf16-sm89', capability=(8, 9), horizon=50)
    with pytest.raises(InputError, match='Historical recipe overrides'):
        merge_options(selected, {'config_overrides': {'lite_compile_sample_actions': False}})
    options = merge_options(selected, {'allow_recipe_overrides': True,
                                      'config_overrides': {'lite_compile_sample_actions': False,
                                                           'lite_compile_preserve_action_linears': False}})
    assert not options['config_overrides']['lite_compile_sample_actions']
    assert selected['customized']


@pytest.mark.parametrize('sm', [(8, 9), (12, 0)])
def test_public_pi05_backend_passes_the_full_recipe(monkeypatch, tmp_path, sm):
    from vvla_pi05 import backend
    from vvla_pi05.lite import pi05
    import torch
    (tmp_path/'config.json').write_text(json.dumps({'chunk_size': 50, 'n_action_steps': 50}))
    monkeypatch.setattr(backend, 'resolve_assets', lambda *a, **k: {'tokenizer': 'fixture'})
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: sm)
    calls = []
    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return SimpleNamespace(_provenance={})
    monkeypatch.setattr(pi05.Pi05LitePolicy, 'from_pretrained', load)
    monkeypatch.setattr(backend, 'Pi05Engine', lambda policy, **kw: policy)
    policy = backend.load_backend(dict(checkpoint=str(tmp_path), device='cuda:0', recipe='bf16', options={}))
    actual = calls[0][1]
    assert actual.pop('device') == 'cuda:0'
    assert actual.pop('tokenizer') == 'fixture'
    assert actual == FIXTURES[f'pi05-bf16-sm{sm[0]}{sm[1]}']['config']
    assert policy._provenance['public_recipe']['loader'] == actual
    assert policy._provenance['public_recipe']['execution']=={'sealed':True}


def test_unsealed_historical_run_cannot_be_reported_as_an_unchanged_control(monkeypatch,tmp_path):
    from vvla_pi05 import backend
    from vvla_pi05.lite import pi05
    import torch
    (tmp_path/'config.json').write_text(json.dumps({'chunk_size':50,'n_action_steps':50}))
    monkeypatch.setattr(backend,'resolve_assets',lambda *a,**k:{})
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(8,9))
    calls=[]
    def load(*a,**k):
        calls.append(k)
        return SimpleNamespace(_provenance={})
    monkeypatch.setattr(pi05.Pi05LitePolicy,'from_pretrained',load)
    monkeypatch.setattr(backend,'Pi05Engine',lambda policy,**kw:policy)
    config=dict(checkpoint=str(tmp_path),device='cuda:0',recipe='bf16',options={'sealed':False})
    with pytest.raises(InputError,match='allow_recipe_overrides'):
        backend.load_backend(config)
    assert calls==[]
    config['options']['allow_recipe_overrides']=True
    policy=backend.load_backend(config)
    receipt=policy._provenance['public_recipe']
    assert receipt['customized'] and receipt['execution']=={'sealed':False}
    assert calls==[dict(device='cuda:0',tokenizer=None,**FIXTURES['pi05-bf16-sm89']['config'])]
    config['options']['sealed']='false'
    with pytest.raises(InputError,match='sealed must be boolean'):backend.load_backend(config)


def test_non_historical_shapes_keep_explicit_optimized_control():
    for horizon in (10, 15):
        result = resolve_recipe('bf16', capability=(8, 9), horizon=horizon)
        assert result['resolved'] == 'bf16-portable'
        assert result['loader']['profile'] == 'full-bf16'
        assert result['loader']['config_overrides']['lite_fused_expert_qkv']
        assert not result['historical_configuration']


def test_reference_precision_is_an_explicit_recipe():
    result = resolve_recipe('bf16-reference', capability=(8, 9), horizon=50)
    assert result['loader']['profile'] == 'bf16'
    assert result['loader']['config_overrides'] == {}


@pytest.mark.parametrize('recipe,sm', [('bf16-sm89',(8,9)), ('bf16-sm120',(12,0)),
                                       ('mlp-fp8-sm120',(12,0))])
def test_oft_presets_retain_full_native_options_and_blas(recipe, sm):
    from vvla_oft.presets import resolve_recipe
    expected=FIXTURES['oft-'+recipe]
    selected=resolve_recipe(recipe,capability=sm)
    assert selected['loader']==expected['config']
    assert selected['blas_library'].lower()==expected['math_state']['blas_library'].split('.')[-1].lower()
    if recipe.startswith('bf16'):
        assert resolve_recipe('bf16',capability=sm)['loader']==expected['config']
    else:
        assert resolve_recipe('mlp-fp8',capability=sm)['loader']==expected['config']


@pytest.mark.parametrize('recipe,sm', [('bf16-sm89',(8,9)), ('bf16-sm120',(12,0)),
                                       ('mlp-fp8-sm120',(12,0))])
def test_public_oft_loader_enables_historical_native_path(monkeypatch, recipe, sm):
    from vvla_oft import backend,lite
    import torch
    calls=[];blas=[]
    monkeypatch.setattr(backend,'resolve_assets',lambda *a,**k:
                        dict(source='source',transformers_source='fork',native='native'))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:sm)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(torch.backends.cuda,'preferred_blas_library',lambda value:blas.append(value))
    def load(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(metadata=lambda:{})
    monkeypatch.setattr(lite,'OFTLitePolicy',load)
    engine=backend.load_backend(dict(recipe=recipe,checkpoint='checkpoint',device='cuda:0',options={}))
    actual=calls[0]
    assert actual.pop('native_path')=='native'
    assert actual==FIXTURES['oft-'+recipe]['config']
    assert blas==[engine.metadata()['public_recipe']['blas_library']]


def test_public_lingbot_default_restores_cap32_private_posttopk_composition(monkeypatch):
    from vvla_lingbot import backend,private_lite
    calls=[]
    monkeypatch.setattr(backend,'resolve_assets',lambda *a,**k:
                        dict(source='source',qwen='qwen',norm_stats='norm'))
    def factory(params,**kwargs):
        calls.append((params,kwargs))
        return SimpleNamespace(metadata=lambda:{})
    monkeypatch.setattr(private_lite,'factory',factory)
    engine=backend.load_backend(dict(recipe='bf16',checkpoint='checkpoint',device='cuda:0',options={}))
    params,kwargs=calls[0]
    assert params['graph_cache_capacity']==32
    assert params['posttopk'] and params['posttopk_sum_order']=='pairwise_02_13'
    assert not params['routed_down_fp8'] and not params['true_batch']
    assert kwargs=={'sealed_guard':True}
    assert engine.metadata()['public_recipe']['composition']=='cap32/private-owner/PostTopK(pairwise_02_13)'


def test_lingbot_independent_candidates_remain_explicit():
    from vvla_lingbot.presets import resolve_recipe
    for name in ('routed-down-fp8','posttopk-down-fp8','true-batch-bf16'):
        selected=resolve_recipe(name)
        assert selected['options']['graph_cache_capacity']==1
        assert not selected['full_weight_release_qualified']
    reference=resolve_recipe('bf16-reference')
    assert not reference['posttopk'] and not reference['options']['sealed_guard']


@pytest.mark.parametrize('precision',['fp8','nvfp4'])
def test_lingbot_legacy_vlm_precision_reaches_public_loader_without_new_overlays(monkeypatch,precision):
    from vvla_lingbot import backend,lite
    calls=[]
    monkeypatch.setattr(backend,'resolve_assets',lambda *a,**k:
        dict(source='source',qwen='qwen',norm_stats='norm',calibration='calibration',packed_weights='packed'))
    def factory(params):
        calls.append(params)
        return SimpleNamespace(metadata=lambda:{})
    monkeypatch.setattr(lite,'LingBotLitePolicy',factory)
    engine=backend.load_backend(dict(recipe='vlm-'+precision,checkpoint='checkpoint',device='cuda:0'))
    selected=engine.metadata()['public_recipe']
    assert selected['mode']=='framework-'+precision
    assert not selected['options']['sealed_guard']
    assert calls[0]['calibration']=='calibration' and calls[0]['packed_weights']=='packed'
    assert calls[0]['graph_cache_capacity']==1
    assert not any(calls[0][key] for key in ('posttopk','routed_down_fp8','true_batch'))
