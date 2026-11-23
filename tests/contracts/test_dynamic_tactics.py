"""No fabricated tactics are used as execution or historical evidence here."""
import json
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from vvla_pi05.lite.dynamic_cublaslt import load_tactics
from vvla_pi05.recipes import resolve_recipe


def test_missing_historical_tactics_are_not_replaced_by_new_selection(tmp_path):
    with pytest.raises(ValueError,match='preserved historical'):load_tactics(None)
    path=tmp_path/'empty.json';path.write_text(json.dumps({'implementation':'source497-dynamic-bf16-lt-v1','batches':{}}))
    with pytest.raises(ValueError,match='complete historical'):load_tactics(path)


def test_blackwell_batch_public_recipe_keeps_all_owners_and_requires_tactics():
    selection=resolve_recipe('prefix-fp8-batch-sm120',capability=(12,0),horizon=50,max_batch=32)
    options=selection['loader']['config_overrides']
    assert selection['required_assets']==['lt_tactics']
    assert options['lite_prefix_fp8'] and options['lite_batch_geometry']=='sm120'
    assert options['lite_packed_input_metadata'] and options['lite_fused_expert_gate_up']
    assert options['lite_fused_residual_adarms'] and options['lite_compile_sample_actions']
    assert options['lite_native_lifecycle_variant']=='v3'


@pytest.mark.parametrize('requested',['bf16','bf16-optimized'])
def test_default_blackwell_batch_without_asset_is_explicitly_portable(requested):
    selected=resolve_recipe(requested,capability=(12,0),horizon=50,max_batch=8)
    assert selected['resolved']=='bf16-portable' and not selected['required_assets']
    assert selected['fallback']==dict(from_recipe='bf16-batch-sm120',reason='missing_lt_tactics')
    options=selected['loader']['config_overrides']
    assert options['lite_batch_geometry']=='portable' and options['lite_compile_sample_actions']
    assert options['lite_fused_expert_qkv'] and options['lite_fused_expert_gate_up']
    assert not selected['historical_configuration']
    supplied=resolve_recipe(requested,capability=(12,0),horizon=50,max_batch=8,lt_tactics_available=True)
    assert supplied['resolved']=='bf16-batch-sm120' and supplied['fallback'] is None


def public_loader(monkeypatch,tmp_path,assets):
    import torch
    from vvla_pi05 import backend
    from vvla_pi05.lite import pi05
    (tmp_path/'config.json').write_text(json.dumps({'chunk_size':50,'n_action_steps':50}))
    monkeypatch.setattr(backend,'resolve_assets',lambda *a,**k:assets)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(12,0))
    calls=[]
    def load(*a,**kwargs):calls.append(kwargs);return SimpleNamespace(_provenance={})
    monkeypatch.setattr(pi05.Pi05LitePolicy,'from_pretrained',load)
    monkeypatch.setattr(backend,'Pi05Engine',lambda policy,**kw:policy)
    return backend,dict(checkpoint=str(tmp_path),recipe='bf16',device='cuda:0',options={'max_batch':8}),calls


def test_public_default_fallback_and_explicit_missing_asset_rejection(monkeypatch,tmp_path):
    from vvla.errors import InputError
    backend,config,calls=public_loader(monkeypatch,tmp_path,{})
    result=backend.load_backend(config)
    assert result._provenance['public_recipe']['resolved']=='bf16-portable'
    assert calls[0]['config_overrides']['lite_batch_geometry']=='portable'
    config['recipe']='bf16-batch-sm120'
    with pytest.raises(InputError,match='requires assets.lt_tactics'):backend.load_backend(config)
    assert len(calls)==1


def test_only_default_can_fallback_for_device_incompatible_tactics(monkeypatch,tmp_path):
    from vvla.errors import UnsupportedConfiguration
    from vvla_pi05.lite import dynamic_cublaslt as module
    backend,config,calls=public_loader(monkeypatch,tmp_path,{'lt_tactics':'fixture-only'})
    monkeypatch.setattr(module,'load_tactics',lambda path:{'_report_sha256':'fixture-hash'})
    def incompatible(*a):raise module.TacticUnavailable('AlgoCheck rejected this device')
    monkeypatch.setattr(module,'preflight_tactics',incompatible)
    result=backend.load_backend(config)
    receipt=result._provenance['public_recipe']
    assert receipt['resolved']=='bf16-portable'
    assert receipt['fallback']['reason']=='incompatible_lt_tactics'
    assert receipt['fallback']['rejected_asset_sha256']=='fixture-hash'
    assert 'lite_dynamic_cublaslt' not in calls[0]['config_overrides']
    config['recipe']='bf16-batch-sm120'
    with pytest.raises(UnsupportedConfiguration,match='AlgoCheck'):backend.load_backend(config)
    assert len(calls)==1


@pytest.mark.parametrize('failure',[RuntimeError('ABI failure'),MemoryError('allocation failure')])
def test_default_does_not_hide_other_preflight_failures(monkeypatch,tmp_path,failure):
    from vvla_pi05.lite import dynamic_cublaslt as module
    backend,config,calls=public_loader(monkeypatch,tmp_path,{'lt_tactics':'fixture-only'})
    monkeypatch.setattr(module,'load_tactics',lambda path:{'_report_sha256':'fixture-hash'})
    def fail(*a):raise failure
    monkeypatch.setattr(module,'preflight_tactics',fail)
    with pytest.raises(type(failure),match=str(failure)):backend.load_backend(config)
    assert not calls


def test_default_rejects_malformed_explicit_asset(monkeypatch,tmp_path):
    path=tmp_path/'invalid.json';path.write_text('{}')
    backend,config,calls=public_loader(monkeypatch,tmp_path,{'lt_tactics':str(path)})
    with pytest.raises(ValueError,match='complete historical'):backend.load_backend(config)
    assert not calls


def test_compatible_tactics_still_reach_the_strong_public_path(monkeypatch,tmp_path):
    from vvla_pi05.lite import dynamic_cublaslt as module
    backend,config,calls=public_loader(monkeypatch,tmp_path,{'lt_tactics':'fixture-only'})
    monkeypatch.setattr(module,'load_tactics',lambda path:{'_report_sha256':'fixture-hash'})
    monkeypatch.setattr(module,'preflight_tactics',lambda *a:{'geometries_checked':24})
    result=backend.load_backend(config);receipt=result._provenance['public_recipe']
    assert receipt['resolved']=='bf16-batch-sm120' and receipt['fallback'] is None
    assert receipt['tactics_preflight']['geometries_checked']==24
    assert calls[0]['config_overrides']['lite_dynamic_cublaslt']=='fixture-only'


@pytest.mark.parametrize('failure_status',[None,15,3])
def test_tactic_preflight_checks_all_geometries_and_releases_descriptors(monkeypatch,failure_status):
    from vvla_pi05.lite import dynamic_cublaslt as module
    made=[];selected=[]
    class Plan:
        def __init__(self,*a):self.closed=False;made.append(self)
        def close(self):self.closed=True
    monkeypatch.setattr(module,'load_api',lambda *a:(object(),{}))
    monkeypatch.setattr(module.torch.cuda,'device',lambda *a:nullcontext())
    monkeypatch.setattr(module.torch,'empty',lambda *a,**k:object())
    monkeypatch.setattr(module.torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(name='fixture',multi_processor_count=1))
    monkeypatch.setattr(module,'DynamicBiasedPlan',Plan)
    monkeypatch.setattr(module.ffi,'Plan',Plan)
    def select(plan,spec):
        selected.append(spec)
        if failure_status and len(selected)==3:
            raise module.ffi.LtError('cublasLtMatmulAlgoCheck',failure_status,'fixture')
    monkeypatch.setattr(module.ffi,'select_algorithm',select)
    report={'batches':{str(b):{label:{'selected_spec':{'fixture':(b,label)}} for label in module.BASE_SHAPES} for b in module.BATCH_SIZES}}
    if failure_status:
        expected=module.TacticUnavailable if failure_status==15 else module.ffi.LtError
        with pytest.raises(expected):module.preflight_tactics(report,'cuda:0')
    else:assert module.preflight_tactics(report,'cuda:0')['geometries_checked']==24
    assert all(plan.closed for plan in made)
    assert len(selected)==(3 if failure_status else 24)
