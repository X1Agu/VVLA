"""Real shared-core context with tiny CPU fixtures, not B32 GPU qualification."""
from copy import deepcopy
import pickle
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import batch_pi05 as batch
from vvla_repro_pi05.lite import batch_checkpoint_engine as sealed
from vvla_repro_pi05.lite.batch_layout import BatchHostMaskLayout,BatchHostLayoutCache
from test_lite_batch_pi05 import _Single
from test_lite_fixed_checkpoint_engine import _Policy as _TinyPolicy
from test_lite_host_layout import _LayoutModel,_inputs


@pytest.fixture
def engine_fixture(monkeypatch):
    from vvla_repro_pi05.lite import batch_suffix_evidence
    monkeypatch.setattr(batch_suffix_evidence,'validate_runtime',lambda *args: {'test_only':'fictional CPU support'})
    policy=_TinyPolicy();policy.processor=_Single()
    policy.config=SimpleNamespace(attention_backend='torch_fa2')
    policy._provenance=dict(recipe_sha256='a'*64,checkpoint_sha256='b'*64)
    # Fictional CPU evidence binds declaration consistency only. Production
    # factory checks real pinned checkpoint and executed B32 backend evidence.
    policy._lite_optimizations=dict(native_lifecycle_manifest={'test_only':'fictional CPU collector'},prefix_fp8=True,
        batch_suffix_evidence={'test_only':'fictional CPU support'})
    policy._prefix_fp8_installation=SimpleNamespace(validate_precision_manifest=lambda:dict(
        schema=1,precision=batch.PRECISION,batch=32,sites=['fictional CPU site']))
    engine=batch._seal_policy(policy)
    yield policy,engine
    if not policy._closed:engine.close()


def test_public_factory_requires_latest_mixed_and_no_external_mutable_refs(engine_fixture):
    policy,engine=engine_fixture
    with pytest.raises(ValueError,match='latest prefix51'):batch.Pi05BatchEngine.from_pretrained('/checkpoint',precision='bf16')
    with pytest.raises(TypeError,match='from_pretrained'):batch.Pi05BatchEngine(policy)
    with pytest.raises(TypeError,match='private checkpoint'):sealed.BatchSealedModelState(policy,object())
    for name in ('model','processor','graph_runner','policy','config','state_dict','parameters','load_weights','to','train'):
        assert not hasattr(engine,name)
        with pytest.raises(AttributeError):setattr(engine,name,object())
    with pytest.raises(TypeError,match='serialized'):pickle.dumps(engine)
    assert not hasattr(engine,'__dict__')


def test_latest_options_are_copied_and_only_custom_b1_geometry_is_substituted():
    options=deepcopy(batch._SELECTED);before=deepcopy(options)
    selected,loader=batch._batch_options(options)
    assert options==before and selected==before
    for name in ('lite_fused_expert_qkv','lite_fused_expert_gate_up','lite_fused_expert_gelu',
                 'lite_fused_residual_adarms','lite_gpu_uint8_prepare','lite_packed_input_metadata','lite_host_layout_cache'):
        assert loader[name] is True
    for name in ('lite_fixed_checkpoint_engine','lite_prefix_fp8','lite_batched_vision','lite_vision_fc2_cublaslt'):
        assert loader[name] is False
    assert loader['lite_suffix_fa2'] is None and loader['lite_expert_cublaslt'] is None
    for name,expected in batch._SELECTED.items():
        bad=False if type(expected)is bool else 'different'
        with pytest.raises(ValueError,match='latest selected'):batch._batch_options({name:bad})
    with pytest.raises(ValueError):batch._batch_options({'lite_vision_fp8':True})
    with pytest.raises(ValueError):batch._batch_options({'lite_prefix_fp8':1})


def test_batch_metadata_has_explicit_cold_only_scope_latest_origin_and_owned_json(engine_fixture):
    policy,engine=engine_fixture
    first=engine.metadata();second=engine.metadata()
    assert first==second and first is not second
    assert first['batch_size']==32 and first['true_batch'] is True and first['precision']==batch.PRECISION
    assert first['origin_b1_recipe_sha256']==batch.ORIGIN_RECIPE and first['origin_b1_source_inventory_sha256']==batch.ORIGIN_SOURCE
    assert first['sampler_dtype']=='float32'
    manifest=first['quantization_manifest']['fixed_checkpoint_engine']
    assert manifest['cold_audit_checks_complete'] is True and manifest['live_checks_complete'] is False
    assert first['recipe_sha256']==batch._digest(first['recipe_components'])
    manifest['public_api'].clear()
    assert engine.metadata()==second
    policy._lite_optimizations['prefix_fp8']=False
    with pytest.raises(RuntimeError,match='seal changed'):engine.metadata()


@pytest.mark.parametrize('fault',['cache_epoch','math','autocast','detached','runner_closed','source'])
def test_batch_scope_guard_keeps_retirement_available(engine_fixture,fault,monkeypatch):
    policy,engine=engine_fixture;state=engine._Pi05BatchEngine__state
    if fault=='cache_epoch':policy.model._inference_cache.clear()
    elif fault=='math':monkeypatch.setattr(sealed,'_math_settings',lambda:('changed',))
    elif fault=='autocast':monkeypatch.setattr(torch,'is_autocast_enabled',lambda *args:True)
    elif fault=='detached':delattr(policy.model,sealed.INSTALLATION)
    elif fault=='runner_closed':policy.graph_runner._closed=True
    else:monkeypatch.setattr(batch,'_sources',lambda:{'changed':'f'*64})
    with pytest.raises(RuntimeError):engine.metadata()
    engine.close();engine.close();assert state._closed and policy._closed


def test_batch_full_audit_detects_original_weight_mutation(engine_fixture):
    policy,engine=engine_fixture
    with torch.no_grad():policy.model.weight.add_(1)
    with pytest.raises(RuntimeError,match='generation/scope changed|generation changed'):engine.metadata()


def test_batch_failed_close_can_retry_after_detaching_private_seal(engine_fixture,monkeypatch):
    policy,engine=engine_fixture;original=policy.close;attempts=[]
    def fail_first():
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('injected cleanup error')
        original()
    monkeypatch.setattr(policy,'close',fail_first)
    with pytest.raises(RuntimeError,match='cleanup error'):engine.close()
    assert engine._Pi05BatchEngine__state._closed and not policy._closed
    engine.close();assert policy._closed and len(attempts)==2


def _batch_inputs(valid,value):
    rows=[_inputs(valid=(valid+slot)%6+1,cameras=(True,False),value=value+slot/8)for slot in range(32)]
    result={key:torch.cat([row[key]for row in rows],0)for key in ('lang_tokens','lang_masks','noise')}
    result['images']=[torch.cat([row['images'][slot]for row in rows],0)for slot in range(2)]
    result['image_masks']=[torch.cat([row['image_masks'][slot]for row in rows],0)for slot in range(2)]
    result['num_steps']=2
    result['host']=BatchHostMaskLayout.from_rows([row['host_layout']for row in rows],result['images'],result['image_masks'],result['lang_masks'])
    return result


def test_batch_seal_still_builds_actual32_dynamic_context_and_positions(engine_fixture):
    policy,engine=engine_fixture;reference=_LayoutModel();cache=BatchHostLayoutCache();outputs=[]
    initial=policy.model.complete_prepares
    for valid,value in [(1,1.),(4,2.5),(2,-.5)]:
        current=_batch_inputs(valid,value)
        template=cache.get(policy.model,current['host'],current['images'],current['image_masks'],current['lang_masks'])
        with policy._lock,policy.graph_runner._call_lock:
            actual=policy.model.prepare_inference_context(current['images'],current['image_masks'],current['lang_masks'],2,
                positions=False,prepared_layout=template.prepared_layout)
        expected=reference.prepare_inference_context(current['images'],current['image_masks'],current['lang_masks'],2,positions=False)
        for key in ('indices','pads','positions'):torch.testing.assert_close(actual['packing'][key],expected['packing'][key],rtol=0,atol=0)
        assert actual['packing']['pads'].shape[0]==32
        for model,context in ((policy.model,actual),(reference,expected)):
            context['positions']=model.prepare_position_context(current['image_masks'],current['lang_masks'],context['packing'])
        args={key:current[key]for key in ('images','image_masks','lang_tokens','lang_masks','noise','num_steps')}
        with torch.inference_mode():
            got=policy.model.sample_actions(**args,inference_context=actual);want=reference.sample_actions(**args,inference_context=expected)
        torch.testing.assert_close(got,want,rtol=0,atol=0);outputs.append(got.clone())
    assert policy.model.complete_prepares==initial and any(not torch.equal(outputs[0],x)for x in outputs[1:])
    engine.metadata();assert policy.model.complete_prepares==initial+1
