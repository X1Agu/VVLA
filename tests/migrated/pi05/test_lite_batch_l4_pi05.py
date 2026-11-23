"""CPU API/mechanism tests. No L4 numerical, capture or task qualification."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest
import torch

from test_lite_batch_pi05 import _Single
from vvla_repro_pi05.lite import batch_l4_pi05 as batch

BATCHES=(1,2,4,8,16,32)

def observations(size,changed=False):
    return [dict(value=i+(1000 if changed and i==size-1 else 0),
        state=np.arange(8,dtype=np.float32)+i) for i in range(size)]

class Runner:
    def __init__(self,size):
        self.size=size;self.calls=0;self.last_call={};self.stats=Counter();self._call_lock=threading.Lock()
    def __call__(self,**kwargs):
        self.calls+=1
        assert kwargs['noise'].shape==(self.size,50,32) and kwargs['num_steps']==10
        assert all(image.shape[0]==self.size for image in kwargs['images'])
        self.last_call=dict(cold=self.calls==1,graph='CPU mechanism fake',metadata_updated=False)
        self.stats['calls']+=1
        return kwargs['noise']+kwargs['lang_tokens'][:,0,None,None]

class Policy:
    def __init__(self,size):
        self.processor=_Single();self.device=torch.device('cpu');self.graph_runner=Runner(size)
        self._lock=threading.Lock();self._closed=False
    def close(self):self._closed=True

@pytest.mark.parametrize('size',BATCHES)
def test_one_batch_call_independent_draws_dynamic_rows_and_owned_retention(size):
    policy=Policy(size);engine=batch._wrap_policy(policy,batch._CAPABILITY,batch_size=size)
    obs=observations(size);seeds=list(range(size));engine.reset_batch([f'episode/{i}' for i in range(size)])
    first=engine.infer_batch(obs,seeds);saved=first['actions'].copy()
    expected=np.concatenate([torch.randn((1,50,32),generator=torch.Generator().manual_seed(seed)).numpy() for seed in seeds])
    np.testing.assert_array_equal(engine.last_noise(),expected)
    noise=engine.last_noise();noise.fill(1000);np.testing.assert_array_equal(engine.last_noise(),expected)
    np.testing.assert_array_equal(engine.infer_batch(obs,seeds)['actions'],saved)
    middle=engine.infer_batch(observations(size,changed=True),seeds)['actions']
    np.testing.assert_array_equal(middle[:-1],saved[:-1]);assert not np.array_equal(middle[-1],saved[-1])
    np.testing.assert_array_equal(engine.infer_batch(obs,seeds)['actions'],saved)
    assert policy.graph_runner.calls==4 and first['actions'].shape==(size,50,7)
    assert first['actions'].flags.owndata and first['actions'].flags.c_contiguous
    assert not np.shares_memory(first['actions'],middle)
    np.testing.assert_array_equal(first['actions'],saved)
    fixture=engine.prepared_fixture_batch(obs,seeds)
    required={'tokens','masks','image_0_bf16_bits','image_1_bf16_bits','noise_fp32','noise_bf16_bits'}
    assert required<=set(fixture)
    assert fixture['noise_fp32'].shape==(size,50,32) and fixture['noise_fp32'].dtype==np.float32
    assert fixture['noise_bf16_bits'].dtype==np.uint16 and fixture['image_0_bf16_bits'].shape[0]==size
    np.testing.assert_array_equal(fixture['noise_fp32'],expected)
    fixture['noise_fp32'].fill(0);np.testing.assert_array_equal(engine.last_noise(),expected)
    with pytest.raises(AttributeError):engine.batch_size=size
    with pytest.raises(RuntimeError,match='scaffold'):engine.metadata()
    engine.close();engine.close()
    with pytest.raises(RuntimeError,match='closed'):engine.last_noise()

@pytest.mark.parametrize('size',BATCHES)
@pytest.mark.parametrize('change',['count','boolean_seed','negative_seed','state','lock'])
def test_fixed_batch_api_rejects_invalid_dynamic_requests(size,change):
    policy=Policy(size);engine=batch._wrap_policy(policy,batch._CAPABILITY,batch_size=size)
    obs=observations(size);seeds=list(range(size))
    if change=='count':obs=obs[:-1]
    if change=='boolean_seed':seeds[-1]=True
    if change=='negative_seed':seeds[-1]=-1
    if change=='state':obs[-1]['state']=np.ones(7,dtype=np.float32)
    if change=='lock':policy._lock.acquire()
    try:
        with pytest.raises((ValueError,RuntimeError)):engine.infer_batch(obs,seeds)
    finally:
        if policy._lock.locked():policy._lock.release()

def test_origin_is_actual_l4_identity_and_never_a_g4_source_inventory(tmp_path):
    origin=batch._origin();path=Path(origin['identity_path'])
    assert origin['identity_sha256']==batch.ORIGIN_IDENTITY_SHA256 and origin['recipe_sha256']==batch.ORIGIN_RECIPE
    assert 'source_inventory_sha256' not in origin
    changed=tmp_path/'identity.json';changed.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='identity bytes'):batch._origin(changed)
    link=tmp_path/'link.json';link.symlink_to(path)
    with pytest.raises(ValueError,match='identity bytes'):batch._origin(link)

def test_exact_bf16_v4_lineage_and_explicit_unqualified_tactics_excluded():
    selected,loader=batch._batch_options(None)
    assert selected['lite_native_lifecycle_variant']=='v2'
    assert selected['lite_fused_expert_gate_up'] is False and selected['lite_fused_residual_adarms'] is False
    assert selected['lite_expert_cublaslt'] is None and selected['lite_suffix_fa2'] is None
    assert selected['lite_prefix_fp8'] is False and selected['lite_vision_fp8'] is False
    assert selected['lite_batched_vision'] is True and loader['lite_batched_vision'] is False
    for overrides in [{'lite_prefix_fp8':True},{'lite_native_lifecycle_variant':'v3'},
                      {'lite_suffix_fa2':{'splits':8}},{'lite_expert_cublaslt':{'profile':'l4_exact_three','include_down':True}},
                      {'lite_fused_residual_adarms':True},{'lite_fused_expert_gate_up':True},
                      {'num_inference_steps':5},{'lite_action_norm_cache':1}]:
        with pytest.raises(ValueError):batch._batch_options(overrides)

def test_private_factory_rejects_public_injection_precision_and_wrong_batch_before_cuda():
    with pytest.raises(TypeError):batch.Pi05L4BatchEngine(Policy(32))
    with pytest.raises(TypeError):batch._wrap_policy(Policy(32),object(),batch_size=32)
    for kwargs in [{'batch_size':3},{'batch_size':True},{'precision':'mixed_bf16_fp8_prefix'},
                   {'profile':'bf16'},{'attention_backend':'eager'},{'cuda_graph':False},
                   {'config_overrides':{'lite_prefix_fp8':True}},{'batch_cublaslt_evidence':{}},
                   {'batch_suffix_evidence':None},{'model':object()}]:
        with pytest.raises((TypeError,ValueError)):batch.Pi05L4BatchEngine.from_pretrained('unused',**kwargs)

def test_bf16_manifest_records_actual_parameter_dtypes_and_rejects_quantized_owners():
    model=torch.nn.Sequential(torch.nn.Linear(3,4,dtype=torch.bfloat16),torch.nn.LayerNorm(4))
    policy=SimpleNamespace(model=model,config=SimpleNamespace(execution_recipe={'prefix_mlp':'none'}))
    manifest=batch._bf16_manifest(policy)
    assert manifest['precision']=='bf16' and manifest['quantized_owners']==[]
    assert set(manifest['parameter_dtype_inventory'])=={'torch.bfloat16','torch.float32'}
    policy._prefix_fp8_gelu_installation=object()
    with pytest.raises(RuntimeError,match='quantization owner'):batch._bf16_manifest(policy)
    policy._prefix_fp8_gelu_installation=None;policy.config.execution_recipe['expert_mlp']='fp8'
    with pytest.raises(RuntimeError,match='precision recipe'):batch._bf16_manifest(policy)

@pytest.mark.parametrize('size',BATCHES)
def test_checkpoint_factory_uses_private_bf16_loader_and_explicit_size_without_gpu(monkeypatch,size):
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy
    origin=batch._origin();original=json.loads(Path(origin['identity_path']).read_bytes())['metadata']
    policy=Policy(size);policy.config=SimpleNamespace(chunk_size=50,num_inference_steps=10,max_action_dim=32,action_dim=7)
    policy._provenance={'checkpoint_sha256':batch.CHECKPOINT_SHA256,'resolved_config':deepcopy(original['resolved_config'])}
    called={};sentinel=object()
    def loader(checkpoint,**kwargs):called['loader']=(checkpoint,kwargs);return policy
    def install(value,selected,batch_size):called['install']=(value,selected,batch_size)
    def seal(value,batch_size,origin_record,hardware):called['seal']=(value,batch_size,origin_record,hardware);return sentinel
    monkeypatch.setattr(Pi05LitePolicy,'from_pretrained',staticmethod(loader))
    monkeypatch.setattr(batch,'_hardware',lambda device:{'name':'CPU test fake, not GPU-qualified'})
    monkeypatch.setattr(batch,'_install_batch_recipe',install);monkeypatch.setattr(batch,'_seal_policy',seal)
    assert batch.Pi05L4BatchEngine.from_pretrained('original-weights',batch_size=size) is sentinel
    assert called['loader'][1]['precision']=='bf16' and called['loader'][1]['config_overrides']['lite_native_lifecycle_variant']=='v2'
    assert called['loader'][1]['config_overrides']['lite_batched_vision'] is False
    assert called['install'][2]==size and called['seal'][1]==size and called['seal'][2]==origin
    policy._provenance['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='original checkpoint'):
        batch.Pi05L4BatchEngine.from_pretrained('wrong',batch_size=size)
    assert policy._closed
