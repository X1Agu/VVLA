"""CPU-only real batching/slot semantics; not GPU or policy qualification."""
from collections import Counter
from types import MethodType, SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from vvla_repro_pi05.lite import batch_pi05 as batch
from vvla_repro_pi05.lite.batch_processor import Pi05BatchProcessor
from vvla_repro_pi05.lite.batch_layout import BatchHostMaskLayout, BatchHostLayoutCache
from vvla_repro_pi05.lite.batch_suffix_fa2 import _pack_joint_pages
from vvla_repro_pi05.lite.batch_vision import install_batch_vision
from vvla_repro_pi05.lite.layout import HostMaskLayout
from vvla_repro_pi05.lite.attention import prepare_pi05_static_metadata
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vvla_repro_pi05.models.pi05.processor_pi05 import Pi05Processor, Pi05RelativeActions


class _Single:
    def __init__(self):
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32, action_dim=7, state_dim=8)
        self._action_norm = None
        self.relative_actions = Pi05RelativeActions(enabled=True, max_action_dim=32)
        self.calls = 0

    def build_model_inputs(self, obs, *, return_host_layout=False):
        self.calls += 1
        value = obs['value']
        images = [torch.full((1,3,2,2), float(value)), torch.full((1,3,2,2), -1.)]
        cameras = (True, False)
        image_masks = [torch.tensor([x]) for x in cameras]
        tokens = torch.tensor([[value, value+1, value+2, 0]], dtype=torch.int64)
        mask = torch.tensor([[True, True, bool(value%2), False]])
        values = (images, image_masks, tokens, mask)
        if return_host_layout:
            return (*values, HostMaskLayout.from_processor(cameras, mask[0].tolist(), images, image_masks, mask))
        return values

    build_model_outputs = Pi05Processor.build_model_outputs


def _observations(offset=0):
    return [dict(value=i+offset, state=np.arange(8,dtype=np.float32)+i+offset) for i in range(32)]


def test_exact_current_sample_axis_and_owned_output():
    single = _Single(); processor = Pi05BatchProcessor(single,32);obs = _observations()
    images, cameras, tokens, masks = processor.build_model_inputs(obs)
    assert single.calls == 32 and tokens.shape == (32,4) and masks.shape == (32,4)
    torch.testing.assert_close(images[0][:,0,0,0], torch.arange(32,dtype=torch.float32),rtol=0,atol=0)
    assert images[1].eq(-1).all() and cameras[0].all() and not cameras[1].any()
    raw = torch.arange(32*50*32,dtype=torch.float32).reshape(32,50,32)/128
    actual = processor.build_model_outputs(raw,obs)
    expected = np.stack([single.build_model_outputs(raw[i:i+1],obs[i]) for i in range(32)])
    np.testing.assert_array_equal(actual,expected)
    assert actual.flags.c_contiguous and actual.flags.owndata and actual.dtype == np.float32
    retained = actual.copy();raw.add_(1)
    np.testing.assert_array_equal(actual,retained)


def test_batch_host_bytes_bind_all32_current_rows_and_bounded_layout():
    processor = Pi05BatchProcessor(_Single(),32)
    images, cameras, tokens, masks, host = processor.build_model_inputs(_observations(), return_host_layout=True)
    assert type(host) is BatchHostMaskLayout and isinstance(host,HostMaskLayout)
    expected = torch.cat([x[:,None] for x in cameras]+[masks],1).numpy().tobytes()
    assert host.mask_bytes == expected
    with pytest.raises(ValueError,match='belong'):
        host.validate([x.clone()for x in images],cameras,masks)
    model=SimpleNamespace(optimization_flags=lambda:(True,True,False,True),
        paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=2,patch_size=1)))),
        config=SimpleNamespace(prefix_pack_bucket_size=4),action_horizon=50)
    cache=BatchHostLayoutCache(capacity=1)
    first=cache.get(model,host,images,cameras,masks)
    assert first.cpu_packing['pads'].shape[0]==32 and first.empty_pattern==(False,True)
    assert cache.get(model,host,images,cameras,masks)is first and cache.hits==1
    cache.clear();assert not cache.entries


class _Runner:
    def __init__(self):
        self.calls=0;self.last_call={};self.stats=Counter();self._call_lock=threading.Lock()
    def __call__(self,**kwargs):
        self.calls+=1
        assert kwargs['noise'].shape==(32,50,32) and kwargs['num_steps']==10
        assert all(x.shape[0]==32 for x in kwargs['images'])
        self.last_call=dict(cold=self.calls==1,graph='fake CPU graph',metadata_updated=False)
        self.stats['calls']+=1
        return kwargs['noise']+kwargs['lang_tokens'][:,0,None,None]


class _Policy:
    def __init__(self):
        self.processor=_Single();self.device=torch.device('cpu');self.graph_runner=_Runner()
        self._lock=threading.Lock();self._closed=False
    def close(self):self._closed=True


def test_one_model_call_32_independent_draws_slot_order_and_last_noise_ownership():
    policy=_Policy();engine=batch._wrap_policy(policy,batch._CAPABILITY)
    obs=_observations();seeds=list(range(32));engine.reset_batch([f'episode/{i}'for i in range(32)])
    with pytest.raises(RuntimeError,match='No successful'):
        engine.last_noise()
    result=engine.infer_batch(obs,seeds)
    assert policy.graph_runner.calls==1 and result['actions'].shape==(32,50,7)
    noise=engine.last_noise();expected=np.concatenate([torch.randn((1,50,32),generator=torch.Generator().manual_seed(s)).numpy()for s in seeds])
    np.testing.assert_array_equal(noise,expected);noise.fill(999)
    np.testing.assert_array_equal(engine.last_noise(),expected)
    previous=result['actions'].copy();again=engine.infer_batch(obs,seeds)
    np.testing.assert_array_equal(again['actions'],previous)
    assert policy.graph_runner.calls==2 and not again['cold']
    changed=engine.infer_batch(_observations(100),list(reversed(seeds)))
    assert not np.array_equal(changed['actions'],previous)
    np.testing.assert_array_equal(result['actions'],previous)
    engine.close();engine.close()
    with pytest.raises(RuntimeError,match='closed'):engine.last_noise()


@pytest.mark.parametrize('change',['count','boolean_seed','negative_seed','shape','locked'])
def test_batch_shape_rng_and_exclusion_guards(change):
    policy=_Policy();engine=batch._wrap_policy(policy,batch._CAPABILITY);obs=_observations();seeds=list(range(32))
    if change=='count':obs=obs[:-1]
    if change=='boolean_seed':seeds[3]=True
    if change=='negative_seed':seeds[3]=-1
    if change=='shape':obs[2]['state']=np.ones(7,dtype=np.float32)
    if change=='locked':policy._lock.acquire()
    try:
        with pytest.raises((RuntimeError,ValueError)):
            engine.infer_batch(obs,seeds)
    finally:
        if policy._lock.locked():policy._lock.release()


def test_b32_suffix_pages_keep_each_current_sequence_and_zero_invalid_tail():
    prefix=7;capacity=prefix+50
    k=torch.arange(32*capacity*256,dtype=torch.float32).reshape(32,1,capacity,256).to(torch.bfloat16)
    v=-k;keep=torch.ones((32,capacity),dtype=torch.bool)
    for slot in range(32):keep[slot,slot%prefix:prefix]=False
    metadata=prepare_pi05_static_metadata(torch.ones((32,50),dtype=torch.bool),keep,all_queries_valid=True).extra
    pages,values,lengths=_pack_joint_pages(k[:,:,:prefix],v[:,:,:prefix],k[:,:,prefix:],v[:,:,prefix:],
        metadata['k_indices'],metadata['cu_k'],capacity)
    assert pages.shape==(32,256,1,256)
    for slot in range(32):
        expected=k[slot,0,keep[slot]]
        assert lengths[slot]==expected.shape[0]
        torch.testing.assert_close(pages[slot,:expected.shape[0],0],expected,rtol=0,atol=0)
        torch.testing.assert_close(values[slot,:expected.shape[0],0],-expected,rtol=0,atol=0)
        assert pages[slot,expected.shape[0]:].eq(0).all()


def test_b32_camera_batch_split_preserves_exact_slot_order_and_cached_camera():
    encoded_batches=[]
    def encode(pixels):
        encoded_batches.append(pixels.shape[0])
        return pixels[:,0,0,0,None,None].expand(-1,4,2).clone()
    inner=SimpleNamespace(training=False,paligemma=SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(image_size=2,patch_size=1,num_channels=3)),
        model=SimpleNamespace(vision_tower=SimpleNamespace(training=False))),embed_image=encode,
        embed_language_tokens=lambda tokens:tokens[:,:,None].float().expand(-1,-1,2))
    class Model:
        embed_prefix=Pi05ForActionPrediction.embed_prefix
    model=Model()
    model.training=False;model.paligemma_with_expert=inner;model.config=SimpleNamespace(max_cameras=3);model.vlm_width=2
    model.invalidate_inference_caches=lambda **_:None
    owner=install_batch_vision(model)
    rows=torch.arange(32,dtype=torch.float32)
    images=[rows[:,None,None,None].expand(-1,3,2,2).clone(),(rows+100)[:,None,None,None].expand(-1,3,2,2).clone(),torch.full((32,3,2,2),-1.)]
    masks=[torch.ones(32,dtype=torch.bool),torch.ones(32,dtype=torch.bool),torch.zeros(32,dtype=torch.bool)]
    tokens=torch.ones((32,3),dtype=torch.int64);language=torch.ones_like(tokens,dtype=torch.bool)
    empty=torch.full((32,4,2),-999.)
    with torch.inference_mode():
        embs,pads,attention=model.embed_prefix(images,masks,tokens,language,empty_features=(None,None,empty))
    assert encoded_batches==[64]
    torch.testing.assert_close(embs[:,:4,0],rows[:,None].expand(-1,4),rtol=0,atol=0)
    torch.testing.assert_close(embs[:,4:8,0],(rows+100)[:,None].expand(-1,4),rtol=0,atol=0)
    assert embs[:,8:12].eq(-999).all() and not pads[:,8:12].any() and not attention.any()
    assert owner.manifest()['batch']==32
    owner.uninstall()
    assert getattr(model.embed_prefix,'__func__')is Pi05ForActionPrediction.embed_prefix
