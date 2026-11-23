"""CPU slot geometry and ownership; no CUDA/model performance qualification."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_lite_batch_pi05 import _Single
from vvla_repro_pi05.lite.attention import prepare_pi05_static_metadata
from vvla_repro_pi05.lite.batch_layout import BatchHostMaskLayout,BatchHostLayoutCache,validate_fixed_batch_size
from vvla_repro_pi05.lite.batch_processor import Pi05BatchProcessor
from vvla_repro_pi05.lite.batch_suffix_fa2 import _pack_joint_pages,_validate_call
from vvla_repro_pi05.lite.batch_vision import install_batch_vision
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

BATCHES=(1,2,4,8,16,32)

@pytest.mark.parametrize('batch',BATCHES)
def test_native_processor_owned_actions_and_current_host_rows_for_fixed_batch(batch):
    single=_Single();processor=Pi05BatchProcessor(single,batch)
    observations=[dict(value=i,state=np.arange(8,dtype=np.float32)+i) for i in range(batch)]
    images,cameras,tokens,masks,host=processor.build_model_inputs(observations,return_host_layout=True)
    assert host.batch_size==batch and len(host.cameras)==batch and tokens.shape==(batch,4)
    assert host.mask_bytes==torch.cat([x[:,None] for x in cameras]+[masks],1).numpy().tobytes()
    raw=torch.arange(batch*50*32,dtype=torch.float32).reshape(batch,50,32)/128
    actual=processor.build_model_outputs(raw,observations)
    expected=np.stack([single.build_model_outputs(raw[i:i+1],observations[i]) for i in range(batch)])
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape==(batch,50,7) and actual.flags.owndata and actual.flags.c_contiguous
    saved=actual.copy();raw.add_(1);np.testing.assert_array_equal(actual,saved)
    with pytest.raises(ValueError,match='belong'):
        host.validate([x.clone() for x in images],cameras,masks)
    model=SimpleNamespace(optimization_flags=lambda:(True,True,False,True),
        paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=2,patch_size=1)))),
        config=SimpleNamespace(prefix_pack_bucket_size=4),action_horizon=50)
    cache=BatchHostLayoutCache(capacity=1,batch_size=batch)
    assert cache.get(model,host,images,cameras,masks).cpu_packing['pads'].shape[0]==batch
    with pytest.raises(AttributeError):cache.batch_size=32
    cache._batch_size=2 if batch!=2 else 1
    with pytest.raises(RuntimeError,match='batch changed'):cache.get(model,host,images,cameras,masks)

@pytest.mark.parametrize('batch',BATCHES)
def test_suffix_pages_keep_independent_current_rows_and_all_shape_guards(batch):
    prefix=7;capacity=prefix+50
    k=torch.arange(batch*capacity*256,dtype=torch.float32).reshape(batch,1,capacity,256).to(torch.bfloat16)
    keep=torch.ones((batch,capacity),dtype=torch.bool)
    for slot in range(batch):keep[slot,slot%prefix:prefix]=False
    metadata=prepare_pi05_static_metadata(torch.ones((batch,50),dtype=torch.bool),keep,all_queries_valid=True)
    q=torch.zeros((batch,8,50,256),dtype=torch.bfloat16)
    _validate_call(q,k[:,:,:prefix],-k[:,:,:prefix],k[:,:,prefix:],-k[:,:,prefix:],metadata,batch_size=batch,require_cuda=False)
    pages,values,lengths=_pack_joint_pages(k[:,:,:prefix],-k[:,:,:prefix],k[:,:,prefix:],-k[:,:,prefix:],
        metadata.extra['k_indices'],metadata.extra['cu_k'],capacity,batch_size=batch)
    assert pages.shape==(batch,256,1,256)
    for slot in range(batch):
        expected=k[slot,0,keep[slot]];assert lengths[slot]==expected.shape[0]
        torch.testing.assert_close(pages[slot,:len(expected),0],expected,rtol=0,atol=0)
        torch.testing.assert_close(values[slot,:len(expected),0],-expected,rtol=0,atol=0)
        assert pages[slot,len(expected):].eq(0).all()
    if batch!=32:
        with pytest.raises(ValueError):
            _validate_call(q,k[:,:,:prefix],-k[:,:,:prefix],k[:,:,prefix:],-k[:,:,prefix:],metadata,require_cuda=False)

@pytest.mark.parametrize('batch',BATCHES)
def test_camera_major_groups_split_back_to_exact_fixed_batch_slots(batch):
    calls=[]
    def encode(pixels):
        calls.append(pixels.shape[0]);return pixels[:,0,0,0,None,None].expand(-1,4,2).clone()
    inner=SimpleNamespace(training=False,paligemma=SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(image_size=2,patch_size=1,num_channels=3)),
        model=SimpleNamespace(vision_tower=SimpleNamespace(training=False))),embed_image=encode,
        embed_language_tokens=lambda tokens:tokens[:,:,None].float().expand(-1,-1,2))
    class Model:embed_prefix=Pi05ForActionPrediction.embed_prefix
    model=Model();model.training=False;model.paligemma_with_expert=inner
    model.config=SimpleNamespace(max_cameras=3);model.vlm_width=2;model.invalidate_inference_caches=lambda **_:None
    owner=install_batch_vision(model,batch_size=batch);rows=torch.arange(batch,dtype=torch.float32)
    images=[rows[:,None,None,None].expand(-1,3,2,2).clone(),(rows+100)[:,None,None,None].expand(-1,3,2,2).clone(),torch.full((batch,3,2,2),-1.)]
    masks=[torch.ones(batch,dtype=torch.bool),torch.ones(batch,dtype=torch.bool),torch.zeros(batch,dtype=torch.bool)]
    tokens=torch.ones((batch,3),dtype=torch.int64);language=torch.ones_like(tokens,dtype=torch.bool)
    empty=torch.full((batch,4,2),-999.)
    with torch.inference_mode():embs,pads,attention=model.embed_prefix(images,masks,tokens,language,empty_features=(None,None,empty))
    assert calls==[2*batch] and owner.manifest()['batch']==batch
    torch.testing.assert_close(embs[:,:4,0],rows[:,None].expand(-1,4),rtol=0,atol=0)
    torch.testing.assert_close(embs[:,4:8,0],(rows+100)[:,None].expand(-1,4),rtol=0,atol=0)
    assert embs[:,8:12].eq(-999).all() and not pads[:,8:12].any() and not attention.any()
    with pytest.raises(AttributeError):owner.batch_size=32
    owner._batch_size=2 if batch!=2 else 1
    with torch.inference_mode(),pytest.raises(RuntimeError,match='batch changed'):
        model.embed_prefix(images,masks,tokens,language,empty_features=(None,None,empty))
    owner.uninstall()

@pytest.mark.parametrize('bad',[True,False,0,3,64,1.,np.int64(8),None])
def test_fixed_batch_requires_an_explicit_supported_integer(bad):
    with pytest.raises(ValueError):validate_fixed_batch_size(bad)

def test_default_host_layout_still_rejects_non32_rows():
    processor=Pi05BatchProcessor(_Single(),2)
    images,cameras,tokens,masks=processor.build_model_inputs([dict(value=i,state=np.zeros(8,dtype=np.float32)) for i in range(2)])
    rows=[processor._single.build_model_inputs(dict(value=i),return_host_layout=True)[4] for i in range(2)]
    with pytest.raises(ValueError):BatchHostMaskLayout.from_rows(rows,images,cameras,masks)
