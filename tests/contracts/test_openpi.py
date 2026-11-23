"""Processor semantics only; independent official-weight parity is not implied."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vvla_pi05.models.pi05.config import Pi05Config
from vvla_pi05.openpi import CAMERAS, OpenPiLiberoProcessor, OpenPITokenizer, resized_rgb_uint8


@pytest.mark.parametrize('batch',[1,2])
def test_public_optimized_openpi_recipe_keeps_math_and_uses_its_own_processor(batch):
    from vvla_pi05.recipes import resolve_recipe
    from vvla_pi05.lite.pi05 import _split_lite_overrides,_validate_compilation_options
    selection=resolve_recipe('bf16',capability=(8,9),horizon=10,max_batch=batch,
                             input_contract='openpi-libero-pi05')
    assert selection['resolved']=='bf16-portable'
    _,options=_split_lite_overrides(selection['loader']['config_overrides'])
    _validate_compilation_options(options,True)
    assert options['fused_expert_qkv'] and options['fused_expert_gate_up'] and options['compile_sample_actions']
    assert not any(options[k] for k in ('action_norm_cache','gpu_uint8_prepare','packed_input_metadata'))


def config(**overrides):
    args=dict(input_contract='openpi-libero-pi05',discrete_state_input=False,
              chunk_size=10,n_action_steps=10,image_resolution=(4,4),max_cameras=3,
              image_feature_keys=list(CAMERAS),tokenizer_max_length=12,
              input_features={'observation.state':{'shape':[8]}},output_features={'action':{'shape':[7]}},
              norm_stats={k:dict(q01=[-2]*n,q99=[4]*n) for k,n in [('state',8),('action',7)]})
    args.update(overrides)
    return Pi05Config(**args)


def test_sentencepiece_no_state_uses_bos_and_separate_newline(tmp_path):
    import sentencepiece as sp
    corpus=tmp_path/'text.txt'
    corpus.write_text('pick up the block\nmove to the left\nopen drawer\n')
    prefix=str(tmp_path/'tiny')
    sp.SentencePieceTrainer.train(input=str(corpus),model_prefix=prefix,vocab_size=32,
                                  hard_vocab_limit=False,minloglevel=2)
    tokenizer=OpenPITokenizer(prefix+'.model')
    prompt=' pick_up\nthe block '
    ids,mask=tokenizer.tokens(prompt,None,50)
    expected=tokenizer.model.encode('pick up the block',add_bos=True)+tokenizer.model.encode('\n')
    assert ids[:len(expected)]==expected and ids[len(expected):]==[0]*(50-len(expected))
    assert mask==[True]*len(expected)+[False]*(50-len(expected))
    assert ids[0]==tokenizer.model.bos_id()
    assert tokenizer.tokens(prompt,None,2)==(expected[:2],[True,True])


def test_camera_mask_uint8_resize_quantile_epsilon_and_no_state_shift():
    calls=[]
    def tokens(prompt,state,limit):
        calls.append((prompt,state));return [1]+[0]*(limit-1),[True]+[False]*(limit-1)
    processor=OpenPiLiberoProcessor(config(),SimpleNamespace(tokens=tokens),torch.device('cpu'))
    raw=dict(state=np.zeros(8,np.float32),prompt='pick',images={
        CAMERAS[0]:np.full((2,4,3),255,np.uint8),CAMERAS[1]:np.full((4,4,3),127,np.uint8)})
    images,masks,ids,valid,host=processor.build_model_inputs(raw,return_host_layout=True)
    assert [x.item() for x in masks]==[True,True,False]
    assert host.cameras==(True,True,False) and host.tokens==tuple(valid[0].tolist())
    assert torch.all(images[0][:,:,0]==-1) and torch.all(images[0][:,:,1:3]==1)
    assert torch.all(images[2]==-1)
    changed=dict(raw,state=np.full(8,99,np.float32))
    assert torch.equal(ids,processor.build_model_inputs(changed)[2])
    assert calls==[('pick',None),('pick',None)]
    normalized=torch.zeros((1,10,32));normalized[:,:,0]=1
    actions=processor.build_model_outputs(normalized,raw)
    assert actions.shape==(10,7)
    np.testing.assert_allclose(actions[:,0],4.000001,rtol=0,atol=1e-6)
    np.testing.assert_allclose(actions[:,1:],1.0000005,rtol=0,atol=1e-6)
    np.testing.assert_array_equal(actions,processor.build_model_outputs(normalized,changed))
    normalized.zero_();assert actions[0,0]>4
    assert processor.normalized_state(raw['state']).shape==(32,)
    assert (processor.normalized_state(raw['state'])[8:]==0).all()


@pytest.mark.parametrize('changes',[dict(chunk_size=50,n_action_steps=50),dict(discrete_state_input=True),
                                    dict(image_feature_keys=list(reversed(CAMERAS)))])
def test_incompatible_openpi_contract_is_rejected(changes):
    with pytest.raises(ValueError):OpenPiLiberoProcessor(config(**changes),None,torch.device('cpu'))


def test_no_state_semantics_cannot_silently_reuse_lerobot_processor():
    with pytest.raises(ValueError):Pi05Config(discrete_state_input=False)
    with pytest.raises(ValueError):resized_rgb_uint8(np.full((3,3,3),np.nan),4,4)
