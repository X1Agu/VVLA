from types import SimpleNamespace
import numpy as np
import torch

from vvla_oft.service import OFTService


def test_supported_processor_caches_text_but_prepares_each_current_image():
    class Tokenizer:
        pad_token_id=0
        def __init__(self):self.calls=0
        def __len__(self):return 100
        def __call__(self,prompt,**kwargs):
            assert kwargs==dict(return_tensors='pt',padding=False,truncation=None,max_length=None)
            self.calls+=1
            return {'input_ids':torch.tensor([[1,2,29871]])}
    class PrismaticProcessor:
        __module__='prismatic.extern.hf.processing_prismatic'
        def __init__(self):self.tokenizer=Tokenizer();self.images=[]
        def __call__(self,*args):raise AssertionError('Split contract should not re-tokenize per image')
        def image_processor(self,image,**kwargs):
            self.images.append(np.array(image,copy=True))
            return {'pixel_values':torch.from_numpy(image.copy()).permute(2,0,1)[None].float()}
    processor=PrismaticProcessor()
    model=torch.nn.Linear(1,1);model.norm_stats={'a':{'proprio':{}}}
    utils=SimpleNamespace(prepare_images_for_vla=lambda values,cfg:values,normalize_proprio=lambda state,stats:state)
    service=OFTService(model,processor,utils,None,prompt_cache_capacity=1)
    obs=dict(full_image=np.ones((4,4,3),np.uint8),wrist_image=np.zeros((4,4,3),np.uint8),
             prompt='Pick cube',state=np.zeros(8,np.float32))
    first=service.prepare_row(obs,'a')
    obs['full_image'][:]=2
    second=service.prepare_row(obs,'a')
    assert processor.tokenizer.calls==1 and len(processor.images)==4
    assert not torch.equal(first[1],second[1]) and first[0]==second[0]
    processor.tokenizer=Tokenizer()
    service.prepare_row(obs,'a')
    assert processor.tokenizer.calls==1
    obs['prompt']='Place cube';service.prepare_row(obs,'a')
    assert len(service._prompt_cache)==1

