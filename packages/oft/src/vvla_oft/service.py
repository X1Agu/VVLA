"""Per-request preprocessing and routing; no observation is normalized in place."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import torch
from vvla_runtime.execution.requests import RequestLedger
from .adapter import pack_prompts,unnormalize_rows,validate_prompt_capacity

class OFTService:
    def __init__(self,model,processor,utils,runner,*,capacity=128,batch_size=1,prepare_inputs=None,
                 prompt_cache_capacity=128):
        self.model=model;self.processor=processor;self.utils=utils;self.runner=runner
        self.capacity=capacity;self.batch_size=batch_size;self.ledger=RequestLedger()
        self.device=next(model.parameters()).device
        self.prepare_inputs=prepare_inputs
        if type(prompt_cache_capacity) is not int or prompt_cache_capacity < 0:
            raise ValueError('prompt_cache_capacity must be nonnegative')
        self.prompt_cache_capacity=prompt_cache_capacity
        self._prompt_cache=OrderedDict()
        self._text_binding=None
        self.paired_images=None

    def clear_prompt_cache(self):
        self._prompt_cache.clear();self._text_binding=None

    def _split_prismatic(self,prompt,images):
        # Only the explicitly supported Prismatic composition is separable.
        # Arbitrary processors may condition text tokens on image content.
        cls=type(self.processor)
        if (cls.__module__!='prismatic.extern.hf.processing_prismatic' or cls.__name__!='PrismaticProcessor'):
            encoded=[self.processor(prompt,image) for image in images]
            return encoded[0]['input_ids'][0].tolist(),[x['pixel_values'] for x in encoded]
        tokens=self._prompt_tokens(prompt)
        pixels=[self.processor.image_processor(image,return_tensors='pt')['pixel_values'] for image in images]
        return tokens,pixels

    def _prompt_tokens(self,prompt):
        tokenizer=self.processor.tokenizer
        binding=(id(self.processor),id(tokenizer),id(type(tokenizer).__call__),
                 id(type(self.processor).__call__),len(tokenizer),
                 *(getattr(tokenizer,n,None) for n in ('padding_side','truncation_side','bos_token_id',
                    'eos_token_id','pad_token_id','model_max_length','add_bos_token','add_eos_token')))
        if binding!=self._text_binding:
            self.clear_prompt_cache();self._text_binding=binding
        tokens=self._prompt_cache.get(prompt)
        if tokens is None:
            # Exact defaults of the supported PrismaticProcessor.__call__.
            tokens=tuple(tokenizer(prompt,return_tensors='pt',padding=False,truncation=None,max_length=None)['input_ids'][0].tolist())
            validate_prompt_capacity(tokens,self.capacity)
            if self.prompt_cache_capacity:
                self._prompt_cache[prompt]=tokens
                while len(self._prompt_cache)>self.prompt_cache_capacity:self._prompt_cache.popitem(last=False)
        else:self._prompt_cache.move_to_end(prompt)
        return list(tokens)

    def prepare_row(self,obs,key):
        if key not in self.model.norm_stats:raise ValueError('Unknown normalization key')
        prompt="In: What action should the robot take to "+obs['prompt'].lower()+"?\nOut:"
        if self.paired_images is None:
            images=self.utils.prepare_images_for_vla([obs['full_image'],obs['wrist_image']],SimpleNamespace(center_crop=True))
            tokens,encoded_pixels=self._split_prismatic(prompt,images)
            pixels=torch.cat(encoded_pixels,dim=1)[0]
        else:
            tokens=self._prompt_tokens(prompt)
            pixels=self.paired_images(obs['full_image'],obs['wrist_image'])
        # Validate capacity per request before assembling a scheduler wave.
        validate_prompt_capacity(tokens,self.capacity)
        state=np.array(obs['state'],copy=True)
        proprio=self.utils.normalize_proprio(state,self.model.norm_stats[key]['proprio'])
        return tokens,pixels,proprio

    def prepare(self,observations,keys,*,pad=True):
        if not observations or len(observations)!=len(keys) or len(observations)>self.batch_size:
            raise ValueError('invalid request batch')
        return self.collate([self.prepare_row(obs,key) for obs,key in zip(observations,keys)],pad=pad)

    def collate(self,rows,*,pad=True):
        if not rows or len(rows)>self.batch_size:raise ValueError('invalid prepared batch')
        prompts=[list(row[0]) for row in rows];pixels=[row[1] for row in rows];proprio=[row[2] for row in rows]
        # Keep token capacity stable while the graph sees exactly the submitted rows.
        count=len(prompts)
        tensors=pack_prompts(prompts,pad_token_id=self.processor.tokenizer.pad_token_id,capacity=self.capacity if pad else None)
        tensors.update(pixel_values=torch.stack(pixels).to(torch.bfloat16),proprio=torch.as_tensor(np.stack(proprio),dtype=torch.bfloat16))
        if self.prepare_inputs is not None:
            return self.prepare_inputs(tensors),count
        return {k:v.to(self.device) for k,v in tensors.items()},count

    def predict(self,tickets,observations,keys):
        if len(tickets)!=len(observations):
            raise ValueError('ticket count mismatch')
        self.ledger.validate_batch(tickets)
        tensors,count=self.prepare(observations,keys)
        return self.predict_prepared(tickets,tensors,count,keys)

    def predict_prepared(self,tickets,tensors,count,keys):
        if len(tickets)!=count or len(keys)!=count:raise ValueError('prepared routing count mismatch')
        self.ledger.validate_batch(tickets)
        normalized=self.runner(**tensors)[:count].float().cpu().numpy().copy()
        self.ledger.validate_batch(tickets)
        actions=unnormalize_rows(normalized,keys,self.model.norm_stats)
        return {ticket:actions[i].copy() for i,ticket in enumerate(tickets)}
