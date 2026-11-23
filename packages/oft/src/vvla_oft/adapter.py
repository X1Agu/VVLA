"""Batched LIBERO L1 adapter, preserving the official shifted action readout.

Only this adapter knows about 8 x 7 action geometry and bidirectional masks.
Padding follows each row's stop token; it never moves valid prompt positions.
"""
import numpy as np
import torch


def normalized_prompt(row):
    row=list(row)
    if not row or any(type(x) is not int or x < 0 for x in row):
        raise ValueError('invalid prompt token ids')
    if row[-1] != 29871:row.append(29871)
    return row


def validate_prompt_capacity(row, capacity, action_tokens=56):
    if type(capacity) is not int or capacity < len(normalized_prompt(row))+action_tokens+1:
        raise ValueError('prompt exceeds capacity; truncation is not allowed')


def pack_prompts(prompts, *, pad_token_id, action_tokens=56, capacity=None):
    if not prompts or type(action_tokens) is not int or action_tokens < 1:
        raise ValueError('nonempty prompts and positive action token count required')
    rows=[]
    for row in prompts:
        row=normalized_prompt(row)
        rows.append(row)
    needed=max(len(row)+action_tokens+1 for row in rows)
    capacity=needed if capacity is None else capacity
    if type(capacity) is not int or capacity < needed:
        raise ValueError('prompt exceeds capacity; truncation is not allowed')
    ids=torch.full((len(rows),capacity),pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    actions=torch.zeros_like(mask)
    starts=[]
    for i,row in enumerate(rows):
        n=len(row)
        ids[i,:n]=torch.tensor(row)
        ids[i,n:n+action_tokens]=1
        ids[i,n+action_tokens]=2
        mask[i,:n+action_tokens+1]=True
        actions[i,n:n+action_tokens]=True
        # Official head reads the last prompt position, then 55 action positions.
        starts.append(n-1)
    return dict(input_ids=ids,attention_mask=mask,action_mask=actions,
                action_starts=torch.tensor(starts,dtype=torch.long))


class OFTAdapter(torch.nn.Module):
    def __init__(self, model, action_head, proprio_projector, *, execution='shared'):
        super().__init__()
        if execution not in ('official_batch','shared'):
            raise ValueError('unsupported execution path')
        self.model=model
        self.action_head=action_head
        self.proprio_projector=proprio_projector
        self.execution=execution
        if model.language_model.config._attn_implementation != 'sdpa':
            raise ValueError('pinned OFT fork requires SDPA for bidirectional attention')
        self.action_tokens=56

    def execution_signature(self):
        return dict(execution=self.execution,action_tokens=self.action_tokens)

    def forward(self, input_ids, attention_mask, action_mask, action_starts, pixel_values, proprio):
        embeddings=self.model.get_input_embeddings()(input_ids)
        projected=self.model._process_vision_features(pixel_values,None,False)
        projected=self.model._process_proprio_features(projected,proprio,self.proprio_projector)
        embeddings=embeddings * ~action_mask.unsqueeze(-1)
        hidden,valid=self.model._build_multimodal_attention(embeddings,projected,attention_mask)
        if self.execution == 'official_batch':
            # Official model forward, including discarded vocabulary logits and
            # layer states. Only batching and per-row readout are generalized.
            result=self.model.language_model(inputs_embeds=hidden,attention_mask=valid,
                input_ids=None,position_ids=None,past_key_values=None,labels=None,
                use_cache=None,output_attentions=False,output_hidden_states=True,return_dict=True)
            hidden=result.hidden_states[-1]
        else:
            # Avoid the fork's data-dependent host check in _update_causal_mask.
            # Each attention layer replaces the input mask with its last row.
            # That row is precisely this additive key-padding mask. Every valid
            # query can attend to every valid key, including future action tokens.
            size=hidden.shape[1]
            key_mask=torch.zeros_like(valid,dtype=hidden.dtype).masked_fill(~valid,torch.finfo(hidden.dtype).min)
            mask=key_mask[:,None,None,:].expand(-1,1,size,-1)
            positions=torch.arange(size,device=hidden.device)
            llm=self.model.language_model.model
            for layer in llm.layers:
                hidden=layer(hidden,attention_mask=mask,position_ids=positions[None,:],
                             past_key_value=None,output_attentions=False,use_cache=False,
                             cache_position=positions)[0]
            hidden=llm.norm(hidden)
        indices=projected.shape[1]+action_starts[:,None]+torch.arange(self.action_tokens,device=hidden.device)[None,:]
        selected=hidden.gather(1,indices[:,:,None].expand(-1,-1,hidden.shape[-1]))
        return self.action_head.predict_action(selected)


def unnormalize_rows(normalized, keys, stats):
    if len(keys)!=len(normalized):
        raise ValueError('one normalization key per real request required')
    outputs=[]
    for actions,key in zip(normalized,keys):
        s=stats[key]['action']
        lo,hi=np.array(s['q01']),np.array(s['q99'])
        mask=np.asarray(s.get('mask',np.ones_like(lo,dtype=bool)))
        outputs.append(np.where(mask,0.5*(actions+1)*(hi-lo+1e-8)+lo,actions))
    return np.stack(outputs)
