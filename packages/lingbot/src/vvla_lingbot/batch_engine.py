"""True batched execution: ragged packed prefixes and disjoint per-row paged KV.
B1 Engine and the frozen B1 evidence remain unchanged. Every row retains its
own prefix mask, multimodal positions, action-state attention and MoE routes.
"""
import hashlib,json,math
import torch
from .engine import Engine,CachedAdaNorm,native

class BatchedEngine(Engine):
    def _validate_batch_inputs(self,images,img_masks,tokens,lang_masks,state,grid):
        if state.shape[0] not in (1,8) or tokens.shape[0]!=state.shape[0] or not img_masks.all().item():
            raise ValueError('Legacy batch engine requires B1/B8 with all cameras; use the explicit true-batch candidate for other sizes')

    def _norm(self,norm,x,step,squared=None):
        if isinstance(norm,CachedAdaNorm):
            # All rows share the same diffusion timestep; each token reduction
            # is independent. Flatten only the leading batch/token dimensions.
            shape=x.shape
            sq=None if squared is None else squared.reshape(1,-1,shape[-1])
            return norm(x.reshape(1,-1,shape[-1]),step,sq).reshape(shape)
        return norm(x,squared=squared)

    @torch.no_grad()
    def _prepare_prefix(self,images,img_masks,tokens,lang_masks,state,grid,*,refresh_vision=True):
        from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks,block_suffix_to_fv_
        B=state.shape[0]
        self._validate_batch_inputs(images,img_masks,tokens,lang_masks,state,grid)
        self.batch=B
        special=[self.qwe.qwenvl.config.image_token_id,self.qwe.qwenvl.config.vision_start_token_id,self.qwe.qwenvl.config.vision_end_token_id]
        if any((tokens==v).any().item() for v in special):raise ValueError('Language contains special vision markers')
        if refresh_vision:
            for name in ['pos_embeds','position_embeddings','cu_seqlens','visual_split_sizes','visual_max_seqlen']:setattr(self.qwe,name,None)
        prefix,pad,ar,pos,visual,deep=self.flow.embed_prefix(images,img_masks,tokens,lang_masks,image_grid_thw=grid)
        if refresh_vision:
            self.qwe.pos_embeds=self.qwe.qwenvl.visual.fast_pos_embed_interpolate(grid.reshape(-1,3))
        self.prefix_pad=pad;self.prefix_pos=pos
        L=prefix.shape[1];device=state.device
        row_indices=[pad[b].nonzero().flatten() for b in range(B)]
        self.ps=[len(x) for x in row_indices];self.p=sum(self.ps);self.max_p=max(self.ps)
        self.prefix_indices=torch.cat([idx+b*L for b,idx in enumerate(row_indices)])
        self.num_images=img_masks.shape[1];self.patch_count=deep[0].numel()//deep[0].shape[-1]//(B*self.num_images)
        self.boundaries=bool(getattr(self.flow.config,'qwen3vl_use_vision_boundaries',True))
        self.image_token_len=self.patch_count+(2 if self.boundaries else 0)
        self.image_end=self.num_images*self.image_token_len;self.language_end=self.image_end+tokens.shape[1]
        assert L>=self.language_end
        self.visual_indices=visual.reshape(-1).index_select(0,self.prefix_indices).nonzero().flatten()
        assert self.visual_indices.numel()==B*self.num_images*self.patch_count
        self.prefix_causal=bool(getattr(self.flow.config,'vlm_causal',False))
        official_mask=make_att_2d_masks(pad,ar)
        for b,idx in enumerate(row_indices):
            expected=torch.ones((len(idx),len(idx)),dtype=torch.bool,device=device)
            if self.prefix_causal:expected=expected.tril()
            assert torch.equal(official_mask[b].index_select(0,idx).index_select(1,idx),expected),'Unsupported prefix mask'
        self.h=self.flow.config.n_action_steps;self.s=self.h+1
        spad=torch.ones((B,self.s),dtype=torch.bool,device=device);sar=torch.zeros_like(spad);sar[:,:2]=True
        self.suffix_pos=self.flow._build_full_position_ids(pos,pad,spad)[:,:,-self.s:]
        smask=torch.cat([pad[:,None,:].expand(B,self.s,L),make_att_2d_masks(spad,sar)],-1)
        if self.flow.block_future_depth_to_action:smask=block_suffix_to_fv_(smask,0,L,self.flow.num_task_tokens)
        smask=self.flow._block_suffix_to_future_video_if_enabled_(smask,suffix_row_start=0,prefix_len=L)
        indices=[];self.kps=[];offset=0
        for b,idx in enumerate(row_indices):
            allowed=smask[b,0,:L].index_select(0,idx)
            assert torch.equal(smask[b,:,:L].index_select(1,idx),allowed[None,:].expand(self.s,-1))
            allowed_idx=allowed.nonzero().flatten();self.kps.append(len(allowed_idx));indices.append(allowed_idx+offset);offset+=len(idx)
        assert torch.equal(smask[:,:,L:],make_att_2d_masks(spad,sar))
        self.cache_prefix_indices=torch.cat(indices)
        dummy=torch.empty((B,L,1,self.qwe.config.action_head_dim),device=device,dtype=torch.float32)
        prefix_rope=self.vlm.rotary_emb(dummy,pos)
        assert all(x.shape[:2]==(B,L) for x in prefix_rope)
        self.rope={'prefix':tuple(x.reshape(B*L,-1).index_select(0,self.prefix_indices).unsqueeze(0).contiguous() for x in prefix_rope)}
        dummy=torch.empty((B,self.s,1,self.qwe.config.action_head_dim),device=device,dtype=torch.float32)
        suffix_rope=tuple(x.contiguous() for x in self.vlm.rotary_emb(dummy,self.suffix_pos))
        self.prefix_cu=torch.tensor([0,*__import__('itertools').accumulate(self.ps)],device=device,dtype=torch.int32)
        return prefix, deep, suffix_rope

    def _allocate_pages(self, capacity, device):
        self.kp = capacity  # FA2 launch bound. seqused_k supplies each row's live length.
        self.page_size = 16
        blocks = math.ceil((capacity + self.s) / self.page_size)
        kv_heads = self.qwe.qwenvl.config.text_config.num_key_value_heads
        head_dim = self.vlm.layers[0].self_attn.head_dim
        shape = (len(self.vlm.layers), self.batch * blocks, self.page_size, kv_heads, head_dim)
        dtype = torch.float16 if self.fp16_expert else torch.bfloat16
        self.kcache = torch.empty(shape, device=device, dtype=dtype)
        self.vcache = torch.empty_like(self.kcache)
        self.block_table = torch.arange(self.batch*blocks, device=device, dtype=torch.int32).view(self.batch, blocks)
        self.scale = torch.tensor(1., device=device)
        self.cu = {n: torch.arange(self.batch+1, device=device, dtype=torch.int32)*n for n in (1, self.h)}

    def _page_inputs(self, device):
        stride = self.block_table.shape[1] * self.page_size
        starts = torch.arange(self.batch, device=device, dtype=torch.int64) * stride
        lengths = torch.tensor(self.kps, device=device, dtype=torch.int32)
        self.prefix_slots = torch.cat([torch.arange(n, device=device) + b*stride for b,n in enumerate(self.kps)])
        suffix = (starts + lengths)[:, None] + torch.arange(self.s, device=device)
        return suffix.flatten(), {n: lengths+n for n in (1, self.s)}

    @torch.no_grad()
    def _prepare(self,images,img_masks,tokens,lang_masks,state,grid):
        prefix, _, suffix_rope = self._prepare_prefix(images,img_masks,tokens,lang_masks,state,grid)
        B, L = prefix.shape[:2]
        device = state.device
        self.template = prefix.detach().clone()
        self.rope['suffix'] = suffix_rope
        self._allocate_pages(max(self.kps), device)
        self.suffix_slots, self.lengths = self._page_inputs(device)
        self.static=[v.clone() for v in [images,img_masks,tokens,lang_masks,state,grid]]
        self.noise=torch.empty((B,self.h,self.flow.config.max_action_dim),device=device,dtype=torch.bfloat16)
        self.recipe.update(batch_size=B,prefix_valid_per_row=self.ps,prefix_valid=self.p,prefix_capacity=L,suffix_tokens=self.s,kv_prefix_tokens_per_row=self.kps,kv_page_size=self.page_size,image_grid_thw=grid.cpu().tolist(),parameter_count=sum(p.numel() for p in self.flow.parameters()),row_isolation='varlen FA2 prefix and disjoint paged suffix KV; official per-row attention masks asserted')
        self._signature=self._state_signature()
        self._constant_tensors=[self.template,self.prefix_indices,self.cache_prefix_indices,self.visual_indices,self.dt,*self.time_embs,*[x for pair in self.rope.values() for x in pair],self.qwe.pos_embeds,*self.qwe.position_embeddings,self.qwe.cu_seqlens,self.prefix_cu,self.block_table,self.prefix_slots,self.suffix_slots,*self.cu.values(),*self.lengths.values()]
        for model in [self.vlm,self.action]:
            for module in model.modules():
                if isinstance(module,CachedAdaNorm):self._constant_tensors.extend(x for pair in module.modulations for x in pair)
        self._constant_token=tuple((id(t),t.data_ptr(),t._version) for t in self._constant_tensors)

    def _attention(self,q,k,v,layer,prefix):
        B=self.batch
        if prefix:
            q=q.squeeze(0).contiguous();k=k.squeeze(0).contiguous();v=v.squeeze(0).contiguous()
            native.cache_write(k.index_select(0,self.cache_prefix_indices).to(self.kcache.dtype),v.index_select(0,self.cache_prefix_indices).to(self.vcache.dtype),self.kcache[layer],self.vcache[layer],self.prefix_slots,self.scale)
            out=native._flash(q,k,v,cu_seqlens_q=self.prefix_cu,cu_seqlens_k=self.prefix_cu,max_seqlen_q=self.max_p,max_seqlen_k=self.max_p,causal=self.prefix_causal,softmax_scale=q.shape[-1]**-.5,fa_version=2)
            return out.reshape(1,self.p,-1)
        native.cache_write(k.reshape(-1,*k.shape[-2:]).contiguous(),v.reshape(-1,*v.shape[-2:]).contiguous(),self.kcache[layer],self.vcache[layer],self.suffix_slots,self.scale)
        parts=[]
        for qpart,n,kextra in [(q[:,:1],1,1),(q[:,1:],self.h,self.s)]:
            qp=qpart.reshape(-1,*q.shape[-2:]).contiguous()
            out=native._flash(qp,self.kcache[layer],self.vcache[layer],cu_seqlens_q=self.cu[n],cu_seqlens_k=None,max_seqlen_q=n,max_seqlen_k=self.kp+kextra,seqused_k=self.lengths[kextra],block_table=self.block_table,causal=False,softmax_scale=q.shape[-1]**-.5,fa_version=2)
            parts.append(out.reshape(B,n,-1))
        return torch.cat(parts,1)

    def _execute(self):
        images,im,tokens,lm,state,grid=self.static;B=self.batch
        image,deep=self.qwe.embed_image(images.flatten(0,1),grid.reshape(-1,3))
        image=image.reshape(B,self.num_images,self.patch_count,-1);prefix=self.template.clone()
        if self.boundaries:prefix[:,:self.image_end].view(B,self.num_images,self.image_token_len,-1)[:,:,1:-1]=image
        else:prefix[:,:self.image_end]=image.flatten(1,2)
        prefix[:,self.image_end:self.language_end]=self.qwe.embed_language_tokens(tokens)
        self._run_prefix(prefix, deep)
        return self._action_loop(state)

    def _run_prefix(self, prefix, deep):
        packed = prefix.reshape(-1, prefix.shape[-1]).index_select(0, self.prefix_indices).unsqueeze(0)
        self._layers(packed, True, [v.reshape(-1, v.shape[-1]) for v in deep])

    def route_audit(self):
        result=[]
        for i,layer in enumerate(self.action.layers):
            ws=layer.mlp.experts._robby_moe_workspace;assert ws is not None
            counts=ws['counts'].cpu();rows=ws['rows'].cpu();slots=ws['slots'].cpu();routes=[]
            for e,n in enumerate(counts.tolist()):
                assert 0<=n<=self.batch*self.s*layer.mlp.top_k
                routes.extend((int(rows[e,j]),int(slots[e,j]),e) for j in range(n))
            expected={(r,s) for r in range(self.batch*self.s) for s in range(layer.mlp.top_k)}
            assert {(r,s) for r,s,e in routes}==expected and len(routes)==len(expected),'Dropped/duplicated MoE route'
            result.append(dict(layer=i,counts=counts.tolist(),routes=len(routes),route_sha256=hashlib.sha256(json.dumps(sorted(routes)).encode()).hexdigest()))
        return result
