"""OFT native execution adapter: original Llama mathematics, optional backends.

All valid tokens remain mutually visible. Packing only removes masked positions;
the original position IDs and shifted 56-position action readout are preserved.
Metadata is prepared outside capture, copied into each graph's owned inputs.
"""
import math
from collections import OrderedDict
import torch
from torch.nn import functional as F
from .adapter import OFTAdapter
from .recipe import OFTExecutionRecipe
from vvla_runtime.execution.state import tensor_signature
from vvla_runtime.execution import native

def penultimate_vision_patches(tower, images):
    """The pinned Prismatic towers request block -2 with norm=False."""
    x = tower.norm_pre(tower.patch_drop(tower._pos_embed(tower.patch_embed(images))))
    for index, block in enumerate(tower.blocks):
        if index == len(tower.blocks) - 1:
            break
        x = block(x)
    return x[:, tower.num_prefix_tokens:]

def batched_camera_features(backbone, pixel_values, *, penultimate=False, primary_op=None, secondary_op=None, side_stream=None):
    """Run both unfused camera views as independent batch items in each vision tower."""
    batch, channels, height, width = pixel_values.shape
    views = backbone.get_num_images_in_input()
    if not backbone.use_fused_vision_backbone or channels != 6 * views:
        raise ValueError("Batched camera views require two RGB normalizations per view")
    pixels = pixel_values.reshape(batch * views, 6, height, width)
    if penultimate:
        primary_op = primary_op or penultimate_vision_patches
        secondary_op = secondary_op or penultimate_vision_patches
        if side_stream is None:
            primary = primary_op(backbone.featurizer, pixels[:, :3])
            secondary = secondary_op(backbone.fused_featurizer, pixels[:, 3:])
        else:
            main = torch.cuda.current_stream(pixel_values.device)
            side_stream.wait_stream(main)
            with torch.cuda.stream(side_stream):
                secondary = secondary_op(backbone.fused_featurizer,pixels[:,3:])
            primary = primary_op(backbone.featurizer,pixels[:,:3])
            main.wait_stream(side_stream)
            secondary.record_stream(main)
    else:
        primary = backbone.featurizer(pixels[:, :3])
        secondary = backbone.fused_featurizer(pixels[:, 3:])
    features = torch.cat((primary, secondary), dim=-1)
    return features.reshape(batch, views * features.shape[1], features.shape[2])

class OFTNativeAdapter(OFTAdapter):
    def __init__(self,model,action_head,proprio_projector,recipe=None,calibration=None):
        super().__init__(model,action_head,proprio_projector,execution='shared')
        self.recipe=recipe or OFTExecutionRecipe()
        self._camera_features = batched_camera_features
        self._primary_op = self._secondary_op = self._vision_stream = None
        if self.recipe.parallel_vision:
            self._vision_stream = torch.cuda.Stream(device=next(model.vision_backbone.parameters()).device)
        if self.recipe.vision_compile != 'off':
            from vvla_runtime.gemm import compile_bf16_function
            if self.recipe.vision_compile_scope == 'both' and not self.recipe.parallel_vision:
                self._camera_features = compile_bf16_function(batched_camera_features,self.recipe.vision_compile)
            else:
                op = compile_bf16_function(penultimate_vision_patches,self.recipe.vision_compile)
                if self.recipe.vision_compile_scope in ('both','primary'):
                    self._primary_op = op
                if self.recipe.vision_compile_scope in ('both','secondary'):
                    self._secondary_op = op
        cfg=model.language_model.config
        if cfg.pretraining_tp!=1 or cfg.rope_scaling is not None:raise ValueError('Only pinned unscaled Llama OFT layout is supported')
        self.projected_count=model.vision_backbone.get_num_patches()*model.vision_backbone.get_num_images_in_input()+1
        self._rotary_cache={}
        self._layout_cache=OrderedDict()
        self._rope_basis=self._rope_signature()
        rotations=[l.self_attn.rotary_emb for l in model.language_model.model.layers]
        if any(type(x)!=type(rotations[0]) or not torch.equal(x.inv_freq,rotations[0].inv_freq) for x in rotations):raise ValueError('OFT shared rotary precompute requires identical layer frequencies')
        from vvla_runtime.precision import install_mlp_recipe
        install_mlp_recipe(self,calibration)
        from vvla_runtime.gemm import install_attention_layout
        install_attention_layout(self)
        self._norm_flags=(self.recipe.fuse_rmsnorm,self.recipe.fuse_residual_square)
        if any(self._norm_flags):
            from vvla_runtime.execution.llama_norm import LlamaRMSNormExecution
            owners=[(layer,name) for layer in model.language_model.model.layers for name in ('input_layernorm','post_attention_layernorm')]
            owners.append((model.language_model.model,'norm'))
            for owner,name in owners:
                module=LlamaRMSNormExecution(getattr(owner,name),fuse_after_mean=self.recipe.fuse_rmsnorm)
                setattr(owner,name,module)
                self._execution_modules.append(module)
        if self.recipe.attention!='sdpa':native.load_native()

    def _rope_signature(self):
        return tuple((tensor_signature(l.self_attn.rotary_emb.inv_freq),l.self_attn.rotary_emb.base,l.self_attn.rotary_emb.dim)
                     for l in self.model.language_model.model.layers)

    def execution_signature(self):
        return dict(recipe=self.recipe.asdict(),execution=self.execution,projected_count=self.projected_count,action_tokens=self.action_tokens,norm_epsilons=tuple(n.variance_epsilon for n in self._norm_modules()),blas_backend=str(torch.backends.cuda.preferred_blas_library()),bf16_reduced_precision=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))

    def _norm_modules(self):
        return [getattr(layer,name) for layer in self.model.language_model.model.layers for name in ('input_layernorm','post_attention_layernorm')]+[self.model.language_model.model.norm]

    def validate_execution_state(self):
        from vvla_runtime.gemm import validate_attention_layout
        validate_attention_layout(self)
        if (self.recipe.fuse_rmsnorm,self.recipe.fuse_residual_square)!=self._norm_flags:
            raise RuntimeError('Norm recipe changed; rebuild adapter and retire graphs')
        from vvla_runtime.precision import validate_mlp_recipe
        validate_mlp_recipe(self)
        if self.execution!='shared':raise ValueError('Native adapter requires shared execution')
        if self.recipe.precompute_positions and self._rope_signature()!=self._rope_basis:
            raise RuntimeError('Rotary basis changed; rebuild adapter and retire its graphs')

    @torch.no_grad()
    def prepare_inputs(self,inputs,*,device=None,validate=True):
        return self._prepare_inputs(inputs, device=device, validate=validate, reuse_layout=False)

    @torch.no_grad()
    def _prepare_inputs(self,inputs,*,device=None,validate=True,reuse_layout=False):
        # Serving calls this on CPU tensors before H2D, then the runner validates
        # the model immediately before capture/replay. Standalone callers retain
        # validation by default. No request content or model output is cached.
        if validate:self.validate_execution_state()
        if self.recipe.attention!='sdpa':native.load_native()
        mask=inputs['attention_mask'].detach().cpu()
        starts=inputs['action_starts'].detach().cpu()
        b,t=mask.shape;p=self.projected_count;size=t+p
        device=inputs['input_ids'].device if device is None else torch.device(device)
        layout_key = None
        if reuse_layout:
            # The serving adapter owns these masks, positions and page tables. Pixel,
            # proprio, text-token and action values are transferred for every request.
            layout_key = (self.recipe, p, self.action_tokens, b, t, device, inputs['pixel_values'].dtype,
                          mask.dtype, starts.dtype, mask.numpy().tobytes(), starts.numpy().tobytes())
            cached = self._layout_cache.get(layout_key)
            if cached is not None:
                extra, source_stream, ready = cached
                if ready is not None:
                    current = torch.cuda.current_stream(device)
                    if current != source_stream:
                        current.wait_event(ready)
                        for tensor in extra.values():
                            tensor.record_stream(current)
                self._layout_cache.move_to_end(layout_key)
                return {**{k:v.to(device,non_blocking=True) for k,v in inputs.items()}, **extra}
        valid=torch.cat((mask[:,:1],torch.ones((b,p),dtype=torch.bool),mask[:,1:]),1)
        if not bool(valid.any(1).all()):raise ValueError('Every request needs valid keys')
        original_head=p+starts[:,None]+torch.arange(self.action_tokens)[None,:]
        if not bool(valid.gather(1,original_head).all()):raise ValueError('Action readout cannot target masked positions')
        if self.recipe.pack_sequence:
            order=torch.argsort((~valid).to(torch.int32),dim=1,stable=True)
            capacity=min(size,math.ceil(int(valid.sum(1).max())/self.recipe.pack_bucket)*self.recipe.pack_bucket)
            indices=order[:,:capacity].contiguous()
            head=torch.argsort(order,dim=1).gather(1,original_head)
        else:
            capacity=size;indices=torch.arange(size)[None,:].expand(b,-1).contiguous();head=original_head
        packed_valid=valid.gather(1,indices)
        if self.recipe.reference_compatible and (b!=1 or not bool(packed_valid.all())):
            raise ValueError('Reference-compatible SDPA requires one unpadded request')
        lengths=packed_valid.sum(1,dtype=torch.int32)
        cu_k=torch.cat((torch.zeros(1,dtype=torch.int32),lengths.cumsum(0,dtype=torch.int32)))
        cu_q=torch.arange(b+1,dtype=torch.int32)*capacity
        k_indices=torch.argsort((~packed_valid).flatten().to(torch.int32),stable=True)
        pages=math.ceil(capacity/self.recipe.page_size)
        table=torch.arange(1,b*pages+1,dtype=torch.int32).reshape(pages,b).t().contiguous()
        rank=(packed_valid.cumsum(1)-1).clamp_min(0)
        slots=table.gather(1,rank//self.recipe.page_size).long()*self.recipe.page_size+rank%self.recipe.page_size
        slots=torch.where(packed_valid,slots,-1).flatten()
        extra=dict(pack_indices=indices,position_ids=indices,packed_valid=packed_valid,cu_q=cu_q,cu_k=cu_k,k_indices=k_indices,head_indices=head,block_table=table,slot_mapping=slots,lengths=lengths)
        extra={k:v.to(device) for k,v in extra.items()}
        if self.recipe.precompute_positions:
            key=(size,device,inputs['pixel_values'].dtype)
            if key not in self._rotary_cache:
                rotary=self.model.language_model.model.layers[0].self_attn.rotary_emb
                probe=torch.empty((1,1,size,rotary.dim),device=device,dtype=key[2])
                self._rotary_cache[key]=rotary(probe,torch.arange(size,device=device)[None,:])
            cos,sin=self._rotary_cache[key]
            gather=extra['position_ids'][:,:,None].expand(-1,-1,cos.shape[-1])
            extra['rope_cos']=cos.expand(b,-1,-1).gather(1,gather)
            extra['rope_sin']=sin.expand(b,-1,-1).gather(1,gather)
        else:
            extra['rope_cos']=torch.empty(0,device=device,dtype=inputs['pixel_values'].dtype)
            extra['rope_sin']=torch.empty_like(extra['rope_cos'])
        if layout_key is not None:
            stream = ready = None
            if device.type == 'cuda':
                stream = torch.cuda.current_stream(device)
                ready = torch.cuda.Event()
                ready.record(stream)
            self._layout_cache[layout_key] = extra, stream, ready
            if len(self._layout_cache) > 32:
                self._layout_cache.popitem(last=False)
        return {**{k:v.to(device) for k,v in inputs.items()},**extra}

    def forward(self,input_ids,attention_mask,action_mask,action_starts,pixel_values,proprio,pack_indices,position_ids,packed_valid,cu_q,cu_k,k_indices,head_indices,rope_cos,rope_sin,block_table,slot_mapping,lengths):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
        embeddings=self.model.get_input_embeddings()(input_ids)
        if self.recipe.batch_camera_views:
            projected = self.model.projector(self._camera_features(
                self.model.vision_backbone, pixel_values, penultimate=self.recipe.skip_unused_vision_block,
                primary_op=self._primary_op, secondary_op=self._secondary_op, side_stream=self._vision_stream))
        else:
            projected = self.model._process_vision_features(pixel_values, None, False)
        projected=self.model._process_proprio_features(projected,proprio,self.proprio_projector)
        if projected.shape[1]!=self.projected_count:raise ValueError('Visual/proprio token geometry changed')
        embeddings=embeddings*~action_mask.unsqueeze(-1)
        hidden,_=self.model._build_multimodal_attention(embeddings,projected,attention_mask)
        if self.recipe.pack_sequence:
            hidden=hidden.gather(1,pack_indices[:,:,None].expand(-1,-1,hidden.shape[-1]))
            hidden=torch.where(packed_valid[:,:,None],hidden,0)
        b,s,width=hidden.shape
        layers=self.model.language_model.model.layers
        mask=torch.zeros_like(packed_valid,dtype=hidden.dtype).masked_fill(~packed_valid,torch.finfo(hidden.dtype).min)[:,None,None,:].expand(-1,1,s,-1)
        # The pinned upstream single-observation API omits the all-valid mask.
        # A zero mask is mathematically equivalent but changes SDPA dispatch and
        # BF16 rounding. Validation occurs before forward/capture in prepare_inputs.
        if self.recipe.reference_compatible:
            mask=None
        cache=None
        if self.recipe.attention=='vllm_paged_fa2':
            attn=layers[0].self_attn;pages=block_table.shape[1]*b+1
            shape=(pages,self.recipe.page_size,attn.num_key_value_heads,attn.head_dim)
            cache=(torch.empty(shape,device=hidden.device,dtype=hidden.dtype),torch.empty(shape,device=hidden.device,dtype=hidden.dtype))
            cache_scale=torch.ones((),device=hidden.device,dtype=torch.float32)
        squared=None
        def norm(module,value,square):
            return module(value,squared=square) if any(self._norm_flags) else module(value)
        def add(residual,update):
            if self.recipe.fuse_residual_square:
                from vvla_runtime.execution.llama_norm import residual_square_bf16
                return residual_square_bf16(residual,update)
            return residual+update,None
        pending = None
        if self.recipe.fuse_residual_rms:
            from vvla_runtime.execution.llama_norm import residual_rmsnorm_bf16
        for layer_index, layer in enumerate(layers):
            if self.recipe.fuse_residual_rms and pending is not None:
                residual, x = residual_rmsnorm_bf16(hidden, pending, layer.input_layernorm)
            else:
                residual = hidden
                x = norm(layer.input_layernorm, hidden, squared)
            a=layer.self_attn;d=a.head_dim
            if self.recipe.fuse_qkv:q,k,v=a.qkv_proj(x)
            else:q,k,v=a.q_proj(x),a.k_proj(x),a.v_proj(x)
            q=q.view(b,s,a.num_heads,d).transpose(1,2)
            k=k.view(b,s,a.num_key_value_heads,d).transpose(1,2)
            v=v.view(b,s,a.num_key_value_heads,d).transpose(1,2)
            cos,sin=(rope_cos,rope_sin) if self.recipe.precompute_positions else a.rotary_emb(v,position_ids)
            if self.recipe.fuse_rope:
                from vvla_runtime.execution.rotary import rotary_pos_emb
                q,k=rotary_pos_emb(q,k,cos,sin)
            else:q,k=apply_rotary_pos_emb(q,k,cos,sin)
            if self.recipe.attention=='sdpa':
                k=repeat_kv(k,a.num_key_value_groups);v=repeat_kv(v,a.num_key_value_groups)
                if mask is not None:
                    q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
                out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False).transpose(1,2).contiguous()
            else:
                q=q.transpose(1,2).reshape(b*s,a.num_heads,d)
                k=k.transpose(1,2).reshape(b*s,a.num_key_value_heads,d)
                v=v.transpose(1,2).reshape(b*s,a.num_key_value_heads,d)
                if cache is None:
                    # A single packed row has all valid keys first; cu_k ends the row.
                    if not (self.recipe.packed_single_kv and b == 1):
                        k, v = k.index_select(0, k_indices), v.index_select(0, k_indices)
                    out=native.flash_attention(q,k,v,cu_q=cu_q,cu_k=cu_k,max_q=s,max_k=s)
                else:
                    native.cache_write(k,v,*cache,slot_mapping,cache_scale)
                    out=native.flash_attention(q,*cache,cu_q=cu_q,cu_k=None,max_q=s,max_k=s,block_table=block_table,lengths=lengths)
                out=out.reshape(b,s,a.num_heads,d)
            out = out.reshape(b, s, width)
            if self.recipe.action_only_final and layer_index == len(layers)-1:
                # Subsequent operations and the final norm act independently on each token.
                indices = head_indices.unsqueeze(-1).expand(-1, -1, width)
                residual, out = residual.gather(1, indices), out.gather(1, indices)
            update = a.o_proj(out)
            if self.recipe.fuse_residual_rms:
                hidden, mlp_input = residual_rmsnorm_bf16(residual, update, layer.post_attention_layernorm)
                pending = layer.mlp(mlp_input)
            else:
                hidden,squared=add(residual,update)
                residual=hidden;hidden=layer.mlp(norm(layer.post_attention_layernorm,hidden,squared))
                hidden,squared=add(residual,hidden)
        final_norm = self.model.language_model.model.norm
        if self.recipe.fuse_residual_rms:
            _, hidden = residual_rmsnorm_bf16(hidden, pending, final_norm)
        else:
            hidden = norm(final_norm, hidden, squared)
        selected = hidden if self.recipe.action_only_final else hidden.gather(1,head_indices[:,:,None].expand(-1,-1,width))
        return self.action_head.predict_action(selected)
