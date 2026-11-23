"""B1 LingBot execution: existing VLA optimizations with official dynamic sparse MoE.

The backbone, vision, action projections and MoE weights remain official modules.
Only VLM dense MLP linears are eligible for the existing FP8/NVFP4 recipe.
"""
from __future__ import annotations
import hashlib, json, math, time, types
from pathlib import Path
import torch
from .diffusion import euler_update, sampling_clock
from torch import nn
from torch.nn import functional as F
from vvla_runtime.execution import native
from vvla_runtime.execution.llama_norm import LlamaRMSNormExecution, residual_square_bf16
from vvla_runtime.precision import QuantizedLinear, FusedLlamaMLP
from vvla_runtime.pointwise import rotary_pos_emb

MODES = ('official-bf16','framework-bf16','framework-fp8','framework-nvfp4')

class CachedAdaNorm(nn.Module):
    """Existing static Ada modulation cache; retain the upstream FP32 reduction."""
    def __init__(self, original, times):
        super().__init__()
        self.weight, self.gamma, self.beta = original.weight, original.gamma, original.beta
        self.variance_epsilon = original.variance_epsilon
        self.modulations = [(self.gamma(t).unsqueeze(1),self.beta(t).unsqueeze(1)) for t in times]
        self.eval()
    def forward(self,x,step,squared=None):
        from .norm_kernel import ada_after_mean
        var=(x.float().pow(2) if squared is None else squared).mean(-1,keepdim=True)
        gamma,beta=self.modulations[step]
        return ada_after_mean(x,var,self.weight,gamma,beta,self.variance_epsilon)

class Engine:
    def __init__(self,flow,mode,calibration=None,audit=None,*,packed_weights=None,denoise_precision='bf16'):
        assert mode in MODES[1:]
        self.flow,self.mode,self.audit=flow,mode,Path(audit) if audit else None
        self.qwe=flow.qwenvl_with_expert
        self.vlm=self.qwe.qwenvl.model.language_model
        self.action=self.qwe.qwen_expert.model
        self.calibration=calibration or {}
        self._packed_weights=packed_weights
        self.denoise_precision = denoise_precision
        self.fp16_prefix = denoise_precision == 'fp16-prefix-expert'
        self.fp16_expert = self.fp16_prefix or denoise_precision == 'fp16-expert'
        self.fp32_heads = self.fp16_expert or denoise_precision == 'fp32-heads'
        self.fp32_accumulator = self.fp32_heads or denoise_precision == 'fp32-clock-accum'
        self.recipe={'mode':mode,'batch_size':1,'precision_scope':'VLM dense MLP only; MoE and all other projections BF16',
                     'quantized_linears':[],'enabled':{},'not_applicable':{},'graph_builds':0}
        self._graph=None;self._meta_key=None;self._signature=None
        assert flow.config.use_cache and not flow.config.action_fp32
        if (not self.vlm.layers or len(self.vlm.layers)!=len(self.action.layers)
                or len(self.vlm.layers)!=self.vlm.config.num_hidden_layers
                or len(self.action.layers)!=self.action.config.num_hidden_layers):
            raise ValueError('LingBot requires matched, complete declared VLM/action layer stacks')
        self.recipe['layers']=len(self.vlm.layers)
        self.recipe['historical_full_depth']=36
        assert all(l.mlp._moe_implementation=='fused' for l in self.action.layers)
        native.load_native()
        self._install_static()
        self._write_recipe()

    @torch.no_grad()
    def _install_static(self):
        clock_dtype = torch.bfloat16 if self.denoise_precision == 'bf16' else torch.float32
        clock_dt, ts = sampling_clock(self.flow.config.num_steps, clock_dtype)
        integrator_dtype = torch.float32 if self.fp32_accumulator else torch.bfloat16
        self.dt = clock_dt.to(device='cuda', dtype=integrator_dtype)
        self.times = [v.cuda().view(1) for v in ts]
        from lingbotvla.models.vla.lingbot_vla.utils import create_sinusoidal_pos_embedding
        embedding_dtype = torch.float32 if self.fp32_heads else torch.bfloat16
        self.time_embs=[create_sinusoidal_pos_embedding(v,self.flow.config.proj_width,min_period=4e-3,max_period=4.0,device=v.device).to(embedding_dtype) for v in self.times]
        self.recipe['denoise_steps'] = len(ts)
        if self.denoise_precision == 'bf16':
            self.recipe['denoise_time_bf16'] = [float(v) for v in ts]
        else:
            self.recipe.update(denoise_precision=self.denoise_precision,
                denoise_time_fp32=[float(v) for v in ts], euler_dtype=str(integrator_dtype),
                euler_dt=float(self.dt), network_dtype='torch.bfloat16',
                embedding_sha256=[hashlib.sha256(v.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
                                  for v in self.time_embs])
        if self.fp32_heads:
            self.recipe.update(precision_scope='BF16 vision, transformer trunk and experts; original FP32 action heads, AdaNorm modulations, router gates and correction biases; FP32 Euler',
                               action_head_dtype='torch.float32', time_embedding_dtype='torch.float32',
                               sampler_input_state='BF16 widened to FP32 at the state projection')
        if self.fp16_expert:
            self.recipe.update(precision_scope='BF16 vision and VLM prefix; original FP16 expert attention and MLP weights, FP32 residual, heads, conditioning, routers and Euler',
                               expert_linear_dtype='torch.float16', expert_residual_dtype='torch.float32',
                               attention_cache_dtype='torch.float16')
        if self.fp16_prefix:
            self.recipe.update(precision_scope='BF16 vision, initial prefix embeddings, raw state and noise; FP16 text and action matrices; FP32 text and action residuals, learned norms, heads, routers and Euler',
                               text_linear_dtype='torch.float16', text_residual_dtype='torch.float32')
        norm_count=0
        for model in [self.vlm,self.action]:
            for layer in model.layers:
                for name in ['input_layernorm','post_attention_layernorm']:
                    old=getattr(layer,name)
                    if hasattr(old,'gamma'):new=CachedAdaNorm(old,self.time_embs)
                    else:new=LlamaRMSNormExecution(old,fuse_after_mean=True)
                    setattr(layer,name,new);norm_count+=1
                for name in ['q_norm','k_norm']:
                    if hasattr(layer.self_attn,name):
                        setattr(layer.self_attn,name,LlamaRMSNormExecution(getattr(layer.self_attn,name),fuse_after_mean=True));norm_count+=1
            if hasattr(model.norm,'gamma'):model.norm=CachedAdaNorm(model.norm,self.time_embs)
            else:model.norm=LlamaRMSNormExecution(model.norm,fuse_after_mean=True)
            norm_count+=1
        store=None
        if self.mode!='framework-bf16' and self._packed_weights is not None:
            from .packed import PackedStore
            packed=Path(self._packed_weights).resolve(strict=True)
            store=PackedStore(packed,self.mode,self.calibration)
        for i,layer in enumerate(self.vlm.layers):
            mlp=layer.mlp
            if self.mode!='framework-bf16':
                precision='fp8' if self.mode=='framework-fp8' else 'nvfp4'
                for name in ['gate_proj','up_proj','down_proj']:
                    key=f'vlm.layers.{i}.mlp.{name}'
                    am=self.calibration.get(key)
                    if precision=='nvfp4' and (am is None or not math.isfinite(am) or am<=0):raise ValueError(f'Missing real-observation calibration for {key}')
                    original=getattr(mlp,name);restored=None if store is None else store.restore(key,original,precision)
                    setattr(mlp,name,restored if restored is not None else QuantizedLinear(original,precision,input_amax=am))
                    self.recipe['quantized_linears'].append(key)
            layer.mlp=FusedLlamaMLP(mlp,down_quant=self.mode=='framework-nvfp4')
        if store is not None:
            store.save(self.vlm.layers)
            self.recipe['packed_artifact']=str(store.path)
            self.recipe['packed_restored']=store.data is not None
        # Shared expert uses exactly the same dense SiLU x up fusion, stays BF16.
        def reject_fallback(*args,**kwargs):raise RuntimeError('Official robby_moe failed; dense/grouped fallback is disabled for this validated sparse recipe')
        for layer in self.action.layers:
            if not self.fp16_expert:
                layer.mlp.shared_expert=FusedLlamaMLP(layer.mlp.shared_expert)
            layer.mlp.experts.forward=reject_fallback
        self.recipe['enabled'].update(fa2_prefix=True,paged_fa2_action=True,prefix_packing=True,
            static_vision_metadata=True,static_mrope=True,static_time_and_ada=True,
            rmsnorm_after_mean=norm_count,residual_square=True,silu_mul=True,
            nvfp4_down_quant=self.mode=='framework-nvfp4',full_predict_cuda_graph=True,
            dynamic_official_sparse_moe=True,parameter_lifetime_checks=True,owned_output=True)
        self.recipe['not_applicable']['siglip_residual_fp32_cast']='Qwen vision LayerNorm consumes BF16 directly; the old SigLIP explicit FP32 norm/cast pair is absent.'
        self.recipe['not_applicable']['empty_camera_feature_cache']='RoboTwin observes all three cameras; no empty image slots.'
        self.recipe['not_applicable']['expert_quantization']='No existing quantized grouped-MoE implementation; approved selective VLM dense MLP coverage only.'
        self.recipe['existing_experimental_not_in_deployed_recipe']={'qkv_merge':{'enabled':False,'evidence':'HANDOFF 3.3 / 4.4: measured candidate, not deployed'},'share_mlp_input_quant':{'enabled':False,'evidence':'HANDOFF 4.4: L4 measured slower; not deployed'},'pretransposed_gemm':{'enabled':False,'evidence':'HANDOFF 4.3: SM120 B32 +0.7% / +4GiB; not selected'}}
        self.recipe['not_applicable']['cross_observation_prefix_cache']='Not an existing framework capability; prefix recomputed every observation.'

    def _state_signature(self):
        # Same safety category as the frozen framework: parameter identity/version
        # changes invalidate the prepared execution. Never silently replay stale weights.
        for module in self.flow.modules():
            if hasattr(module,'assert_fresh'):module.assert_fresh()
        tensors=list(self.flow.named_parameters())+list(self.flow.named_buffers())
        return (tuple((name,id(p),p.data_ptr(),tuple(p.shape),tuple(p.stride()),str(p.dtype),str(p.device),p._version) for name,p in tensors),
                tuple((name,id(m),type(m).__qualname__,m.training) for name,m in self.flow.named_modules()),
                json.dumps(self.flow.config.to_dict(),sort_keys=True,default=str))

    def _write_recipe(self):
        if self.audit:
            self.audit.mkdir(parents=True,exist_ok=True)
            (self.audit/'RECIPE.json').write_text(json.dumps(self.recipe,indent=2))

    @torch.no_grad()
    def _prepare(self,images,img_masks,tokens,lang_masks,state,grid):
        from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks,block_suffix_to_fv_
        assert state.shape[0]==1 and img_masks.all().item(),'Only validated B1/all-three-camera RoboTwin recipe'
        assert tokens.shape[0]==1
        special=[self.qwe.qwenvl.config.image_token_id,self.qwe.qwenvl.config.vision_start_token_id,self.qwe.qwenvl.config.vision_end_token_id]
        if any((tokens==v).any().item() for v in special):raise ValueError('Special image markers in language require a new metadata recipe')
        # Invalidate all vision-grid constants when preparing a new input recipe.
        for name in ['pos_embeds','position_embeddings','cu_seqlens','visual_split_sizes','visual_max_seqlen']:setattr(self.qwe,name,None)
        # Upstream computes constants and the exact multimodal positions once.
        prefix,pad,ar,pos,visual,deep=self.flow.embed_prefix(images,img_masks,tokens,lang_masks,image_grid_thw=grid)
        self.qwe.pos_embeds=self.qwe.qwenvl.visual.fast_pos_embed_interpolate(grid.reshape(-1,3))
        assert all(getattr(self.qwe,n) is not None for n in ['pos_embeds','position_embeddings','cu_seqlens','visual_split_sizes','visual_max_seqlen'])
        self.template=prefix.detach().clone();self.prefix_pad=pad;self.prefix_pos=pos
        self.prefix_indices=pad[0].nonzero().flatten();self.p=int(self.prefix_indices.numel())
        self.packed_pos=pos.index_select(-1,self.prefix_indices)
        self.num_images=img_masks.numel();self.patch_count=deep[0].shape[0]//self.num_images
        self.boundaries=bool(getattr(self.flow.config,'qwen3vl_use_vision_boundaries',True))
        self.image_token_len=self.patch_count+(2 if self.boundaries else 0)
        self.image_end=self.num_images*self.image_token_len
        self.language_end=self.image_end+tokens.shape[1]
        assert self.template.shape[1]>=self.language_end
        packed_visual=visual[0].index_select(0,self.prefix_indices)
        self.visual_indices=packed_visual.nonzero().flatten()
        assert self.visual_indices.numel()==self.num_images*self.patch_count
        self.prefix_causal=bool(getattr(self.flow.config,'vlm_causal',False))
        expected=torch.ones((self.p,self.p),dtype=torch.bool,device=state.device)
        if self.prefix_causal:expected=expected.tril()
        official_mask=make_att_2d_masks(pad,ar)[0].index_select(0,self.prefix_indices).index_select(1,self.prefix_indices)
        assert torch.equal(official_mask,expected),'Unrepresentable prefix mask'
        self.h=self.flow.config.n_action_steps;self.s=self.h+1
        suffix_pad=torch.ones((1,self.s),dtype=torch.bool,device=state.device)
        suffix_ar=torch.zeros_like(suffix_pad);suffix_ar[:,:2]=True
        self.suffix_pos=self.flow._build_full_position_ids(pos,pad,suffix_pad)[:,:,-self.s:]
        smask=torch.cat([pad[:,None,:].expand(1,self.s,pad.shape[1]),make_att_2d_masks(suffix_pad,suffix_ar)],-1)
        if self.flow.block_future_depth_to_action:smask=block_suffix_to_fv_(smask,0,pad.shape[1],self.flow.num_task_tokens)
        smask=self.flow._block_suffix_to_future_video_if_enabled_(smask,suffix_row_start=0,prefix_len=pad.shape[1])
        allowed=smask[0,0,:pad.shape[1]].index_select(0,self.prefix_indices)
        assert torch.equal(smask[0,:,:pad.shape[1]].index_select(1,self.prefix_indices),allowed[None,:].expand(self.s,-1))
        self.cache_prefix_indices=allowed.nonzero().flatten();self.kp=self.cache_prefix_indices.numel()
        assert torch.equal(smask[0,:,pad.shape[1]:],make_att_2d_masks(suffix_pad,suffix_ar)[0])
        self.rope={}
        for label,pids,n in [('prefix',self.packed_pos,self.p),('suffix',self.suffix_pos,self.s)]:
            dummy=torch.empty((1,n,1,self.qwe.config.action_head_dim),device=state.device,dtype=torch.float32)
            self.rope[label]=tuple(v.contiguous() for v in self.vlm.rotary_emb(dummy,pids))
        self.page_size=16;blocks=math.ceil((self.kp+self.s)/self.page_size)
        att=self.vlm.layers[0].self_attn
        kvheads=self.qwe.qwenvl.config.text_config.num_key_value_heads;hd=att.head_dim
        shape=(len(self.vlm.layers),blocks,self.page_size,kvheads,hd)
        self.kcache=torch.empty(shape,device=state.device,dtype=torch.float16 if self.fp16_expert else torch.bfloat16);self.vcache=torch.empty_like(self.kcache)
        self.block_table=torch.arange(blocks,device=state.device,dtype=torch.int32).view(1,-1)
        self.prefix_slots=torch.arange(self.kp,device=state.device,dtype=torch.int64)
        self.suffix_slots=torch.arange(self.kp,self.kp+self.s,device=state.device,dtype=torch.int64)
        self.scale=torch.tensor(1.,device=state.device,dtype=torch.float32)
        self.cu={n:torch.tensor([0,n],device=state.device,dtype=torch.int32) for n in [1,self.h,self.p]}
        self.lengths={n:torch.tensor([n],device=state.device,dtype=torch.int32) for n in [self.kp+1,self.kp+self.s]}
        self.static=[v.clone() for v in [images,img_masks,tokens,lang_masks,state,grid]]
        self.noise=torch.empty((1,self.h,self.flow.config.max_action_dim),device=state.device,dtype=torch.bfloat16)
        self.recipe.update(prefix_valid=self.p,prefix_capacity=prefix.shape[1],suffix_tokens=self.s,kv_prefix_tokens=self.kp,kv_page_size=self.page_size,
                           image_grid_thw=grid.cpu().tolist(),parameter_count=sum(p.numel() for p in self.flow.parameters()))
        self._signature=self._state_signature()
        self._constant_tensors=[self.template,self.prefix_indices,self.cache_prefix_indices,self.visual_indices,self.dt,*self.time_embs,*[x for pair in self.rope.values() for x in pair],self.qwe.pos_embeds,*self.qwe.position_embeddings,self.qwe.cu_seqlens]
        for model in [self.vlm,self.action]:
            for module in model.modules():
                if isinstance(module,CachedAdaNorm):self._constant_tensors.extend(x for pair in module.modulations for x in pair)
        self._constant_token=tuple((id(t),t.data_ptr(),t._version) for t in self._constant_tensors)

    def _norm(self,norm,x,step,squared=None):
        return norm(x,step,squared) if isinstance(norm,CachedAdaNorm) else norm(x,squared=squared)

    def _rope(self,q,k,label):
        cos,sin=self.rope[label]
        qo,ko=rotary_pos_emb(q.float().transpose(1,2),k.float().transpose(1,2),cos,sin)
        return qo.transpose(1,2).to(torch.bfloat16),ko.transpose(1,2).to(torch.bfloat16)

    def _attention(self,q,k,v,layer,prefix):
        q=q.squeeze(0).contiguous();k=k.squeeze(0).contiguous();v=v.squeeze(0).contiguous()
        if prefix:
            native.cache_write(k.index_select(0,self.cache_prefix_indices).to(self.kcache.dtype),v.index_select(0,self.cache_prefix_indices).to(self.vcache.dtype),self.kcache[layer],self.vcache[layer],self.prefix_slots,self.scale)
            out=native._flash(q,k,v,cu_seqlens_q=self.cu[self.p],cu_seqlens_k=self.cu[self.p],max_seqlen_q=self.p,max_seqlen_k=self.p,causal=self.prefix_causal,softmax_scale=q.shape[-1]**-.5,fa_version=2)
        else:
            native.cache_write(k,v,self.kcache[layer],self.vcache[layer],self.suffix_slots,self.scale)
            parts=[]
            for qpart,n,klen in [(q[:1],1,self.kp+1),(q[1:],self.h,self.kp+self.s)]:
                parts.append(native._flash(qpart,self.kcache[layer],self.vcache[layer],cu_seqlens_q=self.cu[n],cu_seqlens_k=None,max_seqlen_q=n,max_seqlen_k=klen,seqused_k=self.lengths[klen],block_table=self.block_table,causal=False,softmax_scale=q.shape[-1]**-.5,fa_version=2))
            out=torch.cat(parts,0)
        return out.reshape(1,q.shape[0],-1)

    def _layers(self,hidden,prefix,deep=None,step=0):
        if prefix and self.fp16_prefix:
            from .fp16_prefix import prefix_layers
            return prefix_layers(self, hidden, deep)
        layers=self.vlm.layers if prefix else self.action.layers;squared=None
        for i,layer in enumerate(layers):
            normalized=self._norm(layer.input_layernorm,hidden,step,squared)
            att=layer.self_attn;shape=(*hidden.shape[:-1],-1,att.head_dim)
            q=att.q_proj(normalized).view(shape);k=att.k_proj(normalized).view(shape);v=att.v_proj(normalized).view(shape)
            if prefix:q=att.q_norm(q);k=att.k_norm(k)
            q,k=self._rope(q,k,'prefix' if prefix else 'suffix')
            out=self._attention(q,k,v,i,prefix)
            res,sq=residual_square_bf16(hidden.contiguous(),att.o_proj(out).contiguous())
            normalized=self._norm(layer.post_attention_layernorm,res,step,sq)
            mlp_out=layer.mlp(normalized)
            if not prefix:mlp_out=mlp_out[0]
            hidden,squared=residual_square_bf16(res,mlp_out.contiguous())
            if prefix and i<len(deep):
                update=hidden.index_select(1,self.visual_indices)+deep[i].unsqueeze(0)
                hidden=hidden.index_copy(1,self.visual_indices,update);squared=None
        return self._norm(self.vlm.norm if prefix else self.action.norm,hidden,step,squared)

    def _execute(self):
        images,im,tokens,lm,state,grid=self.static
        image,deep=self.qwe.embed_image(images.flatten(0,1),grid.reshape(-1,3))
        image=image.reshape(1,self.num_images,self.patch_count,-1)
        prefix=self.template.clone()
        if self.boundaries:
            imgprefix=prefix[:,:self.image_end].view(1,self.num_images,self.image_token_len,-1)
            imgprefix[:,:,1:-1]=image
        else:prefix[:,:self.image_end]=image.flatten(1,2)
        prefix[:,self.image_end:self.language_end]=self.qwe.embed_language_tokens(tokens)
        prefix=prefix.index_select(1,self.prefix_indices)
        deep=[v.reshape(-1,v.shape[-1]) for v in deep]
        self._layers(prefix,True,deep)
        return self._action_loop(state)

    def _action_loop(self, state):
        head_dtype = self.flow.action_in_proj.weight.dtype
        state_emb = self.flow.state_proj(state.to(head_dtype)).unsqueeze(1)
        x = self.noise.float() if self.fp32_accumulator else self.noise.clone()
        for step, time_embedding in enumerate(self.time_embs):
            action = self.flow.action_in_proj(x.to(head_dtype))
            time_features = time_embedding[:, None, :].expand(state.shape[0], action.shape[1], -1)
            action = self.flow.action_time_mlp_in(torch.cat([action, time_features], -1))
            action = self.flow.action_time_mlp_out(F.silu(action))
            suffix = torch.cat([state_emb, action], 1)
            if self.fp16_expert:
                from .fp16_expert import action_layers
                hidden = action_layers(self, suffix, step)
            else:
                hidden = self._layers(suffix.to(self.noise.dtype), False, step=step)
            velocity = self.flow.action_out_proj(hidden[:, -self.h:].to(head_dtype))
            euler_update(x, velocity, self.dt)
        return x

    @torch.no_grad()
    def __call__(self,images,img_masks,tokens,lang_masks,state,noise=None,image_grid_thw=None):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('External capture unsupported')
        tick=time.perf_counter()
        if self._signature is not None and self._state_signature()!=self._signature:raise RuntimeError('Model parameters changed; rebuild engine and graphs')
        if self._signature is not None and tuple((id(t),t.data_ptr(),t._version) for t in self._constant_tensors)!=self._constant_token:raise RuntimeError('Cached metadata or modulation changed; rebuild engine')
        signature_ms=(time.perf_counter()-tick)*1000;build_ms=0.;built=False
        vals=[images,img_masks,tokens,lang_masks,state,image_grid_thw]
        key=tuple((tuple(v.shape),str(v.dtype),str(v.device)) for v in vals)+(bytes(img_masks.cpu().numpy()),bytes(lang_masks.cpu().numpy()),bytes(image_grid_thw.cpu().numpy()))
        rebuild = self._meta_key != key and self._graph is not None
        if self._meta_key!=key:
            build_start=time.perf_counter();built=True
            # A failed drain must retain the previous graph and its buffers.
            # Dropping the graph first made failed rebuild cleanup unowned.
            torch.cuda.synchronize();self._graph=None
            self._prepare(*vals)
            self._meta_key=key
            self.noise.zero_()
            stream=torch.cuda.Stream();self._capture_stream=stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self._execute()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph();self._graph=graph
            with torch.cuda.graph(graph,stream=stream):self.output=self._execute()
            self._graph=graph;self.recipe['graph_builds']+=1;self._write_recipe()
            build_ms=(time.perf_counter()-build_start)*1000
        for dst,src in zip(self.static,vals):dst.copy_(src)
        if noise is None:noise=torch.randn(self.noise.shape,device=state.device,dtype=state.dtype)
        self.noise.copy_(noise);self._graph.replay()
        self.last_call={'graph_build':built,'graph_rebuild':rebuild,'capture_ms':build_ms,'signature_ms':signature_ms,'graph_replay':True}
        return self.output.clone()

    def route_audit(self):
        result=[]
        for i,layer in enumerate(self.action.layers):
            ws=layer.mlp.experts._robby_moe_workspace
            if ws is None:raise RuntimeError('Official sparse workspace absent')
            counts=ws['counts'].cpu();rows=ws['rows'].cpu();slots=ws['slots'].cpu();routes=[]
            for e,n in enumerate(counts.tolist()):
                if n<0 or n>self.s*layer.mlp.top_k:raise RuntimeError('Route capacity violation')
                routes.extend((int(rows[e,j]),int(slots[e,j]),e) for j in range(n))
            pairs={(r,s) for r,s,e in routes}
            expected={(r,s) for r in range(self.s) for s in range(layer.mlp.top_k)}
            if pairs!=expected or len(routes)!=len(expected):raise RuntimeError('Dropped or duplicated token route')
            result.append({'layer':i,'counts':counts.tolist(),'route_sha256':hashlib.sha256(json.dumps(sorted(routes)).encode()).hexdigest(),'routes':len(routes)})
        return result
