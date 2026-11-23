"""Independent OFT execution switches; defaults retain official BF16 semantics."""
from dataclasses import dataclass,asdict,field
@dataclass(frozen=True)
class OFTExecutionRecipe:
    attention:str='sdpa'
    reference_compatible:bool=False
    pack_sequence:bool=False
    pack_bucket:int=16
    precompute_positions:bool=False
    fuse_rope:bool=False
    mlp_precision:str='bf16'
    fp8_scope:str='mlp'
    nvfp4_layers:tuple[int,...]=field(default_factory=lambda:tuple(range(2,30)))
    fuse_silu_mul:bool=False
    fuse_down_quant:bool=False
    fuse_rmsnorm:bool=False
    fuse_residual_square:bool=False
    qkvo_layout:str='original'
    fuse_qkv:bool=False
    share_mlp_quant:bool=False
    batch_camera_views:bool=False
    skip_unused_vision_block:bool=False
    action_only_final:bool=False
    packed_single_kv:bool=False
    fuse_residual_rms:bool=False
    pack_gate_up:bool=False
    vision_compile:str='off'
    mlp_compile:str='off'
    cache_token_layout:bool=False
    vision_compile_scope:str='both'
    parallel_vision:bool=False
    page_size:int=16
    def __post_init__(self):
        if self.qkvo_layout not in ('original','pretransposed'):raise ValueError('Unknown QKVO layout')
        if self.attention not in ('sdpa','vllm_fa2','vllm_paged_fa2'):raise ValueError('Unknown attention backend')
        for name in ('reference_compatible','pack_sequence','precompute_positions','fuse_rope','fuse_silu_mul','fuse_down_quant','fuse_rmsnorm','fuse_residual_square','fuse_qkv','share_mlp_quant','batch_camera_views'):
            if type(getattr(self,name)) is not bool:raise ValueError(name+' must be boolean')
        if self.reference_compatible and (self.attention!='sdpa' or self.pack_sequence or self.precompute_positions or self.fuse_rope or self.fuse_silu_mul or self.fuse_qkv or self.fuse_rmsnorm or self.fuse_residual_square or self.batch_camera_views or self.qkvo_layout!='original'):
            raise ValueError('Reference-compatible execution requires original unpadded SDPA attention')
        if type(self.pack_bucket) is not int or self.pack_bucket<1:raise ValueError('Invalid pack bucket')
        if type(self.page_size) is not int or self.page_size not in (16,32,64):raise ValueError('Invalid page size')
        if self.mlp_precision not in ('bf16','fp8','nvfp4_selective','nvfp4_full'):raise ValueError('Unknown MLP precision')
        for name in ('skip_unused_vision_block','action_only_final','packed_single_kv','fuse_residual_rms','pack_gate_up','cache_token_layout','parallel_vision'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(name+' must be boolean')
            if getattr(self,name) and self.reference_compatible:
                raise ValueError(name+' requires optimized execution')
        if self.skip_unused_vision_block and not self.batch_camera_views:
            raise ValueError('Penultimate vision output requires batched camera views')
        if self.packed_single_kv and not self.pack_sequence:
            raise ValueError('Direct single-row KV requires a packed sequence')
        if self.fuse_residual_rms and not (self.fuse_rmsnorm and self.fuse_residual_square):
            raise ValueError('Fused residual RMSNorm requires the G4 norm recipe')
        if self.pack_gate_up and (self.mlp_precision != 'bf16' or not self.fuse_silu_mul):
            raise ValueError('Packed BF16 gate/up requires BF16 and fused SiLU')
        if any(v not in ('off','fused','tuned') for v in (self.vision_compile,self.mlp_compile)):
            raise ValueError('Invalid OFT compiler mode')
        if self.vision_compile != 'off' and not self.skip_unused_vision_block:
            raise ValueError('Compiled dual vision requires the explicit penultimate-block execution')
        if self.vision_compile_scope not in ('both','primary','secondary') or (self.vision_compile_scope!='both' and self.vision_compile=='off'):
            raise ValueError('Invalid compiled camera tower selection')
        if self.parallel_vision and not self.skip_unused_vision_block:
            raise ValueError('Parallel camera towers require the explicit penultimate-block execution')
        if self.mlp_compile != 'off' and not self.pack_gate_up:
            raise ValueError('Compiled BF16 MLP requires packed gate/up weights')
        if self.fp8_scope not in ('mlp','llama','llama_vision','all_linear'):raise ValueError('Unknown FP8 scope')
        if self.fp8_scope!='mlp' and self.mlp_precision!='fp8':raise ValueError('Expanded linear scopes require FP8 precision')
        if self.fp8_scope!='mlp' and (self.fuse_qkv or self.qkvo_layout!='original'):raise ValueError('FP8 QKVO cannot use BF16 packed layouts')
        if self.share_mlp_quant and (self.mlp_precision!='fp8' or not self.fuse_silu_mul):raise ValueError('Shared activation quantization requires FP8 and fused SiLU MLP')
        if self.fuse_down_quant and (not self.fuse_silu_mul or self.mlp_precision!='nvfp4_full'):raise ValueError('Fused down quantization requires SiLU fusion and full MLP NVFP4')
        if any(type(i) is not int or i<0 for i in self.nvfp4_layers) or len(set(self.nvfp4_layers))!=len(self.nvfp4_layers):raise ValueError('Invalid NVFP4 layer selection')
        object.__setattr__(self,'nvfp4_layers',tuple(self.nvfp4_layers))
    def asdict(self):return asdict(self)


def recipe_from_config(config):
    """Resolve the public worker options before loading weights or capturing graphs."""
    return OFTExecutionRecipe(
        attention=config['backend'],
        pack_sequence=config['backend']!='sdpa',
        pack_bucket=config.get('pack_bucket',16),
        precompute_positions=config['backend']!='sdpa',
        reference_compatible=config.get('reference_compatible',False),
        mlp_precision=config.get('mlp_precision','bf16'),
        fp8_scope=config.get('fp8_scope','mlp'),
        vision_compile=config.get('vision_compile','off'),
        vision_compile_scope=config.get('vision_compile_scope','both'),
        mlp_compile=config.get('mlp_compile','off'),
        qkvo_layout=config.get('qkvo_layout','original'),
        **{name:config.get(name,False) for name in
           ('fuse_rope','fuse_silu_mul','fuse_rmsnorm','fuse_residual_square','fuse_qkv','share_mlp_quant','fuse_down_quant','batch_camera_views',
            'skip_unused_vision_block','action_only_final','packed_single_kv','fuse_residual_rms','pack_gate_up','cache_token_layout','parallel_vision')},
    )
