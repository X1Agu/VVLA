"""Frozen experiment recipes. Every profile is compared on the same paired80."""
from .recipe import OFTExecutionRecipe
BF16_PROFILES={
 'fa2':dict(attention='vllm_fa2'),
 'fa2_pack':dict(attention='vllm_fa2',pack_sequence=True),
 'fa2_positions':dict(attention='vllm_fa2',precompute_positions=True),
 'fa2_rope':dict(attention='vllm_fa2',fuse_rope=True),
 'fa2_all':dict(attention='vllm_fa2',pack_sequence=True,precompute_positions=True,fuse_rope=True),
 'paged_all':dict(attention='vllm_paged_fa2',pack_sequence=True,precompute_positions=True,fuse_rope=True),
}
PRECISION_PROFILES={
 'fp8':dict(**BF16_PROFILES['fa2_all'],mlp_precision='fp8'),
 'nvfp4_selective':dict(**BF16_PROFILES['fa2_all'],mlp_precision='nvfp4_selective'),
 'nvfp4_full':dict(**BF16_PROFILES['fa2_all'],mlp_precision='nvfp4_full'),
 'nvfp4_full_silu':dict(**BF16_PROFILES['fa2_all'],mlp_precision='nvfp4_full',fuse_silu_mul=True),
 'nvfp4_full_down_fused':dict(**BF16_PROFILES['fa2_all'],mlp_precision='nvfp4_full',fuse_silu_mul=True,fuse_down_quant=True),
}
PROFILES=BF16_PROFILES|PRECISION_PROFILES

NORM_PROFILES={
 'control':dict(PRECISION_PROFILES['nvfp4_full_down_fused']),
 'norm':dict(**PRECISION_PROFILES['nvfp4_full_down_fused'],fuse_rmsnorm=True),
 'residual':dict(**PRECISION_PROFILES['nvfp4_full_down_fused'],fuse_residual_square=True),
 'norm_residual':dict(**PRECISION_PROFILES['nvfp4_full_down_fused'],fuse_rmsnorm=True,fuse_residual_square=True),
}
PROFILES.update(NORM_PROFILES)

GEMM_PROFILES={
 'gemm_control':dict(NORM_PROFILES['norm_residual']),
 'pretransposed':dict(**NORM_PROFILES['norm_residual'],qkvo_layout='pretransposed'),
 'qkv_pretransposed':dict(**NORM_PROFILES['norm_residual'],qkvo_layout='pretransposed',fuse_qkv=True),
}
PROFILES.update(GEMM_PROFILES)
