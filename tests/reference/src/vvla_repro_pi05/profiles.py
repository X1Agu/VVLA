"""Explicit serving recipes. Hardware fallback never silently changes precision."""

MODEL_FLAGS = {
    "cache_empty_camera_features": True,
    "precompute_time_and_positions": True,
    "prefix_kv_only": True,
    "pack_prefix": True,
    "vision_bf16": True,
}

NVFP4_BASE_RECIPE = {
    "prefix_mlp": "nvfp4_full", "vision_mlp": "none", "expert_mlp": "none",
    "fuse_prefix_mlp": True, "fuse_prefix_down_quant": True, "fuse_gate_up": False,
    "fuse_prefix_rmsnorm": True, "prefix_rmsnorm_reduction": "torch", "nvfp4_layers": [],
    "nvfp4_backend": "vllm_cutlass", "fuse_vision_residual_cast": True,
}


def apply_profile(deploy, profile, *, calibration=None, artifact=None, artifact_sha256=None):
    stage = deploy["stages"][0]
    model = stage["model_config"]
    if profile == "bf16":
        if any((calibration, artifact, artifact_sha256)):
            raise ValueError("Quantization assets require the blackwell-nvfp4 profile")
        return
    if profile == "fp8-prefix":
        if any((calibration, artifact, artifact_sha256)):
            raise ValueError("FP8 prefix uses original weights and dynamic per-token activation scales")
        model.update(execution_recipe={"prefix_mlp": "fp8"}, precision_hardware_fallback="error")
        return
    model.update(MODEL_FLAGS)
    if profile == "optimized-bf16":
        if any((calibration, artifact, artifact_sha256)):
            raise ValueError("Quantization assets require the blackwell-nvfp4 profile")
        return
    if profile in {"full-bf16", "full-fp8"}:
        if any((calibration, artifact, artifact_sha256)):
            raise ValueError("Full BF16/FP8 profiles require original weights, not NVFP4 assets")
        # Preserve the tested Blackwell fusion semantics, excluding SM120-only
        # NVFP4/down-quantization and the separate gate/up concatenation experiment.
        model.update(
            execution_recipe={
                "prefix_mlp": "fp8" if profile == "full-fp8" else "none",
                "vision_mlp": "none", "expert_mlp": "none",
                "fuse_prefix_mlp": True, "fuse_prefix_down_quant": False,
                "fuse_gate_up": False, "fuse_prefix_rmsnorm": True,
                "prefix_rmsnorm_reduction": "torch", "fuse_vision_residual_cast": True,
            },
            precision_hardware_fallback="error",
        )
        stage["custom_pipeline_args"]["pi05_pointwise_fusions"] = {
            "fuse_rope": True, "fuse_prefix_residual_square": True,
        }
        return
    if profile != "blackwell-nvfp4":
        raise ValueError(f"Unknown profile {profile!r}")
    if bool(calibration) == bool(artifact):
        raise ValueError("Choose exactly one: calibration for original weights, or an existing packed artifact")
    if artifact and not artifact_sha256:
        raise ValueError("Packed artifacts require their manifest SHA-256")
    if artifact_sha256 and not artifact:
        raise ValueError("Artifact manifest SHA-256 requires an artifact path")
    model.update(attention_backend="vllm_paged_fa2", execution_recipe=dict(NVFP4_BASE_RECIPE),
                 precision_hardware_fallback="error")
    if calibration:
        model["quantization_calibration_path"] = calibration
    if artifact:
        model["execution_artifact_path"] = artifact
        model["execution_artifact_sha256"] = artifact_sha256
    stage["custom_pipeline_args"]["pi05_pointwise_fusions"] = {
        "fuse_rope": True, "fuse_prefix_residual_square": True,
    }

