"""Explicit pinned OFT runtime configuration; no model or CUDA imports."""

import json
from pathlib import Path


def validate_runtime_options(od_config):
    """The bridge owns one resident GPU model; Omni cannot shard or offload it."""
    flags = (
        "enable_sleep_mode",
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
        "enable_prefix_caching",
    )
    for name in flags:
        if getattr(od_config, name, False):
            raise ValueError(f"OFT isolated worker does not support {name}")
    for name in ("lora_path", "quantization_config", "diffusion_offload_config", "kv_offload_config"):
        if getattr(od_config, name, None):
            raise ValueError(f"OFT isolated worker does not support {name}")
    parallel = getattr(od_config, "parallel_config", None)
    for name in (
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "data_parallel_size",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
        "text_encoder_tp_size",
        "ulysses_degree",
        "ring_degree",
        "allgather_degree",
    ):
        if getattr(parallel, name, 1) not in (None, 1):
            raise ValueError("OFT isolated worker supports a single GPU replica only")
    if getattr(parallel, "use_hsdp", False) or getattr(parallel, "enable_expert_parallel", False):
        raise ValueError("OFT isolated worker does not support distributed model sharding")


def make_oft_config(
    checkpoint,
    *,
    worker_python,
    source,
    native_path,
    worker_log,
    norm_key,
    action_space,
    batch=1,
    graph=False,
    backend="sdpa",
    audit_path=None,
    reference_compatible=False,
    mlp_precision="bf16",
    fp8_scope="mlp",
    fuse_rope=False,
    fuse_silu_mul=False,
    fuse_rmsnorm=False,
    fuse_residual_square=False,
    qkvo_layout="original",
    fuse_qkv=False,
    share_mlp_quant=False,
    fuse_down_quant=False,
    calibration_path=None,
    calibration_sha256=None,
    diagnostics_path=None,
):
    checkpoint = Path(checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "openvla" or config.get("text_config", {}).get("model_type") != "llama":
        raise ValueError("This adapter requires the pinned Llama OpenVLA-OFT L1 layout")
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    if norm_key not in stats:
        raise ValueError("Explicit normalization key is absent from checkpoint")
    if not isinstance(action_space, str) or not action_space.strip():
        raise ValueError("action_space is required")
    if type(batch) is not int or not 1 <= batch <= 32:
        raise ValueError("batch must be in [1, 32]")
    if type(graph) is not bool or backend not in ("sdpa", "vllm_fa2", "vllm_paged_fa2"):
        raise ValueError("Invalid execution options")
    if type(reference_compatible) is not bool or mlp_precision not in ("bf16", "fp8", "nvfp4_full", "nvfp4_selective"):
        raise ValueError("Invalid reference/MLP precision options")
    if reference_compatible and (batch != 1 or graph or backend != "sdpa"):
        raise ValueError("Reference-compatible mode requires B1 SDPA without Graph")
    fusions = dict(fuse_rope=fuse_rope, fuse_silu_mul=fuse_silu_mul,
                   fuse_rmsnorm=fuse_rmsnorm, fuse_residual_square=fuse_residual_square,
                   fuse_qkv=fuse_qkv, share_mlp_quant=share_mlp_quant, fuse_down_quant=fuse_down_quant)
    if any(type(value) is not bool for value in fusions.values()):
        raise ValueError("OFT fusion switches must be boolean")
    if reference_compatible and any(fusions.values()):
        raise ValueError("Reference-compatible mode requires unfused execution")
    if fp8_scope not in ("mlp", "all_linear") or (fp8_scope == "all_linear" and mlp_precision != "fp8"):
        raise ValueError("all_linear FP8 scope requires FP8 precision")
    if qkvo_layout not in ("original", "pretransposed"):
        raise ValueError("Unknown QKVO layout")
    if reference_compatible and qkvo_layout != "original":
        raise ValueError("Reference-compatible mode requires original QKVO layout")
    if fp8_scope == "all_linear" and (fuse_qkv or qkvo_layout != "original"):
        raise ValueError("FP8 QKVO cannot use BF16 packed layouts")
    if share_mlp_quant and (mlp_precision != "fp8" or not fuse_silu_mul):
        raise ValueError("Shared activation quantization requires FP8 and fused SiLU MLP")
    if fuse_down_quant and (mlp_precision != 'nvfp4_full' or not fuse_silu_mul):
        raise ValueError('Fused down quantization requires full NVFP4 and fused SiLU')
    if mlp_precision.startswith('nvfp4'):
        if calibration_path is None or not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64:
            raise ValueError('NVFP4 requires a frozen calibration path and SHA256')
        calibration_path = str(Path(calibration_path).resolve(strict=True))
    elif calibration_path is not None or calibration_sha256 is not None:
        raise ValueError('Calibration is only valid for NVFP4')
    return {
        "pipeline": "vla_openvla_oft",
        "async_chunk": False,
        "distributed_executor_backend": "mp",
        "dtype": "bfloat16",
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "max_num_seqs": batch,
                "request_batch_max_wait_ms": 10.0 if batch > 1 else 0.0,
                "enforce_eager": not graph,
                "model_class_name": "VlaOpenVlaOftPipeline",
                "model_config": {
                    "policy_server_config": {
                        "needs_session_id": False,
                        "action_space": action_space,
                        "action_horizon": 8,
                        "action_dim": 7,
                        "max_cameras": 2,
                        "image_resolution": [224, 224],
                        "norm_key": norm_key,
                    }
                },
                "custom_pipeline_args": {
                    "worker_python": str(Path(worker_python).absolute()),
                    "source": str(Path(source).resolve(strict=True)),
                    "native_path": str(Path(native_path).resolve(strict=True)),
                    "worker_log": str(Path(worker_log).absolute()),
                    "norm_key": norm_key,
                    "batch": batch,
                    "graph": graph,
                    "backend": backend,
                    "reference_compatible": reference_compatible,
                    "mlp_precision": mlp_precision,
                    "fp8_scope": fp8_scope,
                    "qkvo_layout": qkvo_layout,
                    **fusions,
                    "calibration_path": calibration_path,
                    "calibration_sha256": calibration_sha256,
                    "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
                    "audit_path": str(Path(audit_path).absolute()) if audit_path else None,
                },
            }
        ],
    }
