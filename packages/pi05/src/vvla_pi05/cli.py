"""Runtime diagnostics and deployment configuration; serving stays in the standard CLI."""

import argparse
import json
from pathlib import Path

from vvla_pi05.runtime import OMNI_COMMIT, check_runtime


def make_pi05_config(
    checkpoint: Path, *, tokenizer: str | None, graph: bool, backend: str, batch: int, action_space: str
) -> dict:
    from vvla_pi05.models.pi05.config import Pi05Config

    config = Pi05Config.from_pretrained(checkpoint)
    if batch < 1:
        raise ValueError("batch must be positive")
    # Preserve checkpoint camera slots, action widths, normalization, and relative-action semantics.
    metadata = dict(config.policy_server_config)
    metadata.update(
        image_resolution=list(config.image_resolution),
        needs_session_id=False,
        action_space=action_space,
        action_horizon=config.chunk_size,
        action_dim=config.action_dim,
        max_action_dim=config.max_action_dim,
        max_cameras=config.max_cameras,
    )
    custom = {"pi05_cuda_graph": graph, "pi05_graph_lifecycle_guard": True}
    if tokenizer:
        custom["tokenizer"] = tokenizer
    return {
        "pipeline": "vla_pi05",
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
                "model_class_name": "VlaPi05Pipeline",
                "custom_pipeline_args": custom,
                "model_config": {"attention_backend": backend, "policy_server_config": metadata},
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check pinned packages; optionally execute a small CUDA operation")
    doctor.add_argument("--cuda", action="store_true")
    config = commands.add_parser("config", help="generate a checkpoint-specific π0.5 deployment YAML")
    config.add_argument("checkpoint", type=Path)
    config.add_argument("--output", type=Path, required=True)
    config.add_argument("--tokenizer")
    config.add_argument("--graph", action="store_true")
    config.add_argument("--backend", choices=["eager", "vllm_fa2", "vllm_paged_fa2"], default="vllm_fa2")
    config.add_argument("--batch", type=int, default=1)
    config.add_argument("--profile", choices=["bf16", "fp8-prefix", "optimized-bf16", "full-bf16", "full-fp8", "blackwell-nvfp4"], default="bf16")
    config.add_argument("--calibration", help="activation_amax JSON for the original checkpoint")
    config.add_argument("--artifact", help="packed artifact directory")
    config.add_argument("--artifact-sha256", help="packed artifact manifest SHA-256")
    # This is a robot/checkpoint property, so never infer it from action-vector width.
    config.add_argument("--action-space", required=True)
    oft = commands.add_parser("oft-config", help="generate a pinned LIBERO L1 OFT deployment YAML")
    oft.add_argument("checkpoint", type=Path)
    oft.add_argument("--output", type=Path, required=True)
    for name in ("worker-python", "source", "native-path", "worker-log", "norm-key", "action-space"):
        oft.add_argument("--" + name, required=True)
    oft.add_argument("--batch", type=int, default=1)
    oft.add_argument("--graph", action="store_true")
    oft.add_argument("--backend", choices=["sdpa", "vllm_fa2", "vllm_paged_fa2"], default="sdpa")
    oft.add_argument("--reference-compatible", action="store_true", help="B1 unpadded SDPA; Graph must be disabled")
    oft.add_argument("--mlp-precision", choices=["bf16", "fp8"], default="bf16")
    oft.add_argument("--fp8-scope", choices=["mlp", "all_linear"], default="mlp")
    oft.add_argument("--qkvo-layout", choices=["original", "pretransposed"], default="original")
    oft.add_argument("--share-mlp-quant", action="store_true")
    for fusion in ("rope", "silu-mul", "rmsnorm", "residual-square", "qkv"):
        oft.add_argument("--fuse-" + fusion, action="store_true")
    args = parser.parse_args()
    versions = check_runtime()
    if args.command == "doctor":
        result = {"packages": versions, "omni_commit": OMNI_COMMIT}
        if args.cuda:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            x = torch.eye(32, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(x @ x, x)
            result["cuda"] = {
                "runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "bf16_matmul": "passed",
            }
        print(json.dumps(result, indent=2))
    elif args.command == "oft-config":
        import yaml
        from .oft_config import make_oft_config

        value = make_oft_config(
            args.checkpoint,
            worker_python=args.worker_python,
            source=args.source,
            native_path=args.native_path,
            worker_log=args.worker_log,
            norm_key=args.norm_key,
            action_space=args.action_space,
            batch=args.batch,
            graph=args.graph,
            backend=args.backend,
            reference_compatible=args.reference_compatible,
            mlp_precision=args.mlp_precision,
            fp8_scope=args.fp8_scope,
            fuse_rope=args.fuse_rope,
            fuse_silu_mul=args.fuse_silu_mul,
            fuse_rmsnorm=args.fuse_rmsnorm,
            fuse_residual_square=args.fuse_residual_square,
            qkvo_layout=args.qkvo_layout, fuse_qkv=args.fuse_qkv, share_mlp_quant=args.share_mlp_quant,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        print(f"Wrote {args.output}. Serve with: vllm serve <checkpoint> --omni --deploy-config <this YAML>")
    else:
        import yaml

        value = make_pi05_config(
            args.checkpoint,
            tokenizer=args.tokenizer,
            graph=args.graph,
            backend=args.backend,
            batch=args.batch,
            action_space=args.action_space,
        )
        from vvla_pi05.profiles import apply_profile

        apply_profile(
            value,
            args.profile,
            calibration=args.calibration,
            artifact=args.artifact,
            artifact_sha256=args.artifact_sha256,
        )
        # Avoid silently replacing a deploy configuration somebody has already edited.
        with args.output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        print(f"Wrote {args.output}. Serve with: vllm serve <checkpoint> --omni --deploy-config <this YAML>")


if __name__ == "__main__":
    main()
