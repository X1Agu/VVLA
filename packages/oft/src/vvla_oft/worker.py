"""Private, framed IPC worker. Only this interpreter imports the pinned OFT fork."""

import argparse
import hashlib
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
import traceback

import msgpack
import msgpack_numpy
import numpy as np

MAX_MESSAGE = 64 * 1024 * 1024


def pack(value):
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(value):
    return msgpack.unpackb(value, object_hook=msgpack_numpy.decode, raw=False)


class Worker:
    def __init__(self, config):
        import torch

        from .reference import load_reference, CHECKPOINT_REVISION
        from .calibration import load_calibration
        from .native_adapter import OFTNativeAdapter
        from .recipe import recipe_from_config
        from vvla_runtime.execution.graph import TensorGraphRunner
        from .service import OFTService

        self.config = config
        self.runner = None
        if torch.__version__.split("+")[0] != "2.13.0" or not torch.cuda.is_available():
            raise RuntimeError("OFT worker requires the pinned CUDA Torch 2.13.0 runtime")
        if config["backend"] not in ("sdpa", "vllm_fa2", "vllm_paged_fa2"):
            raise ValueError("Unsupported OFT backend")
        if type(config["batch"]) is not int or not 1 <= config["batch"] <= 32:
            raise ValueError("OFT batch must be in [1, 32]")
        if type(config["graph"]) is not bool:
            raise ValueError("OFT graph must be boolean")
        reference_compatible = config.get("reference_compatible", False)
        if type(reference_compatible) is not bool:
            raise ValueError("reference_compatible must be boolean")
        if reference_compatible and (config["batch"] != 1 or config["graph"] or config["backend"] != "sdpa"):
            raise ValueError("Reference-compatible mode requires B1 SDPA without Graph")
        if config.get("mlp_precision", "bf16") not in ("bf16", "fp8", "nvfp4_full", "nvfp4_selective"):
            raise ValueError("Unsupported MLP precision")
        recipe = recipe_from_config(config)
        calibration = load_calibration(config, CHECKPOINT_REVISION)
        os.environ["VLA_VLLM_NATIVE_PATH"] = config["native_path"]
        # Omni invokes its runner inside inference_mode. Model construction lives
        # in this interpreter so persistent tensors retain version counters.
        model, head, proprio, processor, utils, provenance = load_reference(config["source"], config["checkpoint"])
        adapter = OFTNativeAdapter(model, head, proprio, recipe, calibration).eval()
        revision = hashlib.sha256((Path(config["checkpoint"]) / "config.json").read_bytes()).hexdigest()
        self.runner = TensorGraphRunner(
            adapter, revision=revision, use_graph=config["graph"], diagnostics_path=config.get('diagnostics_path')
        )
        self.service = OFTService(model, processor, utils, self.runner, capacity=128, batch_size=config["batch"],
            prepare_inputs=lambda tensors: adapter.prepare_inputs(tensors, device=next(model.parameters()).device, validate=False))
        self.default_key = config.get("norm_key")
        if self.default_key is not None and self.default_key not in model.norm_stats:
            raise ValueError("Configured normalization key is absent from checkpoint")
        self.record(
            "loaded",
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            torch=torch.__version__,
            module_file=__file__,
            recipe=recipe.asdict(),
            quantization_manifest=adapter.quantization_manifest,
            preprocessing=provenance,
            calibration_sha256=config.get('calibration_sha256'),
        )

    def record(self, event, **fields):
        if self.config.get("audit_path"):
            path = Path(self.config["audit_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event=event, pid=os.getpid(), time_ns=time.time_ns(), **fields)) + "\n")

    def prepare(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("robot_obs must be an object")
        for name in ("full_image", "wrist_image"):
            image = obs.get(name)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} must be an HWC uint8 RGB image")
            if not 1 <= min(image.shape[:2]) <= max(image.shape[:2]) <= 4096:
                raise ValueError(f"{name} dimensions are invalid")
        state = np.asarray(obs.get("state"))
        if state.shape != (8,) or state.dtype.kind not in "fi" or not np.isfinite(state).all():
            raise ValueError("state must contain eight finite numeric values")
        if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
            raise ValueError("prompt must be a nonempty string")
        key = obs.get("norm_key", self.default_key)
        if not isinstance(key, str) or key not in self.service.model.norm_stats:
            raise ValueError("An explicit known norm_key is required")
        return self.service.prepare_row(obs, key), key

    def infer(self, requests):
        if not isinstance(requests, list) or not 1 <= len(requests) <= self.config["batch"]:
            raise ValueError("Invalid scheduler batch size")
        started = time.perf_counter()
        prepared_at = None
        outputs = [None] * len(requests)
        prepared, keys, tickets, indices = [], [], [], []
        try:
            for i, request in enumerate(requests):
                try:
                    row, key = self.prepare(request["observation"])
                    ticket = self.service.ledger.start(request["request_id"])
                    prepared.append(row)
                    keys.append(key)
                    tickets.append(ticket)
                    indices.append(i)
                except (ValueError, TypeError, KeyError) as exc:
                    outputs[i] = {"error": str(exc)}
            if tickets:
                tensors, count = self.service.collate(prepared, pad=not self.config.get("reference_compatible", False))
                prepared_at = time.perf_counter()
                actions = self.service.predict_prepared(tickets, tensors, count, keys)
                for i, ticket in zip(indices, tickets, strict=True):
                    value = actions[ticket].astype(np.float32)
                    if value.shape != (8, 7) or not np.isfinite(value).all():
                        raise RuntimeError("OFT returned invalid action values")
                    outputs[i] = {"actions": value}
            return outputs
        finally:
            import torch
            for ticket in tickets:
                self.service.ledger.finish(ticket)
            self.record(
                "forward",
                batch=len(requests),
                request_ids=[r.get('request_id') for r in requests],
                session_ids=[r.get('observation', {}).get('session_id') for r in requests],
                action_sha256=[None if o is None or 'actions' not in o else hashlib.sha256(o['actions'].tobytes()).hexdigest() for o in outputs],
                prepare_ms=None if prepared_at is None else (prepared_at-started)*1000,
                predict_ms=None if prepared_at is None else (time.perf_counter()-prepared_at)*1000,
                errors=[o is None or "error" in o for o in outputs],
                graph_stats=dict(self.runner.stats),
                active_requests=len(self.service.ledger._active),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.record("closed", graph_stats=dict(self.runner.stats), active_requests=len(self.service.ledger._active))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    connection = Connection(args.fd)
    worker = None
    try:
        config = unpack(connection.recv_bytes(MAX_MESSAGE))
        worker = Worker(config)
        connection.send_bytes(pack({"ready": True, "pid": os.getpid()}))
        while True:
            request = unpack(connection.recv_bytes(MAX_MESSAGE))
            if request["op"] == "close":
                worker.close()
                worker = None
                connection.send_bytes(pack({"closed": True}))
                break
            if request["op"] != "infer":
                raise ValueError("Unknown worker operation")
            connection.send_bytes(pack({"outputs": worker.infer(request["requests"])}))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        traceback.print_exc()
        try:
            connection.send_bytes(pack({"fatal": f"{type(exc).__name__}: {exc}"}))
        except (OSError, EOFError):
            pass
        raise
    finally:
        if worker is not None:
            worker.close()
        connection.close()


if __name__ == "__main__":
    main()
