"""Standard Omni diffusion-stage wrapper for the pinned LIBERO L1 OFT worker."""

from pathlib import Path
import weakref

import numpy as np
from torch import nn
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .oft_bridge import OFTBridge
from .oft_config import validate_runtime_options
from .runtime import check_runtime


def _post_process(output):
    return output


def get_oft_post_process_func(od_config):
    return _post_process


class VlaOpenVlaOftPipeline(nn.Module):
    supports_request_batch = True

    def __init__(self, *, od_config, prefix=""):
        super().__init__()
        check_runtime()
        validate_runtime_options(od_config)
        config = dict(od_config.custom_pipeline_args or {})
        config["checkpoint"] = str(Path(od_config.model).resolve(strict=True))
        for key in ("worker_python", "source", "native_path", "worker_log", "batch", "backend", "graph"):
            if key not in config:
                raise ValueError(f"Missing OFT configuration: {key}")
        if config["graph"] and od_config.enforce_eager:
            raise ValueError("OFT graph requires enforce_eager=False")
        if str(od_config.dtype).removeprefix("torch.") != "bfloat16":
            raise ValueError("This OFT pipeline supports BF16 only")
        self.bridge = OFTBridge(config)
        # Omni drops model_runner and collects it on shutdown. It does not call
        # arbitrary pipeline.close methods; the finalizer owns only the bridge.
        self._finalizer = weakref.finalize(self, self.bridge.close)

    def load_weights(self, weights=()):
        return set()

    def close(self):
        self._finalizer()

    def forward(self, req, **kwargs):
        batched = isinstance(req, DiffusionRequestBatch)
        rows = req.requests if batched else [req]
        outputs = [None] * len(rows)
        indices, requests = [], []
        for i, row in enumerate(rows):
            if getattr(row, "lora_request", None) or getattr(row.sampling_params, "lora_request", None):
                outputs[i] = DiffusionOutput(error="OFT dynamic LoRA is unsupported")
                continue
            obs = (row.sampling_params.extra_args or {}).get("robot_obs")
            if obs is None:
                prompt = getattr(row, "prompt", "")
                prompt = prompt.get("prompt", "") if isinstance(prompt, dict) else prompt
                if prompt == "dummy run":
                    outputs[i] = DiffusionOutput(output={"actions": np.zeros((8, 7), dtype=np.float32)})
                else:
                    outputs[i] = DiffusionOutput(error="OFT requires robot_obs")
                continue
            requests.append({"request_id": str(row.request_id), "observation": obs})
            indices.append(i)
        if requests:
            for i, result in zip(indices, self.bridge.infer(requests), strict=True):
                outputs[i] = (
                    DiffusionOutput(error=result["error"])
                    if "error" in result
                    else DiffusionOutput(output={"actions": result["actions"]})
                )
        return outputs if batched else outputs[0]
