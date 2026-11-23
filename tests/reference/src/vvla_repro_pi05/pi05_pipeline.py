"""Standard Omni adapter around the preserved, namespaced π0.5 implementation."""

import logging
import os
import json
from pathlib import Path
import time

from vvla_repro_pi05.models.pi05.pipeline_pi05 import Pi05Pipeline, get_pi05_post_process_func
from vvla_repro_pi05.runtime import check_runtime
from vvla_repro_pi05.topology import PI05_PIPELINE

__all__ = ["VlaPi05Pipeline", "PI05_PIPELINE", "get_pi05_post_process_func"]

logger = logging.getLogger(__name__)


class VlaPi05Pipeline(Pi05Pipeline):
    """Opt-in pipeline; installing the plugin leaves upstream Pi05Pipeline intact."""

    def __init__(self, *, od_config, prefix=""):
        check_runtime()
        super().__init__(od_config=od_config, prefix=prefix)
        self._integration_audit = (od_config.custom_pipeline_args or {}).get("vla_audit_path")
        logger.info(
            "vllm-vla initialized %s in worker pid=%d (device=%s, dtype=%s, graph=%s)",
            type(self).__name__,
            os.getpid(),
            self._device,
            self._torch_dtype,
            self.graph_runner is not None,
        )
        self._record_event(
            "loaded",
            pipeline=type(self).__name__,
            module=type(self).__module__,
            module_file=__file__,
            device=str(self._device),
            dtype=str(self._torch_dtype),
            backend=self.config.attention_backend,
            graph=self.graph_runner is not None,
            optimization_flags={name: getattr(self.config, name) for name in (
                "cache_empty_camera_features", "precompute_time_and_positions",
                "prefix_kv_only", "pack_prefix", "vision_bf16",
            )},
            execution_recipe=self.model.config.execution_recipe,
            execution_state=getattr(self.model, "_execution_state", {}).get("status"),
            quantization_manifest=getattr(self.model, "_execution_state", {}).get("manifest", []),
            fused_module_types=sorted({type(m).__name__ for m in getattr(self.model, "_execution_state", {}).get("modules", [])}),
            vision_parameter_dtypes=sorted({str(p.dtype) for p in self.model.paligemma_with_expert.paligemma.model.vision_tower.parameters()}),
        )

    def _record_event(self, event, **fields):
        if self._integration_audit:
            path = Path(self._integration_audit)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"event": event, "pid": os.getpid(), "time_ns": time.time_ns(), **fields}) + "\n"
                )

    def forward(self, req, **kwargs):
        rows = getattr(req, "requests", [req])
        request_fields = {
            "batch": len(rows),
            "request_ids": [getattr(row, "request_id", None) for row in rows],
            "has_observations": [bool((row.sampling_params.extra_args or {}).get("robot_obs")) for row in rows],
        }
        try:
            result = super().forward(req, **kwargs)
        except Exception as exc:
            self._record_event(
                "forward",
                **request_fields,
                errors=[True] * len(rows),
                exception_type=type(exc).__name__,
                graph_stats=dict(self.graph_runner.stats) if self.graph_runner else {},
            )
            raise
        outputs = result if isinstance(result, list) else [result]
        self._record_event(
            "forward",
            **request_fields,
            errors=[bool(getattr(output, "error", None)) for output in outputs],
            graph_stats=dict(self.graph_runner.stats) if self.graph_runner else {},
        )
        return result

    def _initialize_model(self):
        model = super()._initialize_model()
        flags = (self.od_config.custom_pipeline_args or {}).get("pi05_pointwise_fusions")
        if flags is not None:
            allowed = {"fuse_rope", "fuse_prefix_residual_square"}
            if not isinstance(flags, dict) or set(flags) - allowed or any(type(v) is not bool for v in flags.values()):
                raise ValueError(f"pi05_pointwise_fusions must contain boolean flags from {sorted(allowed)}")
            # Packed artifacts must be loaded first; no captured graph exists yet.
            from vvla_repro_pi05.models.pi05.precision_pi05 import set_pointwise_fusions

            set_pointwise_fusions(model, **flags)
        return model
