"""Optional postprocessor cache for tiny CPU action-normalization statistics.

This helper leaves the processor and input path untouched. It reuses the
original normalization arithmetic on the action device, replacing only repeated
CPU-statistics uploads. Statistics are checked by value, not tensor version:
NumPy aliases, inference tensors and explicit .data writes are supported too.
"""

import hashlib
from pathlib import Path
import threading
import weakref

import torch

from vvla_pi05.models.pi05.processor_pi05 import NormStats, Pi05Processor, apply_norm


class ActionNormDeviceCache:
    """One immutable device snapshot, lazily refreshed by source value/target.

    Integration is explicit: call ``build_model_outputs(actions, observation)``
    in place of the processor's method only when the optional policy flag is on.
    Call ``close()`` after retiring requests. No class/instance patch, device
    allocation or input modification occurs during construction.
    """

    def __init__(self, processor):
        if getattr(processor.build_model_outputs, "__func__", None) is not Pi05Processor.build_model_outputs:
            raise ValueError("Action norm caching requires the original Pi05 output processor")
        self._processor = weakref.ref(processor)
        self._lock = threading.Lock()
        self._key = self._stats = None
        self._closed = False
        self.refreshes = 0  # Diagnostics, excluded from stable metadata.

    @staticmethod
    def _snapshot(stats):
        if not isinstance(stats, NormStats):
            return None
        # The campaign's checkpoint statistics are tiny CPU FP32 vectors.
        # Leave other source types/devices/grad semantics to the native path.
        bounds = (stats.lower, stats.upper)
        if (stats.mode not in ("mean_std", "min_max")
                or any(type(value) is not torch.Tensor or value.device.type != "cpu"
                       or value.dtype != torch.float32 or value.ndim != 1 or value.requires_grad
                       for value in bounds)
                or bounds[0].shape != bounds[1].shape or not bounds[0].numel()):
            return None
        return stats.mode, bounds[0].numpy().tobytes(), bounds[1].numpy().tobytes()

    def _cached_stats(self, source, actions):
        snapshot = self._snapshot(source)
        if snapshot is None:
            self._key = self._stats = None
            return source
        # Resolve cuda's concrete index from the action tensor, not a process
        # default-device alias. Each refresh allocates a new immutable snapshot;
        # it never overwrites storage potentially used by an earlier request.
        key = (snapshot, actions.device, actions.dtype)
        if key != self._key:
            mode, lower, upper = snapshot
            bounds = [torch.frombuffer(bytearray(raw), dtype=torch.float32)
                      .to(device=actions.device, dtype=actions.dtype, copy=True)
                      for raw in (lower, upper)]
            self._stats = NormStats(mode, *bounds)
            self._key = key
            self.refreshes += 1
        return self._stats

    def build_model_outputs(self, actions, robot_obs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Action norm cache is already serving a request")
        try:
            if self._closed:
                raise RuntimeError("Action norm cache is closed")
            processor = self._processor()
            if processor is None:
                raise RuntimeError("Action norm cache processor was released")
            if getattr(processor.build_model_outputs, "__func__", None) is not Pi05Processor.build_model_outputs:
                raise RuntimeError("Action norm cache output processor changed")
            if actions.requires_grad:
                return processor.build_model_outputs(actions, robot_obs)
            stats = self._cached_stats(processor._action_norm, actions)
            # Deliberately keep native arithmetic and order: unnormalize the
            # padded tensor, then relative actions, then trim/cast/D2H. Do not
            # precombine affine coefficients, fuse operations or move math CPU.
            actions = apply_norm(actions, stats, inverse=True)
            if processor.relative_actions.enabled:
                actions = processor.relative_actions.to_absolute(actions, robot_obs.get("state"))
            return actions.squeeze(0)[..., :processor.config.action_dim].float().cpu().numpy()
        finally:
            self._lock.release()

    def manifest(self):
        return {
            "version": "action-norm-device-value-cache-v1",
            "source_contract": "CPU FP32 vectors; other source contracts use original apply_norm",
            "refresh": "exact CPU value bytes plus mode/action device/action dtype; no version-only shortcut",
            "cache": "one immutable snapshot, replaced on refresh; no model graph capture",
            "arithmetic": "original apply_norm inverse on original action device/dtype, then relative actions and D2H",
            "inputs": "unchanged images, masks, tokens, state and noise",
            "ownership": "same processor NumPy result boundary; policy retains its existing owned output copy",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close action norm cache during a request")
        try:
            self._key = self._stats = None
            self._closed = True
        finally:
            self._lock.release()
