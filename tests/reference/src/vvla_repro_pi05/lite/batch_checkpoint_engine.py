"""Private fixed-checkpoint model-state seal for the new B32 API only.

Cold sealing and explicit audits execute the unchanged full shared collector
and every installed owner's prepare. Dynamic request packing, masks, positions,
noise, graph keys, inference and owned outputs remain in their current paths.
"""
import threading
import weakref

import torch

from .math_settings import math_settings as _math_settings

INSTALLATION = "_lite_batch_checkpoint_engine"
_CAPABILITY = object()


class BatchSealedModelState:
    def __init__(self, policy, capability):
        if capability is not _CAPABILITY:
            raise TypeError("B32 state is constructed only by its private checkpoint factory")
        model, runner = policy.model, policy.graph_runner
        if (runner is None or runner.entries or getattr(model, INSTALLATION, None) is not None
                or getattr(model, "_lite_fixed_checkpoint_engine", None) is not None
                or not policy._lock.locked() or not runner._call_lock.locked()):
            raise RuntimeError("B32 sealing requires a fresh unique native graph owner and both locks")
        self._policy, self._model, self._runner = weakref.ref(policy), weakref.ref(model), weakref.ref(runner)
        self._closed, self._math = False, _math_settings()
        with torch.inference_mode():
            self._state = model._prepare_inference_model_state(policy.device.type)
        self._cache = model._inference_cache
        self._generation, self._signature = self._cache.generation, self._cache.signature
        if (self._signature is None or self._state[1] != self._generation or self._state[2] is not None):
            raise RuntimeError("B32 cold prepare must establish its actual prefix-only initialized model generation")
        from .pi05 import _POLICY_HANDLE_FIELDS
        self._thread_owners = tuple((owner, field) for name in _POLICY_HANDLE_FIELDS
            if (owner := getattr(policy, name, None)) is not None
            for field in ("_thread", "_request_thread") if hasattr(owner, field))
        setattr(model, INSTALLATION, self)

    def _assert_live(self, model=None):
        policy, actual, runner = self._policy(), self._model(), self._runner()
        if (self._closed or policy is None or policy._closed or actual is None or runner is None or runner._closed
                or policy.model is not actual or policy.graph_runner is not runner
                or (model is not None and model is not actual) or getattr(actual, INSTALLATION, None) is not self):
            raise RuntimeError("B32 fixed checkpoint engine is closed or detached")
        if (actual._inference_cache is not self._cache or self._cache.generation != self._generation
                or self._cache.signature is not self._signature):
            raise RuntimeError("B32 fixed checkpoint generation changed; reconstruct the engine")
        if torch.is_autocast_enabled(policy.device.type) or _math_settings() != self._math:
            raise RuntimeError("B32 fixed checkpoint autocast/process math settings changed")
        return policy, actual, runner

    def prepare_model_state(self, model, device_type):
        policy, actual, runner = self._assert_live(model)
        if device_type != policy.device.type or not policy._lock.locked() or not runner._call_lock.locked():
            raise RuntimeError("B32 fixed checkpoint requires serialized policy/native graph ownership")
        for owner, field in self._thread_owners:
            setattr(owner, field, threading.get_ident())
        return self._state

    def audit(self):
        policy, model, runner = self._assert_live()
        if not policy._lock.locked() or not runner._call_lock.locked():
            raise RuntimeError("B32 explicit full audit requires both owner locks")
        with torch.inference_mode():
            state = model._prepare_inference_model_state(policy.device.type)
        if state != self._state:
            raise RuntimeError("B32 full audited generation/scope changed")
        self._assert_live()

    def detach(self, policy, runner):
        if (self._policy() is not policy or not policy._lock.locked()
                or (runner is not None and not runner._call_lock.locked())):
            raise RuntimeError("B32 retirement requires both owner locks")
        self._closed = True
        model = self._model()
        if model is not None and getattr(model, INSTALLATION, None) is self:
            delattr(model, INSTALLATION)
