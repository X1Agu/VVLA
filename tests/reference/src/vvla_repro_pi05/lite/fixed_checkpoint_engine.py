"""Private fixed-checkpoint deployment, with fresh dynamic requests.

The ordinary mutable Pi05LitePolicy remains the default. This separate serving
API owns construction through graph retirement and exposes no model, processor,
runner, installation, tensor storage or mutable configuration. Changing a model
requires close and checkpoint reconstruction. Accessing private implementation
objects, monkeypatching code, or mutating process libraries is outside this API.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import threading
import weakref

import torch

from .fixed_checkpoint_engine_evidence import (
    SOURCE_PATHS, canonical, digest, make_manifest, validate_engine_binding,
)

_CAPABILITY = object()
_INSTALLATION = '_lite_fixed_checkpoint_engine'


def _sources():
    package = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((package / path.split('src/vvla_repro_pi05/', 1)[1]).read_bytes()).hexdigest()
            for name, path in SOURCE_PATHS.items()}


def _math_settings():
    return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            str(torch.backends.cuda.preferred_blas_library()))


class _SealedModelState:
    """Private cold model state; never a cached observation or inference context."""
    def __init__(self, policy, capability):
        if capability is not _CAPABILITY:
            raise TypeError('Fixed checkpoint state is only constructed by its checkpoint factory')
        model, runner = policy.model, policy.graph_runner
        if (runner is None or getattr(model, _INSTALLATION, None) is not None
                or policy._calls != 0 or runner.entries):
            raise RuntimeError('Fixed checkpoint sealing requires a new private native graph policy')
        self._policy, self._model, self._runner = weakref.ref(policy), weakref.ref(model), weakref.ref(runner)
        self._closed = False
        self._math = _math_settings()
        # Required packed-weight initialization and owner generation checks are
        # executed, not fabricated from a retained previous signature.
        with torch.inference_mode():
            self._state = model._prepare_inference_model_state(policy.device.type)
        self._cache = model._inference_cache
        self._generation, self._signature = self._cache.generation, self._cache.signature
        if self._signature is None or self._state[1] != self._generation or self._state[2] is not None:
            raise RuntimeError('Fixed checkpoint sealing requires the actual prefix-only initialized generation')
        from .pi05 import _POLICY_HANDLE_FIELDS
        self._thread_owners = tuple((owner, field) for name in _POLICY_HANDLE_FIELDS
            if (owner := getattr(policy, name, None)) is not None
            for field in ('_thread', '_request_thread') if hasattr(owner, field))
        setattr(model, _INSTALLATION, self)
        self._manifest_bytes = None

    def _assert_live(self, model=None):
        policy, actual, runner = self._policy(), self._model(), self._runner()
        if (self._closed or policy is None or policy._closed or actual is None or runner is None
                or runner._closed or policy.model is not actual or policy.graph_runner is not runner
                or (model is not None and model is not actual) or getattr(actual, _INSTALLATION, None) is not self):
            raise RuntimeError('Fixed checkpoint engine is closed or detached from its private owner')
        if (actual._inference_cache is not self._cache or self._cache.generation != self._generation
                or self._cache.signature is not self._signature):
            raise RuntimeError('Fixed checkpoint model epoch changed; close and construct a new engine')
        if torch.is_autocast_enabled(policy.device.type) or _math_settings() != self._math:
            raise RuntimeError('Fixed checkpoint autocast or process math settings changed')
        return policy, actual, runner

    def prepare_model_state(self, model, device_type):
        policy, actual, runner = self._assert_live(model)
        if device_type != policy.device.type or not policy._lock.locked() or not runner._call_lock.locked():
            raise RuntimeError('Fixed checkpoint requests require their serialized policy and native runner')
        thread = threading.get_ident()
        # Shared owners require current-thread authorization on cold graph
        # construction too. No weights, modules, topology or content are read.
        for owner, field in self._thread_owners:
            setattr(owner, field, thread)
        return self._state

    def audit_base(self):
        policy, model, runner = self._assert_live()
        if not policy._lock.locked() or not runner._call_lock.locked():
            raise RuntimeError('Fixed checkpoint audit requires exclusive policy and runner ownership')
        # Explicit audit hooks still run the original full model collector and
        # original packed-owner validation. They are outside request timings.
        with torch.inference_mode():
            current = model._prepare_inference_model_state(policy.device.type)
        if current != self._state:
            raise RuntimeError('Fixed checkpoint audited generation or scope changed')
        self._assert_live()
        return policy.metadata()

    def manifest(self, base):
        native = base['lite_optimizations']['native_lifecycle_manifest']
        value = make_manifest(source_sha256=_sources(),
            base_quantization_manifest_sha256=digest(base['quantization_manifest']),
            native_lifecycle_manifest_sha256=digest(native))
        validate_engine_binding(value, base_quantization_manifest=base['quantization_manifest'],
            native_lifecycle_manifest=native)
        if self._manifest_bytes is None:
            self._manifest_bytes = canonical(value)
        elif canonical(value) != self._manifest_bytes:
            raise RuntimeError('Fixed checkpoint cold source, build or precision seal changed')
        return value

    def detach(self, policy, runner):
        # Cleanup must remain available after a failed math/epoch guard. No
        # inference validity check may prevent synchronous retirement.
        model = self._model()
        if (self._policy() is not policy or not policy._lock.locked()
                or (runner is not None and not runner._call_lock.locked())):
            raise RuntimeError('Fixed checkpoint retirement requires exclusive ownership')
        self._closed = True
        if model is not None and getattr(model, _INSTALLATION, None) is self:
            delattr(model, _INSTALLATION)


class FixedCheckpointEngine:
    """Fixed checkpoint B1 serving API; model changes require reconstruction."""
    __slots__ = ('__policy', '__state')

    def __init__(self, *args, **kwargs):
        raise TypeError('Use FixedCheckpointEngine.from_pretrained; external models/processors are unsupported')

    def __setattr__(self, name, value):
        raise AttributeError('Fixed checkpoint engine settings are immutable; close and construct a new engine')

    def __reduce_ex__(self, protocol):
        raise TypeError('Fixed checkpoint engines cannot be serialized or expose their private model')

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        if cls is not FixedCheckpointEngine:
            raise TypeError('Fixed checkpoint engine subclasses are unsupported')
        from .pi05 import Pi05LitePolicy
        overrides = dict(kwargs.pop('config_overrides', None) or {})
        if 'lite_fixed_checkpoint_engine' in overrides and overrides['lite_fixed_checkpoint_engine'] is not True:
            raise ValueError('FixedCheckpointEngine.from_pretrained requires its explicit enabled option')
        overrides['lite_fixed_checkpoint_engine'] = True
        return Pi05LitePolicy.from_pretrained(checkpoint, config_overrides=overrides, **kwargs)

    def _live_policy(self):
        state = object.__getattribute__(self, '_FixedCheckpointEngine__state')
        return state._assert_live()[0]

    def infer(self, raw_obs, seed):
        return self._live_policy().infer(raw_obs, seed)

    def predict(self, raw_obs, noise):
        return self._live_policy().predict(raw_obs, noise)

    def reset(self, episode_id=None):
        return self._live_policy().reset(episode_id)

    def prepared_fixture(self, raw_obs, noise):
        return self._live_policy().prepared_fixture(raw_obs, noise)

    def metadata(self):
        policy = self._live_policy()
        state = object.__getattribute__(self, '_FixedCheckpointEngine__state')
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError('Cannot audit during an active fixed checkpoint request')
        runner = policy.graph_runner
        if not runner._call_lock.acquire(blocking=False):
            policy._lock.release()
            raise RuntimeError('Cannot audit during an active native graph request')
        try:
            base = state.audit_base()
            manifest = state.manifest(base)
            saved = policy._lite_optimizations.get('fixed_checkpoint_engine_manifest')
            if (policy._lite_optimizations.get('fixed_checkpoint_engine') is not True
                    or canonical(saved) != canonical(manifest)):
                raise RuntimeError('Fixed checkpoint deployment declaration changed')
            result = deepcopy(base)
            result['quantization_manifest']['fixed_checkpoint_engine'] = manifest
            result['recipe_components']['quantization_manifest_sha256'] = digest(result['quantization_manifest'])
            result['recipe_sha256'] = digest(result['recipe_components'])
            return result
        finally:
            runner._call_lock.release()
            policy._lock.release()

    def validate_precision_manifest(self):
        return self.metadata()['quantization_manifest']

    def precision_backend_report(self):
        return self._live_policy().precision_backend_report()

    def preflight(self, raw_observation, noise, *, repeat=3):
        report = self._live_policy().preflight(raw_observation, noise, repeat=repeat)
        report['metadata'] = self.metadata()
        return report

    def close(self):
        policy = object.__getattribute__(self, '_FixedCheckpointEngine__policy')
        if policy._closed:
            return
        state = object.__getattribute__(self, '_FixedCheckpointEngine__state')
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close during an active fixed checkpoint request')
        runner = policy.graph_runner
        if runner is not None and not runner._call_lock.acquire(blocking=False):
            policy._lock.release()
            raise RuntimeError('Cannot close during an active native graph request')
        try:
            # Actual retirement remains the original synchronous graph/foreign
            # plan cleanup. Detach only this deployment seal before uninstall.
            state.detach(policy, runner)
        finally:
            if runner is not None:
                runner._call_lock.release()
            policy._lock.release()
        policy.close()


def _seal_loaded_policy(policy):
    """Private continuation of the checkpoint factory; not a public model API."""
    from .pi05 import Pi05LitePolicy, _json_hash
    if type(policy) is not Pi05LitePolicy:
        raise TypeError('Fixed checkpoint deployment requires the private checkpoint-created policy')
    engine = object.__new__(FixedCheckpointEngine)
    if not policy._lock.acquire(blocking=False):
        raise RuntimeError('Cannot seal an active policy')
    runner = policy.graph_runner
    if runner is None or not runner._call_lock.acquire(blocking=False):
        policy._lock.release()
        raise RuntimeError('Cannot seal without exclusive native graph ownership')
    state = None
    try:
        state = _SealedModelState(policy, _CAPABILITY)
        base = state.audit_base()
        manifest = state.manifest(base)
        policy._lite_optimizations.update(fixed_checkpoint_engine=True, fixed_checkpoint_engine_manifest=manifest)
        policy._provenance['recipe_sha256'] = _json_hash(dict(config=asdict(policy.config),
            lite_optimizations=policy._lite_optimizations))
        object.__setattr__(engine, '_FixedCheckpointEngine__policy', policy)
        object.__setattr__(engine, '_FixedCheckpointEngine__state', state)
    except BaseException:
        if state is not None and getattr(policy.model, _INSTALLATION, None) is state:
            delattr(policy.model, _INSTALLATION)
            state._closed = True
        raise
    finally:
        runner._call_lock.release()
        policy._lock.release()
    try:
        engine.metadata()  # Actual final sealed metadata, before public release.
    except BaseException as error:
        try:
            engine.close()
        except BaseException as cleanup:
            error.add_note('Fixed checkpoint final-audit cleanup failed: '+str(cleanup))
        raise
    return engine
