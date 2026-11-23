"""Default-off identity reuse within one locked, synchronous prepare call.

Every request still runs the complete native live collector. Registered tensor
identities come from that call, never a previous generation/signature. Private
packed identities are read once on demand; code that writes a private cache
must forget it before the write. Geometry and behavior checks are not cached.

The runner serializes requests and supported mutations. As with the original
collector, this is not an atomic snapshot against a different thread directly
mutating tensors without the runner's lock. No callback/model forward belongs
inside this scope. Standalone helper APIs do not inherit it implicitly.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
import weakref

from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_request_identity_installation'
VERSION = 'locked-prepare-full-native-identity-lookup-v1'
_LOCAL = threading.local()


class _RequestIdentities:
    def __init__(self, model, runner, generation, tensor_state):
        if (type(tensor_state) is not tuple or len(tensor_state) != 3
                or any(type(group) is not tuple for group in tensor_state)):
            raise ValueError('Request identity lookup requires the complete native signature triple')
        # NativeLifecycleSignatureInstallation owns the six-field contract.
        # Values are the original immutable tuples, not Tensor/model references.
        self._registered = {value[0]: value for group in tensor_state[:2] for value in group}
        self._private = {}
        self._model, self._runner = weakref.ref(model), weakref.ref(runner)
        self._generation = generation
        self._active = True

    def _assert_active(self):
        if not self._active or getattr(_LOCAL, 'scope', None) is not self:
            raise RuntimeError('Request identity lookup is expired or belongs to another thread')

    def assert_for(self, model, generation):
        self._assert_active()
        runner = self._runner()
        if (self._model() is not model or generation != self._generation
                or model._inference_cache.generation != generation
                or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked()):
            raise RuntimeError('Request identity lookup owner/model/generation changed')

    def identity(self, tensor):
        self._assert_active()
        tid = id(tensor)
        value = self._registered.get(tid)
        if value is not None:
            return value
        value = self._private.get(tid)
        if value is None:
            value = tensor_identity(tensor)
            self._private[tid] = value
        return value

    def forget(self, tensor):
        """Invalidate BEFORE a supported in-place private-cache write."""
        self._assert_active()
        tid = id(tensor)
        if tid in self._registered:
            raise RuntimeError('Registered weights cannot be written inside request identity scope')
        self._private.pop(tid, None)

    def close(self):
        self._active = False
        self._registered.clear()
        self._private.clear()


class RequestIdentityInstallation:
    def __init__(self, model, native):
        self._model, self._native = weakref.ref(model), weakref.ref(native)
        self.enabled = True

    def _active_model(self):
        from .native_lifecycle import INSTALLATION as NATIVE, NativeLifecycleSignatureInstallation
        model, native = self._model(), self._native()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or type(native) is not NativeLifecycleSignatureInstallation or not native.enabled
                or getattr(model, NATIVE, None) is not native):
            raise RuntimeError('Request identity lookup requires its active complete native collector')
        return model

    @contextmanager
    def prepare_scope(self, tensor_state, generation):
        """Internal core handoff immediately after this request's full collect.

        Do not save the yielded object in a model, graph entry or context. This
        context is only for QKV/MLP/Lt prepare, never their standalone execution.
        """
        from .graph import Pi05NativeGraphRunner
        model = self._active_model()
        if getattr(_LOCAL, 'scope', None) is not None:
            raise RuntimeError('Nested or cross-model request identity scopes are unsupported')
        owners = tuple(r for r in getattr(model, '_graph_runners', ()) if not r._closed)
        if (len(owners) != 1 or type(owners[0]) is not Pi05NativeGraphRunner
                or owners[0].model is not model or not owners[0]._call_lock.locked()
                or model._inference_cache.generation != generation):
            raise RuntimeError('Request identity lookup requires one locked native owner and current generation')
        scope = _RequestIdentities(model, owners[0], generation, tensor_state)
        _LOCAL.scope = scope
        try:
            yield scope
        finally:
            scope.close()
            del _LOCAL.scope

    def manifest(self):
        source = Path(__file__)
        return dict(version=VERSION, enabled=self.enabled, default_enabled=False,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    requirements='complete native lifecycle collector v1/v2; exactly one locked native runner',
                    source='complete live native collector in this same request',
                    fields=['object_id', 'data_ptr', 'version', 'dtype', 'device', 'shape'],
                    scope='locked synchronous QKV/MLP/Lt prepare only; finally-cleared thread-local scope',
                    private_cache='read once per request; explicit forget before every private copy',
                    standalone='always reads live identities; no implicit scope lookup',
                    cross_request_cache=False, retained_tensor_or_model_references=False,
                    concurrency='requests and supported mutations serialized by native owner lock')

    def uninstall(self):
        from .fused_expert_mlp import _mutation_guard
        model = self._active_model()
        with _mutation_guard(model, 'request identity lookup'):
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_request_identity_lookup(model):
    """Install after native lifecycle and before/with an idle native runner."""
    from .fused_expert_mlp import _mutation_guard
    from .native_lifecycle import INSTALLATION as NATIVE, NativeLifecycleSignatureInstallation
    with _mutation_guard(model, 'request identity lookup'):
        native = getattr(model, NATIVE, None)
        if type(native) is not NativeLifecycleSignatureInstallation or not native.enabled:
            raise ValueError('Request identity lookup requires the complete native lifecycle collector')
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Request identity lookup is already installed')
        handle = RequestIdentityInstallation(model, native)
        setattr(model, INSTALLATION, handle)
        try:
            model.invalidate_inference_caches(weights_changed=False)
        except BaseException:
            delattr(model, INSTALLATION)
            handle.enabled = False
            raise
        return handle
