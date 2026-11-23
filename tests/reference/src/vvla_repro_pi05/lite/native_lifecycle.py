"""Default-off CPython/ATen implementation of the full live lifecycle check.

Only loading/installation may compile. Each signature call traverses every live
registration and reads every original tensor-identity field. The extension is
CPU-only even when inspecting CUDA tensors: it never reads their contents or
initializes/synchronizes a GPU. Unsupported Python tensor/registry subclasses
are rejected explicitly; the separate Python collector remains the reference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import threading
import weakref

from .lifecycle import LifecycleSignature, collect_lifecycle_signature

VERSION = "native-live-full-lifecycle-v1"
VERSION_V2 = "native-live-full-lifecycle-v2-allocation"
VERSION_V3 = "native-live-full-lifecycle-v3-validated-value-reuse"
INSTALLATION = "_lite_native_lifecycle_signature_installation"
_SOURCE = Path(__file__).resolve().parent / "csrc/native_lifecycle.cpp"
_FLAGS = ("-O3",)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native_lifecycle(*, verbose=False, variant="v1"):
    """Build/load one host extension now; build failure rejects this candidate.

    Torch's content-addressed build cache contains code, never a model or a
    signature. v3 may reuse values already owned by the caller's inference
    cache after a complete fresh live check, without retaining them here.
    This function must be called before inference/graph capture.
    There is deliberately no lazy load or silent fallback in ``signature``.
    """
    if variant not in ("v1", "v2", "v3"):
        raise ValueError("Native lifecycle variant must be v1 or v2 or v3")
    source = _SOURCE if variant == "v1" else _SOURCE.with_name(f"native_lifecycle_{variant}.cpp")
    import torch
    from torch.utils.cpp_extension import load

    if sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("Native lifecycle requires CPython with the GIL")
    reference = Path(__file__).with_name("lifecycle.py")
    tensor_reference = Path(__file__).resolve().parents[1] / "models/pi05/inference_cache_pi05.py"
    build = {
        "version": {"v1": VERSION, "v2": VERSION_V2, "v3": VERSION_V3}[variant],
        "collector_variant": variant,
        "allocation_strategy": ("original" if variant == "v1" else
                                "constant-name-interning; per-call-64KiB-pmr-arena; owned-reference-move" if variant == "v2" else
                                "fresh native snapshot; caller-owned immutable triple reuse only after complete live comparison"),
        "source_sha256": {p.name: _sha(p) for p in (source, Path(__file__), reference, tensor_reference)},
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "python_version": sys.version,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "extra_cflags": list(_FLAGS),
        "with_cuda": False,
        "compiler_request": os.environ.get("CXX", "default C++ compiler"),
        "cuda_initialized_before_build": torch.cuda.is_initialized(),
    }
    if variant == "v3":
        core_source = tensor_reference.with_name("modeling_pi05.py")
        build["source_sha256"][core_source.name] = _sha(core_source)
        build["handoff_schema"] = "exact full tuple: six current outer fields plus three immutable lifecycle fields"
    # Runtime initialization state is diagnostic, not a build/source identity.
    initialized_before = build.pop("cuda_initialized_before_build")
    identity = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    name = "vvla_native_lifecycle_" + identity[:20]
    try:
        extension = load(name=name, sources=[str(source)], extra_cflags=list(_FLAGS),
                         with_cuda=False, verbose=verbose)
    except Exception as exc:
        raise RuntimeError("Native lifecycle build/load failed; candidate rejected (no fallback)") from exc
    if torch.cuda.is_initialized() != initialized_before:
        raise RuntimeError("Host-only native lifecycle loading unexpectedly initialized CUDA")
    build.update(module_name=name, compiler=extension.compiler(), binary_sha256=_sha(extension.__file__),
                 inventory_cache=False, signature_cache=(variant == "v3"), fallback="explicit rejection")
    if variant == "v3":
        build.update(signature_cache_owner="existing Pi05InferenceCache only; collector retains no result",
                     signature_reuse="existing immutable three subtuples after complete live validation",
                     all_live_fields_read_every_call=True, new_persistent_model_or_tensor_references=False,
                     comparison="exact scalar fields; no hash approximation or Python rich equality",
                     comparison_fallback="materialize the same native snapshot; never walk model twice",
                     registry_then_tensor_reads=True)
    return NativeLifecycleCollector(extension, build)


class NativeLifecycleCollector:
    """Loaded code/toolchain only. No references to model inventory or results."""
    def __init__(self, extension, build_manifest):
        self._extension = extension
        self._manifest = build_manifest

    def signature(self, model, *, previous=None):
        if self._manifest.get("collector_variant") == "v3":
            return LifecycleSignature(*self._extension.collect(model, previous))
        return LifecycleSignature(*self._extension.collect(model))

    def manifest(self):
        # Return a detached JSON document so callers cannot mutate provenance.
        return json.loads(json.dumps(self._manifest, sort_keys=True))


class _FreshLifecycleSnapshot:
    """Ephemeral proof issued only after a complete collector call returns.

    A previously retained signature, even one with the same generation, cannot
    substitute for this scope. The callback cannot retain a usable proof.
    """
    def __init__(self, installation, model, signature):
        self._installation, self._model, self._signature = installation, model, signature
        self._thread = threading.get_ident()
        self._active = True

    def require(self, installation, model):
        if (not self._active or self._thread != threading.get_ident()
                or self._installation is not installation or self._model is not model
                or installation._active_model() is not model
                or installation._fresh_snapshot is not self):
            raise RuntimeError('Lifecycle snapshot is not fresh in this native callback/thread')
        return self._signature

    def _retire(self):
        self._active = False
        self._installation = self._model = self._signature = None


class NativeLifecycleSignatureInstallation:
    def __init__(self, model, collector):
        self._model = weakref.ref(model)
        self._collector = collector
        self.enabled = True
        self._fresh_snapshot = None

    def _active_model(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Native lifecycle signature installation is no longer active")
        return model

    def signature(self, *, previous_full_signature=None):
        model = self._active_model()
        if self._collector._manifest.get("collector_variant") == "v3":
            # Core supplies its cache value. Never read model.__dict__ or cache
            # properties here: an extra Python attribute access can have effects.
            previous = (previous_full_signature[-3:]
                        if type(previous_full_signature) is tuple and len(previous_full_signature) == 9 else None)
            return self._collector.signature(model, previous=previous)
        # v1/v2 deliberately ignore the handoff and keep their original path.
        return self._collector.signature(model)

    def __call__(self):
        return self.signature()

    def consume_fresh_signature(self, consumer, *, previous_full_signature=None):
        """Run an opt-in consumer within this call's fresh v3 snapshot scope.

        Ordinary signature callers retain their unchanged path. This method
        does not cache inventory, remove a read, or alter collector semantics.
        """
        model = self._active_model()
        if self._collector._manifest.get('collector_variant') != 'v3':
            raise ValueError('Fresh inventory handoff requires native lifecycle v3')
        if self._fresh_snapshot is not None:
            raise RuntimeError('Nested native lifecycle snapshot consumers are unsupported')
        proof = None
        self._fresh_snapshot = object()  # Also guard re-entry during collection.
        try:
            signature = self.signature(previous_full_signature=previous_full_signature)
            proof = _FreshLifecycleSnapshot(self, model, signature)
            self._fresh_snapshot = proof
            return consumer(signature, proof)
        finally:
            self._fresh_snapshot = None
            if proof is not None:
                proof._retire()

    def manifest(self):
        result = self._collector.manifest()
        result.update(enabled=self.enabled,
                      returned_fields=["parameters", "buffers", "topology"],
                      tensor_identity_fields=["object_id", "data_ptr", "version", "dtype", "device", "shape"],
                      topology="all module nodes/types and all child/parameter/buffer slots, including None and aliases",
                      restrictions="CPython; exact dict registries; exact Tensor/Parameter; no shadowed identity fields")
        return result

    def uninstall(self):
        from .fused_expert_mlp import _mutation_guard
        model = self._active_model()
        with _mutation_guard(model, "native lifecycle signature"):
            if getattr(model, '_lite_vision_fp8_installation', None) is not None:
                raise RuntimeError('Uninstall vision FP8 before its native lifecycle collector')
            if getattr(model, '_lite_prefix_fp8_inventory_installation', None) is not None:
                raise RuntimeError('Uninstall prefix FP8 inventory before its native lifecycle collector')
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_native_lifecycle_signature(model, *, collector=None, verbose=False):
    """Compile at install, prove full parity once, then expose live checks.

    The consumer must use the complete returned triple. This does not alter
    model methods or select a core path; the independent option defaults off.
    Python/native installations are exclusive to make core selection explicit.
    """
    from torch import nn
    from .fused_expert_mlp import _mutation_guard
    from .lifecycle import INSTALLATION as PYTHON_INSTALLATION
    if not isinstance(model, nn.Module):
        raise TypeError("Native lifecycle installation requires a Torch module")
    with _mutation_guard(model, "native lifecycle signature"):
        if getattr(model, INSTALLATION, None) is not None or getattr(model, PYTHON_INSTALLATION, None) is not None:
            raise ValueError("A lifecycle signature collector is already installed")
        if collector is None:
            collector = load_native_lifecycle(verbose=verbose)
        if not isinstance(collector, NativeLifecycleCollector):
            raise TypeError("Expected a loaded NativeLifecycleCollector")
        if collector.signature(model) != collect_lifecycle_signature(model):
            raise RuntimeError("Native lifecycle differs from the complete Python reference")
        handle = NativeLifecycleSignatureInstallation(model, collector)
        setattr(model, INSTALLATION, handle)
        try:
            model.invalidate_inference_caches(weights_changed=False)
        except BaseException:
            delattr(model, INSTALLATION)
            handle.enabled = False
            raise
        return handle
