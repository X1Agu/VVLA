"""Strict, opt-in compilation of already-prepared native-graph tensor work.

The graph owner retains every lifecycle check, stream dependency, static input,
metadata refresh and output clone. This helper never compiles preprocessing or
prepare_inference_context and never enables a second CUDA graph owner.
"""

import hashlib
from pathlib import Path
from types import MethodType, FunctionType
import threading

import torch

VERSION = "prepared-sample-fullgraph-v2"
INDUCTOR_OPTIONS = {
    "triton.cudagraphs": False,
    "emulate_precision_casts": True,
    "use_fast_math": False,
}
_PACKED_INSTALLATIONS = ("_lite_expert_qkv_installation", "_lite_expert_mlp_installation",
                         "_lite_vision_qkv_installation", "_lite_expert_cublaslt_installation",
                         "_lite_vision_fc2_cublaslt_installation", "_lite_prefix_fp8_installation",
                         "_lite_prefix_nvfp4_installation", "_lite_prefix_fp8_gelu_installation",
                         "_lite_vision_fp8_installation", "_lite_dynamic_fp8_installation")


def _packed_handles(model):
    handles = []
    for name in _PACKED_INSTALLATIONS:
        handle = getattr(model, name, None)
        if handle is None:
            continue
        if name == "_lite_expert_mlp_installation" and getattr(handle, "concatenate", None) is False:
            continue
        if not callable(getattr(handle, "compilation_members", None)):
            raise ValueError("Compiled sampling is not validated with this packed installation")
        handles.append(handle)
    return tuple(handles)


def validate_compile_model(model):
    """Fail before compilation rather than remove checks or allow graph breaks."""
    if model.training:
        raise ValueError("Compiled sampling requires model.eval()")
    _packed_handles(model)


def compilation_manifest(enabled):
    from .compile_environment import receipt
    return {
        "enabled": bool(enabled), "version": VERSION, "backend": "inductor",
        "fullgraph": True, "dynamic": False, "options": dict(INDUCTOR_OPTIONS),
        "scope": "sample_actions with externally prepared context and attention metadata",
        "packed_scope": "locked native owner, current generation and original/packed identities checked outside Dynamo",
        "cuda_graph_owner": "Pi05NativeGraphRunner", "torch": torch.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "libdevice_environment": receipt(),
    }


class _PreparedCompiledRun:
    def __init__(self, model, compiled, handles, *, owner, context, code=None):
        from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity
        self.model, self.compiled, self.handles = model, compiled, handles
        self.owner, self.context = owner, context
        self.code=code
        self.active = True
        self.scope_token = object()
        self.epoch = None if context is None else context["token"]
        self.parameter_snapshot = tuple(tensor_identity(x) for x in model.parameters()) if handles else ()
        self.buffer_snapshot = tuple(tensor_identity(x) for x in model.buffers()) if handles else ()
        self.scope_lock = model.__dict__.setdefault("_lite_packed_compile_lock", threading.Lock()) if handles else None

    def retire(self):
        if self.active and self.code is not None:
            # This code object belongs only to this graph entry. Clear its
            # compiler cache after the graph owner's drain, never global state.
            torch._dynamo.reset_code(self.code)
        self.active = False

    def __call__(self):
        if self.scope_lock is None:
            return self._invoke()
        if not self.scope_lock.acquire(blocking=False):
            raise RuntimeError("Another packed compilation scope is active")
        try:
            return self._invoke()
        finally:
            self.scope_lock.release()

    def _invoke(self):
        from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity
        if not self.active:
            raise RuntimeError("Compiled graph entry was retired")
        validate_compile_model(self.model)
        members = []
        if self.handles:
            owner = self.owner
            if (owner is None or owner.model is not self.model or owner._closed
                    or not owner._call_lock.locked() or owner._token != self.epoch
                    or self.context["token"] != self.epoch
                    or self.model._inference_cache.generation != self.epoch[1]):
                raise RuntimeError("Packed compilation requires its locked owner and current prepared generation")
            if (self.parameter_snapshot != tuple(tensor_identity(x) for x in self.model.parameters())
                    or self.buffer_snapshot != tuple(tensor_identity(x) for x in self.model.buffers())):
                raise RuntimeError("Compiled inputs changed; prepare model lifecycle and recapture")
            if _packed_handles(self.model) != self.handles:
                raise RuntimeError("Packed compilation installation changed")
            for handle in self.handles:
                members.extend(handle.compilation_members(self.epoch[1]))
            if any(member._prepared_compile_token is not None for member in members):
                raise RuntimeError("Another packed compilation scope is active")
        for member in members:
            member._prepared_compile_token = self.scope_token
        try:
            return self.compiled()
        finally:
            for member in members:
                member._prepared_compile_token = None


def compile_prepared_run(model, run, *, backend="inductor", owner=None, context=None):
    """Compile one static entry; tests may inject eager/counting backends.

    This call is lazy. The native runner invokes it twice before starting CUDA
    capture, so fullgraph errors and compilation cost belong to the cold path.
    Failure propagates; no eager fallback or global compiler reset is performed.
    """
    validate_compile_model(model)
    handles = _packed_handles(model)
    if handles and (owner is None or context is None):
        raise ValueError("Packed compilation requires a native owner and prepared context")
    options = INDUCTOR_OPTIONS.copy() if backend == "inductor" else None
    if options is not None:
        from .compile_environment import ensure_compile_environment
        ensure_compile_environment(options)
    if type(run) is not FunctionType:
        raise TypeError('A prepared graph entry requires its own Python function closure')
    # Dynamo attaches its specialization budget to a Python code object. All
    # closures from GraphRunner._make_run otherwise share the same budget and
    # fail after enough B/mask/model entries. Copy unchanged bytecode so each
    # statically captured graph owns its compiler cache. No source rewriting,
    # changed constants, global cache reset or raised compiler limits.
    code=run.__code__.replace()
    isolated=FunctionType(code,run.__globals__,run.__name__,run.__defaults__,run.__closure__)
    isolated.__kwdefaults__=run.__kwdefaults__
    compiled = torch.compile(isolated, backend=backend, fullgraph=True, dynamic=False, options=options)
    return _PreparedCompiledRun(model, compiled, handles, owner=owner, context=context,code=code)


@torch.library.custom_op("vvla_pi05_lite::preserved_fp32_linear", mutates_args=())
def _preserved_fp32_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


@_preserved_fp32_linear.register_fake
def _preserved_fp32_linear_fake(x, weight, bias):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _preserved_linear_forward(module, x):
    if torch.compiler.is_compiling():
        if x.dtype != torch.float32 or module.weight.dtype != torch.float32 or module.bias.dtype != torch.float32:
            raise ValueError("Preserved action linear requires its original FP32 boundary")
        return _preserved_fp32_linear(x, module.weight, module.bias)
    return torch.nn.Linear.forward(module, x)


class PreservedActionLinears:
    def __init__(self, model, modules):
        self.model, self.modules, self.enabled = model, modules, True
        self.forwards = tuple(MethodType(_preserved_linear_forward, module) for module in modules)
        for module, forward in zip(modules, self.forwards, strict=True):
            module.forward = forward

    def manifest(self):
        return {"version": "opaque-native-fp32-action-linear-v1", "enabled": self.enabled,
                "modules": ["action_in_proj", "action_out_proj"],
                "operator": "vvla_pi05_lite::preserved_fp32_linear", "arithmetic": "native torch.nn.functional.linear",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def uninstall(self):
        from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
        with _mutation_guard(self.model, "preserved action linears"):
            if not self.enabled or getattr(self.model, "_lite_preserved_action_linears", None) is not self:
                raise RuntimeError("Preserved action linear installation is inactive")
            if any(module.__dict__.get("forward") is not forward
                   for module, forward in zip(self.modules, self.forwards, strict=True)):
                raise RuntimeError("Preserved action linear forward changed externally")
            for module in self.modules:
                delattr(module, "forward")
            delattr(self.model, "_lite_preserved_action_linears")
            self.enabled = False
            self.model.invalidate_inference_caches(weights_changed=False)


def install_preserved_action_linears(model):
    """Independent diagnostic candidate; original parameters/modules stay owned."""
    from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
    with _mutation_guard(model, "preserved action linears"):
        if getattr(model, "_lite_preserved_action_linears", None) is not None:
            raise ValueError("Preserved action linears already installed")
        modules = tuple(getattr(model, name) for name in ("action_in_proj", "action_out_proj"))
        if any(type(module) is not torch.nn.Linear or module.bias is None
               or module.weight.dtype != torch.float32 or module.bias.dtype != torch.float32
               or "forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks
               for module in modules):
            raise ValueError("Preserved action linears require original FP32 bias-bearing Linear modules without hooks")
        handle = PreservedActionLinears(model, modules)
        model._lite_preserved_action_linears = handle
        model.invalidate_inference_caches(weights_changed=False)
        return handle
