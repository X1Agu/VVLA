"""Opt-in expert BF16 QKV projection packing, preserving original parameters.

Packed weights are private inference caches, never checkpoint state. The shared
context lifecycle refreshes them after its complete original-weight signature
check and before capture/replay. Generation changes update fixed addresses on
the caller's stream, so old graphs never refer to freed packed allocations.
"""

from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = "_lite_expert_qkv_installation"
PACKER = "_lite_expert_qkv"
VERSION = "expert-bf16-qkv-concat-v1"


def _projections(attention):
    modules = tuple(getattr(attention, name) for name in ("q_proj", "k_proj", "v_proj"))
    if any(type(module) is not nn.Linear or module.bias is not None for module in modules):
        raise ValueError("Expert QKV requires original bias-free Linear projections")
    q, k, v = modules
    if (q.in_features != k.in_features or q.in_features != v.in_features
            or k.out_features != v.out_features or q.out_features % k.out_features
            or any(module.out_features % attention.head_dim for module in modules)):
        raise ValueError("Expert QKV requires matching contiguous BF16 weights and GQA geometry")
    # Bind only inside this call, after the original geometry short circuits.
    # Read later weights lazily: an earlier invalid dtype/device/stride must
    # still reject before a later missing weight can raise AttributeError.
    q_weight = q.weight
    for module in modules:
        weight = q_weight if module is q else module.weight
        if (weight.dtype != torch.bfloat16 or weight.device != q_weight.device
                or not weight.is_contiguous()):
            raise ValueError("Expert QKV requires matching contiguous BF16 weights and GQA geometry")
    return modules


class PackedExpertQKV:
    """Private cache attached to attention; not an nn.Module or registered state."""

    def __init__(self, attention):
        self._attention = weakref.ref(attention)
        modules = _projections(attention)
        self.width = modules[0].in_features
        self.sizes = tuple(module.out_features for module in modules)
        self.device = modules[0].weight.device
        self._packed = torch.empty((sum(self.sizes), self.width), device=self.device, dtype=torch.bfloat16)
        self._packed_identity = None
        self._source_identity = None
        self._prepared_compile_token = None

    def validate_geometry(self):
        attention = self._attention()
        if attention is None or getattr(attention, PACKER, None) is not self:
            raise RuntimeError("Expert QKV installation was replaced externally")
        modules = _projections(attention)
        if (tuple(module.out_features for module in modules) != self.sizes
                or any(module.in_features != self.width or module.weight.device != self.device for module in modules)):
            raise ValueError("QKV geometry/device changed; clear graphs and reinstall packing")
        return modules

    @torch.no_grad()
    def refresh(self, *, identity_lookup=None):
        modules = self.validate_geometry()
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        if identity_lookup is not None:
            identity_lookup.forget(self._packed)
        offset = 0
        # Do not allocate a replacement: previous graph executions may still
        # own this address. The graph runner orders the caller stream first.
        for size, module in zip(self.sizes, modules, strict=True):
            self._packed[offset:offset + size].copy_(module.weight)
            offset += size
        self._source_identity = tuple(identity(module.weight) for module in modules)
        self._packed_identity = identity(self._packed)

    def assert_ready(self, *, identity_lookup=None):
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        if self._packed_identity is None or identity(self._packed) != self._packed_identity:
            raise RuntimeError("QKV packed cache is unprepared or modified; rebuild through model lifecycle")

    def project(self, x):
        attention = self._attention()
        modules = (attention.q_proj, attention.k_proj, attention.v_proj)
        if attention.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type):
            return tuple(module(x) for module in modules)
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError("Packed QKV compilation requires the native runner's prepared scope")
        else:
            self.assert_ready()
            if tuple(tensor_identity(module.weight) for module in modules) != self._source_identity:
                raise RuntimeError("QKV source changed; call prepare_inference_context before inference")
        if x.dtype != torch.bfloat16 or x.device != self.device or x.shape[-1] != self.width:
            raise ValueError("QKV projection preserves the original BF16 input boundary")
        return F.linear(x, self._packed).split(self.sizes, dim=-1)


@contextmanager
def _mutation_guard(model):
    if model.training or torch.is_inference_mode_enabled():
        raise ValueError("Install/uninstall QKV in eval mode outside inference_mode")
    with ExitStack() as locks:
        for runner in tuple(getattr(model, "_graph_runners", ())):
            lock = getattr(runner, "_call_lock", None)
            if lock is None or not lock.acquire(blocking=False):
                raise RuntimeError("Cannot mutate QKV during an active/unknown graph request")
            locks.callback(lock.release)
            if runner.entries:
                raise ValueError("Clear all live graph runners before changing QKV packing")
        if not callable(getattr(model, "invalidate_inference_caches", None)):
            raise TypeError("QKV installation requires the shared model lifecycle")
        yield


class ExpertQKVInstallation:
    def __init__(self, model, packers):
        self._model = weakref.ref(model)
        self._packers = packers
        self._generation = None
        self.enabled = True
        self.refreshes = 0  # Diagnostics only, deliberately excluded from identity.

    def prepare(self, generation, *, identity_lookup=None):
        if not self.enabled:
            raise RuntimeError("QKV installation is inactive")
        if identity_lookup is not None:
            identity_lookup.assert_for(self._model(), generation)
        if generation == self._generation:
            for packer in self._packers:
                if getattr(packer._attention(), PACKER, None) is not packer:
                    raise RuntimeError("Expert QKV installation was replaced externally")
                packer.assert_ready(identity_lookup=identity_lookup)
            return
        # Validate all projections before any copy. Even an explicit .data write
        # followed by invalidate must refresh, despite unchanged tensor versions.
        for packer in self._packers:
            packer.validate_geometry()
        for packer in self._packers:
            packer.refresh(identity_lookup=identity_lookup)
        self._generation = generation
        self.refreshes += 1

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / "models/pi05/modeling_pi05.py"
        return {"version": VERSION, "enabled": self.enabled, "dtype": "bfloat16",
                "split_order": ["q", "k", "v"], "head_layout": "BHSD views; original RoPE and KV cat",
                "layers": [{"layer": i, "input_width": p.width, "output_widths": list(p.sizes)}
                           for i, p in enumerate(self._packers)],
                "packed_weight_bytes": sum(sum(p.sizes) * p.width * 2 for p in self._packers),
                "refresh": "original model generation; in-place packed copies before graph execution",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), core)}}

    def compilation_members(self, generation):
        """Validate outside Dynamo; never refresh a stale context implicitly."""
        model = self._model()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or self._generation != generation):
            raise RuntimeError("QKV compilation requires the current prepared generation")
        for packer in self._packers:
            modules = packer.validate_geometry()
            packer.assert_ready()
            if tuple(tensor_identity(module.weight) for module in modules) != packer._source_identity:
                raise RuntimeError("QKV source changed; call prepare_inference_context before inference")
        return tuple(self._packers)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("QKV installation is no longer active")
        with _mutation_guard(model):
            for packer in self._packers:
                if getattr(packer._attention(), PACKER, None) is not packer:
                    raise RuntimeError("A QKV packer was modified externally")
            for packer in self._packers:
                delattr(packer._attention(), PACKER)
                packer._packed = None
                packer._packed_identity = None
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_expert_qkv(model):
    """Install after checkpoint/recipe loading and before constructing graphs."""
    with _mutation_guard(model):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Expert QKV packing is already installed")
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        if not layers:
            raise ValueError("Expert QKV requires at least one expert layer")
        for layer in layers:
            if getattr(layer.self_attn, PACKER, None) is not None:
                raise ValueError("Expert QKV requires unmodified attention packer slots")
            _projections(layer.self_attn)
        packers = [PackedExpertQKV(layer.self_attn) for layer in layers]
        for layer, packer in zip(layers, packers, strict=True):
            setattr(layer.self_attn, PACKER, packer)
        installation = ExpertQKVInstallation(model, packers)
        setattr(model, INSTALLATION, installation)
        model.invalidate_inference_caches(weights_changed=False)
        return installation
