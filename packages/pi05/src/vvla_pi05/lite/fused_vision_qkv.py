"""Opt-in biased SigLIP QKV packing with original module/parameter ownership.

Only the projections in the audited Transformers 5.19 attention forward change.
The packed weight and bias are private, fixed-address inference caches refreshed
at the model's original generation boundary, before any empty-image encoding.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import inspect
from pathlib import Path
from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.models.siglip.modeling_siglip import SiglipAttention, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = "_lite_vision_qkv_installation"
PACKER = "_lite_vision_qkv"
VERSION = "siglip-bf16-biased-qkv-concat-v1"


def _projections(attention):
    if type(attention) is not SiglipAttention:
        raise ValueError("Vision QKV requires the audited Transformers SiglipAttention")
    modules = tuple(getattr(attention, name) for name in ("q_proj", "k_proj", "v_proj"))
    if any(type(module) is not nn.Linear or module.bias is None or "forward" in module.__dict__
           or module._forward_hooks or module._forward_pre_hooks for module in modules):
        raise ValueError("Vision QKV requires original biased Linear projections without forward hooks")
    width, heads, head_dim = attention.embed_dim, attention.num_heads, attention.head_dim
    if (any(type(value) is not int or value < 1 for value in (width, heads, head_dim))
            or heads * head_dim != width):
        raise ValueError("Vision QKV requires valid hidden width / attention head geometry")
    device = modules[0].weight.device
    for module in modules:
        if (module.in_features != width or module.out_features != width
                or tuple(module.weight.shape) != (width, width) or tuple(module.bias.shape) != (width,)
                or any(t.dtype != torch.bfloat16 or t.device != device or not t.is_contiguous()
                       for t in (module.weight, module.bias))):
            raise ValueError("Vision QKV requires matching contiguous BF16 weight AND bias; no implicit bias cast")
    return modules


def _structure(attention):
    return (attention.embed_dim, attention.num_heads, attention.head_dim, attention.scale,
            attention.dropout, attention.is_causal, attention.config._attn_implementation)


def _source_identity(modules):
    return tuple((id(module), tensor_identity(module.weight), tensor_identity(module.bias)) for module in modules)


def _packed_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
    if (self.training or torch.is_grad_enabled()
            or torch.is_autocast_enabled(hidden_states.device.type)):
        return SiglipAttention.forward(self, hidden_states, attention_mask=attention_mask, **kwargs)
    # The remainder is the audited SiglipAttention.forward, with only the three
    # projection calls replaced. In particular kwargs are not newly forwarded
    # to a backend that the original implementation did not receive them on.
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, k, v = getattr(self, PACKER).project(hidden_states)
    queries = q.view(hidden_shape).transpose(1, 2)
    keys = k.view(hidden_shape).transpose(1, 2)
    values = v.view(hidden_shape).transpose(1, 2)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self, queries, keys, values, attention_mask,
        is_causal=self.is_causal, scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.out_proj(attn_output)
    return attn_output, attn_weights


class PackedVisionQKV:
    def __init__(self, attention):
        self._attention = weakref.ref(attention)
        modules = _projections(attention)
        self.width = attention.embed_dim
        self.device = modules[0].weight.device
        self.structure = _structure(attention)
        self._weight = torch.empty((3 * self.width, self.width), dtype=torch.bfloat16, device=self.device)
        self._bias = torch.empty((3 * self.width,), dtype=torch.bfloat16, device=self.device)
        self._source_identity = self._packed_identity = None
        self._prepared_compile_token = None
        self._forward = MethodType(_packed_attention_forward, attention)

    def validate_geometry(self):
        attention = self._attention()
        if (attention is None or getattr(attention, PACKER, None) is not self
                or attention.__dict__.get("forward") is not self._forward):
            raise RuntimeError("Vision QKV installation/forward was replaced externally")
        modules = _projections(attention)
        if _structure(attention) != self.structure or modules[0].weight.device != self.device:
            raise ValueError("Vision QKV structure/device changed; clear graphs and reinstall packing")
        return modules

    def assert_ready(self):
        if (self._packed_identity is None
                or (tensor_identity(self._weight), tensor_identity(self._bias)) != self._packed_identity):
            raise RuntimeError("Vision QKV cache is unprepared or modified; prepare the model lifecycle")

    def assert_source_current(self, modules):
        if _source_identity(modules) != self._source_identity:
            raise RuntimeError("Vision QKV source changed; call prepare_inference_context with its current generation")

    @torch.no_grad()
    def refresh(self):
        modules = self.validate_geometry()
        for row, module in enumerate(modules):
            start, end = row * self.width, (row + 1) * self.width
            self._weight[start:end].copy_(module.weight)
            self._bias[start:end].copy_(module.bias)
        self._source_identity = _source_identity(modules)
        self._packed_identity = (tensor_identity(self._weight), tensor_identity(self._bias))

    def project(self, x):
        attention = self._attention()
        if attention.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type):
            return tuple(getattr(attention, name)(x) for name in ("q_proj", "k_proj", "v_proj"))
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError("Vision QKV compilation requires the native runner's prepared scope")
        else:
            modules = self.validate_geometry()
            self.assert_ready()
            self.assert_source_current(modules)
        if (x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[-1] != self.width
                or x.dtype != torch.bfloat16 or x.device != self.device):
            raise ValueError("Vision QKV requires BF16 [batch, tokens, hidden] on the original device")
        # Bias is consumed by native F.linear, not added to a BF16 GEMM result.
        return F.linear(x, self._weight, self._bias).split(self.width, dim=-1)


@contextmanager
def _mutation_guard(model):
    if model.training or torch.is_inference_mode_enabled():
        raise ValueError("Install/uninstall vision QKV in eval mode outside inference_mode")
    with ExitStack() as locks:
        for runner in tuple(getattr(model, "_graph_runners", ())):
            lock = getattr(runner, "_call_lock", None)
            if lock is None or not lock.acquire(blocking=False):
                raise RuntimeError("Cannot mutate vision QKV during an active/unknown graph request")
            locks.callback(lock.release)
            if runner.entries:
                raise ValueError("Clear all live graph runners before changing vision QKV packing")
        if not callable(getattr(model, "invalidate_inference_caches", None)):
            raise TypeError("Vision QKV installation requires the shared model lifecycle")
        yield


class VisionQKVInstallation:
    def __init__(self, model, packers):
        self._model = weakref.ref(model)
        self._packers = packers
        self._generation = None
        self.enabled = True
        self.refreshes = 0  # Diagnostics, excluded from manifest identity.

    def _assert_active(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Vision QKV installation is inactive")
        return model

    def prepare(self, generation):
        self._assert_active()
        if type(generation) is not int or generation < 0:
            raise ValueError("Vision QKV prepare requires an integer model generation")
        modules = [packer.validate_geometry() for packer in self._packers]
        if generation == self._generation:
            for packer, projections in zip(self._packers, modules, strict=True):
                packer.assert_ready()
                packer.assert_source_current(projections)
            return
        # All geometry is validated before the first write. Explicit epochs
        # refresh even .data writes whose tensor version counters did not move.
        for packer in self._packers:
            packer.refresh()
        self._generation = generation
        self.refreshes += 1

    def compilation_members(self, generation):
        self._assert_active()
        if self._generation != generation:
            raise RuntimeError("Vision QKV compilation requires the current prepared generation")
        for packer in self._packers:
            modules = packer.validate_geometry()
            packer.assert_ready()
            packer.assert_source_current(modules)
        return tuple(self._packers)

    def manifest(self):
        hf = Path(inspect.getsourcefile(SiglipAttention))
        return {
            "version": VERSION, "enabled": self.enabled, "transformers": transformers.__version__,
            "dtype": "bfloat16", "bias_dtype": "bfloat16", "split_order": ["q", "k", "v"],
            "bias_boundary": "one native F.linear(input, packed_weight, packed_bias); no separate bias add",
            "attention": "audited native SiglipAttention interface, head layout, mask and output projection",
            "norms_and_residuals": "unchanged; outside this attention projection helper",
            "layers": [{"layer": i, "width": p.width, "heads": p.structure[1], "head_dim": p.structure[2]}
                       for i, p in enumerate(self._packers)],
            "packed_weight_bytes": sum(3 * p.width * p.width * 2 for p in self._packers),
            "packed_bias_bytes": sum(3 * p.width * 2 for p in self._packers),
            "refresh": "fixed-address copies at original generation, before empty-image feature preparation",
            "compile": "requires prepared native-owner scope; standalone packed compilation fails closed",
            "numerics": "native BF16 boundaries preserved; changed GEMM shape may change GPU reduction rounding",
            "source_sha256": {
                "fused_vision_qkv.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "transformers/models/siglip/modeling_siglip.py": hashlib.sha256(hf.read_bytes()).hexdigest(),
            },
        }

    def uninstall(self):
        model = self._assert_active()
        with _mutation_guard(model):
            for packer in self._packers:
                packer.validate_geometry()
            model.invalidate_inference_caches(weights_changed=False)
            for packer in self._packers:
                attention = packer._attention()
                delattr(attention, "forward")
                delattr(attention, PACKER)
                packer._weight = packer._bias = packer._packed_identity = None
            delattr(model, INSTALLATION)
            self.enabled = False


def install_vision_qkv(model):
    """Install after BF16 checkpoint/recipe loading, before graph construction."""
    with _mutation_guard(model):
        if transformers.__version__ != "5.19.0":
            raise RuntimeError("Vision QKV forward is audited for transformers==5.19.0 only")
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Vision QKV is already installed")
        layers = model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers
        if not len(layers):
            raise ValueError("Vision QKV requires SigLIP encoder layers")
        attentions = tuple(layer.self_attn for layer in layers)
        for attention in attentions:
            _projections(attention)
            if (getattr(attention, PACKER, None) is not None or "forward" in attention.__dict__
                    or getattr(attention.forward, "__func__", None) is not SiglipAttention.forward):
                raise ValueError("Vision QKV requires the original unpatched attention forward")
        packers = [PackedVisionQKV(attention) for attention in attentions]
        installation = VisionQKVInstallation(model, packers)
        model.invalidate_inference_caches(weights_changed=False)
        for attention, packer in zip(attentions, packers, strict=True):
            setattr(attention, PACKER, packer)
            attention.forward = packer._forward
        setattr(model, INSTALLATION, installation)
        return installation
