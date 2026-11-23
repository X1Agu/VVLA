"""Independent expert gate/up packing and GELU-multiply inference candidates.

Both options default off. Original modules, parameters and checkpoint keys are
retained. Packed weights are private caches refreshed by the shared model epoch;
no packed storage is replaced while an old graph might still reference it.
"""

from contextlib import ExitStack, contextmanager
from functools import partial
import hashlib
from pathlib import Path
from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = "_lite_expert_mlp_installation"
VERSION = "expert-bf16-gate-up-gelu-v1"


def _projections(mlp):
    from transformers.activations import GELUTanh
    from transformers.models.gemma.modeling_gemma import GemmaMLP

    if type(mlp) is not GemmaMLP or type(mlp.act_fn) is not GELUTanh:
        raise ValueError("Expert MLP requires the original GemmaMLP and GELUTanh activation")
    activation = mlp.act_fn.act
    if (not isinstance(activation, partial) or activation.func is not F.gelu or activation.args
            or activation.keywords != {"approximate": "tanh"} or "forward" in mlp.act_fn.__dict__):
        raise ValueError("Expert MLP requires the original Torch GELU(tanh) activation")
    modules = tuple(getattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))
    if any(type(module) is not nn.Linear or module.bias is not None or "forward" in module.__dict__
           or module._forward_hooks or module._forward_pre_hooks for module in modules):
        raise ValueError("Expert MLP requires original bias-free Linear projections without hooks")
    gate, up, down = modules
    # Keep shape checks ahead of dtype/device/stride, and do not fetch down's
    # weight until the original gate/up shape comparison has passed.
    gate_weight = gate.weight
    gate_shape = gate_weight.shape
    up_weight = up.weight
    if gate_shape != up_weight.shape:
        raise ValueError("Expert MLP requires compatible contiguous BF16 gate/up/down weights")
    down_weight = down.weight
    if down_weight.shape != (gate.in_features, gate.out_features):
        raise ValueError("Expert MLP requires compatible contiguous BF16 gate/up/down weights")
    for weight in (gate_weight, up_weight, down_weight):
        if (weight.dtype != torch.bfloat16 or weight.device != gate_weight.device
                or not weight.is_contiguous()):
            raise ValueError("Expert MLP requires compatible contiguous BF16 gate/up/down weights")
    return modules


def _behavior_identity(mlp):
    activation = mlp.act_fn
    return (id(activation), id(activation.act),
            tuple((id(module), id(module.__dict__.get("forward")),
                   tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                  for module in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)),
            id(activation.__dict__.get("forward")))


class ExpertMLPCandidate:
    """Private state, not a module/buffer; every original parameter stays owned."""

    def __init__(self, mlp, *, concatenate, fuse_gelu_mul):
        self._mlp = weakref.ref(mlp)
        gate, _, _ = _projections(mlp)
        self.width, self.intermediate = gate.in_features, gate.out_features
        self.device = gate.weight.device
        self._options = (concatenate, fuse_gelu_mul)
        self._installed_options = self._options
        self._original_forward = type(mlp).forward
        self._forward = None
        self._packed = (torch.empty((2 * self.intermediate, self.width), device=self.device, dtype=torch.bfloat16)
                        if concatenate else None)
        self._source_identity = self._packed_identity = self._behavior = None
        self._prepared_compile_token = None

    def validate(self):
        mlp = self._mlp()
        if mlp is None or mlp.__dict__.get("forward") is not self._forward:
            raise RuntimeError("Expert MLP forward was replaced externally")
        if self._options != self._installed_options:
            raise RuntimeError("Expert MLP options changed; clear graphs and reinstall")
        modules = _projections(mlp)
        if (modules[0].weight.shape != (self.intermediate, self.width)
                or any(module.weight.device != self.device for module in modules)):
            raise ValueError("Expert MLP geometry/device changed; clear graphs and reinstall")
        return modules

    @torch.no_grad()
    def refresh(self, *, identity_lookup=None):
        modules = self.validate()
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        if self._packed is not None:
            if identity_lookup is not None:
                identity_lookup.forget(self._packed)
            self._packed[:self.intermediate].copy_(modules[0].weight)
            self._packed[self.intermediate:].copy_(modules[1].weight)
            self._packed_identity = identity(self._packed)
        self._source_identity = tuple(identity(module.weight) for module in modules)
        self._behavior = _behavior_identity(self._mlp())

    def assert_ready(self, *, identity_lookup=None):
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        mlp = self._mlp()
        if (mlp is None or mlp.__dict__.get("forward") is not self._forward
                or self._options != self._installed_options):
            raise RuntimeError("Expert MLP installation was modified externally")
        if self._source_identity is None or _behavior_identity(mlp) != self._behavior:
            raise RuntimeError("Expert MLP is unprepared or its modules changed; prepare through model lifecycle")
        if self._packed is not None and identity(self._packed) != self._packed_identity:
            raise RuntimeError("Expert MLP packed cache was modified; rebuild through model lifecycle")

    def forward(self, x):
        mlp = self._mlp()
        if mlp.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type):
            return self._original_forward(mlp, x)
        if self._options[0]:
            if torch.compiler.is_compiling():
                if self._prepared_compile_token is None:
                    raise RuntimeError("Packed MLP compilation requires the native runner's prepared scope")
            else:
                self.assert_ready()
                modules = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
                if tuple(tensor_identity(module.weight) for module in modules) != self._source_identity:
                    raise RuntimeError("Expert MLP weights changed; call prepare_inference_context before inference")
        # GELU-only reads the live original Linear weights directly. Its pure
        # tensor forward needs no detached-cache identity lookup; the outer model
        # prepare still validates every parameter and this installation's modules.
        if (x.dtype != torch.bfloat16 or x.device != self.device or x.ndim not in (2, 3)
                or x.shape[-1] != self.width or not x.is_contiguous() or any(size < 1 for size in x.shape)):
            raise ValueError("Expert MLP input must be contiguous nonempty BF16 [B,D] or [B,S,D]")
        if self._options[0]:
            gate, up = F.linear(x, self._packed).chunk(2, dim=-1)
        else:
            gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        if self._options[1] and x.is_cuda:
            from vvla_repro_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
            # Existing kernel explicitly rounds GELU to BF16 before multiplication
            # and stores the product in BF16 before the down projection.
            hidden = gelu_tanh_mul(gate, up)
        else:
            hidden = mlp.act_fn(gate) * up
        return mlp.down_proj(hidden)


@contextmanager
def _mutation_guard(model, feature):
    if model.training or torch.is_inference_mode_enabled():
        raise ValueError(f"Install/uninstall {feature} in eval mode outside inference_mode")
    with ExitStack() as locks:
        for runner in tuple(getattr(model, "_graph_runners", ())):
            lock = getattr(runner, "_call_lock", None)
            if lock is None or not lock.acquire(blocking=False):
                raise RuntimeError(f"Cannot mutate {feature} during an active/unknown graph request")
            locks.callback(lock.release)
            if runner.entries:
                raise ValueError(f"Clear all live graph runners before changing {feature}")
        if not callable(getattr(model, "invalidate_inference_caches", None)):
            raise TypeError(f"{feature} requires the shared model lifecycle")
        yield


class ExpertMLPInstallation:
    def __init__(self, model, candidates, options):
        self._model = weakref.ref(model)
        self._candidates = candidates
        self._options = options
        self._generation = None
        self.enabled = True
        self.refreshes = 0

    @property
    def concatenate(self):
        """Compile compatibility: True uses a detached, lifecycle-checked cache."""
        return self._options[0]

    @property
    def fuse_gelu_mul(self):
        return self._options[1]

    def prepare(self, generation, *, identity_lookup=None):
        if not self.enabled or getattr(self._model(), INSTALLATION, None) is not self:
            raise RuntimeError("Expert MLP installation is inactive or replaced")
        if identity_lookup is not None:
            identity_lookup.assert_for(self._model(), generation)
        if generation == self._generation:
            for candidate in self._candidates:
                candidate.assert_ready(identity_lookup=identity_lookup)
            return
        for candidate in self._candidates:
            candidate.validate()
        for candidate in self._candidates:
            candidate.refresh(identity_lookup=identity_lookup)
        self._generation = generation
        self.refreshes += 1

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / "models/pi05"
        paths = (Path(__file__), core / "modeling_pi05.py", core / "fused_mlp_pi05.py")
        return {"version": VERSION, "enabled": self.enabled,
                "concatenate_gate_up": self._options[0], "fuse_gelu_mul": self._options[1],
                "dtype": "bfloat16", "split_order": ["gate", "up"],
                "rounding": "BF16 GEMM outputs -> BF16 GELU(tanh) -> BF16 product -> BF16 down GEMM",
                "enable_fp_fusion": False, "gelu_kernel": "shared prefix gelu_tanh_mul",
                "numerical_limit": "GEMM shape and tanh lowering may change values; GPU parity required",
                "packed_weight_bytes": sum(4 * c.width * c.intermediate for c in self._candidates)
                    if self._options[0] else 0,
                "layers": [{"layer": i, "width": c.width, "intermediate": c.intermediate}
                           for i, c in enumerate(self._candidates)],
                "refresh": "original model generation; same packed addresses before graph execution",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}

    def compilation_members(self, generation):
        model = self._model()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or self._generation != generation):
            raise RuntimeError("MLP compilation requires the current prepared generation")
        if not self.concatenate:
            return ()
        for candidate in self._candidates:
            modules = candidate.validate()
            candidate.assert_ready()
            if tuple(tensor_identity(module.weight) for module in modules) != candidate._source_identity:
                raise RuntimeError("Expert MLP weights changed; call prepare_inference_context before inference")
        return tuple(self._candidates)

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Expert MLP installation is no longer active")
        with _mutation_guard(model, "expert MLP"):
            if any(c._mlp().__dict__.get("forward") is not c._forward for c in self._candidates):
                raise RuntimeError("An installed expert MLP forward was modified externally")
            for candidate in self._candidates:
                delattr(candidate._mlp(), "forward")
                candidate._packed = candidate._packed_identity = None
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_expert_mlp(model, *, concatenate=False, fuse_gelu_mul=False):
    """Install either/both independent candidates; both options default off."""
    if type(concatenate) is not bool or type(fuse_gelu_mul) is not bool:
        raise ValueError("Expert MLP options must be booleans")
    if not (concatenate or fuse_gelu_mul):
        raise ValueError("Enable at least one expert MLP candidate explicitly")
    with _mutation_guard(model, "expert MLP"):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Expert MLP candidate is already installed")
        mlps = [layer.mlp for layer in model.paligemma_with_expert.gemma_expert.model.layers]
        if not mlps or len({id(mlp) for mlp in mlps}) != len(mlps):
            raise ValueError("Expert MLP requires distinct nonempty layers")
        for mlp in mlps:
            if mlp.training or "forward" in mlp.__dict__:
                raise ValueError("Expert MLP requires eval modules with original forwards")
            _projections(mlp)
        candidates = [ExpertMLPCandidate(mlp, concatenate=concatenate, fuse_gelu_mul=fuse_gelu_mul) for mlp in mlps]
        for mlp, candidate in zip(mlps, candidates, strict=True):
            def forward(module, x, _candidate=candidate):
                return _candidate.forward(x)
            candidate._forward = MethodType(forward, mlp)
            mlp.forward = candidate._forward
        installation = ExpertMLPInstallation(model, candidates, (concatenate, fuse_gelu_mul))
        setattr(model, INSTALLATION, installation)
        model.invalidate_inference_caches(weights_changed=False)
        return installation
