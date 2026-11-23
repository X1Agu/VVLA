"""Optional 17-MLP GELU/FP8 conversion extension over the original 51-site owner.

No weights/scales are packed, replaced or cached here. The base owner retains
all immutable-epoch, full-live, generation, math and original parameter checks.
The new arithmetic has its own executed contract evidence and manifest entry.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import weakref

import torch

from . import prefix_fp8 as base_module
from . import prefix_fp8_gelu as primitive
from . import prefix_fp8_gelu_kernel as kernel
from .fused_expert_mlp import _mutation_guard
from .prefix_fp8_gelu_evidence import load_evidence, validate_fusion_manifest

INSTALLATION = '_lite_prefix_fp8_gelu_installation'
VERSION = 'pi05-prefix-fp8-gelu-owned-17-v1'


class _MLPMember:
    def __init__(self, owner, index, module):
        self.owner, self.index, self.module = owner, index, module
        self.sites = tuple(owner.base.sites[index * 3:index * 3 + 3])
        self._prepared_compile_token = None
        self.original_forward = module.__dict__.get('forward')
        self.original_class_forward = type(module).forward
        def forward(module, x, _member=self):
            return _member.call(module, x)
        self.forward = MethodType(forward, module)

    def validate(self):
        from ..models.pi05.precision_pi05 import FusedPrefixMLP
        module = self.module
        if (type(module) is not FusedPrefixMLP or module.training or module.concatenate
                or module.fuse_down_quant or module._fusion_flags != (False, False)
                or module.gate_up_weight is not None or module._forward_hooks or module._forward_pre_hooks
                or type(module).forward is not self.original_class_forward
                or module.__dict__.get('forward') is not self.forward
                or any(site.module is not getattr(module, name) for site, name in
                       zip(self.sites, base_module.PROJECTIONS, strict=True))
                or any(site is not self.owner.base.sites[self.index * 3 + offset]
                       for offset, site in enumerate(self.sites))):
            raise RuntimeError('Prefix GELU exact MLP/site/forward binding changed')

    def call(self, module, x):
        if (module is not self.module or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(x.device.type) or x.dtype != torch.bfloat16
                or x.device != self.sites[0].device or x.ndim != 3 or x.shape[0] != 1
                or x.shape[1] not in primitive.SUPPORTED_CAPACITIES or x.shape[2] != 2048
                or not x.is_contiguous()):
            raise ValueError('Prefix GELU fusion requires the original B1 BF16 inference boundary')
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None or any(site._prepared_compile_token is None for site in self.sites):
                raise RuntimeError('Prefix GELU requires all base and extension prepared compile members')
        else:
            self.owner.base.assert_execution_owner()
            self.owner.validate_bindings()
            for site in self.sites: site.validate()
        # Preserve independent gate/up conversion and GEMMs. Only the rounded
        # GELU*up materialization + down converter become the new exact kernel.
        gate, up = module.gate_proj(x), module.up_proj(x)
        down = self.sites[2]
        y, _, _ = primitive.scaled_down(gate.view(-1, 16384), up.view(-1, 16384), down.quantized, down.scale)
        return y.view(1, x.shape[1], 2048)


class PrefixFP8GeluInstallation:
    def __init__(self, model, base, evidence, contracts, raw_files):
        self._model, self.base = weakref.ref(model), base
        self.enabled = True
        self.evidence, self.contracts = deepcopy(evidence), deepcopy(contracts)
        self.raw_evidence = dict(raw_files)
        self.members = tuple(_MLPMember(self, index, layer.mlp)
                             for index, layer in enumerate(base_module._layers(model)[:17]))
        self._installed_members = self.members
        self._lookup = {id(member.module): member for member in self.members}
        self._bindings = (primitive.gelu_quantize_rows, primitive.scaled_down, kernel.launch)
        self._implementation = primitive.implementation_manifest()
        self._manifest = dict(schema=1, version=VERSION, enabled=True,
            scope='prefix_mlp_gelu_down_conversion_layers_0_16',
            modules=[f'prefix.{i}.mlp' for i in range(17)],
            implementation=deepcopy(self._implementation), contracts=deepcopy(contracts),
            owner_source_sha256=base_module._sha(__file__),
            evidence_validator_sha256=base_module._sha(Path(__file__).with_name('prefix_fp8_gelu_evidence.py')),
            lifecycle=dict(owner='Pi05NativeGraphRunner', base_owner='immutable_51_site_prefix_fp8',
                packed_weights_reused=True, source_parameters_preserved=True,
                independent_activation_cache=False, request_local_outputs=True,
                full_base_live_checks_preserved=True, extension_compile_members=17,
                retirement_order=['graph', 'gelu_extension', 'base_fp8_owner']),
            arithmetic_scope='only GELU_tanh/BF16 multiply/down input quantization; gate/up and GEMM unchanged',
            fallback_permitted=False, quality_validated=False, performance_validated=False)

    def accepts_forward(self, module):
        member = self._lookup.get(id(module))
        return (self.enabled and member is not None and member.module is module
                and module.__dict__.get('forward') is member.forward)

    def accepts_original_forward(self, module, original):
        member = self._lookup.get(id(module))
        return (member is not None and member.original_forward is original
                and original is None and self.accepts_forward(module))

    def validate_bindings(self):
        model = self._model()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or type(self.base) is not base_module.PrefixFP8Installation or not self.base.enabled
                or getattr(model, base_module.INSTALLATION, None) is not self.base
                or self._bindings != (primitive.gelu_quantize_rows, primitive.scaled_down, kernel.launch)
                or len(self.members) != 17 or len(self._lookup) != 17
                or any(a is not b for a, b in zip(self.members, self._installed_members, strict=True))):
            raise RuntimeError('Prefix GELU extension installation/operator/member ownership changed')
        layers = base_module._layers(model)
        for member, layer in zip(self.members, layers[:17], strict=True):
            if member.module is not layer.mlp or self._lookup.get(id(layer.mlp)) is not member:
                raise RuntimeError('Prefix GELU ordered MLP topology changed')
            member.validate()

    def compilation_members(self, generation):
        self.base.assert_execution_owner()
        if generation != self.base._generation:
            raise RuntimeError('Prefix GELU stale prepared generation')
        self.validate_bindings()
        return self.members

    def manifest(self):
        self.validate_bindings()
        if base_module._canonical(primitive.implementation_manifest()) != base_module._canonical(self._implementation):
            raise RuntimeError('Prefix GELU primitive source/implementation changed inside epoch')
        return deepcopy(validate_fusion_manifest(self._manifest, self.base.backend))

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Prefix GELU extension is inactive')
        with _mutation_guard(model, 'prefix GELU extension uninstall'):
            runner = None if self.base._runner is None else self.base._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Drain native graph before restoring prefix GELU forwards')
            # Refuse to overwrite an unrelated external mutation. Original
            # forwards and ownership remain reachable for explicit recovery.
            if any(m.module.__dict__.get('forward') is not m.forward for m in self._installed_members):
                raise RuntimeError('Prefix GELU forward changed externally; cannot safely restore')
            for member in reversed(self._installed_members):
                delattr(member.module, 'forward')
                member._prepared_compile_token = None
            delattr(model, INSTALLATION)
            self.enabled = False
            model.invalidate_inference_caches(weights_changed=False)


def install_prefix_fp8_gelu(model, evidence):
    """After the 51-site base, before any runner binding/capture; no fallback."""
    with _mutation_guard(model, 'prefix GELU extension installation'):
        base = getattr(model, base_module.INSTALLATION, None)
        if (getattr(model, INSTALLATION, None) is not None or type(base) is not base_module.PrefixFP8Installation
                or not base.enabled or base._runner is not None or base._generation is not None
                # The actual model creates this registry lazily when the first
                # graph runner is constructed. Cold installation must not do so.
                or tuple(getattr(model, '_graph_runners', ())) or len(base.sites) != 51
                or base.backend.get('quantizer_variant') != base_module.native.VARIANT):
            raise ValueError('Prefix GELU requires an unbound original SM120 51-site FP8 owner')
        base._validate(full=True)
        hardware = base_module._hardware(base.sites[0].device)
        raw_files, contracts = load_evidence(evidence, hardware)
        owner = PrefixFP8GeluInstallation(model, base, evidence, contracts, raw_files)
        installed = []
        try:
            for member in owner.members:
                if 'forward' in member.module.__dict__:
                    raise ValueError('Prefix GELU cannot replace an existing MLP instance forward')
                member.module.forward = member.forward
                installed.append(member)
            setattr(model, INSTALLATION, owner)
            # Validate the complete extension while rollback is still local;
            # base GEMM authorization alone never authorizes this new kernel.
            owner.manifest()
            model.invalidate_inference_caches(weights_changed=False)
            return owner
        except BaseException:
            # No device work is submitted by installation and no tensor is
            # allocated/repacked. Roll back only bindings installed by us.
            for member in reversed(installed):
                if member.module.__dict__.get('forward') is member.forward:
                    delattr(member.module, 'forward')
            if getattr(model, INSTALLATION, None) is owner: delattr(model, INSTALLATION)
            owner.enabled = False
            raise
