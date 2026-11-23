"""Default-off, version-pinned expert BF16 cuBLASLt candidate.

Installation order: existing QKV and MLP candidates, this helper, native runner,
then bind_runner(runner). Shared-core prepare must call this installation AFTER
QKV/MLP prepare. Compile scopes must include this INSTALLATION. Retire graphs and
uninstall this helper BEFORE uninstalling its QKV/MLP prerequisites.
"""
import hashlib
from pathlib import Path
import threading
from types import MethodType
import weakref

import torch
from torch import nn

from vvla_repro_pi05.lite.cublaslt_runtime import ExpertLtRuntime, expert_cublaslt, profile_specs
from vvla_repro_pi05.lite.fused_expert_mlp import _mutation_guard
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity

INSTALLATION = '_lite_expert_cublaslt_installation'
VERSION = 'expert-cublaslt-lifecycle-v2'


def _fallback(module, x):
    return module.training or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)


class _Site:
    def __init__(self, installation, label, source):
        self.installation, self.label, self.source = installation, label, source
        self.plan = installation.runtime.plans[label]
        self.spec = self.plan.spec
        self.device = source().device
        self._weight = (torch.empty((self.spec['k'], self.spec['n']), dtype=torch.bfloat16, device=self.device)
                        if self.spec['layout'] == 'contiguous_kn' else source())
        self._source_identity = self._weight_identity = None
        self._prepared_compile_token = None

    def validate(self):
        source = self.source()
        if (source.dtype != torch.bfloat16 or source.device != self.device
                or tuple(source.shape) != (self.spec['n'], self.spec['k']) or not source.is_contiguous()):
            raise ValueError('cuBLASLt source shape/dtype/device changed; clear graphs and reinstall')
        return source

    def refresh(self):
        self._refresh_validated(self.validate())

    def _refresh_validated(self, source, *, identity_lookup=None):
        # Only consume a source validated in this synchronous prepare call.
        # Never retain a validation result across requests or generations.
        if self.spec['layout'] == 'contiguous_kn':
            if identity_lookup is not None:
                identity_lookup.forget(self._weight)
            self._weight.copy_(source.T)
        else:
            # Original packed allocations may not be replaced while a graph owns
            # their addresses. Shared source installers retain this same contract.
            if self._weight is not source:
                raise RuntimeError('cuBLASLt NK source allocation changed; clear graphs and reinstall')
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        self._source_identity = identity(source)
        self._weight_identity = identity(self._weight)

    def assert_ready(self):
        if self._source_identity is None:
            raise RuntimeError('cuBLASLt weight cache changed or is unprepared; use model lifecycle')
        self._assert_validated_ready(self.validate())

    def _assert_validated_ready(self, source, *, identity_lookup=None):
        if self._source_identity is None:
            raise RuntimeError('cuBLASLt weight cache changed or is unprepared; use model lifecycle')
        read_identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        identity = read_identity(source)
        # NK aliases the source. Check both expected identities against the same
        # current identity, rather than rereading that one Tensor twice.
        weight_identity = identity if self._weight is source else read_identity(self._weight)
        if identity != self._source_identity or weight_identity != self._weight_identity:
            raise RuntimeError('cuBLASLt weight cache changed or is unprepared; use model lifecycle')

    def __call__(self, x):
        if torch.compiler.is_compiling():
            if self._prepared_compile_token is None:
                raise RuntimeError('cuBLASLt compilation requires the native prepared scope')
        else:
            self.installation.assert_execution_owner()
            self.assert_ready()
        if (x.dtype != torch.bfloat16 or x.device != self.device or x.ndim not in (2, 3)
                or x.shape[-1] != self.spec['k'] or x.numel() != 50 * self.spec['k']
                or not x.is_contiguous()):
            raise ValueError('cuBLASLt candidate requires contiguous B1/H50 BF16 expert inputs')
        return expert_cublaslt(x, self._weight, self.plan.key, self.spec['n'])


class _LayerCalls:
    def __init__(self, installation, layer, packer, candidate):
        self.installation, self.layer, self.packer, self.candidate = installation, layer, packer, candidate
        self.original_qkv = packer.project
        self.original_mlp = candidate.forward
        self.original_out = layer.self_attn.o_proj.forward
        self.sites = {}
        self.sites['qkv'] = _Site(installation, 'qkv', lambda: packer._packed)
        mlp = layer.mlp
        if candidate._options[0]:
            self.sites['gate_up_packed'] = _Site(installation, 'gate_up_packed', lambda: candidate._packed)
        else:
            self.sites['gate'] = _Site(installation, 'gate_or_up', lambda: mlp.gate_proj.weight)
            self.sites['up'] = _Site(installation, 'gate_or_up', lambda: mlp.up_proj.weight)
        if 'down' in installation.runtime.plans:
            self.sites['down'] = _Site(installation, 'down', lambda: mlp.down_proj.weight)
        self.sites['attention_out'] = _Site(installation, 'attention_out', lambda: layer.self_attn.o_proj.weight)
        self.qkv_forward = self.project
        self.mlp_forward = self.mlp
        def out_forward(module, x, _calls=self):
            return _calls.out(module, x)
        self.out_forward = MethodType(out_forward, layer.self_attn.o_proj)

    def install(self):
        self.packer.project = self.qkv_forward
        self.candidate.forward = self.mlp_forward
        self.layer.self_attn.o_proj.forward = self.out_forward

    def validate(self):
        if (self.packer.__dict__.get('project') is not self.qkv_forward
                or self.candidate.__dict__.get('forward') is not self.mlp_forward
                or self.layer.self_attn.o_proj.__dict__.get('forward') is not self.out_forward):
            raise RuntimeError('cuBLASLt projection dispatch was replaced externally')
        projections = self.packer.validate_geometry()
        modules = self.candidate.validate()
        out = self.layer.self_attn.o_proj
        if (type(out) is not nn.Linear or out.bias is not None or out._forward_hooks or out._forward_pre_hooks
                or out.weight.dtype != torch.bfloat16 or tuple(out.weight.shape) != (1024, 2048)):
            raise ValueError('cuBLASLt attention output projection geometry or behavior changed')
        sources = tuple((site, site.validate()) for site in self.sites.values())
        return projections, modules, sources

    def _assert_validated_sources(self, validated, *, identity_lookup=None):
        # Private, stack-local handoff from validate(), under the native request
        # lock. The standalone assert_sources() below still reads live state.
        projections, modules, sources = validated
        identity = tensor_identity if identity_lookup is None else identity_lookup.identity
        self.packer.assert_ready(identity_lookup=identity_lookup)
        self.candidate.assert_ready(identity_lookup=identity_lookup)
        if tuple(identity(module.weight) for module in projections) != self.packer._source_identity:
            raise RuntimeError('cuBLASLt QKV sources changed; prepare model lifecycle')
        if tuple(identity(module.weight) for module in modules) != self.candidate._source_identity:
            raise RuntimeError('cuBLASLt MLP sources changed; prepare model lifecycle')
        for site, source in sources:
            site._assert_validated_ready(source, identity_lookup=identity_lookup)

    def assert_sources(self):
        self.packer.assert_ready()
        self.candidate.assert_ready()
        projections = self.packer.validate_geometry()
        if tuple(tensor_identity(module.weight) for module in projections) != self.packer._source_identity:
            raise RuntimeError('cuBLASLt QKV sources changed; prepare model lifecycle')
        modules = self.candidate.validate()
        if tuple(tensor_identity(module.weight) for module in modules) != self.candidate._source_identity:
            raise RuntimeError('cuBLASLt MLP sources changed; prepare model lifecycle')
        for site in self.sites.values():
            site.assert_ready()

    def project(self, x):
        if _fallback(self.layer.self_attn, x):
            return self.original_qkv(x)
        if not torch.compiler.is_compiling():
            self.assert_sources()
        return self.sites['qkv'](x).split(self.packer.sizes, dim=-1)

    def mlp(self, x):
        mlp = self.layer.mlp
        if _fallback(mlp, x):
            return self.original_mlp(x)
        if not torch.compiler.is_compiling():
            self.assert_sources()
        if self.candidate._options[0]:
            gate, up = self.sites['gate_up_packed'](x).chunk(2, dim=-1)
        else:
            gate, up = self.sites['gate'](x), self.sites['up'](x)
        if self.candidate._options[1] and x.is_cuda:
            from vvla_repro_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
            hidden = gelu_tanh_mul(gate, up)
        else:
            hidden = mlp.act_fn(gate) * up
        return self.sites['down'](hidden) if 'down' in self.sites else mlp.down_proj(hidden)

    def out(self, module, x):
        if _fallback(module, x):
            return self.original_out(x)
        return self.sites['attention_out'](x)

    def restore(self):
        # Both prerequisites originally expose class methods, never instance overrides.
        delattr(self.packer, 'project')
        delattr(self.candidate, 'forward')
        delattr(self.layer.self_attn.o_proj, 'forward')


class ExpertCublasLtInstallation:
    def __init__(self, model, runtime, qkv, mlp, profile, include_down):
        self._model = weakref.ref(model)
        self.runtime, self.qkv, self.mlp = runtime, qkv, mlp
        self.profile, self.include_down = profile, include_down
        self.layers = []
        self.enabled = True
        self._runner = None
        self._generation = self._request_thread = None
        self.refreshes = 0

    def bind_runner(self, runner):
        from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
        model = self._model()
        with _mutation_guard(model, 'expert cuBLASLt owner binding'):
            if self._runner is not None or not self.enabled or type(runner) is not Pi05NativeGraphRunner:
                raise ValueError('Bind expert cuBLASLt exactly once to a native runner')
            owners = tuple(r for r in getattr(model, '_graph_runners', ()) if not r._closed)
            if owners != (runner,) or runner.model is not model or runner._closed:
                raise ValueError('Expert cuBLASLt requires exactly one native graph owner')
            self._runner = weakref.ref(runner)
            self.runtime.bind_owner(self)

    def _check_owner(self, *, enumerate_owners):
        model = self._model()
        runner = None if self._runner is None else self._runner()
        if (not self.enabled or model is None or getattr(model, INSTALLATION, None) is not self
                or runner is None or runner._closed or runner.model is not model
                or not runner._call_lock.locked()):
            raise RuntimeError('Expert cuBLASLt requires its bound native runner and locked request')
        if enumerate_owners:
            owners = tuple(r for r in getattr(model, '_graph_runners', ()) if not r._closed)
            if owners != (runner,):
                raise RuntimeError('Expert cuBLASLt does not permit multiple native graph owners')
        return model

    def assert_execution_owner(self):
        model = self._check_owner(enumerate_owners=False)
        if (self._request_thread != threading.get_ident() or self._generation is None
                or model._inference_cache.generation != self._generation):
            raise RuntimeError('Expert cuBLASLt requires the current prepared owner generation/thread')

    def _validate(self):
        model = self._model()
        self.runtime.validate()
        if (getattr(model, '_lite_expert_qkv_installation', None) is not self.qkv
                or getattr(model, '_lite_expert_mlp_installation', None) is not self.mlp
                or not self.qkv.enabled or not self.mlp.enabled):
            raise RuntimeError('Expert cuBLASLt prerequisites changed; reinstall in lifecycle order')
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers) != len(self.layers) or any(a is not b.layer for a, b in zip(layers, self.layers, strict=True)):
            raise RuntimeError('Expert cuBLASLt layer topology changed')
        return tuple((calls, calls.validate()) for calls in self.layers)

    @torch.no_grad()
    def prepare(self, generation, *, identity_lookup=None):
        model = self._check_owner(enumerate_owners=True)
        if (generation != model._inference_cache.generation
                or self.qkv._generation != generation or self.mlp._generation != generation):
            raise RuntimeError('Prepare expert cuBLASLt after QKV/MLP and the shared model generation')
        if identity_lookup is not None:
            identity_lookup.assert_for(model, generation)
        validated = self._validate()
        if generation == self._generation:
            for calls, values in validated:
                calls._assert_validated_sources(values, identity_lookup=identity_lookup)
        else:
            # Validate all layers before performing any asynchronous weight copy.
            for calls in self.layers:
                calls.packer.assert_ready(identity_lookup=identity_lookup)
                calls.candidate.assert_ready(identity_lookup=identity_lookup)
            for _, (_, _, sources) in validated:
                for site, source in sources:
                    if identity_lookup is None:
                        site._refresh_validated(source)
                    else:
                        site._refresh_validated(source, identity_lookup=identity_lookup)
            self._generation = generation
            self.refreshes += 1
        self._request_thread = threading.get_ident()

    def compilation_members(self, generation):
        self._check_owner(enumerate_owners=True)
        self.assert_execution_owner()
        if generation != self._generation:
            raise RuntimeError('Expert cuBLASLt compilation requires its current prepared generation')
        validated = self._validate()
        for calls, values in validated:
            calls._assert_validated_sources(values)
        return tuple(site for calls in self.layers for site in calls.sites.values())

    def manifest(self):
        source = Path(__file__)
        runtime_source = source.with_name('cublaslt_runtime.py')
        return dict(version=VERSION, enabled=self.enabled, profile=self.profile, include_down=self.include_down,
                    runtime=self.runtime.manifest(), layers=len(self.layers),
                    transposed_weight_bytes=sum(site._weight.numel() * site._weight.element_size()
                        for calls in self.layers for site in calls.sites.values()
                        if site.spec['layout'] == 'contiguous_kn'),
                    rounding='BF16 projections -> BF16 GELU(tanh) -> BF16 product -> BF16 down',
                    ownership='one explicitly bound native runner; locked thread; stream/event ordering by runner',
                    refresh='generation validated externally; persistent KN copies refreshed in place',
                    validation='same-call geometry/source handoff; no cross-request validation cache',
                    numerical_limit='synthetic algorithm gate only; real fixture and closed-loop quality required',
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, runtime_source)})

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Expert cuBLASLt installation is inactive')
        with _mutation_guard(model, 'expert cuBLASLt'):
            self._validate()
            # Runner.clear()/close() drains outstanding work. Merely deleting its
            # entry dictionary is not sufficient to release foreign descriptors.
            runner = None if self._runner is None else self._runner()
            if runner is not None and (runner._event is not None or runner._device is not None):
                raise RuntimeError('Retire native runner work before freeing cuBLASLt resources')
            for calls in self.layers:
                calls.restore()
            delattr(model, INSTALLATION)
            self.enabled = False
            try:
                self.runtime.close()
            finally:
                self.layers.clear()
                self._runner = None
                model.invalidate_inference_caches(weights_changed=False)


def install_expert_cublaslt(model, *, profile, include_down=False, _runtime_factory=None):
    """Explicit opt-in. No fallback to an unmeasured algorithm or another GPU."""
    profile_specs(profile, include_down)
    with _mutation_guard(model, 'expert cuBLASLt'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Expert cuBLASLt is already installed')
        qkv = getattr(model, '_lite_expert_qkv_installation', None)
        mlp = getattr(model, '_lite_expert_mlp_installation', None)
        if qkv is None or mlp is None or not qkv.enabled or not mlp.enabled:
            raise ValueError('Install the original expert QKV and MLP candidates before cuBLASLt')
        if mlp.concatenate != (profile == 'g4_best_three'):
            raise ValueError('L4 profile requires separate gate/up; G4 profile requires packed gate/up')
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        if not layers or len(layers) != len(qkv._packers) or len(layers) != len(mlp._candidates):
            raise ValueError('Expert cuBLASLt requires matching nonempty expert layers')
        device = qkv._packers[0].device
        for layer, packer, candidate in zip(layers, qkv._packers, mlp._candidates, strict=True):
            packer.validate_geometry()
            candidate.validate()
            out = layer.self_attn.o_proj
            if (packer.width != 1024 or packer.sizes != (2048, 256, 256)
                    or candidate.width != 1024 or candidate.intermediate != 4096
                    or packer.device != device or candidate.device != device
                    or 'project' in packer.__dict__ or 'forward' in candidate.__dict__
                    or type(out) is not nn.Linear or out.bias is not None or 'forward' in out.__dict__
                    or out._forward_hooks or out._forward_pre_hooks or tuple(out.weight.shape) != (1024, 2048)
                    or out.weight.dtype != torch.bfloat16 or out.weight.device != device):
                raise ValueError('Expert cuBLASLt requires original pi05 1024/4096/8:1/head256 BF16 geometry')
        factory = ExpertLtRuntime if _runtime_factory is None else _runtime_factory
        runtime = factory(device, profile, include_down=include_down)
        handle = ExpertCublasLtInstallation(model, runtime, qkv, mlp, profile, include_down)
        installed = []
        try:
            for layer, packer, candidate in zip(layers, qkv._packers, mlp._candidates, strict=True):
                handle.layers.append(_LayerCalls(handle, layer, packer, candidate))
            for calls in handle.layers:
                calls.install()
                installed.append(calls)
            setattr(model, INSTALLATION, handle)
            model.invalidate_inference_caches(weights_changed=False)
            return handle
        except BaseException as error:
            for calls in reversed(installed):
                calls.restore()
            if getattr(model, INSTALLATION, None) is handle:
                delattr(model, INSTALLATION)
            handle.enabled = False
            try:
                runtime.close()
            except BaseException as cleanup:
                error.add_note('Partial expert cuBLASLt installation cleanup: ' + str(cleanup))
            raise
