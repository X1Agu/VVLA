"""Deferred to Colab: CPU ABI/lifecycle tests and actual CUDA/compile contracts.

Nothing in this file runs CUDA unless pytest selects a CUDA-marked function.
Production runtime is replaced only in CPU tests; public installs remain CUDA13
and exact library/device/algorithm pinned.
"""
import ctypes as C
from collections import Counter
from copy import deepcopy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP

from vvla_repro_pi05.lite import cublaslt_runtime as rt
from vvla_repro_pi05.lite import expert_cublaslt as candidate
from vvla_repro_pi05.lite.fused_expert_mlp import install_expert_mlp
from vvla_repro_pi05.lite.fused_qkv import install_expert_qkv
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction


class FakeFunction:
    def __init__(self, name, library):
        self.name, self.library = name, library

    def __call__(self, *args):
        library = self.library
        library.calls.append((self.name, args))
        if self.name == 'cublasLtGetStatusName':
            return b'TEST_STATUS'
        if self.name == library.fail:
            return 7
        if self.name.endswith('Create'):
            library.next += 1
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = library.next
        return 0


class FakeLibrary:
    def __init__(self, fail=None):
        self.calls, self.next, self.fail = [], 100, fail

    def __getattr__(self, name):
        value = FakeFunction(name, self)
        setattr(self, name, value)
        return value


@pytest.mark.parametrize('layout,transpose', [('original_nk', 1), ('contiguous_kn', 0)])
def test_runtime_abi_exact_shapes_actual_workspace_size_and_cleanup(layout, transpose):
    assert rt.abi_layout()['heuristic_bytes'] == 96
    library = FakeLibrary()
    plan = rt.Plan(rt.LtAPI(library), 50, 1024, 2560, layout, 'library-default')
    matrices = [args[1:] for name, args in library.calls if name == 'cublasLtMatrixLayoutCreate']
    assert matrices == [(14, 50, 1024, 1024),
                        (14, 2560, 1024, 1024) if transpose else (14, 1024, 2560, 2560),
                        (14, 50, 2560, 2560)]
    pointers = [SimpleNamespace(data_ptr=lambda address=address: address)
                for address in (0x1000, 0x2000, 0x3000)]
    workspace = SimpleNamespace(data_ptr=lambda: 0x4000, numel=lambda: 819200)
    plan.launch(*pointers, rt.Algo(), workspace, SimpleNamespace(cuda_stream=0x9876))
    arguments = library.calls[-1][1]
    assert arguments[-2] == 819200  # Never advertise 256MiB for a tiny allocation.
    assert arguments[-1].value == 0x9876
    assert C.cast(arguments[2], C.POINTER(C.c_float))[0] == 1.0
    assert C.cast(arguments[7], C.POINTER(C.c_float))[0] == 0.0
    plan.close()
    assert len([name for name, _ in library.calls if name.endswith('Destroy')]) == 6
    plan.close()


def test_partial_descriptor_failure_releases_all_successful_resources():
    library = FakeLibrary('cublasLtMatrixLayoutSetAttribute')
    with pytest.raises(rt.LtError):
        rt.Plan(rt.LtAPI(library), 50, 1024, 2560, 'contiguous_kn', 'library-default')
    assert [name for name, _ in library.calls if name.endswith('Destroy')] == [
        'cublasLtMatrixLayoutDestroy', 'cublasLtMatmulDescDestroy', 'cublasLtDestroy']


@pytest.mark.parametrize('field', ['id', 'tile', 'stages', 'swizzle', 'split_k',
                                  'reduction_scheme', 'custom_option', 'numerical_impl_flags'])
def test_fresh_heuristic_matches_every_semantic_field_not_opaque_bytes(field):
    spec = rt.profile_specs('l4_exact_three')['qkv']
    result = rt.Heuristic()
    result.workspaceSize = spec['workspace_bytes']
    info = dict(spec['algorithm'], opaque_algo_hex='not-used-or-deserialized')
    plan = SimpleNamespace(heuristics=lambda: [result], api=SimpleNamespace(algorithm_info=lambda _: info))
    assert rt.select_algorithm(plan, spec)[0] == 0
    info[field] += 1
    with pytest.raises(RuntimeError, match='exactly once'):
        rt.select_algorithm(plan, spec)


def test_heuristic_missing_duplicate_workspace_and_dtype_math_gates_fail_closed():
    spec = rt.profile_specs('g4_best_three')['attention_out']
    result = rt.Heuristic()
    result.workspaceSize = spec['workspace_bytes']
    api = SimpleNamespace(algorithm_info=lambda _: spec['algorithm'])
    for values in ([], [result, result]):
        with pytest.raises(RuntimeError, match='exactly once'):
            rt.select_algorithm(SimpleNamespace(heuristics=lambda: values, api=api), spec)
    result.workspaceSize += 1
    with pytest.raises(RuntimeError):
        rt.select_algorithm(SimpleNamespace(heuristics=lambda: [result], api=api), spec)
    assert spec['algorithm']['numerical_impl_flags'] & 0xff00 == 0x200
    assert spec['algorithm']['reduction_scheme'] == 4  # Explicit existing library-default reduction.
    assert rt.candidate_gate(result, spec['algorithm'], 'fp32') == 'split_reduction_not_FP32'


def test_profiles_are_independent_immutable_and_down_is_separate():
    a = rt.profile_specs('l4_exact_three')
    b = rt.profile_specs('l4_exact_three', True)
    assert 'down' not in a and 'down' in b
    assert a['gate_or_up']['n'] == 4096
    assert rt.profile_specs('g4_best_three')['gate_up_packed']['n'] == 8192
    a['qkv']['algorithm']['id'] = -1
    assert rt.profile_specs('l4_exact_three')['qkv']['algorithm']['id'] == 6
    for args in [('default', False), ('g4_best_three', True), ('l4_exact_three', 1)]:
        with pytest.raises(ValueError):
            rt.profile_specs(*args)


class SmallExpert(nn.Module):
    prepare_inference_context = Pi05ForActionPrediction.prepare_inference_context

    def __init__(self, device='cpu'):
        super().__init__()
        self.config = SimpleNamespace(min_period=.004, max_period=4.0)
        self._inference_cache = Pi05InferenceCache()
        self.paligemma_with_expert = nn.Module()
        p = self.paligemma_with_expert
        p.paligemma = nn.Module()
        p.paligemma.model = nn.Module()
        p.paligemma.model.vision_tower = nn.Linear(1, 1, device=device)
        p.gemma_expert = nn.Module()
        p.gemma_expert.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.head_dim = 256
        for name, k, n in [('q_proj', 1024, 2048), ('k_proj', 1024, 256), ('v_proj', 1024, 256),
                           ('o_proj', 2048, 1024)]:
            setattr(layer.self_attn, name, nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16))
        layer.mlp = GemmaMLP(GemmaConfig(hidden_size=1024, intermediate_size=4096)).to(device=device, dtype=torch.bfloat16)
        p.gemma_expert.model.layers = nn.ModuleList([layer])
        self.eval()

    @property
    def layer(self):
        return self.paligemma_with_expert.gemma_expert.model.layers[0]

    def optimization_flags(self):
        return False, False, False, False, False

    def invalidate_inference_caches(self, **kwargs):
        self._inference_cache.clear()


class FakePlan:
    def __init__(self, spec, device):
        self.spec, self.device = spec, torch.device(device)
        self.workspace = torch.empty(256, dtype=torch.uint8, device=device)
        self.key, self.owner, self.closed = next(rt._IDS), None, False
        rt._PLANS[self.key] = self

    def execute(self, x, weight, workspace, n):
        self.owner.assert_execution_owner()
        assert workspace is self.workspace and n == self.spec['n']
        raw = weight if self.spec['layout'] == 'original_nk' else weight.T.contiguous()
        return torch.nn.functional.linear(x, raw)

    def close(self):
        rt._PLANS.pop(self.key, None)
        self.closed = True


class FakeRuntime:
    def __init__(self, device, profile, *, include_down=False):
        self.plans = {name: FakePlan(spec, device) for name, spec in rt.profile_specs(profile, include_down).items()}
        self.closed = False

    def bind_owner(self, owner):
        for plan in self.plans.values():
            plan.owner = owner

    def validate(self):
        assert not self.closed and all(not plan.closed for plan in self.plans.values())

    def manifest(self):
        return {'test_fake_runtime': True}

    def close(self):
        for plan in self.plans.values():
            plan.close()
        self.closed = True


def make_case(profile='l4_exact_three', include_down=False, *, device='cpu', fake=True):
    model = SmallExpert(device)
    qkv = install_expert_qkv(model)
    mlp = install_expert_mlp(model, concatenate=profile == 'g4_best_three', fuse_gelu_mul=True)
    handle = candidate.install_expert_cublaslt(model, profile=profile, include_down=include_down,
                                               _runtime_factory=FakeRuntime if fake else None)
    runner = Pi05NativeGraphRunner(model)
    handle.bind_runner(runner)
    return model, qkv, mlp, handle, runner


def prepare(model, handle, runner):
    device = next(model.parameters()).device
    context = model.prepare_inference_context(
        [torch.ones((1, 3, 2, 2), device=device)], [torch.ones(1, dtype=torch.bool, device=device)],
        torch.ones((1, 4), dtype=torch.bool, device=device), 10, positions=False)
    # Supports standalone review before the root-owned core hook is integrated.
    handle.prepare(context['token'][1])
    runner._token = context['token']
    return context


def close_case(qkv, mlp, handle, runner):
    runner.close()
    handle.uninstall()
    mlp.uninstall()
    qkv.uninstall()


@pytest.mark.parametrize('profile,include_down', [('l4_exact_three', False), ('l4_exact_three', True),
                                                 ('g4_best_three', False)])
def test_owned_cpu_lifecycle_original_state_and_dispatch(profile, include_down):
    model, qkv, mlp, handle, runner = make_case(profile, include_down)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    parameters = tuple(id(value) for value in model.parameters())
    identity = handle.manifest()
    x = torch.randn(1, 50, 1024).bfloat16()
    y = torch.randn(1, 50, 2048).bfloat16()
    with torch.no_grad(), runner._call_lock:
        prepare(model, handle, runner)
        q, k, v = qkv._packers[0].project(x)
        assert q.stride(-2) == k.stride(-2) == v.stride(-2) == 2560
        for actual, module in zip((q, k, v), (model.layer.self_attn.q_proj, model.layer.self_attn.k_proj,
                                             model.layer.self_attn.v_proj), strict=True):
            torch.testing.assert_close(actual, module(x), rtol=0, atol=0)
        torch.testing.assert_close(model.layer.mlp(x), GemmaMLP.forward(model.layer.mlp, x), rtol=0, atol=0)
        torch.testing.assert_close(model.layer.self_attn.o_proj(y), nn.Linear.forward(model.layer.self_attn.o_proj, y),
                                   rtol=0, atol=0)
        prepare(model, handle, runner)
    assert handle.refreshes == 1 and handle.manifest() == identity
    assert tuple(id(value) for value in model.parameters()) == parameters
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    close_case(qkv, mlp, handle, runner)
    assert not hasattr(model, candidate.INSTALLATION)
    assert 'forward' not in model.layer.self_attn.o_proj.__dict__


def test_inplace_refresh_same_addresses_and_stale_sources_fail_before_dispatch():
    model, qkv, mlp, handle, runner = make_case()
    x = torch.ones(1, 50, 1024, dtype=torch.bfloat16)
    with torch.no_grad(), runner._call_lock:
        first = prepare(model, handle, runner)
        addresses = [site._weight.data_ptr() for site in handle.layers[0].sites.values()]
        model.layer.self_attn.q_proj.weight.add_(.125)
        with pytest.raises(RuntimeError, match='sources changed'):
            qkv._packers[0].project(x)
        second = prepare(model, handle, runner)
        assert second['token'] != first['token'] and handle.refreshes == 2
        torch.testing.assert_close(qkv._packers[0].project(x)[0], model.layer.self_attn.q_proj(x), rtol=0, atol=0)
        assert [site._weight.data_ptr() for site in handle.layers[0].sites.values()] == addresses
        # Explicit .data modification retains the existing explicit invalidation contract.
        model.layer.mlp.up_proj.weight.data.add_(.1)
        model.invalidate_inference_caches()
        prepare(model, handle, runner)
        assert handle.refreshes == 3
    close_case(qkv, mlp, handle, runner)


def test_bind_once_active_owner_topology_tamper_and_retire_contracts():
    model, qkv, mlp, handle, runner = make_case()
    with pytest.raises(RuntimeError, match='locked request'):
        handle.prepare(model._inference_cache.generation)
    with pytest.raises(ValueError, match='exactly once'):
        handle.bind_runner(runner)
    other = Pi05NativeGraphRunner(model)
    with runner._call_lock, pytest.raises(RuntimeError, match='multiple'):
        prepare(model, handle, runner)
    other.close()
    with runner._call_lock:
        prepare(model, handle, runner)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):
        handle.uninstall()
    runner.clear()
    close_case(qkv, mlp, handle, runner)


def test_cache_mutation_and_module_replacement_fail_closed():
    model, qkv, mlp, handle, runner = make_case()
    with torch.no_grad(), runner._call_lock:
        prepare(model, handle, runner)
        site = handle.layers[0].sites['gate']
        site._weight.add_(1)
        with pytest.raises(RuntimeError, match='cache changed'):
            prepare(model, handle, runner)
        model.invalidate_inference_caches()
        prepare(model, handle, runner)
        old = model.layer.self_attn.o_proj
        model.layer.self_attn.o_proj = nn.Linear(2048, 1024, bias=False, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match='dispatch'):
            prepare(model, handle, runner)
        model.layer.self_attn.o_proj = old
    close_case(qkv, mlp, handle, runner)


def test_prebind_uninstall_and_partial_allocation_cleanup(monkeypatch):
    model = SmallExpert()
    qkv, mlp = install_expert_qkv(model), install_expert_mlp(model, fuse_gelu_mul=True)
    handle = candidate.install_expert_cublaslt(model, profile='l4_exact_three', _runtime_factory=FakeRuntime)
    handle.uninstall()  # Partial policy constructor never created a runner.
    created = []
    class RecordingRuntime(FakeRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
    def fail(*args, **kwargs):
        raise MemoryError('synthetic KN allocation failure')
    monkeypatch.setattr(candidate, '_LayerCalls', fail)
    with pytest.raises(MemoryError):
        candidate.install_expert_cublaslt(model, profile='l4_exact_three', _runtime_factory=RecordingRuntime)
    assert created[0].closed and all(plan.closed for plan in created[0].plans.values())
    assert not hasattr(model, candidate.INSTALLATION)
    assert 'project' not in qkv._packers[0].__dict__
    assert 'forward' not in mlp._candidates[0].__dict__
    mlp.uninstall()
    qkv.uninstall()


def test_fake_compile_fullgraph_requires_prepared_scope_and_rejects_retired_entry():
    from vvla_repro_pi05.lite.compilation import compile_prepared_run, _PACKED_INSTALLATIONS
    assert candidate.INSTALLATION in _PACKED_INSTALLATIONS, 'Integrate the root-owned compilation handle tuple first'
    model, qkv, mlp, handle, runner = make_case()
    x = torch.randn(1, 50, 1024).bfloat16()
    with torch.no_grad(), runner._call_lock:
        context = prepare(model, handle, runner)
        run = lambda: model.layer.mlp(x)
        compiled = compile_prepared_run(model, run, backend='eager', owner=runner, context=context)
        torch.testing.assert_close(compiled(), run(), rtol=0, atol=0)
        assert all(site._prepared_compile_token is None for site in handle.layers[0].sites.values())
        model.layer.mlp.gate_proj.weight.add_(.1)
        with pytest.raises(RuntimeError, match='inputs changed'):
            compiled()
        prepare(model, handle, runner)
        with pytest.raises(RuntimeError, match='generation'):
            compiled()
        compiled.retire()
        with pytest.raises(RuntimeError, match='retired'):
            compiled()
    close_case(qkv, mlp, handle, runner)


def make_meta_case(profile='g4_best_three', include_down=False, layers=18):
    # Full production geometry, no weight storage/initialization or GPU access.
    with torch.device('meta'):
        model = SmallExpert('meta')
        model.paligemma_with_expert.gemma_expert.model.layers.extend(
            deepcopy(model.layer) for _ in range(layers - 1))
        qkv = install_expert_qkv(model)
        mlp = install_expert_mlp(model, concatenate=profile == 'g4_best_three', fuse_gelu_mul=True)
        handle = candidate.install_expert_cublaslt(model, profile=profile, include_down=include_down,
                                                   _runtime_factory=FakeRuntime)
        runner = Pi05NativeGraphRunner(model)
        handle.bind_runner(runner)
    with runner._call_lock:
        generation = model._inference_cache.generation
        qkv.prepare(generation)
        mlp.prepare(generation)
        handle.prepare(generation)
    return model, qkv, mlp, handle, runner


@pytest.mark.parametrize('profile,include_down,sites,identities', [
    ('g4_best_three', False, 3, 12), ('l4_exact_three', False, 4, 15),
    ('l4_exact_three', True, 5, 17)])
@pytest.mark.parametrize('operation', ['prepare', 'compilation_members'])
def test_request_local_validation_checks_every_layer_once_without_cross_request_cache(
        monkeypatch, profile, include_down, sites, identities, operation):
    from vvla_repro_pi05.lite import fused_qkv, fused_expert_mlp
    model, qkv, mlp, handle, runner = make_meta_case(profile, include_down)
    counts = Counter()
    def instrument(owner, name, counter):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            counts[counter] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, wrapped)
    for calls in handle.layers:
        instrument(calls.packer, 'validate_geometry', 'qkv_geometry')
        instrument(calls.candidate, 'validate', 'mlp_geometry')
        for site in calls.sites.values():
            instrument(site, 'validate', 'site_geometry')
    for module in (candidate, fused_qkv, fused_expert_mlp):
        instrument(module, 'tensor_identity', 'tensor_identity')
    try:
        with runner._call_lock:
            for request in range(1, 3):
                getattr(handle, operation)(model._inference_cache.generation)
                assert counts == dict(qkv_geometry=18 * request, mlp_geometry=18 * request,
                                      site_geometry=18 * sites * request,
                                      tensor_identity=18 * identities * request)
            # Same generation is not permission to reuse a previous request's
            # original-source identity. No shared-core refresh masks this check.
            with torch.no_grad():
                model.layer.self_attn.q_proj.weight.add_(1)
            with pytest.raises(RuntimeError, match='QKV sources changed'):
                getattr(handle, operation)(model._inference_cache.generation)
    finally:
        close_case(qkv, mlp, handle, runner)


@pytest.mark.parametrize('field', ['_source_identity', '_weight_identity'])
def test_original_nk_alias_still_checks_both_expected_identities(field):
    model, qkv, mlp, handle, runner = make_meta_case(layers=1)
    site = handle.layers[0].sites['attention_out']
    assert site.source() is site._weight
    original = getattr(site, field)
    try:
        setattr(site, field, ('invalid',))
        with runner._call_lock, pytest.raises(RuntimeError, match='cache changed'):
            handle.prepare(model._inference_cache.generation)
        with pytest.raises(RuntimeError, match='cache changed'):
            site.assert_ready()
    finally:
        setattr(site, field, original)
        close_case(qkv, mlp, handle, runner)


@pytest.mark.parametrize('fault,pattern', [
    ('qkv_shape', 'geometry'), ('mlp_hook', 'without hooks'),
    ('activation', 'GELU'), ('options', 'options changed'),
    ('output_hook', 'output projection'), ('site_source', 'source shape')])
def test_merged_validation_preserves_geometry_behavior_and_site_failures(monkeypatch, fault, pattern):
    model, qkv, mlp, handle, runner = make_meta_case(layers=1)
    calls = handle.layers[0]
    try:
        with monkeypatch.context() as change, runner._call_lock:
            if fault == 'qkv_shape':
                change.setattr(model.layer.self_attn.q_proj, 'out_features', 2047)
            elif fault == 'mlp_hook':
                change.setattr(model.layer.mlp.gate_proj, '_forward_hooks', {1: lambda *a: None})
            elif fault == 'activation':
                change.setattr(model.layer.mlp.act_fn, 'forward', lambda x: x)
            elif fault == 'options':
                change.setattr(calls.candidate, '_options', (False, False))
            elif fault == 'output_hook':
                change.setattr(model.layer.self_attn.o_proj, '_forward_pre_hooks', {1: lambda *a: None})
            else:
                change.setattr(calls.sites['qkv'], 'source', lambda: torch.empty(1, device='meta'))
            with pytest.raises((RuntimeError, ValueError), match=pattern):
                handle.prepare(model._inference_cache.generation)
    finally:
        close_case(qkv, mlp, handle, runner)


def test_standalone_checks_and_refresh_read_live_state_after_prior_validation(monkeypatch):
    model, qkv, mlp, handle, runner = make_meta_case(layers=1)
    calls = handle.layers[0]
    site = calls.sites['gate_up_packed']
    try:
        calls.validate()  # This must not populate a persistent trusted snapshot.
        with monkeypatch.context() as change:
            change.setattr(site, 'source', lambda: torch.empty(1, device='meta'))
            for operation in (calls.assert_sources, site.assert_ready, site.refresh):
                with pytest.raises(ValueError, match='source shape'):
                    operation()
        with torch.no_grad():
            model.layer.mlp.down_proj.weight.add_(1)
        with pytest.raises(RuntimeError, match='MLP sources changed'):
            calls.assert_sources()
    finally:
        close_case(qkv, mlp, handle, runner)


def test_new_generation_validates_all_layers_before_first_site_refresh(monkeypatch):
    model, qkv, mlp, handle, runner = make_meta_case(layers=2)
    refreshed = []
    try:
        with runner._call_lock:
            model.invalidate_inference_caches()
            generation = model._inference_cache.generation
            qkv.prepare(generation)
            mlp.prepare(generation)
            for calls in handle.layers:
                for site in calls.sites.values():
                    original = site._refresh_validated
                    def wrapped(source, _original=original, _site=site):
                        refreshed.append(_site)
                        return _original(source)
                    monkeypatch.setattr(site, '_refresh_validated', wrapped)
            last = handle.layers[-1].sites['qkv']
            with monkeypatch.context() as change:
                change.setattr(last, 'source', lambda: torch.empty(1, device='meta'))
                with pytest.raises(ValueError, match='source shape'):
                    handle.prepare(generation)
            assert not refreshed and handle._generation != generation
            handle.prepare(generation)
            assert len(refreshed) == 6 and handle._generation == generation
    finally:
        close_case(qkv, mlp, handle, runner)


def cuda_profile():
    if not torch.cuda.is_available():
        pytest.skip('CUDA contracts are queued remotely')
    name = torch.cuda.get_device_name()
    if 'L4' in name:
        return 'l4_exact_three'
    if 'RTX PRO 6000 Blackwell' in name:
        return 'g4_best_three'
    pytest.skip('Only audited L4/RTX PRO 6000 profiles are supported')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA remote queue only')
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('include_down', [False, True])
def test_cuda_actual_projection_rope_mlp_capture_repeat_and_weight_refresh(compiled, include_down, tmp_path, record_property):
    profile = cuda_profile()
    if include_down and profile != 'l4_exact_three':
        pytest.skip('Independent down algorithm is L4-only')
    from vvla_repro_pi05.lite.compilation import compile_prepared_run, INDUCTOR_OPTIONS
    from vvla_runtime.pointwise import rotary_pos_emb
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model, qkv, mlp, handle, runner = make_case(profile, include_down, device='cuda', fake=False)
    x = torch.randn(1, 50, 1024, device='cuda', dtype=torch.bfloat16) / 8
    y = torch.randn(1, 50, 2048, device='cuda', dtype=torch.bfloat16) / 8
    angle = torch.randn(1, 50, 256, device='cuda')
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    z = torch.randn(1, 50, 4096, device='cuda', dtype=torch.bfloat16) / 8
    calls = handle.layers[0]
    gate_sites = ['gate_up_packed'] if profile == 'g4_best_three' else ['gate', 'up']
    primitive_labels = ['q', 'k', 'v', 'attention_out', *gate_sites]
    if include_down:
        primitive_labels.append('down')
    labels = [*primitive_labels, 'rope_q', 'rope_k', 'mlp']
    primitive_count = len(primitive_labels)
    def run():
        q, k, v = qkv._packers[0].project(x)
        raw = [q, k, v, model.layer.self_attn.o_proj(y)]
        raw.extend(calls.sites[name](x) for name in gate_sites)
        if include_down:
            raw.append(calls.sites['down'](z))
        qr, kr = rotary_pos_emb(q.view(1, 50, 8, 256).transpose(1, 2),
                               k.view(1, 50, 1, 256).transpose(1, 2), cos, sin)
        return (*raw, qr, kr, model.layer.mlp(x))
    def delta(actual, expected):
        difference = actual.float() - expected.float()
        return dict(shape=list(actual.shape), bitwise_equal=bool(torch.equal(actual, expected)),
                    max_abs=float(difference.abs().max()), rms=float(difference.square().mean().sqrt()),
                    changed_elements=int((actual != expected).sum()))
    graph = None
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), runner._call_lock, torch.cuda.stream(stream):
            context = prepare(model, handle, runner)
            eager = run()
            rawq, rawk, rawv = (module(x) for module in (model.layer.self_attn.q_proj, model.layer.self_attn.k_proj,
                                                        model.layer.self_attn.v_proj))
            reference_rotary = rotary_pos_emb(rawq.view(1, 50, 8, 256).transpose(1, 2),
                                              rawk.view(1, 50, 1, 256).transpose(1, 2), cos, sin)
            raw_reference = [rawq, rawk, rawv, nn.Linear.forward(model.layer.self_attn.o_proj, y)]
            if profile == 'g4_best_three':
                raw_reference.append(torch.cat((model.layer.mlp.gate_proj(x), model.layer.mlp.up_proj(x)), -1))
            else:
                raw_reference.extend((model.layer.mlp.gate_proj(x), model.layer.mlp.up_proj(x)))
            if include_down:
                raw_reference.append(nn.Linear.forward(model.layer.mlp.down_proj, z))
            reference = (*raw_reference, *reference_rotary, GemmaMLP.forward(model.layer.mlp, x))
            for actual, expected in zip(eager, reference, strict=True):
                torch.testing.assert_close(actual, expected, rtol=.02, atol=.03)
            execute = compile_prepared_run(model, run, owner=runner, context=context) if compiled else run
            execute()
            warm_reference = tuple(value.clone() for value in execute())
            # An independent control consumes the SAME raw Q/K (and original
            # strided split views), but contains NO Lt operation. This isolates
            # compiler lowering of RoPE from algorithm/GEMM/capture changes.
            rope_control = None
            if compiled:
                def only_rope():
                    return rotary_pos_emb(eager[0].view(1, 50, 8, 256).transpose(1, 2),
                                          eager[1].view(1, 50, 1, 256).transpose(1, 2), cos, sin)
                control = torch.compile(only_rope, backend='inductor', fullgraph=True,
                                        dynamic=False, options=dict(INDUCTOR_OPTIONS))
                rope_control = tuple(value.clone() for value in control())
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = execute()
            graph.replay()
            first = tuple(value.clone() for value in captured)
            graph.replay()
            evidence = {
                'profile': profile, 'compiled': compiled, 'include_down': include_down,
                'capture_vs_warm': {name: delta(a, b) for name, a, b in zip(labels, captured, warm_reference, strict=True)},
                'capture_vs_eager': {name: delta(a, b) for name, a, b in zip(labels, captured, eager, strict=True)},
                'replay_vs_replay': {name: delta(a, b) for name, a, b in zip(labels, captured, first, strict=True)},
                'rope_control_vs_compiled': None if rope_control is None else {
                    name: delta(a, b) for name, a, b in zip(('rope_q', 'rope_k'), rope_control,
                                                          warm_reference[primitive_count:primitive_count + 2], strict=True)},
            }
            evidence_path = tmp_path / 'LT_BOUNDARY_EVIDENCE.json'
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
            record_property('lt_boundary_evidence', json.dumps(evidence, sort_keys=True))
            print('Lt boundary evidence:', evidence_path, json.dumps(evidence, sort_keys=True))
            # Strict capture/ownership contract: every stage must match this
            # execution mode's warm output and repeated graph replays bitwise.
            for actual, expected in zip(captured, first, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(captured, warm_reference, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # No tolerance is added for any cuBLASLt primitive or MLP output.
            exact_indices = [*range(primitive_count), len(labels) - 1]
            if not compiled:
                exact_indices.extend((primitive_count, primitive_count + 1))
            for index in exact_indices:
                torch.testing.assert_close(captured[index], eager[index], rtol=0, atol=0,
                                           msg=lambda message, name=labels[index]: name + ': ' + message)
            if compiled:
                for actual, expected, isolated in zip(
                        captured[primitive_count:primitive_count + 2],
                        eager[primitive_count:primitive_count + 2], rope_control, strict=True):
                    # Before using the pre-existing small-core compiler tolerance,
                    # require exact equality to an Lt-free compiled RoPE control.
                    torch.testing.assert_close(actual, isolated, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=.02, atol=.01)
            stream.synchronize()
            graph.reset()
            graph = None
            if compiled:
                execute.retire()
            addresses = [site._weight.data_ptr() for site in handle.layers[0].sites.values()]
            model.layer.self_attn.q_proj.weight.add_(.001)
            prepare(model, handle, runner)
            assert [site._weight.data_ptr() for site in handle.layers[0].sites.values()] == addresses
            updated = run()
            assert not torch.equal(updated[0], eager[0])
            stream.synchronize()
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        close_case(qkv, mlp, handle, runner)
