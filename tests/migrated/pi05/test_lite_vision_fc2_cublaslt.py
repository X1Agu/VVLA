"""Write-only locally; run these ABI/lifecycle and CUDA contracts on Colab.

The native algorithm and candidate may differ numerically. Execution gates here
are candidate eager/compile/capture/refresh raw-byte equality. Independent real
FP32/FP64 screen evidence is linked separately and is never replaced by that gate.
"""
import ctypes as C
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import vision_fc2_cublaslt as candidate
from vvla_repro_pi05.lite import vision_fc2_cublaslt_runtime as rt
from vvla_repro_pi05.lite.compilation import compile_prepared_run, _PACKED_INSTALLATIONS
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache, tensor_identity
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction


@pytest.fixture(autouse=True)
def meta_autocast_query(monkeypatch):
    # Meta is used only for storage-free CPU lifecycle/trace contracts. It has
    # no autocast implementation; leave every real device query untouched.
    original = torch.is_autocast_enabled
    monkeypatch.setattr(torch, 'is_autocast_enabled',
                        lambda device_type='cuda': False if device_type == 'meta' else original(device_type))


class FakeFunction:
    def __init__(self, name, library):
        self.name, self.library = name, library

    def __call__(self, *args):
        lib = self.library
        lib.calls.append((self.name, args))
        if self.name.endswith('SetAttribute'):
            lib.attributes.append((self.name, args[1], C.string_at(args[2], args[3])))
        if self.name == 'cublasLtGetStatusName':
            return b'TEST_STATUS'
        if self.name == lib.fail:
            return 7
        if self.name.endswith('Create'):
            lib.next += 1
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = lib.next
        return 0


class FakeLibrary:
    def __init__(self, fail=None):
        self.calls, self.attributes, self.next, self.fail = [], [], 100, fail

    def __getattr__(self, name):
        value = FakeFunction(name, self)
        setattr(self, name, value)
        return value


def test_bias_abi_column_layout_pointer_types_operand_order_and_actual_workspace():
    library = FakeLibrary()
    bias = SimpleNamespace(shape=(rt.N,), dtype=torch.bfloat16, is_contiguous=lambda: True, data_ptr=lambda: 0x5000)
    plan = rt.BiasedPlan(rt.ffi.LtAPI(library), bias)
    assert plan.bias is bias
    assert [args[1:] for name, args in library.calls if name == 'cublasLtMatrixLayoutCreate'] == [
        (14, 4304, 1152, 4304), (14, 4304, 512, 4304), (14, 1152, 512, 1152)]
    attrs = {attr: raw for name, attr, raw in library.attributes if name == 'cublasLtMatmulDescSetAttribute'}
    for attr, expected, kind in ((2, 0, C.c_int32), (3, 1, C.c_int32), (4, 0, C.c_int32),
                                  (7, 4, C.c_uint32), (8, 0x5000, C.c_void_p), (26, 14, C.c_int32)):
        assert len(attrs[attr]) == C.sizeof(kind)
        assert kind.from_buffer_copy(attrs[attr]).value == expected
    assert all(C.c_int32.from_buffer_copy(raw).value == 0 for name, _, raw in library.attributes
               if name == 'cublasLtMatrixLayoutSetAttribute')
    x, weight, output = (SimpleNamespace(data_ptr=lambda address=a: address) for a in (0x1000, 0x2000, 0x3000))
    workspace = SimpleNamespace(data_ptr=lambda: 0x4000, numel=lambda: 256)
    plan.launch(x, weight, output, rt.ffi.Algo(), workspace, SimpleNamespace(cuda_stream=0x9876))
    args = library.calls[-1][1]
    assert args[3].value == 0x2000 and args[5].value == 0x1000
    assert args[-2] == 256 and args[-1].value == 0x9876
    assert C.cast(args[2], C.POINTER(C.c_float))[0] == 1
    assert C.cast(args[7], C.POINTER(C.c_float))[0] == 0
    plan.close()
    assert sum(name.endswith('Destroy') for name, _ in library.calls) == 6
    plan.close()


def test_partial_biased_descriptor_cleanup_attempts_all_resources():
    lib = FakeLibrary('cublasLtMatrixLayoutSetAttribute')
    bias = SimpleNamespace(shape=(rt.N,), dtype=torch.bfloat16, is_contiguous=lambda: True, data_ptr=lambda: 0x5000)
    with pytest.raises(rt.ffi.LtError):
        rt.BiasedPlan(rt.ffi.LtAPI(lib), bias)
    assert [name for name, _ in lib.calls if name.endswith('Destroy')] == [
        'cublasLtMatrixLayoutDestroy', 'cublasLtMatmulDescDestroy', 'cublasLtDestroy']


@pytest.mark.parametrize('field', tuple(rt.ALGORITHM))
def test_unique_fresh_top16_exact_semantic_match(field):
    spec = rt.specification()
    result = rt.ffi.Heuristic()
    info = dict(spec['algorithm'], opaque_algo_hex='never_deserialize')
    plan = SimpleNamespace(heuristics=lambda: [result], api=SimpleNamespace(algorithm_info=lambda _: info))
    assert rt.ffi.select_algorithm(plan, spec)[0] == 0
    info[field] += 1
    with pytest.raises(RuntimeError, match='exactly once'):
        rt.ffi.select_algorithm(plan, spec)


def test_algorithm_workspace_duplicates_and_fp32_bf16_math_fail_closed():
    spec = rt.specification()
    result = rt.ffi.Heuristic()
    plan = SimpleNamespace(heuristics=lambda: [result, result],
                           api=SimpleNamespace(algorithm_info=lambda _: dict(spec['algorithm'])))
    with pytest.raises(RuntimeError, match='exactly once'):
        rt.ffi.select_algorithm(plan, spec)
    plan.heuristics = lambda: [result]
    result.workspaceSize = 256
    with pytest.raises(RuntimeError, match='exactly once'):
        rt.ffi.select_algorithm(plan, spec)
    assert spec['algorithm']['numerical_impl_flags'] & 0xff00 == 0x200
    assert spec['algorithm']['numerical_impl_flags'] & 0xff0000 == 0x20000
    assert spec['algorithm']['split_k'] == 1 and spec['algorithm']['reduction_scheme'] == 0
    spec['algorithm']['id'] = -1
    assert rt.specification()['algorithm']['id'] == 21


class SmallVision(nn.Module):
    prepare_inference_context = Pi05ForActionPrediction.prepare_inference_context

    def __init__(self, device='meta'):
        super().__init__()
        self.config = SimpleNamespace(min_period=.004, max_period=4.0)
        self._inference_cache = Pi05InferenceCache()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.model = nn.Module()
        vision = nn.Module()
        vision.encoder = nn.Module()
        layers = []
        for _ in range(27):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.fc2 = nn.Linear(rt.K, rt.N, device=device, dtype=torch.bfloat16)
            layers.append(layer)
        vision.encoder.layers = nn.ModuleList(layers)
        self.paligemma_with_expert.paligemma.model.vision_tower = vision
        self.eval()

    @property
    def layers(self):
        return self.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers

    def optimization_flags(self):
        return False, False, False, False, False

    def invalidate_inference_caches(self, **kwargs):
        self._inference_cache.clear()

    def prepare_attention_metadata(self, *args, **kwargs):
        return ()

    def prepare_position_context(self, *args, **kwargs):
        return None

    def sample_actions(self, images, image_masks, tokens, masks, *, noise, **kwargs):
        return torch.stack(tuple(self.layers[i].mlp.fc2(noise) for i in (0, 13, 26)))


class FakePlan:
    def __init__(self, weight, bias):
        self.key = next(rt._IDS)
        self.weight, self.bias = weight, bias
        self.workspace = torch.empty(256, device=weight.device, dtype=torch.uint8)
        self.closed = False
        self.owner = None
        self.spec = rt.specification()
        rt._PLANS[self.key] = self

    def execute(self, x, weight, bias):
        self.owner().assert_execution_owner()
        assert weight is self.weight and bias is self.bias
        return nn.functional.linear(x, weight, bias)

    def close(self):
        rt._PLANS.pop(self.key, None)
        self.closed = True


class FakeRuntime:
    def __init__(self, device, sources):
        self.plans = [FakePlan(w, b) for w, b in sources]
        self.closed = False

    def bind_owner(self, owner):
        for plan in self.plans:
            plan.owner = weakref.ref(owner)

    def validate(self):
        assert not self.closed and all(not p.closed for p in self.plans)

    def manifest(self):
        return {'fake_runtime': True, 'real_input_screen': deepcopy(rt.REAL_SCREEN)}

    def close(self):
        for plan in self.plans:
            plan.close()
        self.closed = True


def make_case(*, device='meta', fake=True):
    model = SmallVision(device)
    handle = candidate.install_vision_fc2_cublaslt(model, _runtime_factory=FakeRuntime if fake else None)
    runner = Pi05NativeGraphRunner(model)
    handle.bind_runner(runner)
    return model, handle, runner


def prepare(model, runner):
    # CPU masks avoid a meta bool conversion. None of this toy's cache flags are
    # enabled, so the real core still performs the full parameter/buffer scan.
    context = model.prepare_inference_context([], [], torch.ones((1, 4), dtype=torch.bool), 10, positions=False)
    runner._token = context['token']
    return context


def close_case(handle, runner):
    runner.close()
    handle.uninstall()


def test_meta_preserves_original_parameters_state_dict_and_all27_binding():
    model = SmallVision()
    parameters = tuple(model.parameters())
    names = tuple(model.state_dict())
    modules = tuple(layer.mlp.fc2 for layer in model.layers)
    handle = candidate.install_vision_fc2_cublaslt(model, _runtime_factory=FakeRuntime)
    assert tuple(model.parameters()) == parameters and tuple(model.state_dict()) == names
    assert all(layer.mlp.fc2 is module for layer, module in zip(model.layers, modules, strict=True))
    assert len(handle.sites) == 27 and len({id(s.plan) for s in handle.sites}) == 27
    assert handle.manifest()['runtime']['real_input_screen']['policy_quality_passed'] is False
    runner = Pi05NativeGraphRunner(model)
    handle.bind_runner(runner)
    with torch.no_grad(), runner._call_lock:
        context = prepare(model, runner)
        assert handle._generation == context['token'][1]  # actual core hook, no test-side prepare call
        assert len(handle.compilation_members(handle._generation)) == 27
    close_case(handle, runner)
    assert all('forward' not in module.__dict__ for module in modules)
    assert tuple(model.state_dict()) == names


@pytest.mark.parametrize('shape', [(1, 256, rt.K), (2, 255, rt.K), (512, rt.K)])
def test_non_target_empty_camera_geometry_uses_original_linear_without_owner(shape, monkeypatch):
    model, handle, runner = make_case()
    calls = []
    original = nn.Linear.forward
    def spy(module, x):
        calls.append(module)
        return original(module, x)
    monkeypatch.setattr(nn.Linear, 'forward', spy)
    with torch.no_grad():
        output = model.layers[0].mlp.fc2(torch.empty(shape, dtype=torch.bfloat16, device='meta'))
    assert calls == [model.layers[0].mlp.fc2] and output.shape == (*shape[:-1], rt.N)
    close_case(handle, runner)


def test_target_requires_owner_and_live_version_then_core_refresh():
    model, handle, runner = make_case()
    x = torch.empty(rt.INPUT_SHAPE, dtype=torch.bfloat16, device='meta')
    with torch.no_grad():
        with pytest.raises(RuntimeError, match='locked request'):
            model.layers[0].mlp.fc2(x)
        with runner._call_lock:
            prepare(model, runner)
            before = handle._generation
            model.layers[0].mlp.fc2(x)
            model.layers[0].mlp.fc2.bias.add_(1)
            with pytest.raises(RuntimeError, match='changed or unprepared'):
                model.layers[0].mlp.fc2(x)
            with pytest.raises(RuntimeError, match='without a model generation'):
                handle.prepare(before)
            prepare(model, runner)
            assert handle._generation > before
            model.layers[0].mlp.fc2(x)
    close_case(handle, runner)


@pytest.mark.parametrize('change', ['parameter', 'module', 'dtype', 'hook', 'order'])
def test_live_topology_parameter_dtype_and_hook_changes_reject_but_cleanup_survives(change):
    model, handle, runner = make_case()
    old_module = model.layers[0].mlp.fc2
    with torch.no_grad(), runner._call_lock:
        prepare(model, runner)
        if change == 'parameter':
            old_module.bias = nn.Parameter(old_module.bias.clone())
        elif change == 'module':
            model.layers[0].mlp.fc2 = nn.Linear(rt.K, rt.N, device='meta', dtype=torch.bfloat16)
        elif change == 'dtype':
            old_module.to(dtype=torch.float32)
        elif change == 'hook':
            old_module.register_forward_hook(lambda m, a, y: y)
        else:
            model.layers[0], model.layers[1] = model.layers[1], model.layers[0]
        with pytest.raises((RuntimeError, ValueError)):
            prepare(model, runner)
    close_case(handle, runner)
    assert 'forward' not in old_module.__dict__ and handle.runtime.closed


def test_multiple_owner_active_request_and_undrained_work_are_rejected():
    model, handle, runner = make_case()
    another = Pi05NativeGraphRunner(model)
    with runner._call_lock:
        with pytest.raises(RuntimeError, match='multiple native'):
            prepare(model, runner)
        with pytest.raises(RuntimeError, match='active/unknown'):
            handle.uninstall()
    another.close()
    runner._device = torch.device('meta')
    with pytest.raises(RuntimeError, match='Retire native runner work'):
        handle.uninstall()
    assert handle.enabled and all(not p.closed for p in handle.runtime.plans)
    runner._device = None
    close_case(handle, runner)


@pytest.mark.parametrize('change', ['plan', 'site', 'training'])
def test_private_dispatch_pairing_and_eval_state_are_live_checked(change):
    model, handle, runner = make_case()
    modules = tuple(site.module for site in handle.sites)
    with runner._call_lock:
        prepare(model, runner)
        if change == 'plan':
            handle.sites[0].plan = handle.sites[1].plan
        elif change == 'site':
            handle.sites[0] = handle.sites[1]
        else:
            model.layers[0].mlp.fc2.train()
        with pytest.raises((RuntimeError, ValueError)):
            prepare(model, runner)
    # Restore eval solely because uninstall's existing shared mutation guard
    # disallows training model ownership, not to repair the corrupted pairing.
    model.eval()
    close_case(handle, runner)
    assert all('forward' not in module.__dict__ for module in modules)


def test_partial_helper_construction_restores_all_forwards_and_runtime(monkeypatch):
    model = SmallVision()
    made = []
    class RecordedRuntime(FakeRuntime):
        def __init__(self, *args):
            super().__init__(*args)
            made.append(self)
    original = candidate._Site
    calls = []
    def fail(*args):
        calls.append(None)
        if len(calls) == 13:
            raise MemoryError('injected site setup')
        return original(*args)
    monkeypatch.setattr(candidate, '_Site', fail)
    with pytest.raises(MemoryError, match='injected'):
        candidate.install_vision_fc2_cublaslt(model, _runtime_factory=RecordedRuntime)
    assert made[0].closed and all(p.closed for p in made[0].plans)
    assert not hasattr(model, candidate.INSTALLATION)
    assert all('forward' not in layer.mlp.fc2.__dict__ for layer in model.layers)


def test_compiled_meta_fullgraph_scope_bias_dependency_refresh_and_retirement():
    assert candidate.INSTALLATION in _PACKED_INSTALLATIONS
    model, handle, runner = make_case()
    x = torch.empty(rt.INPUT_SHAPE, device='meta', dtype=torch.bfloat16)
    graphs = []
    def backend(graph, examples):
        graphs.append(graph)
        return graph.forward
    try:
        with torch.no_grad(), runner._call_lock:
            context = prepare(model, runner)
            compiled = compile_prepared_run(model, lambda: model.layers[0].mlp.fc2(x),
                                            backend=backend, owner=runner, context=context)
            output = compiled()
            assert output.shape == (2, 256, rt.N)
            nodes = [n for n in graphs[0].graph.nodes if 'vision_fc2_cublaslt' in str(n.target)]
            assert len(graphs) == 1 and len(nodes) == 1 and len(nodes[0].args) == 4
            assert all(s._prepared_compile_token is None for s in handle.sites)
            model.layers[0].mlp.fc2.bias.add_(1)
            with pytest.raises(RuntimeError, match='inputs changed'):
                compiled()
            prepare(model, runner)
            with pytest.raises(RuntimeError, match='generation'):
                compiled()
            compiled.retire()
            with pytest.raises(RuntimeError, match='retired'):
                compiled()
    finally:
        close_case(handle, runner)


def test_real_screen_link_is_independent_and_preserves_old_strict_scope():
    root = Path(__file__).resolve().parents[2]
    path = root / rt.REAL_SCREEN['path']
    if not path.exists():
        pytest.skip('Saved results are not included in this source-only deployment')
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == rt.REAL_SCREEN['sha256']
    report = json.loads(data)
    assert report['algorithm'] == rt.ALGORITHM
    assert report['synthetic'] is False and report['status'] == 'complete'
    assert report['numerical_screen']['eligible'] is True
    assert report['quality_passed'] is False and report['strict_probe_reinterpreted'] is False
    assert len(report['fixture_records']) == 5 and report['site_budget'] == 27


def _require_g4():
    if not torch.cuda.is_available():
        pytest.skip('CUDA deferred to Colab')
    if 'RTX PRO 6000 Blackwell' not in torch.cuda.get_device_name():
        pytest.skip('Pinned algorithm is G4/SM120 only')


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deferred to Colab')
def test_cuda_real_native_runner_stream_handoff_owned_output_and_generation_recapture():
    _require_g4()
    settings = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model = handle = runner = None
    try:
        model = SmallVision('cuda')
        handle = candidate.install_vision_fc2_cublaslt(model)
        runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
        handle.bind_runner(runner)
        generator = torch.Generator(device='cuda').manual_seed(113021)
        x = torch.randn(rt.INPUT_SHAPE, device='cuda', dtype=torch.bfloat16, generator=generator) / 8
        args = dict(images=[], image_masks=[], lang_tokens=torch.ones((1, 4), device='cuda', dtype=torch.int64),
                    lang_masks=torch.ones((1, 4), device='cuda', dtype=torch.bool), noise=x, num_steps=10)
        first = runner(**args)
        first_owned = first.clone()
        old_entry = next(iter(runner.entries.values()))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            args['noise'] = x * .75
            second = runner(**args)
        torch.cuda.current_stream().wait_stream(stream)
        assert runner.stats['captures'] == 1 and runner.stats['stream_handoffs'] >= 1
        assert first.data_ptr() != second.data_ptr() and second.data_ptr() != old_entry.output.data_ptr()
        _exact(first, first_owned)
        with torch.no_grad(), runner._call_lock:
            prepare(model, runner)
            expected = model.sample_actions([], [], None, None, noise=args['noise'])
        _exact(second, expected)
        with torch.no_grad():
            model.layers[0].mlp.fc2.weight.mul_(.875)
            model.layers[13].mlp.fc2.bias.add_(.03125)
        third = runner(**args)
        assert runner.stats['invalidations'] == 1 and runner.stats['captures'] == 2
        assert not old_entry.compiled_callable.active
        with torch.no_grad(), runner._call_lock:
            prepare(model, runner)
            refreshed = model.sample_actions([], [], None, None, noise=args['noise'])
        _exact(third, refreshed)
        _exact(first, first_owned)
        with pytest.raises(ValueError, match='Clear all live'):
            handle.uninstall()
        keys = [p.key for p in handle.runtime.plans]
        runner.clear()  # Must drain event, retire compiled wrappers, then permit descriptor release.
        assert runner._event is None and runner._device is None
        handle.uninstall()
        assert all(key not in rt._PLANS for key in keys)
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            if handle is not None and handle.enabled:
                handle.uninstall()
            torch.backends.cuda.matmul.allow_tf32 = settings[0]
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deferred to Colab')
@pytest.mark.parametrize('compiled', [False, True])
def test_cuda_all27_self_exact_capture_replay_owned_outputs_and_weight_bias_refresh(compiled):
    _require_g4()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model = handle = runner = graph = execute = None
    try:
        model, handle, runner = make_case(device='cuda', fake=False)
        generator = torch.Generator(device='cuda').manual_seed(113019)
        x = torch.randn(rt.INPUT_SHAPE, device='cuda', dtype=torch.bfloat16, generator=generator) / 8
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        def run():
            return tuple(layer.mlp.fc2(x) for layer in model.layers)
        with torch.no_grad(), runner._call_lock, torch.cuda.stream(stream):
            context = prepare(model, runner)
            expected = tuple(y.clone() for y in run())
            execute = compile_prepared_run(model, run, owner=runner, context=context) if compiled else run
            for actual, reference in zip(execute(), expected, strict=True):
                _exact(actual, reference)
            execute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = execute()
            for _ in range(3):
                graph.replay()
                for actual, reference in zip(captured, expected, strict=True):
                    _exact(actual, reference)
            owned = tuple(y.clone() for y in captured)
            x.mul_(.75)
            graph.replay()
            for actual, reference in zip(captured, run(), strict=True):
                _exact(actual, reference)
            for actual, reference in zip(owned, expected, strict=True):
                _exact(actual, reference)
            stream.synchronize()
            graph.reset()
            graph = None
            pointers = [(s.weight.data_ptr(), s.bias.data_ptr()) for s in handle.sites]
            for index in (0, 13, 26):
                model.layers[index].mlp.fc2.weight.mul_(.875)
                model.layers[index].mlp.fc2.bias.add_(.03125)
            with pytest.raises(RuntimeError, match='changed or unprepared'):
                model.layers[0].mlp.fc2(x)
            if compiled:
                with pytest.raises(RuntimeError, match='inputs changed'):
                    execute()
                execute.retire()
            context = prepare(model, runner)
            assert pointers == [(s.weight.data_ptr(), s.bias.data_ptr()) for s in handle.sites]
            refreshed = tuple(y.clone() for y in run())
            execute = compile_prepared_run(model, run, owner=runner, context=context) if compiled else run
            execute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                recaptured = execute()
            graph.replay()
            for actual, reference in zip(recaptured, refreshed, strict=True):
                _exact(actual, reference)
            # Native empty-camera behavior remains untouched with actual tensors.
            native_input = x[:1].contiguous()
            _exact(model.layers[0].mlp.fc2(native_input), nn.Linear.forward(model.layers[0].mlp.fc2, native_input))
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            if graph is not None:
                graph.reset()
            if compiled and execute is not None and hasattr(execute, 'retire'):
                execute.retire()
            if runner is not None:
                runner.close()
            if handle is not None and handle.enabled:
                handle.uninstall()
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
