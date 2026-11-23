"""CPU owner contracts; fictional policies never constitute SM89 qualification."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'src/vvla_repro_pi05/lite/batch_l4_compile_libdevice.py'
spec = importlib.util.spec_from_file_location('l4_compile_owner_cpu_contract', PATH)
owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)


@pytest.fixture
def fixture(monkeypatch):
    proof = dict(library=dict(path='/test-only/l4-toolkit-libdevice', sha256=owner.LIBDEVICE_SHA256))
    class Config:
        emulate_precision_casts = False
        eager_numerics = SimpleNamespace(use_pytorch_libdevice=False)
        fail = False
        @contextmanager
        def patch(self, options):
            assert options == owner.OPTIONS
            old = self.emulate_precision_casts
            self.emulate_precision_casts = options['emulate_precision_casts']
            try:
                if self.fail:
                    raise RuntimeError('original context failed')
                yield
            finally:
                self.emulate_precision_casts = old
    config = Config(); knobs = SimpleNamespace(libdevice_path=None); calls = []
    behavior = dict(fail=False, wrong_library=False, on_helper=None)
    def helper():
        calls.append((config.emulate_precision_casts, config.eager_numerics.use_pytorch_libdevice))
        assert config.emulate_precision_casts is True
        if behavior['on_helper'] is not None: behavior['on_helper']()
        if behavior['fail']:
            raise RuntimeError('original helper failed')
        path = '/foreign' if behavior['wrong_library'] else proof['library']['path']
        monkeypatch.setenv('TRITON_LIBDEVICE_PATH', path); knobs.libdevice_path = path
    tasks = SimpleNamespace(_set_triton_libdevice_path=helper)
    compilation = SimpleNamespace(INDUCTOR_OPTIONS=deepcopy(owner.OPTIONS))
    codecache, checkpoint_engine = SimpleNamespace(), SimpleNamespace()
    bindings = [config, tasks, knobs, compilation, codecache, checkpoint_engine]
    math = deepcopy(owner.MATH)
    def native_math():
        if not owner._exact_policy(math):
            raise RuntimeError('native math changed')
        return deepcopy(math)
    monkeypatch.setattr(owner, 'native_math_policy', native_math)
    monkeypatch.setattr(owner, 'load_qualification', lambda evidence: deepcopy(proof))
    runtime = dict(original_compile_options=deepcopy(owner.OPTIONS), math_policy=deepcopy(owner.MATH),
                   test_only='fictional CPU bindings; no GPU support')
    monkeypatch.setattr(owner, '_runtime', lambda policy, evidence: deepcopy(runtime))
    monkeypatch.setattr(owner, '_bindings', lambda: tuple(bindings))
    monkeypatch.delenv('TRITON_LIBDEVICE_PATH', raising=False)
    class Policy:
        def __init__(self):
            self._closed = False; self._calls = 0
            self.device = SimpleNamespace(type='cuda')
            self.graph_runner = SimpleNamespace(_closed=False, compile_sample_actions=True, entries={})
            self._lite_optimizations = {'compile_sample_actions': True}
            self._lock = threading.Lock()
    policy = Policy()
    value = owner.L4BatchCompileLibdeviceOwner(policy, {'test_only': 'fictional CPU proof'}, batch_size=16)
    return SimpleNamespace(policy=policy, value=value, config=config, knobs=knobs, tasks=tasks,
        compilation=compilation, calls=calls, runtime=runtime, proof=proof, math=math,
        bindings=bindings, behavior=behavior, Policy=Policy)


def test_cold_construction_and_audit_never_initialize_helper_or_warm_model(fixture):
    f = fixture
    before = f.value.manifest()
    assert f.calls == [] and f.policy._calls == 0 and f.policy.graph_runner.entries == {}
    assert f.value.diagnostics() == {'initialized': False}
    assert f.config.emulate_precision_casts is False and f.knobs.libdevice_path is None
    assert before['default_enabled'] is False and before['batch_size'] == 16
    assert before['request_cache'] is False
    assert f.value.audit(f.policy) == before and f.calls == []


def test_first_request_once_and_static_recipe_survives_initialization(fixture):
    f = fixture; before = f.value.manifest()
    with pytest.raises(RuntimeError, match='request lock'):
        f.value.ensure_initialized(f.policy)
    assert f.calls == []
    with f.policy._lock:
        f.value.ensure_initialized(f.policy)
        f.value.ensure_initialized(f.policy)
    assert f.calls == [(True, False)] and f.config.emulate_precision_casts is False
    assert f.value.manifest() == before and f.value.audit(f.policy) == before
    diag = f.value.diagnostics()
    assert diag['initialization_cost_inside_request'] is True
    assert diag['native_math_before'] == diag['native_math_after'] == owner.MATH
    assert set(diag) == {'initialized', 'initialization_ms', 'native_math_before', 'native_math_after',
                         'library_sha256', 'initialization_cost_inside_request'}


def test_manifest_and_diagnostics_do_not_expose_mutable_owner_state(fixture):
    f = fixture
    m = f.value.manifest(); m['actual']['original_compile_options'].clear()
    assert f.value.manifest()['actual']['original_compile_options'] == owner.OPTIONS
    with f.policy._lock:
        f.value.ensure_initialized(f.policy)
    d = f.value.diagnostics(); d['native_math_after'].clear()
    assert f.value.diagnostics()['native_math_after'] == owner.MATH


@pytest.mark.parametrize('fault', ['helper', 'options_replace', 'options_value', 'options_type', 'patch',
    'env', 'knob', 'runner_closed', 'policy_closed', 'foreign_policy', 'compile_route',
    'outside_context', 'pytorch_libdevice', 'compiler_module', 'checkpoint_module'])
def test_cheap_guard_rejects_changed_live_bindings_without_forward(fixture, monkeypatch, fault):
    f = fixture; policy = f.policy
    if fault == 'helper': f.tasks._set_triton_libdevice_path = lambda: None
    elif fault == 'options_replace': f.compilation.INDUCTOR_OPTIONS = dict(f.compilation.INDUCTOR_OPTIONS)
    elif fault == 'options_value': f.compilation.INDUCTOR_OPTIONS['use_fast_math'] = True
    elif fault == 'options_type': f.compilation.INDUCTOR_OPTIONS['use_fast_math'] = 0
    elif fault == 'patch': f.config.patch = lambda *args: None
    elif fault == 'env': monkeypatch.setenv('TRITON_LIBDEVICE_PATH', '/foreign')
    elif fault == 'knob': f.knobs.libdevice_path = '/foreign'
    elif fault == 'runner_closed': f.policy.graph_runner._closed = True
    elif fault == 'policy_closed': f.policy._closed = True
    elif fault == 'foreign_policy': policy = f.Policy()
    elif fault == 'compile_route': f.policy.graph_runner.compile_sample_actions = False
    elif fault == 'outside_context': f.config.emulate_precision_casts = True
    elif fault == 'pytorch_libdevice': f.config.eager_numerics = SimpleNamespace(use_pytorch_libdevice=True)
    elif fault == 'compiler_module': f.bindings[4] = SimpleNamespace()
    else: f.bindings[5] = SimpleNamespace()
    with pytest.raises(RuntimeError):
        with f.policy._lock: f.value.ensure_initialized(policy)
    assert f.calls == []


@pytest.mark.parametrize(('key', 'value'), [('backend_fp32_precision', 'ieee'), ('legacy_allow_tf32', 0),
    ('legacy_allow_tf32', True), ('float32_matmul_precision', 'high'), ('global_fp32_precision', 'tf32'),
    ('fp16_reduction', 1), ('fp16_reduction', False), ('bf16_reduction', False)])
def test_all_original_native_math_getters_are_strict(fixture, key, value):
    f = fixture; f.math[key] = value
    with pytest.raises(RuntimeError, match='native math'):
        with f.policy._lock: f.value.ensure_initialized(f.policy)
    assert f.calls == []


@pytest.mark.parametrize('failure', ['context', 'helper', 'wrong_library'])
def test_failed_initialization_is_poisoned_and_can_be_closed_without_retry(fixture, failure):
    f = fixture
    if failure == 'context': f.config.fail = True
    else: f.behavior['fail' if failure == 'helper' else 'wrong_library'] = True
    with pytest.raises(RuntimeError):
        with f.policy._lock: f.value.ensure_initialized(f.policy)
    assert f.config.emulate_precision_casts is False and f.value.diagnostics()['initialized'] is False
    call_count = len(f.calls)
    with pytest.raises(RuntimeError, match='retired, failed'):
        with f.policy._lock: f.value.ensure_initialized(f.policy)
    assert len(f.calls) == call_count
    f.value.close(); f.value.close()


@pytest.mark.parametrize('failure', ['proof', 'runtime'])
def test_explicit_cold_audit_rehashes_qualification_and_current_runtime(fixture, failure):
    f = fixture
    if failure == 'proof': f.proof['changed'] = True
    else: f.runtime['changed'] = 'compiler/source/library/binary changed'
    with pytest.raises(RuntimeError): f.value.audit(f.policy)


@pytest.mark.parametrize('batch', [1, 8, 32, True, 16.0])
def test_b16_evidence_cannot_authorize_another_geometry(fixture, batch):
    with pytest.raises(ValueError, match='B16'):
        owner.L4BatchCompileLibdeviceOwner(fixture.Policy(), {}, batch_size=batch)


@pytest.mark.parametrize('fault', ['entries', 'calls', 'closed', 'cpu', 'compile_disabled'])
def test_owner_requires_fresh_private_compiled_policy(fixture, fault):
    policy = fixture.Policy()
    if fault == 'entries': policy.graph_runner.entries['already-used'] = object()
    elif fault == 'calls': policy._calls = 1
    elif fault == 'closed': policy._closed = True
    elif fault == 'cpu': policy.device.type = 'cpu'
    else: policy._lite_optimizations['compile_sample_actions'] = False
    with pytest.raises(ValueError): owner.L4BatchCompileLibdeviceOwner(policy, {}, batch_size=16)


def test_retired_owner_rejects_actual_policy(fixture):
    fixture.value.close(); fixture.value.close()
    with pytest.raises(RuntimeError): fixture.value.audit(fixture.policy)


@pytest.mark.parametrize('bad', [None, {}, {'backend_report_sha256': '762768b638a5a90d5ca02a833d7e73977d880e2f6ff01c475fbec5fa661d84ba'},
    {'backend_report': 'missing', 'backend_report_sha256': owner.REPORT_SHA256,
     'probe_directory': 'missing', 'launch_directory': 'missing', 'qualified_source_root': 'missing', 'extra': True}])
def test_g4_or_incomplete_qualification_is_rejected_before_loading_files(bad):
    with pytest.raises(ValueError): owner.load_qualification(bad)


def test_getter_uses_original_native_policy_without_any_setter(monkeypatch):
    matmul = SimpleNamespace(fp32_precision='none', allow_tf32=False,
        allow_fp16_reduced_precision_reduction=True, allow_bf16_reduced_precision_reduction=True)
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), fp32_precision='none'),
        get_float32_matmul_precision=lambda: 'highest')
    before = deepcopy(vars(matmul))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    assert owner.native_math_policy() == owner.MATH
    assert vars(matmul) == before and fake.backends.fp32_precision == 'none'


def test_qualification_file_reader_rejects_links_and_relative_directories(tmp_path):
    path = tmp_path / 'proof.json'; path.write_text('{}')
    linked = tmp_path / 'linked.json'; linked.symlink_to(path)
    with pytest.raises(ValueError): owner._sha(linked)
    with pytest.raises(ValueError): owner._checked_directory('relative')
    linked_dir = tmp_path / 'directory'; linked_dir.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): owner._checked_directory(linked_dir)


def test_qualification_rehashes_raw_bytes_before_interpreting_any_claim(tmp_path, monkeypatch):
    import hashlib
    report = tmp_path / 'report.json'; report.write_text('{}')
    probe = tmp_path / 'probe'; probe.mkdir()
    launch = tmp_path / 'launch'; launch.mkdir()
    source = tmp_path / 'source'; source.mkdir()
    # Fictional CPU source trees have no connection to a real qualification.
    files = {}
    for index in range(346):
        path = source / (str(index) + '.py'); path.write_text('# CPU fixture\n')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker = source / 'SOURCE_PAYLOAD.json'; marker.write_text(json.dumps({'files_sha256': files}))
    monkeypatch.setattr(owner, 'SOURCE', hashlib.sha256(marker.read_bytes()).hexdigest())
    raw = probe / 'initial.npz'; raw.write_bytes(b'fictional CPU raw artifact')
    monkeypatch.setattr(owner, 'REPORT_SHA256', hashlib.sha256(report.read_bytes()).hexdigest())
    monkeypatch.setattr(owner, 'PROBE_FILES', {'initial.npz': hashlib.sha256(raw.read_bytes()).hexdigest()})
    monkeypatch.setattr(owner, 'LAUNCH_FILES', {})
    raw.write_bytes(b'changed raw output')
    evidence = dict(backend_report=str(report), backend_report_sha256=owner.REPORT_SHA256,
                    probe_directory=str(probe), launch_directory=str(launch), qualified_source_root=str(source))
    with pytest.raises(ValueError, match='artifact changed: initial.npz'):
        owner.load_qualification(evidence)


def test_actual_runtime_gate_hashes_optional_graph_source_without_importing_vllm(tmp_path, monkeypatch):
    """Exercise the runtime source gate with no optional graph/vLLM modules."""
    import socket
    import subprocess
    package = '_l4_runtime_no_vllm'
    for name in (package, package+'.lite', package+'.models', package+'.models.pi05'):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    def module(name, filename):
        value = ModuleType(package+name); value.__file__ = str(tmp_path/filename)
        path = Path(value.__file__); path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# CPU fixture\n')
        monkeypatch.setitem(sys.modules, value.__name__, value)
        return value
    pi05 = module('.lite.pi05', 'lite/pi05.py')
    graph = module('.lite.graph', 'lite/graph.py')
    model = module('.models.pi05.modeling_pi05', 'models/pi05/modeling_pi05.py')
    Path(model.__file__).with_name('graph_pi05.py').write_text('raise RuntimeError("must never import optional wrapper")\n')
    class Policy: pass
    class Runner: pass
    pi05.Pi05LitePolicy = Policy; graph.Pi05NativeGraphRunner = Runner
    policy = Policy(); policy.device = 'cuda'; policy.graph_runner = Runner()
    monkeypatch.setattr(owner, '__package__', package+'.lite')
    props = SimpleNamespace(uuid='test-only', name='NVIDIA L4', major=8, minor=9)
    torch = SimpleNamespace(__version__=owner.RUNTIME['torch'], __file__=str(tmp_path/'torch/__init__.py'),
        version=SimpleNamespace(git_version=owner.RUNTIME['git'], cuda=owner.RUNTIME['cuda']),
        cuda=SimpleNamespace(get_device_properties=lambda device: props))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(__version__='3.7.1'))
    monkeypatch.setattr(socket, 'gethostname', lambda: 'test-host')
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: '580.82.07\n')
    config = SimpleNamespace(__file__=str(tmp_path/'config.py'), emulate_precision_casts=False,
        eager_numerics=SimpleNamespace(use_pytorch_libdevice=False))
    tasks = SimpleNamespace(__file__=str(tmp_path/'compile_tasks.py'))
    compilation = SimpleNamespace(__file__=str(tmp_path/'compilation.py'), INDUCTOR_OPTIONS=deepcopy(owner.OPTIONS))
    codecache = SimpleNamespace(__file__=str(tmp_path/'codecache.py'))
    checkpoint = SimpleNamespace(__file__=str(tmp_path/'batch_checkpoint_engine.py'))
    monkeypatch.setattr(owner, '_bindings', lambda: (config,tasks,None,compilation,codecache,checkpoint))
    monkeypatch.setattr(owner, 'native_math_policy', lambda: deepcopy(owner.MATH))
    expected = {'compile_tasks.py':owner.HELPER_SHA256, 'codecache.py':owner.CODECACHE_SHA256,
        'config.py':owner.CONFIG_MODULE_SHA256, 'compilation.py':owner.COMPILATION_SHA256,
        'batch_checkpoint_engine.py':owner.CHECKPOINT_ENGINE_SHA256, 'pi05.py':owner.POLICY_SHA256,
        'graph_pi05.py':owner.GRAPH_SHA256, 'modeling_pi05.py':owner.MODEL_SHA256,
        'graph.py':owner.NATIVE_RUNNER_SHA256, 'libdevice.bc':owner.LIBDEVICE_SHA256}
    visited = []
    def sha(path):
        path = Path(path); visited.append(path)
        return expected.get(path.name, 'test-only-current-binary-sha')
    monkeypatch.setattr(owner, '_sha', sha)
    proof = dict(hardware=dict(gpu_name='NVIDIA L4',gpu_uuid='GPU-test-only',driver='580.82.07',hostname='test-host'),
                 library=dict(path=str(tmp_path/'libdevice.bc')))
    actual = owner._runtime(policy, proof)
    assert actual['sources_sha256']['graph_pi05.py'] == owner.GRAPH_SHA256
    assert Path(model.__file__).with_name('graph_pi05.py') in visited
    assert package+'.models.pi05.graph_pi05' not in sys.modules
    assert 'vllm' not in sys.modules


def test_actual_fresh_child_no_vllm_source_gate():
    """Required on the target ours-env; local scaffolds do not qualify it."""
    import os
    import subprocess
    if os.environ.get('VLA_REQUIRE_REAL_NO_VLLM') != '1':
        pytest.skip('Actual target ours-env regression is explicitly required by its launcher')
    code = '''import importlib.util,json,sys,torch
assert importlib.util.find_spec('vllm') is None
assert not torch.cuda.is_initialized()
from vvla_repro_pi05.lite import pi05,graph,batch_l4_compile_libdevice as owner
from vvla_repro_pi05.models.pi05 import modeling_pi05
model,wrapper=owner._model_source_paths(pi05,modeling_pi05)
assert owner._sha(wrapper)==owner.GRAPH_SHA256
assert owner._sha(model)==owner.MODEL_SHA256
assert owner._sha(pi05.__file__)==owner.POLICY_SHA256
assert owner._sha(graph.__file__)==owner.NATIVE_RUNNER_SHA256
assert 'vvla_repro_pi05.models.pi05.graph_pi05' not in sys.modules
assert 'vllm' not in sys.modules and not torch.cuda.is_initialized()
print(json.dumps(dict(real_fresh_child=True,vllm_available=False,cuda_initialized=False,
    model=str(model),source_only_graph=str(wrapper),graph_sha256=owner._sha(wrapper))))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt['real_fresh_child'] is True and receipt['vllm_available'] is False
    assert receipt['cuda_initialized'] is False and receipt['graph_sha256'] == owner.GRAPH_SHA256


def test_source_only_graph_binding_rejects_cross_package_and_link(tmp_path):
    root = tmp_path.resolve(); lite=root/'lite'; core=root/'models/pi05'
    lite.mkdir(); core.mkdir(parents=True)
    policy=lite/'pi05.py'; model=core/'modeling_pi05.py'; wrapper=core/'graph_pi05.py'
    for path in (policy,model,wrapper):path.write_text('# CPU fixture\n')
    policy_module=SimpleNamespace(__file__=str(policy));model_module=SimpleNamespace(__file__=str(model))
    assert owner._model_source_paths(policy_module,model_module)==(model,wrapper)
    foreign=root/'foreign/modeling_pi05.py';foreign.parent.mkdir();foreign.write_text('# foreign\n')
    with pytest.raises(ValueError,match='containment'):
        owner._model_source_paths(policy_module,SimpleNamespace(__file__=str(foreign)))
    wrapper.unlink();wrapper.symlink_to(foreign)
    with pytest.raises(ValueError,match='containment'):
        owner._model_source_paths(policy_module,model_module)
