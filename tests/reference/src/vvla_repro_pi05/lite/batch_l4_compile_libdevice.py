"""Opt-in SM89/B16 owner of the original Inductor libdevice initialization.

This module is independent of the SM120 owner. Its qualification is the actual
L4 native-math-v2 seven-request diagnostic, not a full quality or timing claim.
Construction/audit read provenance only. The one original helper invocation
belongs inside the first raw request's timer and the existing policy lock.
No cache lookup, guard, key, result, model, observation or action is retained.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import weakref

IMPLEMENTATION = 'pi05-l4-b16-original-inductor-libdevice-before-cache-v1'
REPORT_SHA256 = '421b6da22c3a458932d415d6d2fdc671538b4f54c76c32f45c961b68a6832bcb'
SOURCE = '8b8ab363e58d8a8746eaf734e5492108cac24332bd84b35f9ac2a7ca945686cd'
CONFIG = '745572afd472f52bf99a7e064375db4fce2fc3017360d7547f557660dcbe51a3'
RECIPE = '560e2ea8e7d55a1e1218b6059a14e65594ea2e2f7990081230e7bd42d94a9222'
FORMAL = 'e42d3d529e1be205ec105ada6a0325b4499e329fa23eaec1d27f257aba4136f1'
LIBDEVICE_SHA256 = '91334d6e12748f6cb5bbf0a1cd965a56bcd93dc4f496d2e5c5f8c6e523094356'
BUNDLED_SHA256 = '5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee'
HELPER_SHA256 = '824244b91623128e299df93db614e1f046b9ddee2f0ca8a84c6e7707bbacf5c2'
CODECACHE_SHA256 = '1de34a274c58a2b826a6c46829e94f1df5b24a3d83841a6c4ff8a691a50d8c59'
CONFIG_MODULE_SHA256 = '4cdbe0025fd4890e2dfaa6395e57a08971c33889265cbcd26b33a1cf9ee1fa7f'
COMPILATION_SHA256 = '4066f24920bd5ce7062c517a2a74e7abba8fe43a2b37adf1d63f84439e38c01c'
CHECKPOINT_ENGINE_SHA256 = 'daafeb5420d978ade83fceddff1db0e4090011454e47e1c66d2329d179e35e50'
POLICY_SHA256 = 'a5e4729f33291fed58f87ce50474099a93a964fe4c5758064354e369997eb531'
GRAPH_SHA256 = 'c3922112b0860b76271ff87a2da83d1c37d9e145e533007173516b072517ca5f'
MODEL_SHA256 = 'fc53f4e4272e0e441b56fbc08949c80dbf213eea7459bbba535e5addacdcf443'
NATIVE_RUNNER_SHA256 = '1b9fb828a9acaaef835219100c40d7191aa7a0fa1652eb9d610e7a53e96a908d'
OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
RUNTIME = {'torch': '2.13.0+cu130', 'git': 'cf30153c4c131c8164ee7798e5022d810682e2cb', 'cuda': '13.0'}
MATH = {'backend_fp32_precision': 'none', 'legacy_allow_tf32': False,
        'float32_matmul_precision': 'highest', 'global_fp32_precision': 'none',
        'fp16_reduction': True, 'bf16_reduction': True}
LABELS = ('initial', 'changing-1', 'changing-2', 'changing-3', 'changing-4', 'changing-5', 'initial-restored')

# Exact producer bytes, retained unchanged after the independent CPU audit.
# The raw NPZ files are hashed without loading arrays or executing cache files.
PROBE_FILES = {'STATUS.json': '0338e9eb48d3296b24ded5a67c1a89347eeb2ca594e9d0c868a37f21e399f887',
 'LIBDEVICE_INITIALIZATION.json': '6c0d322b4fb4b39e8e85a14c49f91269684d2f5bc5da0334ee609ee4bdbcce7b',
 'MATH_POLICY.json': '24da299c82b74ca450cb8af6c7c282050315e275eaf97ebecd828e920476f2ff',
 'CACHE_CALLS.json': '5a337d68ab9a86dbf2f7b1f23146476eb8eb0477167a2dee5875e099d0b66957',
 'CLEANUP.json': '6d986bc5ca8e1ca9d780bd9c4d7e6d2ad9ddd0789a2ef23ce8af9d1b3b2c37a9',
 'METADATA_INITIAL.json': '10ce68338f182b450cfad3c3abd86c820e4721efc701482767b6f90bd053cfb2',
 'METADATA_FINAL.json': '10ce68338f182b450cfad3c3abd86c820e4721efc701482767b6f90bd053cfb2',
 'FORMAL_IDENTITY.json': '6069715c00f65a6f34228a5e2ba1a01c27976716ccf1a0d63e26b2eedc190a7e',
 'SOURCE_PAYLOAD.json': '8b8ab363e58d8a8746eaf734e5492108cac24332bd84b35f9ac2a7ca945686cd',
 'CONFIG.json': '745572afd472f52bf99a7e064375db4fce2fc3017360d7547f557660dcbe51a3',
 'DRIVER.py': 'd804a3974ff1ca1282a94f1684f2c1796d852deb2f7b1a0fcec65112d8194c4c',
 'HELPER.py': '7703a44ba3c5ad99624e0bc4654270d4ebb46c248b92a5ed29034b91c5a8d6d4',
 'initial.json': '11ac03d49bc8eb238930af9307356b69c9e5224a07930a21f6b064f2cb5a6b05',
 'initial.npz': '4a118a488d448abced773310fac9c46eb7d4cffa6cbf79e0ec3618797e36b94d',
 'changing-1.json': '9dc1b16002d1aace700d58726d8f1db008ad2f048e9e48a65c09f9200420a3b9',
 'changing-1.npz': '69b9e711b4ee12e363c918118d0237abc5dafd56a2518e81fa5cf66e761889df',
 'changing-2.json': '950803558d100aec92699492e1987586dcce4235dd20f0ea3cdac32c096e0b1b',
 'changing-2.npz': 'bcac681804d5134fe5274b10a10bb84f083e49c0059b628243e0ee5aa0afe963',
 'changing-3.json': '3be3161fda2e71ba8ff0e77a24ba141f684a72769535fb1eb8ea00e48b4c551c',
 'changing-3.npz': 'd0a07fd135811edec224a843090aa9000b630a1978c03e06c288da7bc8fa294e',
 'changing-4.json': '41b22e1c9ddb4ca75bda8f76601679cc9946437b9505f29cd44a9b090696d56a',
 'changing-4.npz': '66a7ec3677df9002da1a4cefe8f94c928d1d03f8172b6143ac101ce495a12d93',
 'changing-5.json': '3ad592791bc883eec2c677919d2a5b3aab62c9c1d04feed3527b60ca3f13d1ed',
 'changing-5.npz': '027488b6b2bab7bd20f2075e5675368abe8e82920fd137f1cb9947d87a6d29ff',
 'initial-restored.json': '7cfd13937557421c79275ce092cc78e784eaa83836a8c4acc3a24d0aa851821a',
 'initial-restored.npz': '4a118a488d448abced773310fac9c46eb7d4cffa6cbf79e0ec3618797e36b94d'}
LAUNCH_FILES = {'CHILD_EXIT.json': '8d12bd8649fbc52fc50cff6ab73b5e0582b6a77add4d57d3bfd74b16b2d8d4a3',
 'exit.json': '5b3296fce01441777c86e33e6f7b71261f3d6dfec3da4d1618e61d5057e4f564',
 'GPU_AFTER.json': 'e1b8f12a566bbe29c8fda4764bf7e5de1b3e10854c98f0652c23565e0cdb44d5'}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require a regular qualified file: ' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _exact_policy(value):
    return (type(value) is dict and set(value) == set(MATH)
            and all(type(value[key]) is type(expected) and value[key] == expected
                    for key, expected in MATH.items()))


def _checked_directory(value):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError('An explicit regular qualification directory is required')
    return path.resolve(strict=True)


def load_qualification(evidence):
    """Read one fixed L4 proof, including actual retirement; no CUDA imports."""
    fields = {'backend_report', 'backend_report_sha256', 'probe_directory', 'launch_directory', 'qualified_source_root'}
    if (type(evidence) is not dict or set(evidence) != fields
            or evidence['backend_report_sha256'] != REPORT_SHA256):
        raise ValueError('Require the exact independently audited L4 native-math-v2 diagnostic')
    report_path = Path(evidence['backend_report'])
    if not report_path.is_absolute() or _sha(report_path) != REPORT_SHA256:
        raise ValueError('Independent L4 diagnostic audit changed')
    probe = _checked_directory(evidence['probe_directory'])
    launch = _checked_directory(evidence['launch_directory'])
    source = _checked_directory(evidence['qualified_source_root'])
    if _sha(source / 'SOURCE_PAYLOAD.json') != SOURCE:
        raise ValueError('Original qualified L4 source346 inventory changed')
    inventory = _read(source / 'SOURCE_PAYLOAD.json')['files_sha256']
    if len(inventory) != 346:
        raise ValueError('Original L4 qualification requires all346 source files')
    for name, expected in inventory.items():
        relative = Path(name); path = source / relative
        if (relative.is_absolute() or '..' in relative.parts
                or not path.resolve(strict=True).is_relative_to(source) or _sha(path) != expected):
            raise ValueError('Original qualified L4 source member changed: ' + name)
    for root, files in ((probe, PROBE_FILES), (launch, LAUNCH_FILES)):
        for name, expected in files.items():
            if _sha(root / name) != expected:
                raise ValueError('Original L4 qualification artifact changed: ' + name)
    report, state = _read(report_path), _read(probe / 'STATUS.json')
    initialization = _read(probe / 'LIBDEVICE_INITIALIZATION.json')
    math = _read(probe / 'MATH_POLICY.json')
    calls = _read(probe / 'CACHE_CALLS.json')
    if (report.get('status') != 'independent_cpu_raw_and_bindings_audit_passed'
            or report.get('qualification_profile') != 'native-math-v2'
            or report.get('source_inventory_sha256') != SOURCE or report.get('config_sha256') != CONFIG
            or report.get('recipe_sha256') != RECIPE or report.get('candidate_formal_content_sha256') != FORMAL
            or report.get('batch_size') != 16 or report.get('source_files') != 346
            or report.get('production_readiness_claim') is not False
            or report.get('closed_loop_quality_result') is not False
            or state.get('status') != 'complete' or state.get('actual_cache_hit') is not True
            or state.get('final_bindings_verified') is not True or state.get('all_original_outputs_and_noise_exact') is not True
            or state.get('runtime') != RUNTIME or state.get('cache_reset') is not False
            or state.get('prewarm') is not False or state.get('closed_loop') is not False
            or state.get('source_inventory_sha256') != SOURCE or state.get('config_sha256') != CONFIG
            or state.get('recipe_sha256') != RECIPE or state.get('formal_content_sha256') != FORMAL
            or state.get('batch_size') != 16 or state.get('groups') != 5
            or state.get('calls') != [_read(probe / (label + '.json')) for label in LABELS]
            or [row.get('label') for row in state['calls']] != list(LABELS)
            or any(any(value is not True for value in row['checks'].values()) for row in state['calls'])
            or _read(probe / 'CLEANUP.json') != {'adapter_closed': True, 'passed': True}):
        raise ValueError('L4 diagnostic is incomplete or outside its qualified source/B16 scope')
    if (math != state['math_policy_snapshots'] or math != report['actual_native_math_snapshots']
            or set(math) != {'before_constructor', 'after_constructor', 'after_calls', 'after_all_calls'}
            or [row['label'] for row in math['after_calls']] != list(LABELS)
            or not all(_exact_policy(value) for value in [math['before_constructor'], math['after_constructor'],
                math['after_all_calls'], *[row['policy'] for row in math['after_calls']]])):
        raise ValueError('Actual original native-math policy snapshots changed')
    if (initialization != state['initialization'] or initialization != report['actual_initialization']
            or initialization['helper_name'] != 'torch._inductor.runtime.compile_tasks._set_triton_libdevice_path'
            or initialization['helper_source_sha256'] != HELPER_SHA256
            or initialization['original_compile_options'] != OPTIONS):
        raise ValueError('Original compiler helper/options proof changed')
    for phase, precision, library in (('before', False, BUNDLED_SHA256), ('inside_before', True, BUNDLED_SHA256),
            ('inside_after', True, LIBDEVICE_SHA256), ('after', False, LIBDEVICE_SHA256)):
        row = initialization[phase]
        if (row['emulate_precision_casts'] is not precision or row['use_pytorch_libdevice'] is not False
                or row['key'] != 'libdevice-' + library or len(row['extern_libs']) != 1
                or row['extern_libs'][0]['name'] != 'libdevice' or row['extern_libs'][0]['sha256'] != library):
            raise ValueError('Original helper precision context/library transition changed')
    if (calls != state['cache_calls'] or len(calls) != 1 or calls[0]['kind'] != 'load_with_key'
            or calls[0].get('graph_returned') is not True or calls[0]['cache_info'].get('cache_state') != 'hit'
            or state.get('actual_first_FX_cache_hit_delta') != 1):
        raise ValueError('Original fully guarded FX cache did not actually return its graph')
    event = calls[0]['evidence_file']
    if (Path(event).name != event or _sha(probe / event) != calls[0]['evidence_sha256']
            or calls[0]['evidence_sha256'] != report['actual_cache_evidence']['events'][0]['evidence_sha256']):
        raise ValueError('Original cache observation changed')
    child, outer, idle = (_read(launch / name) for name in ('CHILD_EXIT.json', 'exit.json', 'GPU_AFTER.json'))
    if (child.get('confirmed_exit') is not True or child.get('exit_code') != 0
            or child.get('pid') != state['pid'] or outer.get('confirmed_exit') is not True
            or outer.get('exit_code') != 0 or idle.get('idle') is not True or idle.get('compute_apps') != ''):
        raise ValueError('Actual diagnostic process did not retire normally and leave the card idle')
    return dict(schema=1, backend_report=str(report_path.resolve()), backend_report_sha256=REPORT_SHA256,
        source_inventory_sha256=SOURCE, config_sha256=CONFIG, recipe_sha256=RECIPE,
        formal_content_sha256=FORMAL, batch_size=16, hardware=deepcopy(state['hardware']),
        runtime=deepcopy(RUNTIME), library=deepcopy(initialization['after']['extern_libs'][0]),
        original_compile_options=deepcopy(OPTIONS), math_policy=deepcopy(MATH),
        qualified_source_inventory_sha256=SOURCE, qualified_source_files=346,
        probe_files_sha256=deepcopy(PROBE_FILES), launch_files_sha256=deepcopy(LAUNCH_FILES),
        cache_evidence_sha256=calls[0]['evidence_sha256'],
        qualification_scope='same-card SM89 B16 seven original real requests exact and a guarded FX hit; no production, full80 quality, warm or all-request speedup inheritance')


def _bindings():
    from torch._inductor import config, codecache
    from torch._inductor.runtime import compile_tasks
    from triton import knobs
    from . import compilation, batch_checkpoint_engine
    return config, compile_tasks, knobs.nvidia, compilation, codecache, batch_checkpoint_engine


def _patch_function(config):
    method = config.patch
    return getattr(method, '__func__', method)


def native_math_policy():
    """Getters only: assigning legacy allow_tf32=False changes none to ieee."""
    import torch
    value = dict(backend_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
        legacy_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        global_fp32_precision=torch.backends.fp32_precision,
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if not _exact_policy(value):
        raise RuntimeError('Original L4 native-math none/legacyFalse/highest policy changed')
    return value


def _model_source_paths(policy_module, model_module):
    """Bind a source-only optional wrapper within the actual model package."""
    policy = Path(policy_module.__file__).resolve(strict=True)
    model = Path(model_module.__file__).resolve(strict=True)
    core = policy.parent.parent / 'models' / 'pi05'
    wrapper = core / 'graph_pi05.py'
    if (policy.name != 'pi05.py' or policy.parent.name != 'lite'
            or model != core / 'modeling_pi05.py' or wrapper.is_symlink()
            or wrapper.resolve(strict=True).parent != core):
        raise ValueError('Actual policy/model package or optional graph source containment changed')
    return model, wrapper


def _runtime(policy, proof):
    import socket
    import subprocess
    import torch
    import triton
    from . import pi05, graph
    from ..models.pi05 import modeling_pi05
    model_source, graph_source = _model_source_paths(pi05, modeling_pi05)
    config, tasks, knobs, compilation, codecache, checkpoint_engine = _bindings()
    props = torch.cuda.get_device_properties(policy.device)
    uuid = str(props.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    driver = subprocess.check_output(['nvidia-smi', '--id=' + uuid, '--query-gpu=driver_version',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    hardware = dict(gpu_name=props.name, gpu_uuid=uuid, driver=driver, hostname=socket.gethostname())
    runtime = dict(torch=str(torch.__version__), git=torch.version.git_version, cuda=torch.version.cuda)
    expected_sources = {'compile_tasks.py': (tasks.__file__, HELPER_SHA256),
        'codecache.py': (codecache.__file__, CODECACHE_SHA256), 'config.py': (config.__file__, CONFIG_MODULE_SHA256),
        'compilation.py': (compilation.__file__, COMPILATION_SHA256),
        'batch_checkpoint_engine.py': (checkpoint_engine.__file__, CHECKPOINT_ENGINE_SHA256),
        # The native Lite runner does not execute the vLLM graph wrapper. Bind
        # its original source bytes beside the actually loaded model without
        # importing that optional wrapper and its unrelated vLLM dependency.
        'pi05.py': (pi05.__file__, POLICY_SHA256),
        'graph_pi05.py': (graph_source, GRAPH_SHA256),
        'modeling_pi05.py': (model_source, MODEL_SHA256),
        'graph.py': (graph.__file__, NATIVE_RUNNER_SHA256)}
    actual_sources = {name: _sha(path) for name, (path, _) in expected_sources.items()}
    if (type(policy) is not pi05.Pi05LitePolicy or type(policy.graph_runner) is not graph.Pi05NativeGraphRunner
            or hardware != proof['hardware'] or props.name != 'NVIDIA L4' or (props.major, props.minor) != (8, 9)
            or runtime != RUNTIME or triton.__version__ != '3.7.1'
            or any(actual_sources[name] != expected for name, (_, expected) in expected_sources.items())
            or _sha(proof['library']['path']) != LIBDEVICE_SHA256
            or _wire(compilation.INDUCTOR_OPTIONS) != _wire(OPTIONS)
            or config.emulate_precision_casts is not False or config.eager_numerics.use_pytorch_libdevice is not False):
        raise ValueError('Actual SM89 card/compiler/runtime/precision/library differs from L4 qualification')
    binary_root = Path(torch.__file__).resolve().parent
    return dict(runtime=runtime, triton=triton.__version__, hardware=hardware, capability=[8, 9],
        math_policy=native_math_policy(), original_compile_options=deepcopy(OPTIONS),
        sources_sha256={**actual_sources, 'batch_l4_compile_libdevice.py': _sha(__file__)},
        torch_binary_sha256={name: _sha(binary_root / name) for name in ('lib/libtorch_cuda.so', 'lib/libtorch_cpu.so')})


class L4BatchCompileLibdeviceOwner:
    """One private owner; no compiler initialization occurs in its constructor."""
    __slots__ = ('_policy', '_evidence', '_proof', '_manifest', '_anchors', '_initialized', '_closed', '_failed', '_diagnostics')

    def __init__(self, policy, evidence, *, batch_size):
        if (type(batch_size) is not int or batch_size != 16 or policy._closed or policy.device.type != 'cuda'
                or policy.graph_runner._closed or policy.graph_runner.compile_sample_actions is not True
                or policy.graph_runner.entries or policy._calls
                or policy._lite_optimizations.get('compile_sample_actions') is not True):
            raise ValueError('L4 compile initialization requires a fresh actual private CUDA B16 policy')
        proof = load_qualification(evidence)
        actual = _runtime(policy, proof)
        config, tasks, knobs, compilation, codecache, checkpoint_engine = _bindings()
        self._policy = weakref.ref(policy); self._evidence = deepcopy(evidence); self._proof = proof
        self._anchors = (config, tasks, knobs, compilation, codecache, checkpoint_engine,
            tasks._set_triton_libdevice_path, _patch_function(config), compilation.INDUCTOR_OPTIONS, policy.graph_runner)
        self._manifest = dict(schema=1, implementation=IMPLEMENTATION, batch_size=16,
            qualification=deepcopy(proof), actual=actual, default_enabled=False,
            initialization='once inside the first timed raw request after fresh preprocessing/FP32 noise and before the unchanged graph runner',
            cache='original key/lookup/full guards/results retained; no reset, deletion, prewarm, fallback or key/result substitution',
            precision='original Inductor options/context and strict native math getters; no legacy precision setters',
            request_cache=False, graph_and_compilation_sources_unchanged=True,
            validation='cold full model seal and explicit source/library/runtime audit; current owner/options/helper/env/knob/math checks on every request',
            quality_scope='seven-request mechanism evidence only; a new source/recipe qualification and complete80 remain required')
        self._initialized = self._closed = self._failed = False
        self._diagnostics = dict(initialized=False)
        self._guard(policy)

    def _guard(self, policy):
        if self._closed or self._failed or self._policy() is not policy or policy._closed or policy.graph_runner._closed:
            raise RuntimeError('L4 compiler owner is retired, failed or attached to another policy')
        config, tasks, knobs, compilation, codecache, checkpoint_engine = _bindings()
        actual = (config, tasks, knobs, compilation, codecache, checkpoint_engine,
            tasks._set_triton_libdevice_path, _patch_function(config), compilation.INDUCTOR_OPTIONS, policy.graph_runner)
        if any(left is not right for left, right in zip(actual, self._anchors, strict=True)):
            raise RuntimeError('Original L4 compiler helper/options/owner binding changed')
        options = compilation.INDUCTOR_OPTIONS
        expected_options = self._manifest['actual']['original_compile_options']
        if (type(options) is not dict or set(options) != set(expected_options)
                or _wire(OPTIONS) != _wire(expected_options) or _wire(MATH) != _wire(self._manifest['actual']['math_policy'])
                or any(type(options[key]) is not bool or options[key] is not expected for key, expected in expected_options.items())
                or config.emulate_precision_casts is not False or config.eager_numerics.use_pytorch_libdevice is not False
                or policy.graph_runner.compile_sample_actions is not True
                or policy._lite_optimizations.get('compile_sample_actions') is not True):
            raise RuntimeError('Original L4 compiled sampling route or precision options changed')
        allowed = (self._proof['library']['path'],) if self._initialized else (None, self._proof['library']['path'])
        if os.environ.get('TRITON_LIBDEVICE_PATH') not in allowed or knobs.libdevice_path not in allowed:
            raise RuntimeError('Original L4 libdevice environment or Triton knob changed')
        native_math_policy()
        return config, tasks, knobs

    def manifest(self):
        return deepcopy(self._manifest)

    def audit(self, policy):
        self._guard(policy)
        proof = load_qualification(self._evidence)
        if _wire(proof) != _wire(self._proof) or _wire(_runtime(policy, proof)) != _wire(self._manifest['actual']):
            raise RuntimeError('L4 qualification or actual compiler/library/source/runtime changed')
        return self.manifest()

    def ensure_initialized(self, policy):
        config, tasks, knobs = self._guard(policy)
        if not policy._lock.locked():
            raise RuntimeError('L4 libdevice initialization requires the original serialized request lock')
        if self._initialized:
            return
        before = native_math_policy(); started = time.perf_counter_ns()
        try:
            with config.patch(self._manifest['actual']['original_compile_options']):
                if config.emulate_precision_casts is not True or config.eager_numerics.use_pytorch_libdevice is not False:
                    raise RuntimeError('Original Inductor precision context was not entered')
                tasks._set_triton_libdevice_path()
            after = native_math_policy()
            if (before != after or config.emulate_precision_casts is not False
                    or config.eager_numerics.use_pytorch_libdevice is not False
                    or os.environ.get('TRITON_LIBDEVICE_PATH') != self._proof['library']['path']
                    or knobs.libdevice_path != self._proof['library']['path']):
                raise RuntimeError('Original helper failed to establish qualified libdevice under restored native math')
        except BaseException:
            self._failed = True
            raise
        self._initialized = True
        self._diagnostics = dict(initialized=True, initialization_ms=(time.perf_counter_ns() - started) / 1e6,
            native_math_before=before, native_math_after=after, library_sha256=LIBDEVICE_SHA256,
            initialization_cost_inside_request=True)

    def diagnostics(self):
        return deepcopy(self._diagnostics)

    def close(self):
        self._closed = True
        self._policy = lambda: None
        self._anchors = ()
