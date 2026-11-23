"""Cold-only original Inductor libdevice initialization for the private B32 API.

The original compiler options request eager precision. Torch's asynchronous
compiler sets CUDA's libdevice after an earlier FX cache lookup. This owner
performs that same official initialization before the first lookup, inside
those same options. It never changes cache keys, guards, compiled results,
model arithmetic options, or the graph/compilation implementation.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import weakref

IMPLEMENTATION = 'pi05-b32-original-inductor-libdevice-before-cache-v1'
REPORT_SHA256 = '762768b638a5a90d5ca02a833d7e73977d880e2f6ff01c475fbec5fa661d84ba'
SOURCE = '9be109d82c27dffd5dec1f7f5eca4d717e39f92a84185cf299d163ef634b12f4'
CONFIG = 'fe43bdcbfbe1e2cde0322caeb02a4a9301e42df55a03f5071955471dc7ccbed9'
FORMAL = 'd06dd65b6f27f8a8623b5621a4973e360ab5429a064816072704b1e60ed8a7b0'
LIBDEVICE_SHA256 = '91334d6e12748f6cb5bbf0a1cd965a56bcd93dc4f496d2e5c5f8c6e523094356'
BUNDLED_SHA256 = '5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee'
HELPER_SHA256 = '824244b91623128e299df93db614e1f046b9ddee2f0ca8a84c6e7707bbacf5c2'
COMPILATION_SHA256 = '4066f24920bd5ce7062c517a2a74e7abba8fe43a2b37adf1d63f84439e38c01c'
FIRST_ARRAYS_SHA256 = 'a3c35c45bc0824d3609b4ff6e1043a4e59d11e07daecbd709025cdac9f85011a'
OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
REQUIRED_RUNTIME = {'torch': '2.13.0+cu130', 'git': 'cf30153c4c131c8164ee7798e5022d810682e2cb', 'cuda': '13.0'}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require a real qualified file: ' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def load_qualification(evidence):
    """Bind one actually completed cold diagnostic, without inheriting quality."""
    if (type(evidence) is not dict or set(evidence) != {'backend_report', 'backend_report_sha256'}
            or evidence['backend_report_sha256'] != REPORT_SHA256):
        raise ValueError('Original libdevice initialization requires its exact passed G4 diagnostic')
    path = Path(evidence['backend_report'])
    if _sha(path) != REPORT_SHA256:
        raise ValueError('Passed libdevice diagnostic changed')
    report = _read(path)
    checks = {'actions_exact', 'noise_exact', 'owned_cpu_output', 'raw_unchanged', 'valid_noise'}
    if (report.get('status') != 'complete' or report.get('actual_cache_hit') is not True
            or report.get('final_bindings_verified') is not True
            or report.get('source_inventory_sha256') != SOURCE
            or report.get('config_sha256') != CONFIG or report.get('formal_content_sha256') != FORMAL
            or report.get('runtime') != REQUIRED_RUNTIME or set(report.get('checks', {})) != checks
            or any(value is not True for value in report['checks'].values())
            or report.get('cache_reset') is not False or report.get('prewarm') is not False
            or report.get('quality_passed') is not False):
        raise ValueError('Cold diagnostic is incomplete or outside its qualified scope')
    root = path.parent
    first, cleanup, initialization = _read(root/'FIRST_REQUEST.json'), _read(root/'CLEANUP.json'), _read(root/'LIBDEVICE_INITIALIZATION.json')
    if (cleanup != {'passed': True} or first.get('cold') is not True
            or first.get('checks') != report['checks'] or first.get('elapsed_ms') != report['first_request_ms']
            or first.get('arrays_sha256') != FIRST_ARRAYS_SHA256
            or _sha(root/'FIRST_REQUEST.npz') != FIRST_ARRAYS_SHA256
            or _wire(initialization) != _wire(report['initialization'])
            or initialization['original_compile_options'] != OPTIONS
            or initialization['helper_source_sha256'] != HELPER_SHA256
            or initialization['helper_name'] != 'torch._inductor.runtime.compile_tasks._set_triton_libdevice_path'):
        raise ValueError('Original helper, actual arrays or normal cleanup evidence changed')
    for phase, precision, library in (('before', False, BUNDLED_SHA256), ('inside_before', True, BUNDLED_SHA256),
            ('inside_after', True, LIBDEVICE_SHA256), ('after', False, LIBDEVICE_SHA256)):
        row = initialization[phase]
        if (row['emulate_precision_casts'] is not precision or row['use_pytorch_libdevice'] is not False
                or row['key'] != 'libdevice-' + library or len(row['extern_libs']) != 1
                or row['extern_libs'][0]['name'] != 'libdevice' or row['extern_libs'][0]['sha256'] != library):
            raise ValueError('Official helper precision context or external library proof differs')
    calls = _read(root/'CACHE_CALLS.json')
    if _wire(calls) != _wire(report['cache_calls']) or not any(row.get('graph_returned') is True
            and row['cache_info'].get('cache_state') == 'hit' for row in calls):
        raise ValueError('Original cache did not actually return its guarded graph')
    for row in calls:
        relative = Path(row['evidence_file'])
        if relative.is_absolute() or '..' in relative.parts or _sha(root/relative) != row['evidence_sha256']:
            raise ValueError('Original cache observation bytes changed')
    return dict(schema=1, backend_report=str(path.resolve()), backend_report_sha256=REPORT_SHA256,
        source_inventory_sha256=SOURCE, config_sha256=CONFIG, formal_content_sha256=FORMAL,
        hardware=deepcopy(report['hardware']), runtime=deepcopy(report['runtime']),
        library=deepcopy(initialization['after']['extern_libs'][0]),
        helper_source_sha256=HELPER_SHA256, original_compile_options=deepcopy(OPTIONS),
        artifacts_sha256={name:_sha(root/name)for name in ('FIRST_REQUEST.json','FIRST_REQUEST.npz',
            'CLEANUP.json','LIBDEVICE_INITIALIZATION.json','CACHE_CALLS.json')},
        qualification_scope='one frozen real initial32 cold cache hit and exact actions/noise; no complete320 quality or formal timing inheritance')


def _bindings():
    from torch._inductor import config
    from torch._inductor.runtime import compile_tasks
    from triton import knobs
    from . import compilation
    return config, compile_tasks, knobs.nvidia, compilation


def _patch_function(config):
    method=config.patch
    return getattr(method,'__func__',method)


def _runtime(policy, proof):
    import torch
    import triton
    from torch._inductor import codecache
    from torch.utils import _triton
    from .prefix_fp8 import _hardware
    config, tasks, knobs, compilation = _bindings()
    runtime = dict(torch=str(torch.__version__), git=torch.version.git_version, cuda=torch.version.cuda)
    hardware = _hardware(policy.device)
    actual_hardware = dict(gpu_uuid=hardware['uuid'],gpu_name=hardware['name'],
        driver=hardware['driver_version'],hostname=hardware['hostname'])
    library = proof['library']
    if (runtime != REQUIRED_RUNTIME or actual_hardware != proof['hardware']
            or triton.__version__ != '3.7.1' or _sha(library['path']) != LIBDEVICE_SHA256
            or _sha(tasks.__file__) != HELPER_SHA256
            or _sha(compilation.__file__) != COMPILATION_SHA256
            or _wire(compilation.INDUCTOR_OPTIONS) != _wire(OPTIONS)
            or config.emulate_precision_casts is not False
            or config.eager_numerics.use_pytorch_libdevice is not False):
        raise ValueError('Original compiler/runtime/card/precision options or libdevice differs')
    binary_root = Path(torch.__file__).resolve().parent
    return dict(runtime=runtime,triton=triton.__version__,hardware=actual_hardware,
        library=deepcopy(library),original_compile_options=deepcopy(OPTIONS),
        sources_sha256={name:_sha(path)for name,path in {
            'batch_compile_libdevice.py':__file__, 'compilation.py':compilation.__file__,
            'compile_tasks.py':tasks.__file__, 'codecache.py':codecache.__file__, '_triton.py':_triton.__file__}.items()},
        torch_binary_sha256={name:_sha(binary_root/name)for name in ('lib/libtorch_cuda.so','lib/libtorch_cpu.so')})


class BatchCompileLibdeviceOwner:
    """Private engine owner; immutable qualification, first-request work only."""
    __slots__=('_policy','_evidence','_proof','_manifest','_anchors','_initialized','_closed','_diagnostics')

    def __init__(self, policy, evidence):
        if (policy._closed or policy.device.type != 'cuda'
                or policy.graph_runner.compile_sample_actions is not True
                or policy._lite_optimizations.get('compile_sample_actions') is not True):
            raise ValueError('Libdevice initialization requires the actual private live CUDA policy')
        proof = load_qualification(evidence)
        actual = _runtime(policy,proof)
        config,tasks,knobs,compilation = _bindings()
        self._policy=weakref.ref(policy);self._evidence=deepcopy(evidence);self._proof=proof
        self._anchors=(config,tasks,knobs,compilation,tasks._set_triton_libdevice_path,_patch_function(config),compilation.INDUCTOR_OPTIONS,policy.graph_runner)
        self._manifest=dict(schema=1,implementation=IMPLEMENTATION,qualification=deepcopy(proof),actual=actual,
            initialization='once in the timed first raw request after current preprocessing/noise and before original graph runner',
            precision='official helper under original Inductor options; global precision flags restored by the original config context',
            cache='original lookup/key/full guards/results; no deletion, reset, prewarm or replacement',
            graph_and_compilation_sources_unchanged=True,request_cache=False,
            validation='complete cold seal and explicit file/runtime audit; cheap current helper/options/env/knob guards per request',
            quality_scope='cold initial32 diagnostic only; new complete closed-loop320 still required')
        self._initialized=False;self._closed=False;self._diagnostics=dict(initialized=False)
        self._guard(policy)

    def _guard(self,policy):
        if self._closed or self._policy() is not policy or policy._closed or policy.graph_runner._closed:
            raise RuntimeError('Libdevice cold owner is retired or attached to another policy')
        config,tasks,knobs,compilation = _bindings()
        actual=(config,tasks,knobs,compilation,tasks._set_triton_libdevice_path,_patch_function(config),compilation.INDUCTOR_OPTIONS,policy.graph_runner)
        if any(left is not right for left,right in zip(actual,self._anchors,strict=True)):
            raise RuntimeError('Original compiler helper/options binding changed')
        options=compilation.INDUCTOR_OPTIONS
        expected=self._manifest['actual']['original_compile_options']
        if (type(options)is not dict or set(options)!=set(expected) or _wire(OPTIONS)!=_wire(expected)
                or any(type(options[key])is not bool or options[key]is not value for key,value in expected.items())):
            raise RuntimeError('Original Inductor precision options changed')
        allowed=(self._proof['library']['path'],) if self._initialized else (None,self._proof['library']['path'])
        if os.environ.get('TRITON_LIBDEVICE_PATH')not in allowed or knobs.libdevice_path not in allowed:
            raise RuntimeError('Original libdevice environment or Triton knob changed')
        if (policy.graph_runner.compile_sample_actions is not True
                or policy._lite_optimizations.get('compile_sample_actions') is not True
                or config.emulate_precision_casts is not False
                or config.eager_numerics.use_pytorch_libdevice is not False):
            raise RuntimeError('Original compiled sampling route or outside precision context changed')
        return config,tasks,knobs

    def manifest(self):
        return deepcopy(self._manifest)

    def audit(self,policy):
        self._guard(policy)
        proof=load_qualification(self._evidence)
        if _wire(proof)!=_wire(self._proof) or _wire(_runtime(policy,proof))!=_wire(self._manifest['actual']):
            raise RuntimeError('Cold compiler/library qualification or actual source/binary changed')
        return self.manifest()

    def ensure_initialized(self,policy):
        config,tasks,knobs=self._guard(policy)
        if not policy._lock.locked():
            raise RuntimeError('Libdevice initialization requires the original serialized request lock')
        if self._initialized:return
        before=dict(emulate_precision_casts=bool(config.emulate_precision_casts),
            use_pytorch_libdevice=bool(config.eager_numerics.use_pytorch_libdevice))
        started=time.perf_counter_ns()
        with config.patch(self._manifest['actual']['original_compile_options']):
            tasks._set_triton_libdevice_path()
        after=dict(emulate_precision_casts=bool(config.emulate_precision_casts),
            use_pytorch_libdevice=bool(config.eager_numerics.use_pytorch_libdevice))
        if before!=after or os.environ.get('TRITON_LIBDEVICE_PATH')!=self._proof['library']['path'] or knobs.libdevice_path!=self._proof['library']['path']:
            raise RuntimeError('Official original helper did not establish the qualified library under restored precision options')
        self._initialized=True
        self._diagnostics=dict(initialized=True,initialization_ms=(time.perf_counter_ns()-started)/1e6,
            global_precision_before=before,global_precision_after=after,library_sha256=LIBDEVICE_SHA256,
            initialization_cost_inside_request=True)

    def diagnostics(self):
        return deepcopy(self._diagnostics)

    def close(self):
        self._closed=True;self._policy=lambda:None
