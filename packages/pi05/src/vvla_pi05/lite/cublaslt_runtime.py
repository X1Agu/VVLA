"""Pinned cuBLASLt ABI for opt-in expert BF16 inference.

The descriptor/ABI implementation is promoted from the separately audited
probe_cublaslt_algorithms.py. No profiling/harness dependency is imported.
All library validation, descriptor creation and heuristic queries happen at load.
"""
import ctypes as C
import hashlib
import itertools
from pathlib import Path
import re
import weakref

import torch

from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


WORKSPACE_BYTES = 256 * 1024**2
TOP = 16
# Verified against installed NVIDIA cuBLAS 13.1.1 headers on L4, read-only.
HEADER_REFERENCE_SHA256 = {
    'cublasLt.h': '8be43fbf48625b98b8562d7df394e6af0d7ee1c05beeb5333577d5e33182fa3e',
    'cublas_api.h': 'f6b443f43358de591584d38810d60fcf7c1080677a6ba1605a99e8b58b349890',
    'library_types.h': '7ccccd9cf42214386170704865517bbbf83702d6bbb5e0c0cba3fc3a48cbd29d',
}
FLASHRT_REFERENCE = dict(path='FlashRT/csrc/gemm/gemm_runner.cu',
    sha256='906059e6b24d5038439d6c62ecaa5e35e6e6a268f9e1e688494692c0f59d1574',
    behavior='row-major BF16_NN, compute32F, 256MiB workspace, configurable top-N heuristic autotune')
ENUMS = dict(CUDA_R_16BF=14, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68,
             CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLASLT_ORDER_ROW=1,
             CUBLASLT_MATRIX_LAYOUT_ORDER=1, CUBLASLT_MATMUL_DESC_POINTER_MODE=2,
             CUBLASLT_MATMUL_DESC_TRANSA=3, CUBLASLT_MATMUL_DESC_TRANSB=4,
             CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=1,
             CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK=3,
             CUBLASLT_ALGO_CONFIG_ID=0, CUBLASLT_ALGO_CONFIG_TILE_ID=1,
             CUBLASLT_ALGO_CONFIG_SPLITK_NUM=2, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME=3,
             CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING=4, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION=5,
             CUBLASLT_ALGO_CONFIG_STAGES_ID=6, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS=15)


class Algo(C.Structure):
    _fields_ = [('data', C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [('algo', Algo), ('workspaceSize', C.c_size_t), ('state', C.c_int),
                ('wavesCount', C.c_float), ('reserved', C.c_int * 4)]


def abi_layout():
    if (C.sizeof(C.c_void_p), C.sizeof(Algo), C.sizeof(Heuristic), C.alignment(Heuristic)) != (8, 64, 96, 8):
        raise RuntimeError('Only verified 64-bit NVIDIA cuBLASLt ABI is supported')
    offsets = {key: getattr(Heuristic, key).offset for key, _ in Heuristic._fields_}
    if offsets != dict(algo=0, workspaceSize=64, state=72, wavesCount=76, reserved=80):
        raise RuntimeError('Heuristic ABI offsets differ')
    return dict(algo_bytes=64, heuristic_bytes=96, offsets=offsets)


def verify_headers(include):
    """Fail before FFI calls if installed declarations/enums changed."""
    texts = {name: (Path(include) / name).read_text() for name in HEADER_REFERENCE_SHA256}
    joined = '\n'.join(texts.values())
    clean = re.sub(r'/\*.*?\*/', '', joined, flags=re.S)
    for name, value in ENUMS.items():
        if not re.search(r'\b' + name + r'\s*=\s*' + str(value) + r'\s*[,}]', clean):
            raise RuntimeError('Unverified installed CUDA enum: ' + name)
    compact = re.sub(r'\s+', '', clean)
    structs = ('typedefstruct{uint64_tdata[8];}cublasLtMatmulAlgo_t;',
               'typedefstruct{cublasLtMatmulAlgo_talgo;size_tworkspaceSize;cublasStatus_tstate;floatwavesCount;intreserved[4];}cublasLtMatmulHeuristicResult_t;')
    if any(value not in compact for value in structs):
        raise RuntimeError('Unverified installed cuBLASLt structure declarations')
    if '#defineCUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F(0x02ull<<8)' not in compact:
        raise RuntimeError('Unverified accumulator capability flag')
    version = [int(re.search(r'#define\s+CUBLAS_VER_' + key + r'\s+(\d+)', joined).group(1))
               for key in ('MAJOR', 'MINOR', 'PATCH')]
    if version[0] != 13:
        raise RuntimeError('This ctypes binding is audited for CUDA/cuBLAS major13 only')
    return dict(include=str(Path(include).resolve()), header_sha256={name: file_sha(Path(include) / name) for name in texts},
                header_version=version[0] * 10000 + version[1] * 100 + version[2], abi=abi_layout())


def installed_library():
    # Use the exact library already mapped by Torch, avoiding a different system
    # CUDA installation. Call only after lock/idle check and Torch CUDA init.
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/libcublasLt.so.' in line and line.split()[-1].startswith('/')}
    resolved = {Path(path).resolve() for path in paths}
    if len(resolved) != 1:
        raise RuntimeError('Expected one Torch-loaded libcublasLt mapping, found: ' + repr(sorted(map(str, resolved))))
    library = resolved.pop()
    include = library.parent.parent / 'include'
    return library, include


class LtError(RuntimeError):
    def __init__(self, operation, status, name):
        super().__init__(f'{operation}: {name} ({status})')
        self.operation, self.status, self.status_name = operation, int(status), name


class AlgorithmUnavailable(RuntimeError):
    """An explicit semantic tactic cannot satisfy the current device contract."""


class LtAPI:
    def __init__(self, library):
        self.lib = library
        p, i, z = C.c_void_p, C.c_int, C.c_size_t
        pp = C.POINTER(p)
        specs = {
            'cublasLtCreate': [pp], 'cublasLtDestroy': [p],
            'cublasLtMatmulDescCreate': [pp, i, i], 'cublasLtMatmulDescDestroy': [p],
            'cublasLtMatrixLayoutCreate': [pp, i, C.c_uint64, C.c_uint64, C.c_int64],
            'cublasLtMatrixLayoutDestroy': [p], 'cublasLtMatmulPreferenceCreate': [pp],
            'cublasLtMatmulPreferenceDestroy': [p],
            'cublasLtMatmulDescSetAttribute': [p, i, p, z],
            'cublasLtMatrixLayoutSetAttribute': [p, i, p, z],
            'cublasLtMatmulPreferenceSetAttribute': [p, i, p, z],
            'cublasLtMatmulAlgoGetHeuristic': [p, p, p, p, p, p, p, i, C.POINTER(Heuristic), C.POINTER(i)],
            'cublasLtMatmulAlgoConfigGetAttribute': [C.POINTER(Algo), i, p, z, C.POINTER(z)],
            'cublasLtMatmulAlgoCapGetAttribute': [C.POINTER(Algo), i, p, z, C.POINTER(z)],
            'cublasLtMatmulAlgoInit': [p, i, i, i, i, i, i, i, C.POINTER(Algo)],
            'cublasLtMatmulAlgoConfigSetAttribute': [C.POINTER(Algo), i, p, z],
            'cublasLtMatmulAlgoCheck': [p, p, p, p, p, p, C.POINTER(Algo), C.POINTER(Heuristic)],
            'cublasLtMatmul': [p, p, p, p, p, p, p, p, p, p, p, p, C.POINTER(Algo), p, z, p],
        }
        for name, args in specs.items():
            fn = getattr(library, name)
            fn.argtypes, fn.restype = args, i
        library.cublasLtGetVersion.argtypes, library.cublasLtGetVersion.restype = [], z
        library.cublasLtGetStatusName.argtypes, library.cublasLtGetStatusName.restype = [i], C.c_char_p

    def check(self, operation, *args):
        status = int(getattr(self.lib, operation)(*args))
        if status:
            name = self.lib.cublasLtGetStatusName(status)
            raise LtError(operation, status, name.decode() if name else 'unknown')

    def attribute(self, operation, handle, attr, value):
        self.check(operation, handle, attr, C.byref(value), C.sizeof(value))

    def algorithm_info(self, algo):
        values = {}
        for attr, name in enumerate(('id', 'tile', 'split_k', 'reduction_scheme', 'swizzle', 'custom_option', 'stages')):
            value, written = C.c_int32(), C.c_size_t()
            self.check('cublasLtMatmulAlgoConfigGetAttribute', C.byref(algo), attr,
                       C.byref(value), C.sizeof(value), C.byref(written))
            if written.value != 4:
                raise RuntimeError('Algorithm config attribute ABI size mismatch')
            values[name] = value.value
        flags, written = C.c_uint64(), C.c_size_t()
        self.check('cublasLtMatmulAlgoCapGetAttribute', C.byref(algo), 15,
                   C.byref(flags), C.sizeof(flags), C.byref(written))
        if written.value != 8:
            raise RuntimeError('Numerical flags ABI size mismatch')
        values.update(numerical_impl_flags=flags.value, opaque_algo_hex=bytes(algo).hex())
        return values


class Plan:
    """Owns descriptors/handle only; caller owns Torch buffers and CUDA stream."""
    def __init__(self, api, m, k, n, layout, reduction_policy):
        if layout not in ('original_nk', 'contiguous_kn') or reduction_policy not in ('fp32', 'library-default'):
            raise ValueError('Unsupported layout or reduction policy')
        self.api, self.resources = api, []
        self.m, self.k, self.n, self.layout = m, k, n, layout
        self.alpha, self.beta = C.c_float(1), C.c_float(0)
        try:
            self.handle = self.create('cublasLtCreate', 'cublasLtDestroy')
            self.desc = self.create('cublasLtMatmulDescCreate', 'cublasLtMatmulDescDestroy', 68, 0)
            for attr, value in ((2, 0), (3, 0), (4, 1 if layout == 'original_nk' else 0)):
                api.attribute('cublasLtMatmulDescSetAttribute', self.desc, attr, C.c_int32(value))
            self.a = self.matrix(m, k, k)
            self.b = self.matrix(n, k, k) if layout == 'original_nk' else self.matrix(k, n, n)
            self.d = self.matrix(m, n, n)
            self.pref = self.create('cublasLtMatmulPreferenceCreate', 'cublasLtMatmulPreferenceDestroy')
            api.attribute('cublasLtMatmulPreferenceSetAttribute', self.pref, 1, C.c_uint64(WORKSPACE_BYTES))
            api.attribute('cublasLtMatmulPreferenceSetAttribute', self.pref, 3,
                          C.c_uint32(2 if reduction_policy == 'fp32' else 7))
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Partial plan cleanup: ' + str(cleanup))
            raise

    def create(self, create, destroy, *args):
        handle = C.c_void_p()
        self.api.check(create, C.byref(handle), *args)
        if not handle.value:
            raise RuntimeError(create + ' returned a null descriptor')
        self.resources.append((destroy, handle))
        return handle

    def matrix(self, rows, cols, ld):
        result = self.create('cublasLtMatrixLayoutCreate', 'cublasLtMatrixLayoutDestroy', 14, rows, cols, ld)
        self.api.attribute('cublasLtMatrixLayoutSetAttribute', result, 1, C.c_int32(1))
        return result

    def heuristics(self):
        values, count = (Heuristic * TOP)(), C.c_int()
        self.api.check('cublasLtMatmulAlgoGetHeuristic', self.handle, self.desc, self.a, self.b,
                       self.d, self.d, self.pref, TOP, values, C.byref(count))
        if not 0 <= count.value <= TOP:
            raise RuntimeError('Invalid returned heuristic count')
        return [values[index] for index in range(count.value)]

    def launch(self, x, weight, output, algo, workspace, stream):
        self.api.check('cublasLtMatmul', self.handle, self.desc, C.byref(self.alpha),
                       C.c_void_p(x.data_ptr()), self.a, C.c_void_p(weight.data_ptr()), self.b,
                       C.byref(self.beta), C.c_void_p(output.data_ptr()), self.d,
                       C.c_void_p(output.data_ptr()), self.d, C.byref(algo),
                       C.c_void_p(workspace.data_ptr()), workspace.numel(), C.c_void_p(stream.cuda_stream))

    def close(self):
        errors = []
        while self.resources:
            operation, handle = self.resources.pop()
            try:
                self.api.check(operation, handle)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('cuBLASLt resource destruction failures: ' + '; '.join(errors))


def candidate_gate(heuristic, info, reduction_policy):
    if heuristic.state:
        return 'heuristic_state_' + str(heuristic.state)
    if heuristic.workspaceSize > WORKSPACE_BYTES:
        return 'workspace_requirement_exceeds_256MiB'
    if info['numerical_impl_flags'] & 0xff00 != 0x0200:
        return 'accumulator_not_exclusively_FP32'
    if reduction_policy == 'fp32' and info['reduction_scheme'] not in (0, 2):
        return 'split_reduction_not_FP32'
    return None


# Algorithms are semantic configs, not serialized implementation-private bytes.
LIBRARY_SHA256 = '1489aeb75d55ee55782798343ac66f57771d561cc1fe1e0491568deefa0e6515'
VERSION = 'expert-cublaslt-pinned-bf16-v1'


def _spec(k, n, layout, *, algo, tile, stages, swizzle=0, split_k=1, reduction=0, workspace=0):
    return dict(m=50, k=k, n=n, layout=layout, workspace_bytes=workspace,
                algorithm=dict(id=algo, tile=tile, stages=stages, swizzle=swizzle,
                               split_k=split_k, reduction_scheme=reduction, custom_option=0,
                               numerical_impl_flags=131586))


PROFILES = {
    'l4_exact_three': {
        'qkv': _spec(1024, 2560, 'contiguous_kn', algo=6, tile=15, stages=17, swizzle=1),
        'gate_or_up': _spec(1024, 4096, 'contiguous_kn', algo=6, tile=18, stages=15),
        'attention_out': _spec(2048, 1024, 'contiguous_kn', algo=6, tile=15, stages=17),
    },
    'g4_best_three': {
        'qkv': _spec(1024, 2560, 'original_nk', algo=21, tile=15, stages=12),
        'gate_up_packed': _spec(1024, 8192, 'contiguous_kn', algo=21, tile=15, stages=12),
        'attention_out': _spec(2048, 1024, 'original_nk', algo=21, tile=15, stages=25,
                               split_k=9, reduction=4, workspace=819200),
    },
}
L4_DOWN = _spec(4096, 1024, 'contiguous_kn', algo=6, tile=15, stages=17)


def profile_specs(profile, include_down=False):
    import copy
    if profile not in PROFILES or type(include_down) is not bool:
        raise ValueError('Select l4_exact_three or g4_best_three and a boolean include_down')
    if include_down and profile != 'l4_exact_three':
        raise ValueError('The independent down candidate was measured only for L4')
    specs = copy.deepcopy(PROFILES[profile])
    if include_down:
        specs['down'] = copy.deepcopy(L4_DOWN)
    return specs


def select_algorithm(plan, spec):
    """Resolve the same semantic algorithm even when heuristic rankings change.

    SM count affects the heuristic shortlist (e.g. L4 versus laptop Ada). Its
    omission is not an unsupported algorithm. Reconstruct through NVIDIA's
    public API and check the actual device/descriptors, never replay opaque
    serialized algorithm bytes or silently choose a different tactic.
    """
    matches = []
    for rank, result in enumerate(plan.heuristics()):
        if result.state:
            continue
        info = plan.api.algorithm_info(result.algo)
        if (candidate_gate(result, info, 'library-default') is None
                and {key: info[key] for key in spec['algorithm']} == spec['algorithm']
                and int(result.workspaceSize) == spec['workspace_bytes']):
            matches.append((rank, result, info))
    if len(matches) > 1:
        raise RuntimeError('Ambiguous duplicate cuBLASLt semantic algorithm')
    if matches:
        return matches[0]
    result = Heuristic()
    api, expected = plan.api, spec['algorithm']
    api.check('cublasLtMatmulAlgoInit', plan.handle, ENUMS['CUBLAS_COMPUTE_32F'],
              ENUMS['CUDA_R_32F'], *([ENUMS['CUDA_R_16BF']] * 4), expected['id'], C.byref(result.algo))
    for attribute, field in enumerate(('id', 'tile', 'split_k', 'reduction_scheme',
                                       'swizzle', 'custom_option', 'stages')):
        if field != 'id':
            api.attribute('cublasLtMatmulAlgoConfigSetAttribute', C.byref(result.algo),
                          attribute, C.c_int32(expected[field]))
    # The header guarantees AlgoCheck leaves result.algo unchanged. It fills
    # state/workspace using these fresh descriptors on the current device.
    api.check('cublasLtMatmulAlgoCheck', plan.handle, plan.desc, plan.a, plan.b,
              plan.d, plan.d, C.byref(result.algo), C.byref(result))
    info = api.algorithm_info(result.algo)
    if (candidate_gate(result, info, 'library-default') is not None
            or {key: info[key] for key in expected} != expected
            or int(result.workspaceSize) != spec['workspace_bytes']):
        raise AlgorithmUnavailable('Explicit cuBLASLt algorithm differs from the requested numerical/workspace contract')
    return None, result, info


_PLANS = {}
_IDS = itertools.count(1)


class BoundPlan:
    """One immutable descriptor set and Torch-owned scratch, shared serially."""
    def __init__(self, api, spec, device):
        import copy
        self.plan = self.workspace = self.owner = None
        self.key = next(_IDS)
        self.spec, self.device = spec, torch.device(device)
        self._installed_spec = copy.deepcopy(spec)
        self.closed = False
        try:
            self.plan = Plan(api, spec['m'], spec['k'], spec['n'], spec['layout'], 'library-default')
            rank, result, info = select_algorithm(self.plan, spec)
            # Keep the fresh API-created struct alive; never deserialize its bytes.
            self.result, self.rank = result, rank
            self.info = {key: value for key, value in info.items() if key != 'opaque_algo_hex'}
            self.workspace = torch.empty(max(256, spec['workspace_bytes']), dtype=torch.uint8, device=self.device)
            self._workspace_identity = tensor_identity(self.workspace)
            _PLANS[self.key] = self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Partial cuBLASLt bound-plan cleanup: ' + str(cleanup))
            raise

    def bind_owner(self, owner):
        if self.closed or self.owner is not None:
            raise RuntimeError('cuBLASLt plan owner is already bound or closed')
        self.owner = weakref.ref(owner)

    def validate(self):
        if (self.closed or _PLANS.get(self.key) is not self or self.spec != self._installed_spec
                or tensor_identity(self.workspace) != self._workspace_identity):
            raise RuntimeError('cuBLASLt immutable plan or private workspace changed; retire and reinstall')

    def execute(self, x, weight, workspace, n):
        owner = None if self.owner is None else self.owner()
        if self.closed or owner is None:
            raise RuntimeError('cuBLASLt plan is unbound or retired')
        owner.assert_execution_owner()  # Scalar Python state only; no topology traversal.
        spec = self.spec
        if (workspace is not self.workspace or workspace.dtype != torch.uint8
                or workspace.device != self.device or workspace.numel() < max(256, spec['workspace_bytes'])
                or not workspace.is_contiguous()):
            raise RuntimeError('cuBLASLt workspace ownership or layout changed')
        expected_weight = (spec['n'], spec['k']) if spec['layout'] == 'original_nk' else (spec['k'], spec['n'])
        if (n != spec['n'] or x.device != self.device or weight.device != self.device
                or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != expected_weight or not weight.is_contiguous()
                or x.ndim not in (2, 3) or x.shape[-1] != spec['k']
                or x.numel() != spec['m'] * spec['k'] or not x.is_contiguous()):
            raise ValueError('cuBLASLt expert execution requires its fixed B1/H50 BF16 shape and layout')
        if x.data_ptr() % 16 or weight.data_ptr() % 16 or workspace.data_ptr() % 256:
            raise ValueError('cuBLASLt candidate requires audited aligned tensor addresses')
        output = x.new_empty((*x.shape[:-1], n))
        stream = torch.cuda.current_stream(x.device)
        self.plan.launch(x, weight, output, self.result.algo, workspace, stream)
        return output

    def close(self):
        if self.closed:
            return
        _PLANS.pop(self.key, None)
        self.closed = True
        try:
            if self.plan is not None:
                self.plan.close()
        finally:
            self.workspace = self.owner = None


# Scratch is private implementation state, like a BLAS handle's workspace. Its
# initial/final bytes are never observable and no input tensor is mutated. Do
# not pass it as a mutable Tensor argument: AOT functionalization would clone it
# per call and defeat its loader-owned, serialized lifetime.
@torch.library.custom_op('vvla_pi05_lite::expert_cublaslt', mutates_args=())
def expert_cublaslt(x: torch.Tensor, weight: torch.Tensor,
                   plan_id: int, out_features: int) -> torch.Tensor:
    plan = _PLANS.get(plan_id)
    if plan is None:
        raise RuntimeError('cuBLASLt compiled plan was retired')
    return plan.execute(x, weight, plan.workspace, out_features)


@expert_cublaslt.register_fake
def _expert_cublaslt_fake(x, weight, plan_id, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


class ExpertLtRuntime:
    """Loader-only ABI/environment checks and plan allocation; no autotuning."""
    def __init__(self, device, profile, *, include_down=False):
        self.plans = {}
        self.profile, self.specs = profile, profile_specs(profile, include_down)
        self.device = torch.device(device)
        if (self.device.type != 'cuda' or not str(torch.__version__).startswith('2.13.')
                or torch.version.cuda is None or not torch.version.cuda.startswith('13.')):
            raise ValueError('Expert cuBLASLt candidate is pinned to Torch2.13 CUDA13')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Create cuBLASLt plans before native graph capture')
        self.device = torch.device('cuda', torch.cuda.current_device() if self.device.index is None else self.device.index)
        name = torch.cuda.get_device_name(self.device)
        expected = (8,9) if profile == 'l4_exact_three' else (12,0)
        if torch.cuda.get_device_capability(self.device) != expected:
            raise ValueError(f'cuBLASLt algorithm profile {profile} requires compute capability {expected}')
        if torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
            raise ValueError('Pinned candidate requires TF32 disabled and existing library-default BF16 reduction enabled')
        with torch.cuda.device(self.device):
            # Initialize Torch BLAS and locate the exact already-mapped library.
            torch.empty((1, 1), device=self.device).mm(torch.empty((1, 1), device=self.device))
            library, include = installed_library()
            evidence = verify_headers(include)
            digest = file_sha(library)
            self.api = LtAPI(C.CDLL(str(library)))
            version = int(self.api.lib.cublasLtGetVersion())
            if version // 10000 != 13:
                raise RuntimeError('cuBLASLt runtime major is outside the checked ABI')
            self.evidence = dict(library_sha256=digest, runtime_version=version,
                                 header_sha256=evidence['header_sha256'], device_name=name,
                                 quality_qualified=False, admission='checked ABI and fresh semantic algorithm validation')
            try:
                for label, spec in self.specs.items():
                    self.plans[label] = BoundPlan(self.api, spec, self.device)
            except BaseException as error:
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note('Partial cuBLASLt runtime cleanup: ' + str(cleanup))
                raise

    def bind_owner(self, owner):
        for plan in self.plans.values():
            plan.bind_owner(owner)

    def validate(self):
        if set(self.plans) != set(self.specs):
            raise RuntimeError('cuBLASLt runtime plan set changed')
        for plan in self.plans.values():
            plan.validate()

    def manifest(self):
        return dict(version=VERSION, profile=self.profile, evidence=dict(self.evidence),
                    math=dict(inputs='BF16', weights='BF16', output='BF16', main_accumulator='FP32',
                              reduction_policy='library-default', quantized=False, tf32=False),
                    plans={key: dict(spec=plan.spec, selected_rank=plan.rank, algorithm=plan.info,
                                     workspace_bytes=plan.workspace.numel()) for key, plan in self.plans.items()})

    def close(self):
        failures = []
        for plan in reversed(tuple(self.plans.values())):
            try:
                plan.close()
            except BaseException as error:
                failures.append(str(error))
        self.plans.clear()
        if failures:
            raise RuntimeError('cuBLASLt runtime cleanup: ' + '; '.join(failures))
