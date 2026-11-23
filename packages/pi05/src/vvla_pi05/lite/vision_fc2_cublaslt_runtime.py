"""Pinned, biased BF16 SigLIP fc2 ABI; no profiling dependency or autotuning.

Original NK weights, one explicit BF16 bias epilogue, FP32 accumulation. This
algorithm intentionally need not equal native F.linear bit for bit. Its separate
real-input accuracy screen is provenance, not a full-policy quality certificate.
"""
import ctypes as C
from copy import deepcopy
import itertools
from pathlib import Path
import re
import weakref

import torch

from vvla_pi05.lite import cublaslt_runtime as ffi
from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

M, K, N = 512, 4304, 1152
INPUT_SHAPE = (2, 256, K)
VERSION = 'vision-fc2-original-nk-biased-lt-v1'
BIAS_ENUMS = dict(CUBLASLT_ORDER_COL=0, CUBLASLT_MATMUL_DESC_EPILOGUE=7,
                  CUBLASLT_EPILOGUE_BIAS=4, CUBLASLT_MATMUL_DESC_BIAS_POINTER=8,
                  CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE=26)
ALGORITHM = dict(id=21, tile=15, stages=18, split_k=1, reduction_scheme=0,
                 swizzle=0, custom_option=0, numerical_impl_flags=131586)
REAL_SCREEN = dict(
    path='benchmarks/pi05-lite-bf16-20261105/results/g4/profiling/vision-fc2-real-accuracy-v2/SUMMARY.json',
    sha256='2b827bb13f27f85046e20cca0f25970a6856f107f5a010f5a31d3679da6bfc6a',
    fixtures=5, sites_per_fixture=27, eligible_sites=135,
    scope='pretrained inputs from action-byte-equivalent native instrumentation; candidate outputs not propagated',
    oracle='full dense FP32 TF32-off/highest, fixed FP64 tiles; per-input non-inferiority and improvement',
    policy_quality_passed=False, native_bitwise=False)


def specification():
    return dict(m=M, k=K, n=N, layout='original_nk', workspace_bytes=0,
                algorithm=dict(ALGORITHM))


def verify_bias_headers(include):
    evidence = ffi.verify_headers(include)
    text = (Path(include) / 'cublasLt.h').read_text()
    clean = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    for name, value in BIAS_ENUMS.items():
        if not re.search(r'\b' + name + r'\s*=\s*' + str(value) + r'\s*[,}]', clean):
            raise RuntimeError('Unverified installed CUDA bias enum: ' + name)
    for phrase in ('Bias vector length must match matrix D rows count.',
                   'Bias vector is broadcasted to all columns and added before applying final'):
        if phrase not in text:
            raise RuntimeError('Unverified installed bias broadcast contract')
    evidence['bias_enums'] = dict(BIAS_ENUMS)
    return evidence


class BiasedPlan(ffi.Plan):
    """Y.T = W @ X.T, all column-major descriptors; bias broadcasts over N."""
    def __init__(self, api, bias):
        if (tuple(bias.shape) != (N,) or bias.dtype != torch.bfloat16
                or not bias.is_contiguous() or bias.data_ptr() % 256):
            raise ValueError('fc2 bias requires contiguous BF16[N] at a fixed 256-byte aligned address')
        self.api, self.resources = api, []
        self.m, self.k, self.n, self.layout = M, K, N, 'original_nk'
        self.bias, self.bias_pointer = bias, bias.data_ptr()
        self.alpha, self.beta = C.c_float(1), C.c_float(0)
        try:
            self.handle = self.create('cublasLtCreate', 'cublasLtDestroy')
            self.desc = self.create('cublasLtMatmulDescCreate', 'cublasLtMatmulDescDestroy', 68, 0)
            for attr, value in ((2, 0), (3, 1), (4, 0)):
                api.attribute('cublasLtMatmulDescSetAttribute', self.desc, attr, C.c_int32(value))
            api.attribute('cublasLtMatmulDescSetAttribute', self.desc, 7, C.c_uint32(4))
            api.attribute('cublasLtMatmulDescSetAttribute', self.desc, 8, C.c_void_p(self.bias_pointer))
            api.attribute('cublasLtMatmulDescSetAttribute', self.desc, 26, C.c_int32(14))
            self.a, self.b, self.d = self.matrix(K, N, K), self.matrix(K, M, K), self.matrix(N, M, N)
            self.pref = self.create('cublasLtMatmulPreferenceCreate', 'cublasLtMatmulPreferenceDestroy')
            api.attribute('cublasLtMatmulPreferenceSetAttribute', self.pref, 1, C.c_uint64(ffi.WORKSPACE_BYTES))
            api.attribute('cublasLtMatmulPreferenceSetAttribute', self.pref, 3, C.c_uint32(7))
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Partial biased fc2 descriptor cleanup: ' + str(cleanup))
            raise

    def matrix(self, rows, cols, ld):
        result = self.create('cublasLtMatrixLayoutCreate', 'cublasLtMatrixLayoutDestroy', 14, rows, cols, ld)
        self.api.attribute('cublasLtMatrixLayoutSetAttribute', result, 1, C.c_int32(0))
        return result

    def launch(self, x, weight, output, algo, workspace, stream):
        self.api.check('cublasLtMatmul', self.handle, self.desc, C.byref(self.alpha),
                       C.c_void_p(weight.data_ptr()), self.a, C.c_void_p(x.data_ptr()), self.b,
                       C.byref(self.beta), C.c_void_p(output.data_ptr()), self.d,
                       C.c_void_p(output.data_ptr()), self.d, C.byref(algo),
                       C.c_void_p(workspace.data_ptr()), workspace.numel(), C.c_void_p(stream.cuda_stream))


_PLANS = {}
_IDS = itertools.count(1)


class BoundPlan:
    def __init__(self, api, weight, bias):
        self.key, self.closed, self.owner = next(_IDS), False, None
        self.plan = self.workspace = None
        self.weight, self.bias, self.device = weight, bias, weight.device
        self.weight_pointer, self.bias_pointer = weight.data_ptr(), bias.data_ptr()
        self.spec = specification()
        self._installed_spec = deepcopy(self.spec)
        try:
            self.plan = BiasedPlan(api, bias)
            self.rank, self.result, info = ffi.select_algorithm(self.plan, self.spec)
            self.info = {key: value for key, value in info.items() if key != 'opaque_algo_hex'}
            if self.info['numerical_impl_flags'] & 0xff0000 != 0x020000:
                raise RuntimeError('Pinned fc2 algorithm is not exclusively BF16 input math')
            self.workspace = torch.empty(256, dtype=torch.uint8, device=self.device)
            self._workspace_identity = tensor_identity(self.workspace)
            _PLANS[self.key] = self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Partial biased fc2 plan cleanup: ' + str(cleanup))
            raise

    def bind_owner(self, owner):
        if self.closed or self.owner is not None:
            raise RuntimeError('fc2 plan is already bound or closed')
        self.owner = weakref.ref(owner)

    def validate(self):
        if (self.closed or _PLANS.get(self.key) is not self or self.spec != self._installed_spec
                or tensor_identity(self.workspace) != self._workspace_identity
                or self.weight.data_ptr() != self.weight_pointer or self.bias.data_ptr() != self.bias_pointer):
            raise RuntimeError('fc2 immutable plan/source allocation/workspace changed; retire and reinstall')

    def execute(self, x, weight, bias):
        owner = None if self.owner is None else self.owner()
        if self.closed or owner is None:
            raise RuntimeError('fc2 plan is unbound or retired')
        owner.assert_execution_owner()
        if (tuple(x.shape) != INPUT_SHAPE or tuple(weight.shape) != (N, K) or tuple(bias.shape) != (N,)
                or any(t.dtype != torch.bfloat16 or t.device != self.device or not t.is_contiguous()
                       for t in (x, weight, bias))
                or weight.data_ptr() != self.weight_pointer or bias.data_ptr() != self.bias_pointer):
            raise ValueError('fc2 execution requires original fixed-address BF16 weight/bias and [2,256,4304]')
        if (x.data_ptr() % 16 or weight.data_ptr() % 16 or bias.data_ptr() % 256
                or self.workspace.data_ptr() % 256):
            raise ValueError('fc2 execution requires audited aligned addresses')
        output = x.new_empty((2, 256, N))
        self.plan.launch(x, weight, output, self.result.algo, self.workspace, torch.cuda.current_stream(x.device))
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
            # Descriptors must die before the strong reference to their bias.
            self.plan = self.workspace = self.owner = self.weight = self.bias = None


@torch.library.custom_op('vvla_pi05_lite::vision_fc2_cublaslt', mutates_args=())
def vision_fc2_cublaslt(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       plan_id: int) -> torch.Tensor:
    plan = _PLANS.get(plan_id)
    if plan is None:
        raise RuntimeError('Biased vision fc2 compiled plan was retired')
    return plan.execute(x, weight, bias)


@vision_fc2_cublaslt.register_fake
def _vision_fc2_fake(x, weight, bias, plan_id):
    return x.new_empty((*x.shape[:-1], N))


class VisionFC2LtRuntime:
    def __init__(self, device, sources):
        self.plans = []
        self.device = torch.device(device)
        if (self.device.type != 'cuda' or not str(torch.__version__).startswith('2.13.')
                or torch.version.cuda is None or not torch.version.cuda.startswith('13.')):
            raise ValueError('Vision fc2 cuBLASLt is pinned to Torch2.13 CUDA13')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Create vision fc2 plans before CUDA graph capture')
        self.device = torch.device('cuda', torch.cuda.current_device() if self.device.index is None else self.device.index)
        name = torch.cuda.get_device_name(self.device)
        capability = torch.cuda.get_device_capability(self.device)
        if capability != (12, 0):
            raise ValueError('Vision fc2 algorithm profile requires SM120')
        self._validate_math()
        with torch.cuda.device(self.device):
            torch.empty((1, 1), device=self.device).mm(torch.empty((1, 1), device=self.device))
            library, include = ffi.installed_library()
            evidence = verify_bias_headers(include)
            digest = ffi.file_sha(library)
            self.api = ffi.LtAPI(C.CDLL(str(library)))
            version = int(self.api.lib.cublasLtGetVersion())
            if version // 10000 != 13:
                raise RuntimeError('Vision fc2 cuBLASLt major is outside the checked ABI')
            self.evidence = dict(library_sha256=digest, runtime_version=version,
                                 header_sha256=evidence['header_sha256'], bias_enums=evidence['bias_enums'],
                                 device_name=name, capability=list(capability),quality_qualified=False,
                                 admission='checked ABI and fresh matching heuristic')
            try:
                for weight, bias in sources:
                    if (tuple(weight.shape) != (N, K) or tuple(bias.shape) != (N,)
                            or any(t.dtype != torch.bfloat16 or t.device != self.device or not t.is_contiguous()
                                   for t in (weight, bias))):
                        raise ValueError('fc2 loader requires original contiguous BF16 weight and bias')
                    self.plans.append(BoundPlan(self.api, weight, bias))
                self._installed_plans = tuple(self.plans)
            except BaseException as error:
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note('Partial vision fc2 runtime cleanup: ' + str(cleanup))
                raise

    @staticmethod
    def _validate_math():
        if torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
            raise ValueError('Pinned vision fc2 requires TF32 off and library-default BF16 reduction enabled')

    def bind_owner(self, owner):
        for plan in self.plans:
            plan.bind_owner(owner)

    def validate(self):
        self._validate_math()
        if (len(self.plans) != len(self._installed_plans)
                or any(a is not b for a, b in zip(self.plans, self._installed_plans, strict=True))):
            raise RuntimeError('Vision fc2 immutable plan set changed')
        for plan in self.plans:
            plan.validate()

    def manifest(self):
        return dict(version=VERSION, evidence=deepcopy(self.evidence), real_input_screen=deepcopy(REAL_SCREEN),
                    math=dict(input='BF16', weight='BF16', bias='BF16', output='BF16', main_accumulator='FP32',
                              tf32=False, quantized=False, epilogue='BIAS before BF16 output',
                              reduction_policy='library-default; selected splitK1/reduction0'),
                    plans=[dict(spec=deepcopy(p.spec), selected_rank=p.rank, algorithm=dict(p.info),
                                workspace_bytes=p.workspace.numel()) for p in self.plans])

    def close(self):
        failures = []
        for plan in reversed(getattr(self, '_installed_plans', tuple(self.plans))):
            try:
                plan.close()
            except BaseException as error:
                failures.append(str(error))
        self.plans.clear()
        if failures:
            raise RuntimeError('Vision fc2 runtime cleanup: ' + '; '.join(failures))
