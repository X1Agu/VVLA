"""Explicit SM120 NVFP4 primitives backed by the pinned vLLM CUDA extension.

No model, weight owner, calibration fit, fallback, or serving engine is created.
Load the extension explicitly before packing/compilation/capture. The extension
is process-global and cannot be safely unloaded; a failed import poisons this
loader until process exit. The primitive retains modules/evidence, never tensors.

Global scales are caller-owned positive finite FP32 scalars, validated cold.
Hot calls validate metadata only. Values are real E2M1 nibbles, block scales are
E4M3 per 16 elements, and the native GEMM returns BF16. Scale padding is zeroed
on every call; this initialization is part of conversion-inclusive timing.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import threading

import torch

VERSION = 'pi05-prefix-nvfp4-native-sm120-v1'
VARIANT = 'vllm032-cutlass-nvfp4-sm120-v1'
IMPLEMENTATION = 'vllm_stable_native_nvfp4_opaque_v1'
QUANTIZE_OPERATOR = 'vvla_repro_pi05_lite::prefix_nvfp4_quantize_rows_v1'
SCALED_MM_OPERATOR = 'vvla_repro_pi05_lite::prefix_nvfp4_native_scaled_mm_v1'
SOURCE_FILES = ('prefix_nvfp4_native.py',)
SUPPORTED_CAPACITIES = (576, 640)
SUPPORTED_GEOMETRIES = ((2048, 16384), (16384, 2048))  # K, N
REQUIRED_RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0], vllm='0.32.0')
NATIVE_EXTENSION_SHA256 = 'ae9c8a293020e422e0b6b61d1024263dc476a772337f5d39d46d18baa7ce0ad9'
NATIVE_WHEEL_SHA256 = '5e2ecdc5b0baf72921c288486417ca82778456a448e4fe60b62ee43c0dfc8611'
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
_LOCK = threading.Lock()
_BACKEND = None
_LOAD_FAILURE = None


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scale_shape(rows, width):
    """Byte-view shape of the native CUTLASS 128x4 swizzled block scales."""
    return ((rows + 127) // 128 * 128, (width + 63) // 64 * 4)


def implementation_manifest():
    root = Path(__file__).resolve().parent
    return deepcopy(dict(version=VERSION, variant=VARIANT, implementation=IMPLEMENTATION,
        source_files_sha256={name: _sha(root / name) for name in SOURCE_FILES},
        required_runtime=REQUIRED_RUNTIME, native_extension_sha256=NATIVE_EXTENSION_SHA256,
        native_wheel_sha256=NATIVE_WHEEL_SHA256, supported_capacities=list(SUPPORTED_CAPACITIES),
        supported_geometries=[list(shape) for shape in SUPPORTED_GEOMETRIES],
        compile_options=COMPILE_OPTIONS, operators=dict(quantizer=QUANTIZE_OPERATOR, gemm=SCALED_MM_OPERATOR),
        native_operators=['_C::scaled_fp4_quant.out', '_C::cutlass_scaled_fp4_mm'],
        dispatch='pinned vLLM stable extension CUDA operators inside opaque custom ops',
        input_dtype='bfloat16', packed_dtype='uint8', packed_encoding='two E2M1 values per byte',
        block_scale_dtype='float8_e4m3fn', block_size=16, global_scale_dtype='float32',
        accumulator_dtype='float32', output_dtype='bfloat16', weight_layout='contiguous_NK_packed_K/2',
        scale_layout='CUTLASS swizzle128x4; paddedM=ceil(M/128)*128; paddedK=ceil(K/64)*64',
        scale_padding='zero-initialized int32 storage on every call; included in online conversion',
        global_scale='caller-owned frozen calibration; typical 2688/amax, never fitted in a hot call',
        alpha='caller-owned FP32 1/(activation_global_scale*weight_global_scale)',
        bias=None, fallback_permitted=False, installation='none; no 51-site binding or quality claim',
        backend_evidence='new same-card actual-wrapper primitive report required'))


def validate_runtime(device):
    if (device.type != 'cuda' or str(torch.__version__) != REQUIRED_RUNTIME['torch']
            or torch.version.cuda != REQUIRED_RUNTIME['cuda']
            or torch.version.git_version != REQUIRED_RUNTIME['torch_git']):
        raise RuntimeError('Prefix NVFP4 requires the explicit Torch2.13.0/cu130 runtime')
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        raise RuntimeError('Prefix NVFP4 supports SM120 only; no architecture fallback')
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Prefix NVFP4 requires explicit dtypes; autocast unsupported')


def _device_index(device):
    return torch.cuda.current_device() if device.index is None else device.index


def _real_file(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Native source must be a regular file inside the explicit root: ' + str(path))
    return path.resolve()


def _inspect_native_files(native_path):
    """Cold file evidence only; never import the requested package here.

The extension itself has a hard-coded digest. Python/metadata files are checked
against the installed RECORD and recorded as actual bytes, not represented as
an independent re-verification of the entire wheel archive.
"""
    root = Path(native_path)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('Provide an absolute real native target directory')
    root = root.resolve(strict=True)
    package = root / 'vllm'
    if package.is_symlink() or not package.is_dir():
        raise ValueError('Native target has no real vllm package directory')
    binaries = list(package.glob('_C_stable_libtorch*.so'))
    if len(binaries) != 1:
        raise ValueError('Require exactly one pinned stable vLLM extension')
    binary = _real_file(binaries[0], root)
    if _sha(binary) != NATIVE_EXTENSION_SHA256:
        raise ValueError('Native extension bytes do not match the fixed SM120 candidate')
    infos = list(root.glob('vllm-*.dist-info'))
    if len(infos) != 1 or infos[0].is_symlink() or not infos[0].is_dir():
        raise ValueError('Require one actual vLLM distribution in the native target')
    info = infos[0]
    distribution = importlib.metadata.PathDistribution(info)
    if distribution.metadata['Name'].lower() != 'vllm' or distribution.version != REQUIRED_RUNTIME['vllm']:
        raise ValueError('Native distribution is not pinned vLLM 0.32.0')
    record_path = _real_file(info / 'RECORD', root)
    records = {}
    for row in csv.reader(io.StringIO(record_path.read_text())):
        if len(row) != 3 or row[0] in records:
            raise ValueError('Malformed or duplicate native installed RECORD entry')
        records[row[0]] = row[1:]
    sources = {}
    # Verify Python sources before importing any vLLM code. Do not import all of
    # them: the actual import below remains the minimal stable-extension route.
    import base64
    paths = [*package.rglob('*.py'), binary, info / 'METADATA', info / 'WHEEL']
    for path in paths:
        path = _real_file(path, root)
        name, digest = path.relative_to(root).as_posix(), _sha(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
        if records.get(name) != ['sha256=' + encoded, str(path.stat().st_size)]:
            raise ValueError('Native installed source/extension differs from RECORD: ' + name)
        sources[name] = dict(sha256=digest, bytes=path.stat().st_size)
    if 'vllm/__init__.py' not in sources:
        raise ValueError('Missing native package initializer')
    receipt_path = root / 'OFT_NATIVE_RECEIPT.json'
    receipt_sha = None
    if receipt_path.exists():
        _real_file(receipt_path, root)
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        if (receipt.get('schema') != 2 or receipt.get('version') != REQUIRED_RUNTIME['vllm']
                or receipt.get('archive_sha256') != NATIVE_WHEEL_SHA256
                or any(receipt.get('files', {}).get(name) != value for name, value in sources.items())):
            raise ValueError('Existing native preparation receipt does not bind these installed bytes')
        receipt_sha = hashlib.sha256(raw).hexdigest()
    return root, binary, dict(native_path=str(root), extension_path=str(binary),
        extension_sha256=NATIVE_EXTENSION_SHA256, installed_source_files=sources,
        installed_record_sha256=_sha(record_path), preparation_receipt_sha256=receipt_sha,
        archive_identity_scope='receipt reference only; archive is not re-read by this primitive loader')


def _module_origins(root):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'vllm' and not name.startswith('vllm.'):
            continue
        if module is None:
            raise RuntimeError('Partially imported vLLM module cannot establish a native backend')
        file = getattr(module, '__file__', None)
        if file is None:
            raise RuntimeError('Unknown preloaded vLLM namespace: ' + name)
        path = _real_file(Path(file), root)
        if any(Path(value).resolve() != path.parent for value in getattr(module, '__path__', ())):
            raise RuntimeError('vLLM package search path differs from its pinned source')
        result[name] = dict(path=str(path), sha256=_sha(path))
    return result


def _operators():
    try:
        return torch.ops._C.scaled_fp4_quant.out, torch.ops._C.cutlass_scaled_fp4_mm.default
    except AttributeError as error:
        raise RuntimeError('Pinned native NVFP4 operators are unavailable; no fallback') from error


def _validate_operator_schemas(quantize, gemm):
    q, mm = quantize._schema, gemm._schema
    if (q.name != '_C::scaled_fp4_quant' or q.overload_name != 'out'
            or [str(arg.type) for arg in q.arguments] != ['Tensor', 'Tensor', 'bool', 'Tensor', 'Tensor']
            or [arg.name for arg in q.arguments[-2:]] != ['output', 'output_scale']
            or any(arg.alias_info is None or not arg.alias_info.is_write for arg in q.arguments[-2:])
            or mm.name != '_C::cutlass_scaled_fp4_mm'
            or [str(arg.type) for arg in mm.arguments] != ['Tensor'] * 6
            or mm.arguments[0].alias_info is None or not mm.arguments[0].alias_info.is_write
            or q.returns or mm.returns):
        raise RuntimeError('Native NVFP4 operator ABI differs from the explicitly supported out-parameter contract')
    return dict(quantizer=str(q), gemm=str(mm))


class NativeBackend:
    """Process-lifetime module/evidence owner. It owns no CUDA tensor storage."""
    def __init__(self, root, extension, package, device_index, quantize, gemm, manifest):
        self._root, self._extension, self._package = root, extension, package
        self._device_index, self._quantize, self._gemm = device_index, quantize, gemm
        self._manifest = deepcopy(manifest)

    def validate(self, device):
        validate_runtime(device)
        if (_LOAD_FAILURE is not None or _BACKEND is not self or _device_index(device) != self._device_index
                or sys.modules.get('vllm') is not self._package
                or sys.modules.get('vllm._C_stable_libtorch') is not self._extension
                or _operators() != (self._quantize, self._gemm)):
            raise RuntimeError('Native NVFP4 process/backend/device binding changed')

    def manifest(self):
        self.validate(torch.device('cuda', self._device_index))
        return deepcopy(self._manifest)


def load_native_backend(native_path, device):
    """Explicit cold load only; import/ABI failure requires a fresh process.

    Repeated calls for the same path/device revalidate cold file identity. No
    sys.path change is left behind and no model-loader/FA2/Omni import is needed.
    An already-loaded stable extension from this exact root may be reused.
    """
    global _BACKEND, _LOAD_FAILURE
    device = torch.device(device)
    validate_runtime(device)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Load native NVFP4 outside CUDA Graph capture')
    with _LOCK:
        if _LOAD_FAILURE is not None:
            raise RuntimeError('Earlier native import failed; restart the process') from _LOAD_FAILURE
        root, binary, evidence = _inspect_native_files(native_path)
        index = _device_index(device)
        if _BACKEND is not None:
            _BACKEND.validate(device)
            if _BACKEND._root != root or _BACKEND._manifest['files'] != evidence:
                raise RuntimeError('Cannot replace a process-global native NVFP4 backend')
            return _BACKEND
        _module_origins(root)  # Reject foreign preloads before changing search paths.
        if 'vllm._C_stable_libtorch' not in sys.modules:
            try:
                _operators()
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Existing NVFP4 dispatcher operators have no verified stable-extension owner')
        previous_path = list(sys.path)
        try:
            sys.path.insert(0, str(root))
            package = importlib.import_module('vllm')
            if package.__version__ != REQUIRED_RUNTIME['vllm']:
                raise RuntimeError('Actual imported vLLM version differs from its distribution')
            extension = importlib.import_module('vllm._C_stable_libtorch')
            origins = _module_origins(root)
            if Path(extension.__file__).resolve() != binary:
                raise RuntimeError('Imported stable extension is not the verified binary')
            for row in origins.values():
                name = Path(row['path']).relative_to(root).as_posix()
                if evidence['installed_source_files'].get(name, {}).get('sha256') != row['sha256']:
                    raise RuntimeError('Imported native module bytes changed after cold verification: ' + name)
            quantize, gemm = _operators()
            schemas = _validate_operator_schemas(quantize, gemm)
            props = torch.cuda.get_device_properties(device)
            manifest = dict(version=VERSION, variant=VARIANT, runtime=deepcopy(REQUIRED_RUNTIME),
                device_index=index, files=evidence, imported_modules=origins, native_schemas=schemas,
                hardware=dict(name=props.name, uuid=str(props.uuid), total_memory=props.total_memory,
                              capability=list(torch.cuda.get_device_capability(device))),
                implementation=implementation_manifest(), import_succeeded=True,
                execution_validated=False, quality_validated=False)
            _BACKEND = NativeBackend(root, extension, package, index, quantize, gemm, manifest)
            return _BACKEND
        except BaseException as error:
            # dlopen/dispatcher registrations cannot be rolled back safely.
            _LOAD_FAILURE = error
            raise
        finally:
            sys.path[:] = previous_path


def _require_backend(device):
    if _BACKEND is None:
        raise RuntimeError('Explicitly load the pinned NVFP4 backend before inference or capture')
    _BACKEND.validate(device)
    return _BACKEND


def _validate_scale(scale, device):
    if (scale.dtype != torch.float32 or scale.device != device or scale.ndim != 1
            or tuple(scale.shape) != (1,) or scale.stride() != (1,) or scale.requires_grad):
        raise ValueError('NVFP4 global scale/alpha must be one same-device contiguous inference FP32 value')


def validate_cold_scale(scale):
    """Caller invokes before publishing a calibrated/persistent scale epoch."""
    _validate_scale(scale, scale.device)
    _require_backend(scale.device)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Scale content validation is cold-only')
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError('NVFP4 global scale/alpha must be finite and positive')
    return dict(shape=[1], dtype='float32', content_sha256=hashlib.sha256(
        scale.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())


def validate_quantizer_input(value, global_scale, *, weight=False):
    shape = tuple(value.shape)
    valid_shape = shape in ((16384, 2048), (2048, 16384)) if weight else (
        len(shape) == 2 and shape[0] in SUPPORTED_CAPACITIES and shape[1] in (2048, 16384))
    if (value.device.type != 'cuda' or value.dtype != torch.bfloat16 or value.ndim != 2
            or not valid_shape or value.stride() != (shape[-1], 1) or value.requires_grad):
        raise ValueError('NVFP4 quantizer requires an explicit contiguous inference CUDA BF16 prefix matrix')
    _validate_scale(global_scale, value.device)


def validate_scaled_mm_inputs(qx, qw, sx, sw, alpha):
    if any(t.ndim != 2 or t.device != qx.device or t.device.type != 'cuda' or t.requires_grad
           for t in (qx, qw, sx, sw)):
        raise ValueError('NVFP4 native operands must be same-device inference CUDA matrices')
    m, packed_k = qx.shape
    k, n = packed_k * 2, qw.shape[0]
    if (m not in SUPPORTED_CAPACITIES or (k, n) not in SUPPORTED_GEOMETRIES or qw.shape[1] != packed_k
            or qx.dtype != torch.uint8 or qw.dtype != torch.uint8
            or sx.dtype != torch.float8_e4m3fn or sw.dtype != torch.float8_e4m3fn
            or qx.stride() != (packed_k, 1) or qw.stride() != (packed_k, 1)
            or tuple(sx.shape) != scale_shape(m, k) or tuple(sw.shape) != scale_shape(n, k)
            or sx.stride() != (sx.shape[1], 1) or sw.stride() != (sw.shape[1], 1)):
        raise ValueError('NVFP4 native packed geometry/dtype/swizzled scale layout is unsupported')
    _validate_scale(alpha, qx.device)


def _quantize(value, global_scale):
    backend = _require_backend(value.device)
    m, k = value.shape
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=value.device)
    padded_m, scale_columns = scale_shape(m, k)
    storage = torch.zeros((padded_m, scale_columns // 4), dtype=torch.int32, device=value.device)
    backend._quantize(value, global_scale, True, output=q, output_scale=storage)
    return q, storage.view(torch.float8_e4m3fn)


@torch.library.custom_op(QUANTIZE_OPERATOR, mutates_args=())
def quantize_rows(value: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_quantizer_input(value, global_scale)
    return _quantize(value, global_scale)


@quantize_rows.register_fake
def _fake_quantize_rows(value, global_scale):
    validate_quantizer_input(value, global_scale)
    return (torch.empty((value.shape[0], value.shape[1] // 2), dtype=torch.uint8, device=value.device),
            torch.empty(scale_shape(*value.shape), dtype=torch.float8_e4m3fn, device=value.device))


@torch.no_grad()
def quantize_weight(weight, global_scale):
    """Cold original-NK packing; returns new versioned tensors, no retained cache."""
    validate_quantizer_input(weight.detach(), global_scale, weight=True)
    _require_backend(weight.device)
    if torch.is_inference_mode_enabled() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Pack native NVFP4 weights outside inference_mode and CUDA Graph capture')
    validate_cold_scale(global_scale)
    if not bool(torch.isfinite(weight).all()):
        raise ValueError('NVFP4 cannot pack nonfinite original BF16 weights')
    return _quantize(weight.detach(), global_scale)


@torch.library.custom_op(SCALED_MM_OPERATOR, mutates_args=())
def native_scaled_mm(qx: torch.Tensor, qw: torch.Tensor, sx: torch.Tensor,
                     sw: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    validate_scaled_mm_inputs(qx, qw, sx, sw, alpha)
    backend = _require_backend(qx.device)
    output = torch.empty((qx.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=qx.device)
    backend._gemm(output, qx, qw, sx, sw, alpha)
    return output


@native_scaled_mm.register_fake
def _fake_native_scaled_mm(qx, qw, sx, sw, alpha):
    validate_scaled_mm_inputs(qx, qw, sx, sw, alpha)
    return torch.empty((qx.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=qx.device)


def scaled_linear(value, quantized_weight, weight_scale, input_global_scale, alpha):
    """Every call includes online BF16→NVFP4 conversion, scales and native GEMM."""
    qx, sx = quantize_rows(value, input_global_scale)
    return native_scaled_mm(qx, quantized_weight, sx, weight_scale, alpha), qx, sx
