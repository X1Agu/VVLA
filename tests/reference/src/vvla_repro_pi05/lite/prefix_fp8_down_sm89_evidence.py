"""Pure evidence for the explicit SM89 down17 production-core recipe.

No Torch imports, model state, or implicit capability selection. Callers must
bind original report bytes and actual installed source/library/device state;
backend diagnostics alone do not prove a live installation or policy quality.
Legacy 51-site validators remain separate, with unchanged schemas.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
VARIANT = 'sm89-down-scalar-fp32-native-v1'
VERSION = 'torch213-prefix-down-fp8-native-sm89-backend-v1'
IMPLEMENTATION = 'sm89_down_native_opaque_scalar_fp32_postscale_v1'
CORE_FILES = ('prefix_fp8_down_sm89.py', 'prefix_fp8_down_sm89_quantizer_kernel.py',
              'prefix_fp8_down_sm89_postscale_kernel.py')
SHARED_FILES = ('probe_prefix_fp8.py',)
DECLARATIONS = ('_meta_registrations.py', 'include/ATen/native/cuda/RowwiseScaledMM.h',
                'include/ATen/ops/_scaled_mm_ops.h', '_inductor/kernel/mm.py', '_inductor/utils.py')
BINARY_FILES = ('lib/libtorch_cuda.so',)
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[8, 9])
OPERATORS = dict(quantizer='vvla_repro_pi05_lite::prefix_down_sm89_quantize_rows_v1',
                 gemm='vvla_repro_pi05_lite::prefix_down_sm89_scalar_mm_fp32_v1',
                 postscale='vvla_repro_pi05_lite::prefix_down_sm89_postscale_bf16_v1')
DISPATCH = 'ATEN torch._scaled_mm singleton FP32 scales inside opaque op; native cuBLASLt'
POSTSCALE = 'two ordered mul.rn.f32 sx[row] then sw[column]; sole BF16 output cast'
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='PTX cvt.rn.satfinite.e4m3x2.f32; duplicated lane; low byte bitcast',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu')
INTENDED_SCOPE = dict(component='prefix', projections=['down_proj'], layer_indices=list(range(17)),
                      intended_sites=17, gate_up_precision='bfloat16')
SCALAR_ONE = dict(shape=[1], dtype='float32', value=1.0,
    tensor_kind='ordinary versioned Tensor; inference tensors forbidden',
    ownership='caller-owned immutable; cold validate_unit_scale required',
    hot_validation='metadata only; caller binds private identity/compile lifetime')
SCALAR_ONE_SHA256 = hashlib.sha256(b'\x00\x00\x80\x3f').hexdigest()
TRIALS = ('native_before', 'fp8_eager', 'conversion_eager', 'conversion_compiled',
          'prequantized_gemm_eager', 'prequantized_gemm_compiled', 'fp8_compiled', 'native_after')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Unexpected/missing fields in ' + label)


def _exact(value, expected, label):
    if type(value) is not type(expected):
        raise ValueError('Wrong type in ' + label)
    if isinstance(expected, dict):
        _keys(value, expected, label)
        for key in expected:
            _exact(value[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError('Wrong length in ' + label)
        for item, wanted in zip(value, expected, strict=True):
            _exact(item, wanted, label)
    elif value != expected:
        raise ValueError('Contract mismatch in ' + label)


def _sha(value, label):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid SHA256 in ' + label)


def _hashes(value, names, label):
    _keys(value, names, label)
    for name in names:
        _sha(value[name], label + '.' + name)


def _hardware(value):
    _keys(value, ('name', 'uuid', 'driver_version', 'memory_mib', 'hostname'), 'backend.hardware')
    cards = {'NVIDIA L4': ('l4', [8, 9]),
             'NVIDIA RTX PRO 6000 Blackwell Server Edition': ('pro6000', [12, 0])}
    if type(value['name']) is not str or value['name'] not in cards:
        raise ValueError('Unsupported backend physical GPU')
    if type(value['uuid']) is not str or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value['uuid']) is None:
        raise ValueError('Invalid backend physical GPU UUID')
    if type(value['memory_mib']) is not int or value['memory_mib'] <= 0:
        raise ValueError('Invalid backend GPU memory')
    if any(type(value[k]) is not str or not value[k] for k in ('driver_version', 'hostname')):
        raise ValueError('Missing backend host/driver')
    return cards[value['name']]


def expected_native_implementation_manifest(core_files_sha256):
    """Expected declarations only; not actual source/runtime evidence."""
    _hashes(core_files_sha256, CORE_FILES, 'SM89 down core sources')
    return deepcopy(dict(version='pi05-prefix-down-fp8-native-sm89-v1', variant=VARIANT,
        implementation=IMPLEMENTATION, core_files_sha256=core_files_sha256,
        required_runtime=RUNTIME, supported_capacities=[576, 640],
        supported_geometries=[[16384, 2048]], intended_scope=INTENDED_SCOPE,
        quantizer_arithmetic=ARITHMETIC, compile_options=COMPILE_OPTIONS,
        operators=OPERATORS, dispatch=DISPATCH, weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]', input_dtype='bfloat16',
        quantized_dtype='float8_e4m3fn', scale_dtype='float32', accumulator_dtype='float32',
        intermediate_dtype='float32', output_dtype='bfloat16', postscale=POSTSCALE,
        scalar_one=SCALAR_ONE, use_fast_accum=False, allow_tf32=False, bias=None,
        scale_result=None, old_rowwise_output_exact_required=False,
        old_rowwise_no_bias_positive_zero_add_preserved=False, fallback_permitted=False,
        backend_evidence='new same-card actual package-core report required',
        installation='none; intended 17-site scope only, no policy or quality claim'))


def recipe_for_variant(variant):
    _exact(variant, VARIANT, 'SM89 down explicit variant')
    return deepcopy(dict(weight='E4M3fn per-output-channel', activation='E4M3fn per-token',
        scale='FP32', scale_floor=1e-12, output='BF16', use_fast_accum=False, bias=None,
        quantizer='FP32 abs/amax/divide448/clamp_min1e-12/divide/clamp/cast',
        quantizer_implementation=IMPLEMENTATION, backend_implementation=IMPLEMENTATION,
        backend_variant=VARIANT, rotations=5,
        gemm='opaque ATEN torch._scaled_mm singleton scales=1; native cuBLASLt; output FP32',
        postscale=POSTSCALE, intermediate_dtype='float32', allow_tf32=False,
        old_rowwise_no_bias_positive_zero_add_preserved=False, old_rowwise_output_exact_required=False,
        quantizer_arithmetic=ARITHMETIC,
        output_cast_implementation=ARITHMETIC['output_cast'],
        weight_packing_implementation='original Torch eager quantize_rows; cold only',
        compile_options=COMPILE_OPTIONS, intended_scope=INTENDED_SCOPE, scalar_one=SCALAR_ONE,
        operators=OPERATORS, dispatch=DISPATCH))


def validate_backend_manifest(backend):
    _keys(backend, ('operator', 'dispatch', 'torch_version', 'torch_git_version', 'cuda_version',
        'hardware', 'capability', 'compile_options', 'installed_declarations', 'installed_binary_sha256',
        'probe_source_sha256', 'probe_recipe_sha256', 'report_sha256', 'quantizer_variant',
        'quantizer_files_sha256', 'supported_capacities', 'shared_helper_files_sha256'), 'SM89 down backend')
    for key, expected in dict(operator='torch._scaled_mm', dispatch='ATEN',
        torch_version=RUNTIME['torch'], torch_git_version=RUNTIME['torch_git'], cuda_version=RUNTIME['cuda'],
        capability=RUNTIME['capability'], compile_options=COMPILE_OPTIONS,
        quantizer_variant=VARIANT, supported_capacities=[576, 640]).items():
        _exact(backend[key], expected, 'SM89 down backend.' + key)
    card, _ = _hardware(backend['hardware'])
    _exact(card, 'l4', 'SM89 down physical GPU')
    for key, names in (('quantizer_files_sha256', CORE_FILES), ('shared_helper_files_sha256', SHARED_FILES),
                       ('installed_declarations', DECLARATIONS), ('installed_binary_sha256', BINARY_FILES)):
        _hashes(backend[key], names, 'SM89 down backend.' + key)
    for key in ('probe_source_sha256', 'probe_recipe_sha256', 'report_sha256'):
        _sha(backend[key], 'SM89 down backend.' + key)
    _exact(backend['probe_recipe_sha256'], digest(recipe_for_variant(VARIANT)), 'SM89 down recipe binding')
    return backend


def _unit_evidence(value):
    _exact(value, dict(passed=True, dtype='float32', shape=[1], stride=[1], value=1.0,
        inference_tensor=False, version_counter_available=True, raw_sha256=SCALAR_ONE_SHA256,
        ownership='caller-owned; no core retention', validation='cold exact bytes'), 'SM89 down unit scale')


def _trial(value, rotations):
    if type(value) is not dict:
        raise ValueError('Missing SM89 down graph trial')
    _exact(value.get('self_exact'), True, 'SM89 down graph replay bytes')
    _exact(value.get('calls_per_sequence'), rotations, 'SM89 down graph projection count')
    sequence, per_projection = value.get('raw_sequence_ms'), value.get('raw_ms_per_projection')
    for values in (sequence, per_projection):
        if (type(values) is not list or len(values) != 30 or any(type(x) not in (int, float)
                or not math.isfinite(x) or x <= 0 for x in values)):
            raise ValueError('Incomplete/nonfinite SM89 down raw event evidence')
    if any(a / rotations != b for a, b in zip(sequence, per_projection, strict=True)):
        raise ValueError('SM89 down event normalization differs from recorded rotation count')


def validate_backend_payload(payload, backend, hardware):
    """Recorded backend evidence only; callers bind raw bytes and live runtime."""
    validate_backend_manifest(backend)
    if type(payload) is not dict:
        raise ValueError('SM89 down report must be a dictionary')
    _exact(backend['hardware'], hardware, 'SM89 down actual hardware')
    for key, expected in dict(version=VERSION, backend_variant=VARIANT, quantizer_variant=VARIANT,
        status='complete', diagnostic_only=True, precision='fp8_backend_probe_bf16_output',
        formal_manifest_eligible=False, synthetic_activations=True, frozen_fixture_validated=False,
        policy_loaded=False, quality_passed=False, production_changed=False,
        native_fallback_permitted=False, cleanup_errors=[], hardware=hardware,
        recipe=recipe_for_variant(VARIANT)).items():
        _exact(payload.get(key), expected, 'SM89 down report.' + key)
    for field, key in (('torch', 'torch_version'), ('torch_git', 'torch_git_version'), ('cuda', 'cuda_version'),
                       ('capability', 'capability'), ('installed_declarations', 'installed_declarations'),
                       ('installed_binary_sha256', 'installed_binary_sha256')):
        _exact(payload.get(field), backend[key], 'SM89 down report.' + field)
    source = payload.get('source')
    if type(source) is not dict:
        raise ValueError('Missing SM89 down actual source evidence')
    for field, key in (('script_sha256', 'probe_source_sha256'),
                       ('core_files_sha256', 'quantizer_files_sha256'),
                       ('shared_helper_files_sha256', 'shared_helper_files_sha256')):
        _exact(source.get(field), backend[key], 'SM89 down report source.' + field)
    _exact(payload.get('core_implementation_manifest'),
        expected_native_implementation_manifest(backend['quantizer_files_sha256']), 'SM89 down actual core manifest')
    if 'checkpoint' in payload:
        if type(payload['checkpoint']) is not dict:
            raise ValueError('Invalid SM89 down checkpoint evidence')
        _exact(payload['checkpoint'].get('sha256'), CHECKPOINT_SHA256, 'SM89 down original checkpoint')
    timing = payload.get('timing')
    if type(timing) is not dict:
        raise ValueError('Missing SM89 down timing evidence')
    _exact(timing.get('warmup'), 10, 'SM89 down warmup')
    _exact(timing.get('samples'), 30, 'SM89 down measured samples')
    l2 = payload.get('l2_cache_bytes')
    if type(l2) is not int or l2 < 0:
        raise ValueError('Missing SM89 down physical L2 size')
    rows = payload.get('rows')
    if type(rows) is not list or len(rows) != 2:
        raise ValueError('SM89 down requires exactly two executed down shapes')
    for m, row in zip((576, 640), rows, strict=True):
        if type(row) is not dict:
            raise ValueError('SM89 down shape evidence must be a dictionary')
        _exact(row.get('shape'), dict(capacity=m, projection='down', m=m, k=16384, n=2048),
               'SM89 down fixed shape')
        for key, expected in dict(status='backend_contracts_passed', synthetic_activations=True,
            native_before_after_exact=True, quantizer_variant=VARIANT, backend_variant=VARIANT,
            rotations=5, distinct_fp8_weight_bytes=5 * 2048 * 16384, cache_limit_bytes=max(128 * 1024 * 1024, l2),
            fp32_temporary_bytes_per_projection=4 * m * 2048,
            fp32_temporary_write_read_bytes_per_projection=8 * m * 2048).items():
            _exact(row.get(key), expected, 'SM89 down row.' + key)
        if row['distinct_fp8_weight_bytes'] <= row['cache_limit_bytes']:
            raise ValueError('SM89 down distinct weights must exceed L2 and 128MiB')
        _unit_evidence(row.get('unit_scale_validation'))
        cold = row.get('cold_unit_scale_seconds')
        if type(cold) not in (int, float) or not math.isfinite(cold) or cold < 0:
            raise ValueError('Missing SM89 down cold unit-scale validation timing')
        for field in ('input_sha256', 'weight_sha256'):
            hashes = row.get(field)
            if type(hashes) is not list or len(hashes) != 5:
                raise ValueError('Missing five SM89 down input/weight hashes')
            for value in hashes:
                _sha(value, 'SM89 down ' + field)
            if len(set(hashes)) != 5:
                raise ValueError('SM89 down rotations must use distinct input/weight bytes')
        checks = row.get('checks')
        if type(checks) is not dict:
            raise ValueError('Missing SM89 down numerical checks')
        for key in ('native_operator_available', 'finite_fp8', 'compiled_quantization_exact', 'fused_quantization_exact'):
            _exact(checks.get(key), True, 'SM89 down checks.' + key)
        for name in ('analytical_semantics', 'scalar_analytical_semantics'):
            check = checks.get(name)
            if type(check) is not dict:
                raise ValueError('Missing SM89 down analytical gate')
            for key in ('passed', 'exact'):
                _exact(check.get(key), True, 'SM89 down analytical gate.' + key)
        scalar = checks['scalar_analytical_semantics']
        _exact(scalar.get('accumulator_dtype'), 'torch.float32', 'SM89 down actual accumulator')
        _sha(scalar.get('actual_sha256'), 'SM89 down actual analytical output')
        trials = row.get('trials')
        _keys(trials, TRIALS, 'SM89 down exactly eight trials')
        for trial in TRIALS:
            _trial(trials[trial], 5)
        for mode in ('eager', 'compiled'):
            trial = trials['prequantized_gemm_' + mode]
            _exact(trial.get('includes_fp32_temporary_and_postscale'), True, 'SM89 down complete GEMM/postscale timing')
            _exact(trial.get('fp32_temporary_bytes'), 5 * 4 * m * 2048, 'SM89 down temporary byte accounting')
        numerics = row.get('numerics')
        if type(numerics) is not list or len(numerics) != 5:
            raise ValueError('Missing five SM89 down numerical rotations')
        for record in numerics:
            if type(record) is not dict or type(record.get('native_vs_quantized_fp32_oracle')) is not dict:
                raise ValueError('Missing SM89 down independent quantized-input oracle')
            _exact(record['native_vs_quantized_fp32_oracle'].get('passed'), True, 'SM89 down independent oracle')
            _sha(record.get('scalar_output_sha256'), 'SM89 down scalar output bytes')
            _sha(record.get('original_rowwise_output_sha256'), 'SM89 down old rowwise comparison bytes')
    return payload


