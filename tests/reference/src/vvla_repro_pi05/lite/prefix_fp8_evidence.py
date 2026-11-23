"""Pure frozen prefix-FP8 backend evidence; no Torch or benchmark dependency.

These checks validate recorded declarations and complete diagnostic evidence.
Callers must separately bind original report bytes and current source/library/
physical-device state. This module does not establish a live quantized installation
or policy quality. No model inventory or tensor state is retained.
"""
import hashlib
import json
import math
import re

CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
FLASHRT_REFERENCE = 'flashrt_bf16'
EXTERNAL_BASE_RECIPE_SCOPE = 'actual_lite_execution_recipe_before_reference_binding'
DOWN_SM89_VARIANT = 'sm89-down-scalar-fp32-native-v1'
_DOWN_SM89_EVIDENCE = None


def down_sm89_evidence():
    """Load the adjacent pure source, including in audit-only file imports.

    This caches only a code module, never runtime/model/tensor identities. The
    explicit variant dispatch below cannot select a recipe by GPU capability.
    """
    global _DOWN_SM89_EVIDENCE
    if _DOWN_SM89_EVIDENCE is None:
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().with_name('prefix_fp8_down_sm89_evidence.py')
        spec = importlib.util.spec_from_file_location('_vvla_prefix_fp8_down_sm89_evidence', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DOWN_SM89_EVIDENCE = module
    return _DOWN_SM89_EVIDENCE


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False,
                   'max_autotune': False, 'max_autotune_gemm': False, 'max_autotune_gemm_backends': 'ATEN'}
PROBE_RECIPE = dict(weight='E4M3fn per-output-channel', activation='E4M3fn per-token', scale='FP32',
                    scale_floor=1e-12, output='BF16', use_fast_accum=False, bias=None,
                    quantizer='FP32 abs/amax/divide448/clamp_min1e-12/divide/clamp/cast',
                    compile_options=COMPILE_OPTIONS)
DECLARATIONS = ('_meta_registrations.py', 'include/ATen/native/cuda/RowwiseScaledMM.h',
                'include/ATen/ops/_scaled_mm_ops.h', '_inductor/kernel/mm.py', '_inductor/utils.py')
BINARY_FILES = ('lib/libtorch_cuda.so',)
QUANTIZER_FILES = ('prefix_fp8_quantizer.py', 'prefix_fp8_quantizer_kernel.py')
QUANTIZER_VARIANTS = {'torch-eager': ('torch_eager_ops', 'torch213-rowwise-prefix-fp8-backend-v1'),
                      'triton-eager-rn': ('triton_eager_rn_custom_op_v1', 'torch213-rowwise-prefix-fp8-backend-v2')}
PRODUCTION_VARIANT = 'triton-eager-rn-native-sm120-v1'
PRODUCTION_FILES = ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')
PRODUCTION_HELPERS = ('probe_prefix_fp8.py',)
PRODUCTION_COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
PRODUCTION_RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
PRODUCTION_OPERATORS = dict(quantizer='vvla_repro_pi05_lite::prefix_fp8_quantize_rows_v1',
                            gemm='vvla_repro_pi05_lite::prefix_fp8_native_scaled_mm_v1')
PRODUCTION_DISPATCH = 'ATEN torch._scaled_mm inside opaque custom op'
QUANTIZER_VARIANTS[PRODUCTION_VARIANT] = (
    'triton_eager_rn_native_opaque_v1', 'torch213-rowwise-prefix-fp8-native-sm120-backend-v1')
QUANTIZER_ARITHMETIC = dict(
    scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187,
    scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu',
)


def recipe_for_variant(variant):
    import copy
    if type(variant) is str and variant == DOWN_SM89_VARIANT:
        return down_sm89_evidence().recipe_for_variant(variant)
    if type(variant) is not str or variant not in QUANTIZER_VARIANTS:
        raise ValueError('Unsupported quantizer implementation variant')
    value = copy.deepcopy(PROBE_RECIPE)
    value['quantizer_implementation'] = QUANTIZER_VARIANTS[variant][0]
    if variant in ('triton-eager-rn', PRODUCTION_VARIANT):
        value['quantizer_arithmetic'] = copy.deepcopy(QUANTIZER_ARITHMETIC)
    if variant == PRODUCTION_VARIANT:
        value.update(compile_options=copy.deepcopy(PRODUCTION_COMPILE_OPTIONS),
                     operators=copy.deepcopy(PRODUCTION_OPERATORS), dispatch=PRODUCTION_DISPATCH)
    return value


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


def expected_native_implementation_manifest(quantizer_files_sha256):
    """Expected declarations, not actual source inspection or passed evidence."""
    from copy import deepcopy
    _hashes(quantizer_files_sha256, PRODUCTION_FILES, 'production native sources')
    return deepcopy(dict(version='pi05-prefix-fp8-native-sm120-v1', variant=PRODUCTION_VARIANT,
        implementation=QUANTIZER_VARIANTS[PRODUCTION_VARIANT][0],
        quantizer_files_sha256=quantizer_files_sha256, required_runtime=PRODUCTION_RUNTIME,
        supported_capacities=[576, 640], supported_geometries=[[2048, 16384], [16384, 2048]],
        quantizer_arithmetic=QUANTIZER_ARITHMETIC, compile_options=PRODUCTION_COMPILE_OPTIONS,
        operators=PRODUCTION_OPERATORS, dispatch=PRODUCTION_DISPATCH,
        weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose',
        input_dtype='bfloat16', quantized_dtype='float8_e4m3fn', scale_dtype='float32',
        accumulator_dtype='float32', output_dtype='bfloat16', use_fast_accum=False,
        bias=None, scale_result=None, fallback_permitted=False,
        backend_evidence='new same-card executed production-wrapper report required',
        installation='none; no policy, weight owner, or quality claim'))


def validate_evidence_config(value, *, backend_report_sha256=None, bf16_control=None):
    """Validate explicit cold-load inputs; callers bind real bytes/control state.

    No file is read here. The path is not an acceptance claim. Supplying expected
    hashes binds this declared input to already validated actual state.
    """
    external = type(value) is dict and 'reference_kind' in value
    _keys(value, ('backend_report', 'backend_report_sha256', 'bf16_control')
          + (('reference_kind',) if external else ()), 'prefix_fp8_evidence')
    if external:
        # An external BF16 baseline is a comparison anchor, never a claim that
        # FlashRT's execution recipe is the Lite policy's BF16 implementation.
        _exact(value['reference_kind'], FLASHRT_REFERENCE, 'prefix_fp8_evidence.reference_kind')
    path = value['backend_report']
    if type(path) is not str or not path or '\x00' in path:
        raise ValueError('prefix_fp8_evidence.backend_report must be an explicit path string')
    _sha(value['backend_report_sha256'], 'prefix_fp8_evidence.backend_report_sha256')
    _hashes(value['bf16_control'], ('recipe_sha256', 'config_sha256', 'identity_sha256', 'content_sha256'),
            'prefix_fp8_evidence.bf16_control')
    if backend_report_sha256 is not None:
        _exact(value['backend_report_sha256'], backend_report_sha256, 'configured actual backend report SHA')
    if bf16_control is not None:
        _exact(value['bf16_control'], bf16_control, 'configured selected BF16 control')
    return value


def validate_backend_manifest(backend):
    """Validate a backend evidence binding without claiming a model installation."""
    if type(backend) is not dict:
        raise ValueError('Backend evidence binding must be a dictionary')
    if type(backend.get('quantizer_variant')) is str and backend['quantizer_variant'] == DOWN_SM89_VARIANT:
        return down_sm89_evidence().validate_backend_manifest(backend)
    production = backend.get('quantizer_variant') == PRODUCTION_VARIANT
    keys = ('operator', 'dispatch', 'torch_version', 'torch_git_version', 'cuda_version',
                    'hardware', 'capability', 'compile_options', 'installed_declarations',
                    'installed_binary_sha256', 'probe_source_sha256', 'probe_recipe_sha256',
                    'report_sha256', 'quantizer_variant', 'quantizer_files_sha256', 'supported_capacities')
    _keys(backend, keys + (('shared_helper_files_sha256',) if production else ()), 'backend')
    capacities = backend['supported_capacities']
    if (type(capacities) is not list or not capacities
            or any(type(m) is not int or not 64 <= m <= 1024 or m % 64 for m in capacities)
            or capacities != sorted(set(capacities))):
        raise ValueError('Explicit sorted unique supported prefix capacities required')
    variant = backend['quantizer_variant']
    recipe = recipe_for_variant(variant)
    files = PRODUCTION_FILES if production else QUANTIZER_FILES if variant == 'triton-eager-rn' else ()
    _hashes(backend['quantizer_files_sha256'], files, 'quantizer source files')
    _exact(backend['operator'], 'torch._scaled_mm', 'backend.operator')
    _exact(backend['dispatch'], 'ATEN', 'backend.dispatch')
    _exact(backend['compile_options'], PRODUCTION_COMPILE_OPTIONS if production else COMPILE_OPTIONS,
           'backend.compile_options')
    for key, pattern in (('torch_version', r'2\.13\.[0-9]+(?:\+[A-Za-z0-9.]+)?'),
                         ('cuda_version', r'13\.[0-9]+'), ('torch_git_version', '[0-9a-f]{40}')):
        if type(backend[key]) is not str or re.fullmatch(pattern, backend[key]) is None:
            raise ValueError('Unsupported backend ' + key)
    card, capability = _hardware(backend['hardware'])
    _exact(backend['capability'], capability, 'backend.capability')
    if production:
        _exact(card, 'pro6000', 'production backend physical GPU')
        _exact(capacities, [576, 640], 'production supported prefix capacities')
        for key, wanted in (('torch_version', PRODUCTION_RUNTIME['torch']),
                            ('torch_git_version', PRODUCTION_RUNTIME['torch_git']),
                            ('cuda_version', PRODUCTION_RUNTIME['cuda']),
                            ('capability', PRODUCTION_RUNTIME['capability'])):
            _exact(backend[key], wanted, 'production backend.' + key)
        _hashes(backend['shared_helper_files_sha256'], PRODUCTION_HELPERS, 'production backend shared helper sources')
    _hashes(backend['installed_declarations'], DECLARATIONS, 'backend.installed_declarations')
    _hashes(backend['installed_binary_sha256'], BINARY_FILES, 'backend.installed_binary_sha256')
    for key in ('probe_source_sha256', 'probe_recipe_sha256', 'report_sha256'):
        _sha(backend[key], 'backend.' + key)
    _exact(backend['probe_recipe_sha256'], digest(recipe), 'backend.probe_recipe_sha256')
    return backend


def validate_backend_payload(payload, backend, hardware):
    """Validate standalone backend evidence, without packed-site/live-model claims.

    The caller must separately verify original report bytes against report_sha256
    and bind the backend fields to its actual installed libraries/source/device.
    Formal policy evidence additionally requires its full installation validator;
    the campaign's validate_backend_report() checks the complete explicit site
    scope (legacy 51 sites, or the separate SM89 down17 recipe) before this call.
    """
    if (type(backend) is dict and type(backend.get('quantizer_variant')) is str
            and backend['quantizer_variant'] == DOWN_SM89_VARIANT):
        return down_sm89_evidence().validate_backend_payload(payload, backend, hardware)
    validate_backend_manifest(backend)
    if type(payload) is not dict:
        raise ValueError('Backend report must be a dictionary')
    variant = backend['quantizer_variant']
    recipe = recipe_for_variant(variant)
    for key, expected in dict(version=QUANTIZER_VARIANTS[variant][1], quantizer_variant=variant, status='complete',
                              diagnostic_only=True, precision='fp8_backend_probe_bf16_output',
                              synthetic_activations=True, frozen_fixture_validated=False, policy_loaded=False,
                              quality_passed=False, production_changed=False, native_fallback_permitted=False,
                              cleanup_errors=[], hardware=hardware, recipe=recipe).items():
        _exact(payload.get(key), expected, 'backend_report.' + key)
    _exact(backend['hardware'], hardware, 'manifest backend hardware')
    for field, key in (('torch', 'torch_version'), ('torch_git', 'torch_git_version'), ('cuda', 'cuda_version'),
                       ('capability', 'capability'), ('installed_declarations', 'installed_declarations'),
                       ('installed_binary_sha256', 'installed_binary_sha256')):
        _exact(payload.get(field), backend[key], 'backend_report.' + field)
    source = payload.get('source', {})
    if type(source) is not dict:
        raise ValueError('Backend source evidence must be a dictionary')
    _exact(source.get('script_sha256'), backend['probe_source_sha256'], 'backend probe source')
    _exact(source.get('quantizer_files_sha256', {}), backend['quantizer_files_sha256'], 'backend quantizer source files')
    if variant == PRODUCTION_VARIANT:
        _exact(source.get('shared_helper_files_sha256'), backend['shared_helper_files_sha256'],
               'production backend shared helper source binding')
        _exact(payload.get('core_implementation_manifest'),
               expected_native_implementation_manifest(backend['quantizer_files_sha256']),
               'production actual core implementation declarations')
    _exact(digest(payload['recipe']), backend['probe_recipe_sha256'], 'backend probe recipe')
    if 'checkpoint' in payload:
        _exact(payload['checkpoint'].get('sha256'), CHECKPOINT_SHA256, 'backend probe checkpoint')
    rows = payload.get('rows')
    if type(rows) is not list or not rows:
        raise ValueError('Missing executed backend shape rows')
    timing = payload.get('timing')
    if type(timing) is not dict:
        raise ValueError('Missing backend timing evidence')
    _exact(timing.get('warmup'), 10, 'backend warmup')
    _exact(timing.get('samples'), 30, 'backend samples')
    l2_bytes = payload.get('l2_cache_bytes')
    if type(l2_bytes) is not int or l2_bytes < 0:
        raise ValueError('Missing physical backend L2 cache size')
    seen = set()
    for row in rows:
        if type(row) is not dict:
            raise ValueError('Backend shape row must be a dictionary')
        shape = row.get('shape', {})
        _keys(shape, ('capacity', 'projection', 'm', 'k', 'n'), 'backend shape')
        m, projection = shape['m'], shape['projection']
        if type(m) is not int or m < 64 or m > 1024 or m % 64 or type(projection) is not str:
            raise ValueError('Unsupported backend probe shape')
        if projection not in ('gate_or_up', 'down') or (m, projection) in seen:
            raise ValueError('Wrong/duplicate backend projection')
        seen.add((m, projection))
        k, n = (2048, 16384) if projection == 'gate_or_up' else (16384, 2048)
        _exact(shape, dict(capacity=m, projection=projection, m=m, k=k, n=n), 'backend shape')
        for key, expected in dict(status='backend_contracts_passed', synthetic_activations=True,
                                  native_before_after_exact=True).items():
            _exact(row.get(key), expected, 'backend row.' + key)
        if variant == PRODUCTION_VARIANT:
            _exact(row.get('quantizer_variant'), PRODUCTION_VARIANT, 'production row variant')
        checks = row.get('checks', {})
        for key in ('native_operator_available', 'finite_fp8', 'compiled_quantization_exact'):
            _exact(checks.get(key), True, 'backend checks.' + key)
        if variant in ('triton-eager-rn', PRODUCTION_VARIANT):
            _exact(checks.get('fused_quantization_exact'), True, 'backend fused quantization')
        for key in ('passed', 'exact'):
            _exact(checks.get('analytical_semantics', {}).get(key), True, 'backend analytical semantics')
            if variant == PRODUCTION_VARIANT:
                _exact(checks.get('core_analytical_semantics', {}).get(key), True,
                       'production native operator analytical semantics')
        if variant == PRODUCTION_VARIANT:
            _sha(checks['core_analytical_semantics'].get('actual_sha256'),
                 'production native analytical output bytes')
        rotations = row.get('rotations')
        cache_limit = row.get('cache_limit_bytes')
        if (type(rotations) is not int or not 4 <= rotations <= 17
                or type(cache_limit) is not int or cache_limit != max(128 * 1024 * 1024, l2_bytes)):
            raise ValueError('Backend rotations/cache evidence is incomplete')
        _exact(row.get('distinct_fp8_weight_bytes'), rotations * n * k, 'backend distinct weight bytes')
        if row['distinct_fp8_weight_bytes'] <= cache_limit:
            raise ValueError('Backend rotations do not exceed cache capacity')
        for field in ('input_sha256', 'weight_sha256'):
            hashes = row.get(field)
            if type(hashes) is not list or len(hashes) != rotations:
                raise ValueError('Missing per-rotation backend ' + field)
            for value in hashes:
                _sha(value, 'backend.' + field)
            if len(set(hashes)) != rotations:
                raise ValueError('Backend rotations must have distinct input and weight bytes')
        trials = ['native_before', 'fp8_eager', 'fp8_compiled', 'native_after']
        if variant in ('triton-eager-rn', PRODUCTION_VARIANT):
            trials += ['conversion_eager', 'conversion_compiled', 'prequantized_gemm_eager', 'prequantized_gemm_compiled']
        for trial in trials:
            evidence = row.get('trials', {}).get(trial, {})
            _exact(evidence.get('self_exact'), True, 'backend graph trial')
            _exact(evidence.get('calls_per_sequence'), rotations, 'backend graph projection count')
            sequence, per_projection = evidence.get('raw_sequence_ms'), evidence.get('raw_ms_per_projection')
            for values in (sequence, per_projection):
                if (type(values) is not list or len(values) != 30 or any(type(x) not in (int, float)
                        or not math.isfinite(x) or x <= 0 for x in values)):
                    raise ValueError('Incomplete/nonfinite backend raw event evidence')
            if any(a / rotations != b for a, b in zip(sequence, per_projection, strict=True)):
                raise ValueError('Backend event normalization differs from recorded rotation count')
        numerics = row.get('numerics')
        if type(numerics) is not list or type(row.get('rotations')) is not int or len(numerics) != row['rotations'] or not numerics:
            raise ValueError('Incomplete backend numerical rotations')
        for record in numerics:
            _exact(record.get('native_vs_quantized_fp32_oracle', {}).get('passed'), True, 'backend quantized-input oracle')
    if seen != {(m, projection) for m in backend['supported_capacities'] for projection in ('gate_or_up', 'down')}:
        raise ValueError('Backend rows must exactly cover declared supported capacities and both prefix geometries')
    return payload


