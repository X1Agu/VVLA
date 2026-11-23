"""Strict B32 prefix-backend evidence; pure stdlib, no B1 acceptance reuse.

The winning B1 provenance below identifies the recipe being extended. It is
never a B32 performance/quality denominator. Runtime execution, native binary
bindings and all four new flattened shapes require a new same-card report.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'pi05-b32-prefix-fp8-backend-v1'
BATCH_SIZE = 32
CAPACITIES = (576, 640)
ROWS = (18432, 20480)
GEOMETRIES = ((2048, 16384), (16384, 2048))
VARIANT = 'b32-triton-eager-rn-native-sm120-v1'
CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
ORIGIN_B1 = dict(source_inventory_sha256='7d3886dbbbe4def09aadf4e66f3b8e445298a83aced58cae1487f5df6da4a9c1',
    recipe_sha256='71dc611b5230da387b05713727fb6368b434ec0818d4856d18e0480f5e581046',
    formal_content_sha256='dbd9f58dca3730d1fb9f3882f5cfc6ac8769d896d3acfbb0e147ad93b1dd8046',
    scope='winning_B1_provenance_only;not_B32_baseline_or_quality_evidence')
SHARED_B1_OWNER_SHA256 = '91ce10642c4eec5ae12aa8369288d867fa61495d2978f48cd65f3d57415279c5'
RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
DECLARATIONS = ('_meta_registrations.py', 'include/ATen/native/cuda/RowwiseScaledMM.h',
    'include/ATen/ops/_scaled_mm_ops.h', '_inductor/kernel/mm.py', '_inductor/utils.py')
BINARIES = ('lib/libtorch_cuda.so',)
PROBE_PATH = 'benchmarks/pi05-b32-20261110/profiling/probe_batch_prefix_fp8_native.py'
PACKAGE_PREFIX = 'vllm-vla-plugin/src/vvla_repro_pi05/'
SOURCE_FILES = tuple(PACKAGE_PREFIX + name for name in (
    'lite/batch_prefix_fp8.py', 'lite/batch_prefix_fp8_native.py', 'lite/batch_prefix_fp8_native_kernel.py',
    'lite/batch_prefix_fp8_evidence.py', 'lite/prefix_fp8.py', 'lite/compilation.py',
    'lite/graph.py', 'lite/fused_expert_mlp.py',
    'models/pi05/fused_mlp_pi05.py', 'models/pi05/precision_pi05.py',
    'models/pi05/inference_cache_pi05.py')) + (PROBE_PATH,)
CHECKS = ('quantizer_eager_exact', 'quantizer_compiled_exact', 'analytical_scale_axes_exact',
    'eager_equals_raw_scaled_mm', 'compiled_equals_eager', 'captured_equals_eager',
    'changed_input_exact', 'joint_weight_scale_refresh_exact', 'previous_outputs_unchanged',
    'no_output_alias', 'graph_retired')
COMPILE_OPTIONS = {'triton.cudagraphs':False,'emulate_precision_casts':True,'use_fast_math':False}
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187,scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu')
PRECISION_BOUNDARIES = dict(vision='unchanged_selected_bf16_control',
    prefix_attention='unchanged_selected_bf16_control',expert_projections='unchanged_selected_bf16_control',
    normalization='unchanged_selected_control_FP32_boundaries',residual_euler='unchanged_selected_control_FP32_boundaries',
    action_time_projections='unchanged_selected_control_FP32_boundaries')
LIVE_CHECKS = ['module_forward_bindings','original_weight_identity','quantized_weight_identity',
    'weight_scale_identity','backend_identity','unique_runner_generation','no_extra_quantized_sites',
    'unchanged_nonprefix_precision','supported_capacity_dispatch']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value, expected, label):
    try:
        valid = canonical(value) == canonical(expected)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('B32 prefix evidence mismatch: ' + label)


def keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('B32 prefix evidence missing/extra fields: ' + label)


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('B32 prefix evidence requires a SHA256')
    return value


def hashes(value, expected):
    keys(value, expected, 'source/binary hashes')
    for item in value.values(): sha(item)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[4]
    result = {name: file_sha(root / name) for name in SOURCE_FILES}
    exact(result[PACKAGE_PREFIX+'lite/prefix_fp8.py'], SHARED_B1_OWNER_SHA256, 'shared winning B1 owner source')
    return result


def validate_config(value):
    keys(value, ('backend_report', 'backend_report_sha256', 'origin_b1'), 'configuration')
    if type(value['backend_report']) is not str or not value['backend_report'] or '\0' in value['backend_report']:
        raise ValueError('B32 backend report must have an explicit path')
    sha(value['backend_report_sha256'])
    exact(value['origin_b1'], ORIGIN_B1, 'B1 origin, never a B32 denominator')
    return value


def validate_hardware(value):
    keys(value, ('name', 'uuid', 'driver_version', 'memory_mib', 'hostname'), 'physical hardware')
    exact(value['name'], 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'SM120 card')
    if type(value['uuid']) is not str or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value['uuid']) is None:
        raise ValueError('B32 backend needs actual GPU UUID')
    if type(value['memory_mib']) is not int or value['memory_mib'] <= 0:
        raise ValueError('B32 backend needs actual physical memory')
    if any(type(value[key]) is not str or not value[key] for key in ('driver_version', 'hostname')):
        raise ValueError('B32 backend needs actual host/driver')


def expected_native_manifest(sources):
    files=('batch_prefix_fp8_native.py','batch_prefix_fp8_native_kernel.py')
    return dict(version='pi05-b32-prefix-fp8-native-sm120-v1',variant=VARIANT,
        implementation='b32_triton_eager_rn_native_opaque_v1',
        quantizer_files_sha256={name:sources[PACKAGE_PREFIX+'lite/'+name] for name in files},
        required_runtime=deepcopy(RUNTIME),supported_capacities=list(ROWS),
        supported_geometries=[list(x) for x in GEOMETRIES],batch_size=32,per_slot_capacities=list(CAPACITIES),
        quantizer_arithmetic=deepcopy(ARITHMETIC),compile_options=deepcopy(COMPILE_OPTIONS),
        operators=dict(quantizer='vvla_repro_pi05_lite::batch32_prefix_fp8_quantize_rows_v1',
                       gemm='vvla_repro_pi05_lite::batch32_prefix_fp8_native_scaled_mm_v1'),
        dispatch='ATEN torch._scaled_mm inside opaque custom op',weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose',input_dtype='bfloat16',
        quantized_dtype='float8_e4m3fn',scale_dtype='float32',accumulator_dtype='float32',output_dtype='bfloat16',
        use_fast_accum=False,bias=None,scale_result=None,fallback_permitted=False,
        backend_evidence='new same-card executed production-wrapper report required',
        installation='none; no policy, weight owner, or quality claim')


def validate_report(payload, *, actual_hardware, actual_sources, actual_torch_files, actual_native_manifest):
    keys(payload, ('schema', 'implementation', 'status', 'diagnostic_only', 'synthetic_activations',
        'policy_loaded', 'quality_passed', 'origin_b1', 'runtime', 'hardware', 'source_files_sha256',
        'torch_files_sha256', 'native_manifest', 'rows', 'cleanup'), 'executed backend report')
    expected = dict(schema=1, implementation=VERSION, status='complete', diagnostic_only=True,
        synthetic_activations=True, policy_loaded=False, quality_passed=False, origin_b1=ORIGIN_B1,
        runtime=RUNTIME, hardware=actual_hardware, source_files_sha256=actual_sources,
        torch_files_sha256=actual_torch_files, native_manifest=actual_native_manifest,
        cleanup=dict(graphs_retired=True, settings_restored=True, errors=[]))
    validate_hardware(actual_hardware)
    hashes(actual_sources, SOURCE_FILES)
    exact(actual_sources[PACKAGE_PREFIX+'lite/prefix_fp8.py'],SHARED_B1_OWNER_SHA256,'winning shared owner bytes')
    hashes(actual_torch_files, (*DECLARATIONS, *BINARIES))
    exact(actual_native_manifest,expected_native_manifest(actual_sources),'native operator declarations')
    for name, value in expected.items(): exact(payload[name], value, 'backend report.' + name)
    if type(payload['rows']) is not list or len(payload['rows']) != 4:
        raise ValueError('B32 backend must execute exactly four new shapes')
    seen = set()
    for row in payload['rows']:
        keys(row, ('shape', 'checks', 'hashes', 'quantized_fp32_oracle', 'timing_ms', 'sample_arrays'), 'shape row')
        shape = row['shape']
        keys(shape, ('batch', 'capacity', 'm', 'k', 'n'), 'shape')
        item = (shape['m'], shape['k'], shape['n'])
        if any(type(shape[key]) is not int for key in shape) or item in seen or shape['capacity'] not in CAPACITIES:
            raise ValueError('B32 shape is duplicate, noninteger or unsupported')
        if shape['batch'] != BATCH_SIZE or shape['m'] != BATCH_SIZE * shape['capacity'] or item[1:] not in GEOMETRIES:
            raise ValueError('B32 actual flattened M/K/N differs')
        seen.add(item)
        keys(row['sample_arrays'],('path','sha256'),'saved actual sample arrays')
        exact(row['sample_arrays']['path'],f'RAW-{shape["m"]}-{shape["k"]}-{shape["n"]}.npz','sample filename')
        sha(row['sample_arrays']['sha256'])
        exact(row['checks'], {name: True for name in CHECKS}, 'shape executed checks')
        hashes(row['hashes'], ('input', 'weight', 'quantized_input', 'activation_scale', 'output', 'refreshed_output'))
        oracle = row['quantized_fp32_oracle']
        keys(oracle, ('passed', 'atol', 'rtol', 'violations', 'max_abs', 'rmse'), 'quantized FP32 oracle')
        exact({key:oracle[key] for key in ('passed','atol','rtol','violations')},
              dict(passed=True, atol=0.03125, rtol=0.015625, violations=0), 'FP32 oracle rule')
        for key in ('max_abs', 'rmse'):
            if type(oracle[key]) not in (int, float) or not math.isfinite(oracle[key]) or oracle[key] < 0:
                raise ValueError('Nonfinite B32 numerical evidence')
        keys(row['timing_ms'], ('eager_conversion_inclusive', 'compiled_graph_conversion_inclusive'), 'raw timings')
        for values in row['timing_ms'].values():
            if type(values) is not list or len(values) != 10 or any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('B32 backend requires actual ten raw per-shape timings')
    exact(sorted(seen), sorted((m,k,n) for m in ROWS for k,n in GEOMETRIES), 'complete B32 M/K/N coverage')
    return payload


def validate_manifest(value):
    """Pure declaration/site validation; callers separately audit live tensors."""
    keys(value, ('schema', 'precision', 'implementation', 'checkpoint_sha256', 'scope', 'batch',
        'per_slot_capacities', 'flattened_rows', 'origin_b1', 'backend', 'precision_boundaries',
        'nonlinear_boundary', 'lifecycle', 'sites'), 'live installation manifest')
    for key, expected in dict(schema=1, precision='mixed_bf16_fp8_prefix', implementation='pi05-b32-prefix-rowwise-fp8-v1',
        checkpoint_sha256=CHECKPOINT_SHA256, scope='prefix_mlp_layers_0_16', batch=32,
        per_slot_capacities=list(CAPACITIES), flattened_rows=list(ROWS), origin_b1=ORIGIN_B1,
        nonlinear_boundary='independent BF16 gate/up -> gelu_tanh_mul -> BF16 down input').items():
        exact(value[key], expected, 'installation.' + key)
    exact(value['precision_boundaries'],PRECISION_BOUNDARIES,'all unchanged precision boundaries')
    backend=value['backend']
    keys(backend,('report_sha256','hardware','runtime','source_files_sha256','torch_files_sha256',
                  'native_manifest','executed_rows_sha256'),'installation backend')
    sha(backend['report_sha256']); sha(backend['executed_rows_sha256'])
    validate_hardware(backend['hardware']); exact(backend['runtime'],RUNTIME,'installation runtime')
    hashes(backend['source_files_sha256'],SOURCE_FILES); hashes(backend['torch_files_sha256'],(*DECLARATIONS,*BINARIES))
    exact(backend['source_files_sha256'][PACKAGE_PREFIX+'lite/prefix_fp8.py'],SHARED_B1_OWNER_SHA256,'shared B1 source')
    exact(backend['native_manifest'],expected_native_manifest(backend['source_files_sha256']),'installed native manifest')
    exact(value['lifecycle'],dict(version='pi05-b32-prefix-fp8-live-owner-v1',
        source_sha256=backend['source_files_sha256'][PACKAGE_PREFIX+'lite/batch_prefix_fp8.py'],enabled=True,
        owner='Pi05NativeGraphRunner',original_parameters_preserved=True,
        content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
        live_checks=LIVE_CHECKS,fallback_permitted=False,
        serving_scope='ordinary_prepare_full_live_checks;private_sealed_engine_cold_and_explicit_full_audit'),
        'live lifecycle declaration')
    if type(value['sites']) is not list or len(value['sites']) != 51:
        raise ValueError('B32 prefix manifest requires actual ordered51 sites')
    for index, site in enumerate(value['sites']):
        projection = ('gate_proj','up_proj','down_proj')[index % 3]
        shape = [2048,16384] if index % 3 == 2 else [16384,2048]
        keys(site, ('name','module_path','module_type','bias','original_weight','quantized_weight','weight_scale'), 'site')
        for key, expected in dict(name=f'prefix.{index//3}.{projection}',
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{index//3}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None).items(): exact(site[key], expected, 'site.'+key)
        for key, dtype, expected_shape in (('original_weight','bfloat16',shape), ('quantized_weight','float8_e4m3fn',shape),
                                          ('weight_scale','float32',[shape[0],1])):
            record=site[key]; keys(record, ('shape','stride','dtype','content_sha256'), 'tensor record')
            exact(record['shape'],expected_shape,'tensor shape'); exact(record['dtype'],dtype,'tensor dtype')
            exact(record['stride'],[expected_shape[1],1] if key!='weight_scale' else [1,1],'tensor stride')
            sha(record['content_sha256'])
    return value
