"""Pure evidence contract for optional 54-site vision FP8, no ML imports.

Executed primitive support is explicitly distinct from real-policy quality.
The original report binds the actual two source files and installed binary;
a later source payload may add an owner without rewriting that old report.
"""
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'pi05-vision-54-fp8-live-owner-v2'
VISION_BOUNDARY = '54_fc1_fc2_FP8_M512; BF16_M256_only_verified_cold_empty_camera; original_activation_exact_owned_bias'
SHAPES = [[512, 1152, 4304], [512, 4304, 1152]]
CORE_FILES = ('vision_fp8_native.py', 'vision_fp8_native_kernel.py')
RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
LIFECYCLE = dict(owner='Pi05NativeGraphRunner', base_owner='immutable_51_site_prefix_fp8',
    native_collector='v3', original_parameters_preserved=True, original_activation_preserved=True,
    execution_bias='cold exact BF16 clone; independent ordinary nongrad storage; original and clone epochs both guarded',
    live_checks=['54_original_weights', '54_original_biases', '54_quantized_weights', '54_weight_scales',
                 '54_independent_execution_biases',
                 'ordered_modules_and_forwards', 'activation_objects_and_forwards', 'base_epoch',
                 'native_v3_owner', 'runtime_and_math', 'unique_runner_generation_thread'],
    content_hash_scope='immutable_cold_epoch; mutation/invalidation requires graph retirement and reinstall',
    retirement_order=['graph', 'vision_fp8', 'base_fp8_owner', 'native_lifecycle'])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value, expected, label):
    if canonical(value) != canonical(expected): raise ValueError('Vision FP8 evidence mismatch: ' + label)


def keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected): raise ValueError('Vision FP8 fields: ' + label)


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Vision FP8 requires SHA256')
    return value


def validate_evidence_config(value):
    keys(value, ('backend_report', 'backend_report_sha256'), 'config')
    if (type(value['backend_report']) is not str or '\x00' in value['backend_report']
            or not Path(value['backend_report']).is_absolute()):
        raise ValueError('Vision FP8 backend_report must be an explicit absolute path')
    sha(value['backend_report_sha256'])
    return value


def validate_contract_record(record):
    keys(record, ('report_sha256', 'source_sha256', 'implementation', 'runtime',
                  'installed_binary_sha256', 'hardware', 'contract_math', 'scope'), 'contracts')
    sha(record['report_sha256']); sha(record['source_sha256'])
    exact(record['scope'], 'synthetic_primitive_support_only; not quality or performance', 'scope')
    keys(record['runtime'], (*RUNTIME, 'triton'), 'runtime')
    exact({k: record['runtime'][k] for k in RUNTIME}, RUNTIME, 'runtime')
    if type(record['runtime']['triton']) is not str or not record['runtime']['triton']:
        raise ValueError('Missing actual Triton version')
    keys(record['installed_binary_sha256'], ('lib/libtorch_cuda.so',), 'binary')
    sha(record['installed_binary_sha256']['lib/libtorch_cuda.so'])
    keys(record['hardware'], ('name', 'uuid', 'driver_version', 'memory_mib', 'hostname'), 'hardware')
    hardware = record['hardware']
    if (hardware['name'] != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
            or type(hardware['uuid']) is not str or not re.fullmatch('GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', hardware['uuid'])
            or type(hardware['memory_mib']) is not int or hardware['memory_mib'] < 1
            or any(type(hardware[k]) is not str or not hardware[k] for k in ('driver_version', 'hostname'))):
        raise ValueError('Wrong vision physical GPU identity')
    exact(record['contract_math'], dict(tf32=False, precision='highest', use_fast_accum=False,
        compile_options=OPTIONS), 'contract math')
    core = record['implementation']
    keys(core, ('version', 'variant', 'implementation', 'source_files_sha256', 'required_runtime',
        'supported_m', 'supported_geometries', 'quantizer_arithmetic', 'compile_options', 'operators',
        'dispatch', 'weight_layout', 'scale_layout', 'input_dtype', 'quantized_dtype', 'scale_dtype',
        'accumulator_dtype', 'output_dtype', 'epilogue', 'bias', 'activation', 'use_fast_accum',
        'scale_result', 'fallback_permitted', 'cold_weight_packing', 'installation'), 'primitive')
    keys(core['source_files_sha256'], CORE_FILES, 'primitive sources')
    for value in core['source_files_sha256'].values(): sha(value)
    expected = dict(version='pi05-vision-biased-fp8-native-sm120-v1',
        variant='vision-biased-rowwise-fp8-native-sm120-v1', implementation='vision_eager_rn_native_opaque_bias_v1',
        required_runtime=RUNTIME, supported_m=512, supported_geometries=[[1152,4304],[4304,1152]],
        compile_options=OPTIONS, operators=dict(quantizer='vvla_repro_pi05_lite::vision_fp8_quantize_rows_v1',
            gemm='vvla_repro_pi05_lite::vision_fp8_native_biased_mm_v1'),
        dispatch='ATEN torch._scaled_mm inside opaque custom op', weight_layout='contiguous_NK_then_transpose',
        scale_layout='activation[M,1];weight[N,1]_then_transpose', input_dtype='bfloat16',
        quantized_dtype='float8_e4m3fn', scale_dtype='float32', accumulator_dtype='float32', output_dtype='bfloat16',
        epilogue='FP32 accumulator * activation scale * weight scale + original BF16 bias -> sole BF16 cast',
        bias='explicit original contiguous BF16[N]; no post-BF16 add',
        activation='unchanged original SigLIP activation, outside this primitive',
        use_fast_accum=False, scale_result=None, fallback_permitted=False,
        cold_weight_packing='Torch eager FP32 abs/amax/divide448/clamp_min/divide/clamp/E4M3fn',
        installation='none; no policy, site installation, performance or quality claim',
        quantizer_arithmetic=dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
            scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
            normalization='FP32 div.rn(value, scale)',
            output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
            padding='masked zero loads neutral to absmax; masked output stores; original K unchanged'))
    for key, value in expected.items(): exact(core[key], value, 'primitive.' + key)
    return record


def validate_backend_bytes(raw, extension_or_record, hardware=None):
    record = extension_or_record.get('contracts', extension_or_record)
    validate_contract_record(record)
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != record['report_sha256']:
        raise ValueError('Vision primitive requires unchanged original report bytes')
    report = json.loads(raw)
    for key, value in dict(version='vision-biased-fp8-native-sm120-contracts-v1', status='complete',
        support_verified=True, graph_retired=True, settings_restored=True, diagnostic_only=True,
        benchmark=False, synthetic_inputs=True, model_loaded=False, production_changed=False,
        quality_passed=False, shapes=SHAPES).items(): exact(report.get(key), value, key)
    for source, target in (('core_implementation_manifest', 'implementation'), ('runtime', 'runtime'),
                           ('hardware', 'hardware'), ('installed_binary_sha256', 'installed_binary_sha256'),
                           ('contract_math', 'contract_math')):
        exact(report.get(source), record[target], source)
    exact(report.get('source'), {'script_sha256':record['source_sha256']}, 'probe source')
    if hardware is not None: exact(record['hardware'], hardware, 'actual physical GPU')
    if (any(type(report.get(k)) not in (int,float) or not math.isfinite(report[k]) for k in ('started','finished'))
            or report['finished'] < report['started']): raise ValueError('Invalid primitive execution interval')
    if type(report.get('rows')) is not list or len(report['rows']) != 2: raise ValueError('Two actual shapes required')
    for row, shape in zip(report['rows'], SHAPES, strict=True):
        exact(row.get('shape'), shape, 'actual shape'); exact(row.get('status'), 'contracts_passed', 'shape status')
        exact(row.get('graph_retired'), True, 'shape retirement')
        for key in ('passed','raw_byte_exact','signed_zero','floor','last_valid_column_amax','masked_padding'):
            exact(row['quantization'].get(key), True, 'quantization.' + key)
        exact(row['quantization'].get('finite_bf16_patterns'), 65280, 'finite patterns')
        for key in ('passed','scale_axes_exact','bias_cancellation_exact'):
            exact(row['analytic'].get(key), True, 'analytic.' + key)
        for key in ('eager_exact','compiled_exact','capture_repeat_exact','same_pointer_refresh_exact',
                    'restoration_exact','retained_outputs_exact','cold_weight_version_counters'):
            exact(row['execution'].get(key), True, 'execution.' + key)
        tiles = row['high_precision'].get('tiles')
        exact(row['high_precision'].get('quality_gate'), False, 'not policy quality')
        if type(tiles) is not list or len(tiles) != 3: raise ValueError('Missing independent FP64 tiles')
        for tile, (r,c) in zip(tiles, ((0,0),(256,shape[2]//2),(504,shape[2]-16)), strict=True):
            for key, value in dict(row=r,column=c,shape=[8,16],full_k=shape[1],oracle_dtype='float64').items():
                exact(tile.get(key), value, 'FP64 tile.' + key)
            for metric in ('original_bf16_math','quantized_operand_math'):
                keys(tile[metric],('max_abs','rmse','relative_rmse'),'FP64 metrics')
                for value in tile[metric].values():
                    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                        raise ValueError('Nonfinite primitive numerical evidence')
            sha(tile['original_oracle_sha256']); sha(tile['packed_oracle_sha256'])
    return report


def tensor_record(record, shape, dtype):
    keys(record, ('shape','stride','dtype','content_sha256'), 'tensor record')
    stride = [shape[1],1] if len(shape) == 2 else [1]
    if len(shape) == 2 and shape[1] == 1: stride = [1,1]
    exact(record['shape'],shape,'tensor shape'); exact(record['stride'],stride,'tensor stride')
    exact(record['dtype'],dtype,'tensor dtype'); sha(record['content_sha256'])


def validate_vision_manifest(extension, backend=None):
    keys(extension, ('schema','version','enabled','scope','contracts','sites','lifecycle',
        'owner_source_sha256','evidence_validator_sha256','native_lifecycle_manifest_sha256',
        'base_owner_source_sha256','activation','dispatch','quality_validated','performance_validated'), 'extension')
    for key, value in dict(schema=1,version=VERSION,enabled=True,scope='vision_mlp_fc1_fc2_layers_0_26',
        lifecycle=LIFECYCLE,activation='original activation_fn object and original BF16 input/output boundary',
        dispatch=VISION_BOUNDARY,quality_validated=False,performance_validated=False).items():
        exact(extension[key],value,key)
    for key in ('owner_source_sha256','evidence_validator_sha256','native_lifecycle_manifest_sha256','base_owner_source_sha256'):
        sha(extension[key])
    contracts = validate_contract_record(extension['contracts'])
    if type(extension['sites']) is not list or len(extension['sites']) != 54: raise ValueError('Exactly54 vision sites required')
    for index, site in enumerate(extension['sites']):
        layer, name = index//2, ('fc1','fc2')[index%2]
        n,k = (4304,1152) if name == 'fc1' else (1152,4304)
        keys(site, ('name','module_path','module_type','original_weight','original_bias','execution_bias','quantized_weight','weight_scale'), 'site')
        exact(site['name'],f'vision.{layer}.{name}','site name')
        exact(site['module_path'],f'paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.{layer}.mlp.{name}','site path')
        exact(site['module_type'],'torch.nn.modules.linear.Linear','module type')
        for field, shape, dtype in (('original_weight',[n,k],'bfloat16'),('original_bias',[n],'bfloat16'),('execution_bias',[n],'bfloat16'),
                ('quantized_weight',[n,k],'float8_e4m3fn'),('weight_scale',[n,1],'float32')):
            tensor_record(site[field],shape,dtype)
        exact(site['execution_bias'],site['original_bias'],'cold exact bias clone bytes/layout')
    if backend is not None:
        exact(backend['quantizer_variant'],'triton-eager-rn-native-sm120-v1','base backend')
        exact(backend['hardware'],contracts['hardware'],'base physical GPU')
        exact(backend['installed_binary_sha256']['lib/libtorch_cuda.so'],contracts['installed_binary_sha256']['lib/libtorch_cuda.so'],'base binary')
        exact(dict(torch=backend['torch_version'],cuda=backend['cuda_version'],
                   torch_git=backend['torch_git_version'],capability=backend['capability']),RUNTIME,'base runtime')
        exact(backend['compile_options'],OPTIONS,'base compile options')
    return extension
