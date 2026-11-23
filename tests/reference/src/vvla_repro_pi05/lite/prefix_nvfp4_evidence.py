"""Pure NVFP4 input/calibration/support evidence checks; never imports Torch.

Passing these checks does not establish a live installation or policy quality.
The owner separately binds actual original bytes, runtime, source and tensors.
"""
import json
import hashlib
import math
from pathlib import PurePosixPath
import re
import struct

PRECISION = 'mixed_bf16_nvfp4_prefix'
VARIANT = 'vllm032-cutlass-nvfp4-sm120-v1'
CALIBRATION_VERSION = 'pi05-prefix-nvfp4-single-fixture-calibration-v1'
CALIBRATION_FORMULA = 'torch.div(FP32_tensor([2688.0]), FP32_tensor([max(actual_input_amax, 1e-12)])) on the selected CUDA device'
WEIGHT_FORMULA = 'torch.div(FP32_tensor([2688.0]), clamp_min(amax(abs(weight.float())),1e-12).reshape(1))'
ALPHA_FORMULA = 'torch.div(FP32_tensor([1.0]), FP32(input_global_scale * weight_global_scale))'
CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
SITE_ORDER = tuple(f'prefix.{layer}.{projection}' for layer in range(17)
                  for projection in ('gate_proj', 'up_proj', 'down_proj'))
CONTROL_KEYS = ('recipe_sha256', 'config_sha256', 'identity_sha256', 'content_sha256')
RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0], vllm='0.32.0')
EXTENSION_SHA256 = 'ae9c8a293020e422e0b6b61d1024263dc476a772337f5d39d46d18baa7ce0ad9'
WHEEL_SHA256 = '5e2ecdc5b0baf72921c288486417ca82778456a448e4fe60b62ee43c0dfc8611'
NATIVE_SCHEMAS = dict(
    quantizer='_C::scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout, *, Tensor(a!) output, Tensor(b!) output_scale) -> ()',
    gemm='_C::cutlass_scaled_fp4_mm(Tensor($0! -> ) out, Tensor a, Tensor b, Tensor block_scale_a, Tensor block_scale_b, Tensor alpha) -> ()')


def expected_native_implementation(source_files_sha256):
    """Exact frozen primitive declaration, with actual source hashes supplied."""
    return dict(version='pi05-prefix-nvfp4-native-sm120-v1', variant=VARIANT,
        implementation='vllm_stable_native_nvfp4_opaque_v1', source_files_sha256=source_files_sha256,
        required_runtime=RUNTIME, native_extension_sha256=EXTENSION_SHA256, native_wheel_sha256=WHEEL_SHA256,
        supported_capacities=[576, 640], supported_geometries=[[2048, 16384], [16384, 2048]],
        compile_options={'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False},
        operators=dict(quantizer='vvla_repro_pi05_lite::prefix_nvfp4_quantize_rows_v1',
                       gemm='vvla_repro_pi05_lite::prefix_nvfp4_native_scaled_mm_v1'),
        native_operators=['_C::scaled_fp4_quant.out', '_C::cutlass_scaled_fp4_mm'],
        dispatch='pinned vLLM stable extension CUDA operators inside opaque custom ops',
        input_dtype='bfloat16', packed_dtype='uint8', packed_encoding='two E2M1 values per byte',
        block_scale_dtype='float8_e4m3fn', block_size=16, global_scale_dtype='float32',
        accumulator_dtype='float32', output_dtype='bfloat16', weight_layout='contiguous_NK_packed_K/2',
        scale_layout='CUTLASS swizzle128x4; paddedM=ceil(M/128)*128; paddedK=ceil(K/64)*64',
        scale_padding='zero-initialized int32 storage on every call; included in online conversion',
        global_scale='caller-owned frozen calibration; typical 2688/amax, never fitted in a hot call',
        alpha='caller-owned FP32 1/(activation_global_scale*weight_global_scale)', bias=None,
        fallback_permitted=False, installation='none; no 51-site binding or quality claim',
        backend_evidence='new same-card actual-wrapper primitive report required')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(actual, expected, label):
    if type(actual) is not type(expected) or canonical(actual) != canonical(expected):
        raise ValueError('NVFP4 evidence mismatch: ' + label)


def keys(value, required, label):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError('Unexpected/missing NVFP4 fields: ' + label)


def sha(value, label):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid NVFP4 SHA256: ' + label)


def control(value):
    keys(value, CONTROL_KEYS, 'bf16_control')
    for name in CONTROL_KEYS:
        sha(value[name], 'bf16_control.' + name)


def validate_evidence_config(value, *, backend_report_sha256=None, calibration_sha256=None, bf16_control=None):
    keys(value, ('backend_report', 'backend_report_sha256', 'calibration', 'calibration_sha256', 'bf16_control'),
         'prefix_nvfp4_evidence')
    for name in ('backend_report', 'calibration'):
        if type(value[name]) is not str or not value[name] or '\0' in value[name]:
            raise ValueError('NVFP4 evidence requires an explicit path: ' + name)
        sha(value[name + '_sha256'], name + '_sha256')
    control(value['bf16_control'])
    for name, expected in (('backend_report_sha256', backend_report_sha256),
                           ('calibration_sha256', calibration_sha256), ('bf16_control', bf16_control)):
        if expected is not None:
            exact(value[name], expected, 'configuration binding ' + name)
    return value


def _hardware(value):
    keys(value, ('name', 'uuid', 'driver_version', 'memory_mib', 'hostname'), 'hardware')
    exact(value['name'], 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'SM120 card name')
    if (type(value['uuid']) is not str
            or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value['uuid']) is None
            or type(value['memory_mib']) is not int or value['memory_mib'] <= 0
            or any(type(value[name]) is not str or not value[name] for name in ('hostname', 'driver_version'))):
        raise ValueError('Invalid physical NVFP4 hardware evidence')


def _uuid_bytes(value):
    """Torch may omit nvidia-smi's GPU- prefix; retain both original strings."""
    if type(value) is not str or re.fullmatch(
            r'(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value) is None:
        raise ValueError('Invalid native actual GPU UUID syntax')
    return bytes.fromhex(value.removeprefix('GPU-').replace('-', ''))


def validate_calibration(value):
    keys(value, ('schema', 'version', 'precision', 'scope', 'site_order', 'capacity', 'checkpoint_sha256',
        'hardware', 'control', 'bf16_control', 'activation_report_sha256', 'single_fixture',
        'quality_validated', 'calibration_formula', 'sites'), 'calibration')
    for name, expected in dict(schema=1, version=CALIBRATION_VERSION, precision=PRECISION,
            scope='prefix_mlp_layers_0_16', site_order=list(SITE_ORDER), capacity=576,
            checkpoint_sha256=CHECKPOINT_SHA256, single_fixture=True, quality_validated=False,
            calibration_formula=CALIBRATION_FORMULA).items():
        exact(value[name], expected, 'calibration.' + name)
    _hardware(value['hardware'])
    control(value['bf16_control'])
    sha(value['activation_report_sha256'], 'activation_report_sha256')
    provenance = value['control']
    if type(provenance) is not dict:
        raise ValueError('NVFP4 calibration needs original input/control provenance')
    exact(provenance.get('episode_id'), 'libero_spatial/0/0', 'calibration single fixture')
    sha(provenance.get('config_sha256'), 'control raw config file SHA256')
    for source, target in (('config_canonical_sha256', 'config_sha256'), ('formal_identity_sha256', 'identity_sha256'),
                            ('formal_content_sha256', 'content_sha256')):
        exact(provenance.get(source), value['bf16_control'][target], 'calibration/control ' + source)
    for name in ('formal_frozen_sha256', 'preflight_identity_sha256', 'preflight_inventory_sha256',
                 'noise_sha256', 'raw_observation_sha256'):
        sha(provenance.get(name), 'control.' + name)
    for name in ('raw_fixture', 'prepared_fixture'):
        record = provenance.get(name)
        keys(record, ('path', 'sha256'), 'control.' + name)
        sha(record['sha256'], name)
        if type(record['path']) is not str or not record['path']:
            raise ValueError('Missing original fixture path')
    keys(value['sites'], SITE_ORDER, 'calibration.sites')
    for index, name in enumerate(SITE_ORDER):
        row = value['sites'][name]
        keys(row, ('input_amax', 'input_floor_applied', 'floor', 'input_global_scale', 'input_npz',
                   'input_sha256', 'input_shape', 'weight_sha256', 'weight_shape', 'checkpoint_suffix'), name)
        amax = row['input_amax']
        if type(amax) is not float or not math.isfinite(amax) or amax < 0:
            raise ValueError('Actual finite nonnegative input amax required: ' + name)
        exact(row['floor'], 1e-12, name + '.floor')
        exact(row['input_floor_applied'], amax < 1e-12, name + '.input_floor_applied')
        scale = row['input_global_scale']
        keys(scale, ('value', 'fp32_bits'), name + '.input_global_scale')
        if (type(scale['value']) is not float or not math.isfinite(scale['value']) or scale['value'] <= 0
                or type(scale['fp32_bits']) is not str or re.fullmatch('0x[0-9a-f]{8}', scale['fp32_bits']) is None):
            raise ValueError('Finite positive actual FP32 calibration scale required: ' + name)
        decoded = struct.unpack('>f', int(scale['fp32_bits'], 16).to_bytes(4, 'big'))[0]
        if decoded != scale['value']:
            raise ValueError('Calibration FP32 bits/value disagree: ' + name)
        k, n = (16384, 2048) if index % 3 == 2 else (2048, 16384)
        exact(row['input_shape'], [1, 576, k], name + '.input_shape')
        exact(row['weight_shape'], [n, k], name + '.weight_shape')
        projection = name.rsplit('.', 1)[1]
        exact(row['checkpoint_suffix'], f'paligemma.model.language_model.layers.{index // 3}.mlp.{projection}.weight',
              name + '.checkpoint_suffix')
        keys(row['input_npz'], ('path', 'sha256'), name + '.input_npz')
        # The actual tap stores shared gate/up inputs once, under gate_proj.
        # Either same-layer gate/up artifact is legal; the original path/hash
        # remain exact in validate_calibration_binding. Down has its own input.
        projections = ('down_proj',) if projection == 'down_proj' else ('gate_proj', 'up_proj')
        expected_files = tuple(f'prefix_{index // 3:02d}_{part}.npz' for part in projections)
        if type(row['input_npz']['path']) is not str or row['input_npz']['path'] not in expected_files:
            raise ValueError('Unexpected calibration input artifact path: ' + name)
        for record, field in ((row, 'input_sha256'), (row, 'weight_sha256'), (row['input_npz'], 'sha256')):
            sha(record[field], name + '.' + field)
    return value


def validate_backend_report(payload, native_manifest, hardware):
    """Bind the completed direct-ops support smoke to an actual cold backend.

    Its synthetic hot-weight latency is intentionally ignored. It does not
    certify this wrapper's CUDA contracts, calibration, or closed-loop quality.
    """
    if type(payload) is not dict:
        raise ValueError('NVFP4 backend report must be an object')
    for name, expected in dict(status='complete', support_verified=True, diagnostic_only=True,
                              policy_executed=False, quality_passed=None, synthetic_inputs=True).items():
        exact(payload.get(name), expected, 'native smoke.' + name)
    sha(payload.get('source_sha256'), 'native smoke.source_sha256')
    _hardware(hardware)
    expected = dict(torch_version=native_manifest['runtime']['torch'], torch_git=native_manifest['runtime']['torch_git'],
        cuda=native_manifest['runtime']['cuda'], capability=[12, 0], gpu_name=hardware['name'], uuid=hardware['uuid'],
        vllm_version=native_manifest['runtime']['vllm'], extension=native_manifest['files']['extension_path'],
        extension_sha256=native_manifest['files']['extension_sha256'],
        schemas=[native_manifest['native_schemas']['quantizer'], native_manifest['native_schemas']['gemm']])
    exact(payload.get('backend'), expected, 'actual native ABI/runtime/hardware')
    rows = payload.get('rows')
    if type(rows) is not list or len(rows) != 4:
        raise ValueError('Complete four-shape NVFP4 native support report required')
    shapes = ((576, 2048, 16384), (640, 2048, 16384), (576, 16384, 2048), (640, 16384, 2048))
    for row, shape in zip(rows, shapes, strict=True):
        if type(row) is not dict:
            raise ValueError('Malformed native smoke row')
        exact([row.get(name) for name in ('m', 'k', 'n')], list(shape), 'native smoke shape')
        exact(row.get('analytic_exact'), True, 'native smoke analytic')
        exact(row.get('capture_refresh_restore_exact'), True, 'native smoke capture/refresh')
    return payload


def validate_backend_payload(payload, backend, hardware):
    validate_backend_manifest(backend)
    exact(backend['hardware'], hardware, 'backend physical hardware')
    exact(payload.get('source_sha256'), backend['report_source_sha256'], 'backend original report source')
    return validate_backend_report(payload, backend['native_runtime'], hardware)


def validate_backend_manifest(value):
    keys(value, ('variant', 'report_sha256', 'report_source_sha256', 'hardware', 'native_runtime',
                 'support_scope', 'implementation'), 'backend')
    exact(value['variant'], VARIANT, 'backend.variant')
    for name in ('report_sha256', 'report_source_sha256'):
        sha(value[name], name)
    _hardware(value['hardware'])
    exact(value['support_scope'], 'direct native operators; policy quality and actual wrapper contracts are separate',
          'backend.support_scope')
    runtime = value['native_runtime']
    keys(runtime, ('version', 'variant', 'runtime', 'device_index', 'files', 'imported_modules', 'native_schemas',
                   'hardware', 'implementation', 'import_succeeded', 'execution_validated', 'quality_validated'), 'native_runtime')
    exact(runtime['version'], 'pi05-prefix-nvfp4-native-sm120-v1', 'native runtime version')
    exact(runtime['variant'], VARIANT, 'native runtime variant')
    exact(runtime['runtime'], dict(torch='2.13.0+cu130', cuda='13.0',
        torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0], vllm='0.32.0'), 'native runtime versions')
    if type(runtime['device_index']) is not int or runtime['device_index'] < 0:
        raise ValueError('Actual native CUDA device index required')
    for name, expected in (('import_succeeded', True), ('execution_validated', False), ('quality_validated', False)):
        exact(runtime[name], expected, 'native loader evidence scope ' + name)
    files = runtime['files']
    keys(files, ('native_path', 'extension_path', 'extension_sha256', 'installed_source_files',
                  'installed_record_sha256', 'preparation_receipt_sha256', 'archive_identity_scope'), 'native files')
    exact(files['extension_sha256'], 'ae9c8a293020e422e0b6b61d1024263dc476a772337f5d39d46d18baa7ce0ad9',
          'pinned native binary')
    for name in ('native_path', 'extension_path'):
        if type(files[name]) is not str or not PurePosixPath(files[name]).is_absolute():
            raise ValueError('Native file provenance requires actual absolute paths')
    sha(files['installed_record_sha256'], 'native installed RECORD')
    if files['preparation_receipt_sha256'] is not None:
        sha(files['preparation_receipt_sha256'], 'native preparation receipt')
    exact(files['archive_identity_scope'], 'receipt reference only; archive is not re-read by this primitive loader', 'native archive scope')
    if type(files['installed_source_files']) is not dict or not files['installed_source_files']:
        raise ValueError('Actual native source-file evidence required')
    for path, record in files['installed_source_files'].items():
        if type(path) is not str or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            raise ValueError('Unsafe native source record')
        keys(record, ('sha256', 'bytes'), 'native source file')
        sha(record['sha256'], 'native source file')
        if type(record['bytes']) is not int or record['bytes'] < 0:
            raise ValueError('Invalid native source size')
    root = PurePosixPath(files['native_path'])
    extension = PurePosixPath(files['extension_path'])
    if (extension.parent != root / 'vllm'
            or extension.name != '_C_stable_libtorch.abi3.so'):
        raise ValueError('Pinned native extension path differs from the explicit native root')
    extension_record = files['installed_source_files'].get(extension.relative_to(root).as_posix())
    if extension_record is None or extension_record['sha256'] != files['extension_sha256']:
        raise ValueError('Native binary is not bound to the installed source inventory')
    imported = runtime['imported_modules']
    if (type(imported) is not dict or not {'vllm', 'vllm._C_stable_libtorch'} <= set(imported)):
        raise ValueError('Actual imported native module evidence required')
    for name, record in imported.items():
        if type(name) is not str or (name != 'vllm' and not name.startswith('vllm.')):
            raise ValueError('Unexpected imported native module')
        keys(record, ('path', 'sha256'), 'imported native module')
        if type(record['path']) is not str or not PurePosixPath(record['path']).is_relative_to(root):
            raise ValueError('Imported native module escapes its explicit root')
        relative = PurePosixPath(record['path']).relative_to(root).as_posix()
        source = files['installed_source_files'].get(relative)
        if source is None or source['sha256'] != record['sha256']:
            raise ValueError('Imported native module is not bound to the installed byte inventory')
    exact(imported['vllm']['path'], str(root / 'vllm/__init__.py'), 'actual package initializer')
    exact(imported['vllm._C_stable_libtorch']['path'], str(extension), 'actual imported extension')
    exact(runtime['native_schemas'], NATIVE_SCHEMAS, 'fixed actual native operator schemas')
    keys(runtime['hardware'], ('name', 'uuid', 'total_memory', 'capability'), 'native hardware')
    if type(runtime['hardware']['total_memory']) is not int or runtime['hardware']['total_memory'] <= 0:
        raise ValueError('Actual native GPU memory required')
    exact(runtime['hardware']['name'], value['hardware']['name'], 'native actual GPU name')
    if _uuid_bytes(runtime['hardware']['uuid']) != _uuid_bytes(value['hardware']['uuid']):
        raise ValueError('NVFP4 evidence mismatch: native actual GPU UUID')
    exact(runtime['hardware']['capability'], [12, 0], 'native actual SM')
    implementation = value['implementation']
    exact(implementation, runtime['implementation'], 'actual wrapper source/implementation')
    for name, expected in dict(version='pi05-prefix-nvfp4-native-sm120-v1', variant=VARIANT,
        implementation='vllm_stable_native_nvfp4_opaque_v1', fallback_permitted=False, input_dtype='bfloat16',
        output_dtype='bfloat16', packed_dtype='uint8', block_scale_dtype='float8_e4m3fn', block_size=16,
        global_scale_dtype='float32', accumulator_dtype='float32', supported_capacities=[576, 640],
        supported_geometries=[[2048, 16384], [16384, 2048]],
        compile_options={'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}).items():
        exact(implementation.get(name), expected, 'wrapper ' + name)
    keys(implementation['source_files_sha256'], ('prefix_nvfp4_native.py',), 'actual wrapper source')
    sha(implementation['source_files_sha256']['prefix_nvfp4_native.py'], 'actual wrapper source')
    exact(implementation, expected_native_implementation(implementation['source_files_sha256']),
          'entire actual primitive implementation declaration')
    return value


def validate_calibration_binding(payload, manifest, hardware):
    """Bind parsed original calibration to the installed 51-site declaration.

    The caller must first hash the original bytes against manifest.calibration
    sha256. This pure check deliberately does not claim live tensor ownership.
    """
    validate_quantization_manifest(manifest)
    validate_calibration(payload)
    exact(payload['hardware'], hardware, 'calibration actual physical GPU')
    exact(manifest['backend']['hardware'], hardware, 'manifest actual physical GPU')
    summary = manifest['calibration']
    for field in ('version', 'activation_report_sha256', 'scope', 'single_fixture', 'capacity',
                  'quality_validated', 'bf16_control'):
        exact(summary[field], payload[field], 'calibration summary.' + field)
    exact(summary['formula'], payload['calibration_formula'], 'calibration formula')
    for row in manifest['sites']:
        original = payload['sites'][row['name']]
        exact(row['calibration'], original, row['name'] + '.original calibration record')
        exact(row['original_weight']['content_sha256'], original['weight_sha256'],
              row['name'] + '.packed source weight SHA')
    return payload


def _tensor_record(value, shape, dtype, label):
    keys(value, ('shape', 'stride', 'dtype', 'content_sha256'), label)
    exact(value['shape'], shape, label + '.shape')
    exact(value['stride'], [shape[1], 1] if len(shape) == 2 else [1], label + '.stride')
    exact(value['dtype'], dtype, label + '.dtype')
    sha(value['content_sha256'], label + '.content_sha256')


def validate_quantization_manifest(value):
    keys(value, ('schema', 'precision', 'implementation', 'checkpoint_sha256', 'scope', 'precision_boundaries',
                 'backend', 'calibration', 'quantizer', 'lifecycle', 'sites'), 'manifest')
    for name, expected in dict(schema=1, precision=PRECISION, implementation='pi05-prefix-nvfp4-cutlass-v1',
                               checkpoint_sha256=CHECKPOINT_SHA256, scope='prefix_mlp_layers_0_16').items():
        exact(value[name], expected, 'manifest.' + name)
    exact(value['precision_boundaries'], dict(vision='unchanged_selected_bf16_control',
        prefix_attention='unchanged_selected_bf16_control', expert_projections='unchanged_selected_bf16_control',
        normalization='unchanged_selected_control_FP32_boundaries', residual_euler='unchanged_selected_control_FP32_boundaries',
        action_time_projections='unchanged_selected_control_FP32_boundaries'), 'unmodified precision boundaries')
    validate_backend_manifest(value['backend'])
    cal = value['calibration']
    keys(cal, ('sha256', 'version', 'activation_report_sha256', 'scope', 'single_fixture', 'capacity',
               'quality_validated', 'formula', 'calibration_source_scope', 'bf16_control'), 'manifest calibration')
    sha(cal['sha256'], 'calibration raw bytes')
    sha(cal['activation_report_sha256'], 'calibration actual input report')
    control(cal['bf16_control'])
    for name, expected in dict(version=CALIBRATION_VERSION, scope='prefix_mlp_layers_0_16', single_fixture=True,
        capacity=576, quality_validated=False, formula=CALIBRATION_FORMULA,
        calibration_source_scope='one BF16 fixture, not formal calibration').items():
        exact(cal[name], expected, 'manifest calibration.' + name)
    quantizer = value['quantizer']
    keys(quantizer, ('source_sha256', 'nonlinear_source_sha256', 'variant', 'format', 'input_dtype', 'output_dtype',
        'accumulator_dtype', 'weight_layout', 'block_size', 'scale_layout', 'input_global_scale_formula',
        'weight_global_scale_formula', 'alpha_formula', 'online_conversion', 'weight_packing', 'gate_up', 'nonlinear_boundary'), 'quantizer')
    exact(quantizer['source_sha256'], digest(value['backend']['implementation']['source_files_sha256']), 'quantizer source digest')
    sha(quantizer['nonlinear_source_sha256'], 'original GELU source')
    for name, expected in dict(variant=VARIANT,
        format='NVFP4 E2M1 values plus E4M3 block scales and FP32 global scales', input_dtype='bfloat16',
        output_dtype='bfloat16', accumulator_dtype='float32', weight_layout='contiguous_NK_packed_K/2', block_size=16,
        scale_layout='CUTLASS swizzle128x4 padded per native ABI', input_global_scale_formula=CALIBRATION_FORMULA,
        weight_global_scale_formula=WEIGHT_FORMULA, alpha_formula=ALPHA_FORMULA, online_conversion=True,
        weight_packing='cold_only', gate_up='independent_quantization',
        nonlinear_boundary='BF16 gate/up -> gelu_tanh_mul -> BF16 down input').items():
        exact(quantizer[name], expected, 'quantizer.' + name)
    lifecycle = value['lifecycle']
    keys(lifecycle, ('version', 'source_sha256', 'enabled', 'owner', 'original_parameters_preserved',
                    'content_hash_scope', 'live_checks', 'fallback_permitted'), 'lifecycle')
    sha(lifecycle['source_sha256'], 'actual owner source')
    for name, expected in dict(version='pi05-prefix-nvfp4-live-owner-v1', enabled=True, owner='Pi05NativeGraphRunner',
        original_parameters_preserved=True, content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
        live_checks=['module_forward_bindings', 'original_weight_identity', 'quantized_weight_identity',
            'weight_scale_identity', 'backend_identity', 'unique_runner_generation', 'no_extra_quantized_sites',
            'unchanged_nonprefix_precision', 'supported_capacity_dispatch', 'input_global_scale_identity',
            'weight_global_scale_identity', 'alpha_identity', 'calibration_byte_identity'], fallback_permitted=False).items():
        exact(lifecycle[name], expected, 'lifecycle.' + name)
    if type(value['sites']) is not list or len(value['sites']) != 51:
        raise ValueError('Exact 51-site NVFP4 installation required')
    for index, (row, name) in enumerate(zip(value['sites'], SITE_ORDER, strict=True)):
        keys(row, ('name', 'module_path', 'module_type', 'bias', 'original_weight', 'quantized_weight',
                   'weight_scale', 'input_global_scale', 'weight_global_scale', 'alpha', 'scalar_values', 'calibration'), name)
        k, n = (16384, 2048) if index % 3 == 2 else (2048, 16384)
        projection = name.rsplit('.', 1)[1]
        for field, expected in dict(name=name,
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{index // 3}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None).items():
            exact(row[field], expected, name + '.' + field)
        _tensor_record(row['original_weight'], [n, k], 'bfloat16', name + '.original_weight')
        _tensor_record(row['quantized_weight'], [n, k // 2], 'uint8', name + '.quantized_weight')
        _tensor_record(row['weight_scale'], [n, k // 16], 'float8_e4m3fn', name + '.weight_scale')
        keys(row['scalar_values'], ('input_global_scale', 'weight_global_scale', 'alpha'), name + '.scalar_values')
        for field in ('input_global_scale', 'weight_global_scale', 'alpha'):
            _tensor_record(row[field], [1], 'float32', name + '.' + field)
            scalar = row['scalar_values'][field]
            keys(scalar, ('value', 'fp32_bits'), name + '.' + field)
            if (type(scalar['value']) is not float or not math.isfinite(scalar['value']) or scalar['value'] <= 0
                    or type(scalar['fp32_bits']) is not str or re.fullmatch('0x[0-9a-f]{8}', scalar['fp32_bits']) is None):
                raise ValueError('Invalid actual NVFP4 scalar record')
            raw = int(scalar['fp32_bits'], 16).to_bytes(4, 'little')
            exact(struct.unpack('<f', raw)[0], scalar['value'], name + '.' + field + ' bits')
            exact(hashlib.sha256(raw).hexdigest(), row[field]['content_sha256'], name + '.' + field + ' raw hash')
        exact(row['scalar_values']['input_global_scale'], row['calibration']['input_global_scale'], name + '.frozen input scale')
        exact(row['original_weight']['content_sha256'], row['calibration']['weight_sha256'], name + '.calibration weight')
    return value
