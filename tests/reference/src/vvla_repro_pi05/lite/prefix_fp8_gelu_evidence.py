"""Pure, explicit evidence for the optional GELU/FP8 primitive and owner.

No Torch imports. File hashes bind executed contracts, not model quality or a
speed claim. Offline audit uses saved bytes; loading additionally checks actual
installed candidate/reference/test source bytes against that executed snapshot.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET

VERSION = 'pi05-prefix-fp8-gelu-owned-17-v1'
SCOPE = 'primitive_contracts_only; not model quality or performance'
FILES = ('RESULT.json', 'SOURCE_PAYLOAD.json', 'cpu.xml', 'cuda.xml',
         'cpu-RESULT.json', 'cuda-RESULT.json', 'exit.json', 'ROOT_RETIREMENT.json')
LITE = 'vllm-vla-plugin/src/vvla_repro_pi05/lite/'
SOURCE_FILES = (LITE + 'prefix_fp8_gelu.py', LITE + 'prefix_fp8_gelu_kernel.py',
    LITE + 'prefix_fp8_native.py', LITE + 'prefix_fp8_native_kernel.py',
    'vllm-vla-plugin/src/vvla_repro_pi05/models/pi05/fused_mlp_pi05.py',
    'vllm-vla-plugin/tests/test_lite_prefix_fp8_gelu.py')
COVERAGE = {
    'cpu': {'test_exact_current_geometry_accepted_without_cuda_work': 2,
            'test_changed_geometry_or_precision_rejected_without_cast_or_fallback': 10,
            'test_new_manifest_binds_actual_sources_and_does_not_claim_existing_authorization': 1,
            'test_fake_fullgraph_keeps_one_converter_native_gemm_and_fresh_outputs': 2},
    'cuda': {'test_cuda_converter_matches_selected_gelu_and_quantizer_raw_bytes_and_ownership': 1,
             'test_cuda_compiled_capture_repeat_refresh_restore_matches_selected_down_exactly': 2},
}
RUNTIME = dict(torch='2.13.0+cu130', cuda='13.0',
    torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}
ARITHMETIC = dict(scale='FP32 multiply by rounded reciprocal, then clamp_min(1e-12)',
    scale_reciprocal_fp32=0.0022321429569274187, scale_reciprocal_fp32_bits='0x3b124925',
    normalization='FP32 div.rn(value, scale)',
    output_cast='E4M3fn round-to-nearest-even after clamp[-448,448]',
    scalar_division_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu')
LIFECYCLE = dict(owner='Pi05NativeGraphRunner', base_owner='immutable_51_site_prefix_fp8',
    packed_weights_reused=True, source_parameters_preserved=True, independent_activation_cache=False,
    request_local_outputs=True, full_base_live_checks_preserved=True, extension_compile_members=17,
    retirement_order=['graph', 'gelu_extension', 'base_fp8_owner'])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value, expected, field):
    if canonical(value) != canonical(expected): raise ValueError('GELU evidence mismatch: ' + field)


def keys(value, expected, field):
    if type(value) is not dict or set(value) != set(expected): raise ValueError('GELU evidence fields: ' + field)


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('GELU evidence requires SHA256')
    return value


def hardware_check(value):
    keys(value, ('name', 'uuid', 'driver_version', 'memory_mib', 'hostname'), 'hardware')
    if (value['name'] != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
            or type(value['uuid']) is not str or not re.fullmatch('GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value['uuid'])
            or type(value['memory_mib']) is not int or value['memory_mib'] <= 0
            or any(type(value[k]) is not str or not value[k] for k in ('driver_version', 'hostname'))):
        raise ValueError('GELU contracts require actual SM120 physical identity')


def validate_evidence_config(config):
    keys(config, ('contracts_root', 'result_sha256', 'source_manifest_sha256'), 'config')
    if (type(config['contracts_root']) is not str or '\x00' in config['contracts_root']
            or not Path(config['contracts_root']).is_absolute()):
        raise ValueError('GELU contracts_root must be an explicit absolute path')
    sha(config['result_sha256']); sha(config['source_manifest_sha256'])
    return config


def _record(record):
    keys(record, ('schema', 'source_manifest_sha256', 'result_sha256', 'files_sha256',
                  'source_files_sha256', 'hardware', 'cpu_cases', 'cuda_cases', 'scope'), 'contract record')
    exact(record['schema'], 1, 'contract schema'); exact(record['cpu_cases'], 15, 'CPU count')
    exact(record['cuda_cases'], 3, 'CUDA count'); exact(record['scope'], SCOPE, 'scope')
    keys(record['files_sha256'], FILES, 'raw contract files')
    keys(record['source_files_sha256'], SOURCE_FILES, 'executed source coverage')
    for value in (*record['files_sha256'].values(), *record['source_files_sha256'].values(),
                  record['source_manifest_sha256'], record['result_sha256']): sha(value)
    exact(record['files_sha256']['RESULT.json'], record['result_sha256'], 'result binding')
    exact(record['files_sha256']['SOURCE_PAYLOAD.json'], record['source_manifest_sha256'], 'source binding')
    hardware_check(record['hardware'])


def validate_evidence_bytes(blobs, contract_record, hardware):
    """Offline: never import current model code or trust a reconstructed report."""
    _record(contract_record); keys(blobs, FILES, 'original byte files')
    hardware_check(hardware); exact(contract_record['hardware'], hardware, 'physical GPU')
    for name in FILES:
        if type(blobs[name]) is not bytes: raise ValueError('GELU contracts require original bytes')
        exact(hashlib.sha256(blobs[name]).hexdigest(), contract_record['files_sha256'][name], name)
    values = {name: json.loads(raw) for name, raw in blobs.items() if name.endswith('.json')}
    result, source = values['RESULT.json'], values['SOURCE_PAYLOAD.json']
    keys(source, ('base_commit', 'files'), 'source payload')
    if type(source['base_commit']) is not str or re.fullmatch('[0-9a-f]{40}', source['base_commit']) is None:
        raise ValueError('GELU source commit malformed')
    if type(source['files']) is not dict: raise ValueError('GELU source inventory malformed')
    for name, item in source['files'].items():
        if type(name) is not str or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe executed source path')
        keys(item, ('sha256', 'bytes'), 'source file'); sha(item['sha256'])
        if type(item['bytes']) is not int or item['bytes'] < 0: raise ValueError('Invalid source size')
    for name in SOURCE_FILES:
        exact(source['files'][name]['sha256'], contract_record['source_files_sha256'][name], name)
    keys(result, ('all_owned_processes_retired', 'contracts_only', 'finished', 'hardware', 'model_loaded',
        'passed', 'performance_measured', 'production_changed', 'source', 'started', 'status', 'tests'), 'RESULT')
    for key, value in dict(status='complete', all_owned_processes_retired=True, contracts_only=True,
                           model_loaded=False, performance_measured=False, production_changed=False, passed=18).items():
        exact(result[key], value, 'RESULT.' + key)
    exact(result['hardware'], hardware, 'RESULT.hardware')
    exact(result['source'], dict(files=len(source['files']), source_manifest_sha256=contract_record['source_manifest_sha256']), 'RESULT.source')
    if any(type(result[key]) not in (int, float) or not math.isfinite(result[key]) for key in ('started', 'finished')) or result['finished'] < result['started']:
        raise ValueError('Invalid GELU contract execution interval')
    keys(result['tests'], ('cpu', 'cuda'), 'test stages')
    pids = []
    for label in ('cpu', 'cuda'):
        row = result['tests'][label]
        expected = dict(actual_cuda=label == 'cuda', coverage=COVERAGE[label],
            junit_sha256=contract_record['files_sha256'][label + '.xml'], model_loaded=False,
            no_skip=True, passed=sum(COVERAGE[label].values()), performance_measured=False)
        exact(row, expected, label + ' acceptance')
        cases = ET.fromstring(blobs[label + '.xml']).findall('.//testcase')
        if any(any(child.tag in ('failure', 'error', 'skipped') for child in case) for case in cases):
            raise ValueError('GELU failed/skipped/error contract')
        if any('test_lite_prefix_fp8_gelu' not in case.get('classname', '').split('.') for case in cases):
            raise ValueError('Wrong GELU testcase module')
        names = [case.get('name', '').split('[', 1)[0] for case in cases]
        exact({name: names.count(name) for name in COVERAGE[label]}, COVERAGE[label], label + ' actual coverage')
        exact(len(cases), expected['passed'], label + ' actual cases')
        stage = values[label + '-RESULT.json']
        keys(stage, ('confirmed_exit', 'error', 'finished', 'log_sha256', 'pid', 'returncode'), label + ' process result')
        exact(stage['confirmed_exit'], True, label + ' retired'); exact(stage['error'], None, label + ' error')
        exact(stage['returncode'], 0, label + ' exit'); sha(stage['log_sha256'])
        if type(stage['pid']) is not int or stage['pid'] <= 1: raise ValueError('Invalid contract PID')
        if (type(stage['finished']) not in (int, float) or not math.isfinite(stage['finished'])
                or not result['started'] <= stage['finished'] <= result['finished']):
            raise ValueError('Invalid contract stage interval')
        pids.append(stage['pid'])
    keys(values['exit.json'], ('exit_code', 'finished'), 'controller exit')
    exact(values['exit.json']['exit_code'], 0, 'controller exit')
    retirement = values['ROOT_RETIREMENT.json']
    keys(retirement, ('all_retired', 'compute', 'exit_sha256', 'observed_at', 'pids',
                      'result_sha256', 'source_manifest_sha256'), 'root retirement')
    for value in (values['exit.json']['finished'], retirement['observed_at']):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid contract retirement time')
    if not result['finished'] <= values['exit.json']['finished'] <= retirement['observed_at']:
        raise ValueError('Contract retirement precedes completion')
    exact(retirement['all_retired'], True, 'root retirement'); exact(retirement['compute'], '', 'root GPU idle')
    exact(retirement['result_sha256'], contract_record['result_sha256'], 'root result binding')
    exact(retirement['exit_sha256'], contract_record['files_sha256']['exit.json'], 'root exit binding')
    exact(retirement['source_manifest_sha256'], contract_record['source_manifest_sha256'], 'root source binding')
    if (type(retirement['pids']) is not list or len(retirement['pids']) != 3
            or any(type(pid) is not int or pid <= 1 for pid in retirement['pids'])
            or len(set(retirement['pids'])) != 3 or retirement['pids'][1:] != pids):
        raise ValueError('Root retirement does not cover exact controller/children')
    return contract_record


def load_evidence(config, hardware, source_root=None):
    validate_evidence_config(config); hardware_check(hardware)
    root = Path(config['contracts_root']).resolve()
    blobs = {}
    for name in FILES:
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe/missing GELU evidence file: ' + name)
        if path.stat().st_size > 8 * 1024 * 1024: raise ValueError('Unexpected GELU evidence size')
        blobs[name] = path.read_bytes()
    source = json.loads(blobs['SOURCE_PAYLOAD.json'])
    record = dict(schema=1, source_manifest_sha256=config['source_manifest_sha256'], result_sha256=config['result_sha256'],
        files_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()},
        source_files_sha256={name: source['files'][name]['sha256'] for name in SOURCE_FILES},
        hardware=hardware, cpu_cases=15, cuda_cases=3, scope=SCOPE)
    validate_evidence_bytes(blobs, record, hardware)
    actual_root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[4]
    for name in SOURCE_FILES:
        path = actual_root / name
        item = source['files'][name]
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(actual_root)
                or path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']):
            raise ValueError('Current GELU/reference/test source differs from executed contracts: ' + name)
    return blobs, record


def validate_fusion_manifest(ext, backend=None):
    keys(ext, ('schema', 'version', 'enabled', 'scope', 'modules', 'implementation', 'contracts',
        'owner_source_sha256', 'evidence_validator_sha256', 'lifecycle', 'arithmetic_scope',
        'fallback_permitted', 'quality_validated', 'performance_validated'), 'fusion extension')
    for key, value in dict(schema=1, version=VERSION, enabled=True,
        scope='prefix_mlp_gelu_down_conversion_layers_0_16', modules=[f'prefix.{i}.mlp' for i in range(17)],
        lifecycle=LIFECYCLE,
        arithmetic_scope='only GELU_tanh/BF16 multiply/down input quantization; gate/up and GEMM unchanged',
        fallback_permitted=False, quality_validated=False, performance_validated=False).items():
        exact(ext[key], value, key)
    sha(ext['owner_source_sha256']); sha(ext['evidence_validator_sha256']); _record(ext['contracts'])
    files = ext['contracts']['source_files_sha256']
    implementation = dict(version='pi05-prefix-gelu-fp8-fusion-sm120-v1',
        operator='vvla_repro_pi05_lite::prefix_gelu_fp8_quantize_rows_v1',
        source_sha256={name: files[LITE + name] for name in ('prefix_fp8_gelu.py', 'prefix_fp8_gelu_kernel.py')},
        reference_source_sha256={**{name: files[LITE + name] for name in ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')},
            'models/pi05/fused_mlp_pi05.py': files['vllm-vla-plugin/src/vvla_repro_pi05/models/pi05/fused_mlp_pi05.py']},
        required_runtime=RUNTIME, compile_options=OPTIONS, supported_capacities=[576, 640], width=16384, output_width=2048,
        input_dtype='bfloat16', output_dtypes=['float8_e4m3fn', 'float32'],
        intermediate_rounding=['GELU_tanh_to_BF16_RTNE', 'rounded_GELU_times_up_to_BF16_RTNE'],
        quantizer_arithmetic=ARITHMETIC, gemm_operator='vvla_repro_pi05_lite::prefix_fp8_native_scaled_mm_v1',
        use_fast_accum=False, fp_fusion=False, hidden_materialization=False, persistent_tensors=False,
        output_ownership='fresh_nonaliasing_outputs', fallback_permitted=False,
        installation='none; explicit primitive only',
        evidence_requirement='new executed exact candidate contracts and owner evidence; old backend report is insufficient',
        performance_validated=False, quality_validated=False)
    exact(ext['implementation'], implementation, 'actual primitive implementation')
    if backend is not None:
        exact(backend['hardware'], ext['contracts']['hardware'], 'base/extension physical GPU')
        exact(backend['quantizer_variant'], 'triton-eager-rn-native-sm120-v1', 'base variant')
        exact(backend['quantizer_files_sha256'], {name: files[LITE + name]
            for name in ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')}, 'unchanged base native operators')
        exact(backend['compile_options'], OPTIONS, 'base compile options')
        exact(dict(torch=backend['torch_version'], cuda=backend['cuda_version'],
            torch_git=backend['torch_git_version'], capability=backend['capability']), RUNTIME, 'base runtime')
    return ext
