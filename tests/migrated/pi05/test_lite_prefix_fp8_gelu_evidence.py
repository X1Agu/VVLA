"""Pure CPU rejection contracts; all generated execution evidence is fictional."""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import pytest

from prefix_fp8_gelu_fixtures import evidence, make_backend, make_evidence, make_extension, raw, sha


def reseal(data):
    """Rebind bytes so content validation, rather than stale SHA, rejects a lie."""
    blobs = data['blobs']
    result = json.loads(blobs['RESULT.json'])
    for label in ('cpu', 'cuda'):
        result['tests'][label]['junit_sha256'] = sha(blobs[label + '.xml'])
    blobs['RESULT.json'] = raw(result)
    retirement = json.loads(blobs['ROOT_RETIREMENT.json'])
    retirement['result_sha256'] = sha(blobs['RESULT.json'])
    retirement['exit_sha256'] = sha(blobs['exit.json'])
    blobs['ROOT_RETIREMENT.json'] = raw(retirement)
    data['record']['result_sha256'] = sha(blobs['RESULT.json'])
    data['record']['files_sha256'] = {name: sha(value) for name, value in blobs.items()}
    data['config']['result_sha256'] = data['record']['result_sha256']


def test_load_binds_exact_raw_files_actual_sources_and_independent_extension(tmp_path):
    data = make_evidence(tmp_path)
    blobs, record = evidence.load_evidence(data['config'], data['hardware'], data['source_root'])
    assert blobs == data['blobs'] and record == data['record']
    extension = make_extension(record)
    assert evidence.validate_fusion_manifest(extension, make_backend(record)) == extension
    assert evidence.validate_evidence_bytes(blobs, record, data['hardware']) == record
    # Offline audit must remain valid after the deployed source is no longer
    # available; it binds the executed inventory and full original bytes.
    (data['source_root'] / evidence.SOURCE_FILES[0]).unlink()
    assert evidence.validate_evidence_bytes(blobs, record, data['hardware']) == record
    with pytest.raises(ValueError, match='Current GELU'):
        evidence.load_evidence(data['config'], data['hardware'], data['source_root'])


@pytest.mark.parametrize('change', ('missing_file', 'changed_byte', 'wrong_gpu', 'stale_result', 'boolean_cases'))
def test_raw_or_record_mismatch_rejects(tmp_path, change):
    data = make_evidence(tmp_path)
    if change == 'missing_file': del data['blobs']['cuda.xml']
    elif change == 'changed_byte': data['blobs']['cpu.xml'] += b' '
    elif change == 'wrong_gpu': data['hardware']['uuid'] = 'GPU-22222222-2222-3333-4444-555555555555'
    elif change == 'stale_result': data['record']['result_sha256'] = '0' * 64
    elif change == 'boolean_cases': data['record']['cpu_cases'] = True
    with pytest.raises(ValueError):
        evidence.validate_evidence_bytes(data['blobs'], data['record'], data['hardware'])


@pytest.mark.parametrize('change', ('cuda_skipped', 'missing_case', 'wrong_module', 'false_actual_cuda',
    'bool_total', 'not_retired', 'wrong_child', 'child_failure', 'controller_failure', 'quality_claim'))
def test_rehashed_forged_completion_cannot_replace_executed_coverage(tmp_path, change):
    data = make_evidence(tmp_path)
    blobs = data['blobs']
    if change in ('cuda_skipped', 'missing_case', 'wrong_module'):
        xml = ET.fromstring(blobs['cuda.xml'])
        first = xml.find('.//testcase')
        if change == 'cuda_skipped': ET.SubElement(first, 'skipped', message='fictional skip')
        elif change == 'missing_case': xml.find('testsuite').remove(first)
        else: first.set('classname', 'unrelated.test_module')
        blobs['cuda.xml'] = ET.tostring(xml)
    elif change in ('false_actual_cuda', 'bool_total', 'quality_claim'):
        result = json.loads(blobs['RESULT.json'])
        if change == 'false_actual_cuda': result['tests']['cuda']['actual_cuda'] = False
        elif change == 'bool_total': result['passed'] = True
        else: result['production_changed'] = True
        blobs['RESULT.json'] = raw(result)
    elif change in ('not_retired', 'wrong_child'):
        retirement = json.loads(blobs['ROOT_RETIREMENT.json'])
        if change == 'not_retired': retirement['all_retired'] = False
        else: retirement['pids'][2] = 104
        blobs['ROOT_RETIREMENT.json'] = raw(retirement)
    else:
        name = 'cuda-RESULT.json' if change == 'child_failure' else 'exit.json'
        value = json.loads(blobs[name])
        value['returncode' if change == 'child_failure' else 'exit_code'] = 1
        blobs[name] = raw(value)
    reseal(data)
    with pytest.raises(ValueError):
        evidence.validate_evidence_bytes(data['blobs'], data['record'], data['hardware'])


@pytest.mark.parametrize('change', ('missing_contracts', 'arithmetic', 'subset', 'old_report_only',
    'wrong_native_source', 'sm89', 'nvfp4', 'compile_route', 'unknown_field'))
def test_old_backend_never_authorizes_new_fusion_by_itself(tmp_path, change):
    data = make_evidence(tmp_path)
    extension, backend = make_extension(data['record']), make_backend(data['record'])
    if change == 'missing_contracts': del extension['contracts']
    elif change == 'arithmetic': extension['implementation']['fp_fusion'] = True
    elif change == 'subset': extension['modules'].pop()
    elif change == 'old_report_only': extension = deepcopy(backend)
    elif change == 'wrong_native_source': backend['quantizer_files_sha256']['prefix_fp8_native.py'] = '0' * 64
    elif change == 'sm89': backend['quantizer_variant'] = 'sm89-down-scalar-fp32-native-v1'
    elif change == 'nvfp4': backend['quantizer_variant'] = 'nvfp4'
    elif change == 'compile_route': backend['compile_options']['use_fast_math'] = True
    else: extension['silently_allowed'] = True
    with pytest.raises(ValueError): evidence.validate_fusion_manifest(extension, backend)


def test_config_paths_source_tamper_and_symlink_do_not_bypass_cold_binding(tmp_path):
    data = make_evidence(tmp_path)
    config = deepcopy(data['config'])
    config['contracts_root'] = 'relative'
    with pytest.raises(ValueError): evidence.validate_evidence_config(config)
    config = deepcopy(data['config']); config['result_sha256'] = True
    with pytest.raises(ValueError): evidence.validate_evidence_config(config)
    path = data['source_root'] / evidence.SOURCE_FILES[-1]
    path.write_bytes(b'changed test, same declared revision')
    with pytest.raises(ValueError, match='Current GELU'):
        evidence.load_evidence(data['config'], data['hardware'], data['source_root'])
    from pathlib import Path
    path = Path(data['config']['contracts_root']) / 'cpu.xml'
    original = tmp_path / 'outside.xml'
    original.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(original)
    with pytest.raises(ValueError, match='Unsafe/missing'):
        evidence.load_evidence(data['config'], data['hardware'], data['source_root'])
