"""CPU synthetic evidence rejection; no actual CUDA qualification claim."""
import json
from pathlib import Path

import pytest

from vvla_repro_pi05.lite import batch_coalesced_evidence as evidence


@pytest.fixture
def qualified_shape(tmp_path):
    report = dict(version=evidence.VERSION, status='complete', batch_size=32,
        source_inventory_sha256=evidence.SOURCE, config_sha256=evidence.CONFIG,
        formal_content_sha256=evidence.FORMAL, preflight_content_sha256=evidence.PREFLIGHT,
        candidate_source_sha256=evidence.sha(Path(evidence.__file__).with_name('batch_coalesced_prepare.py')),
        accepted_for_full_call_screen=True, decision='eligible_for_full_call_screen',
        policy_loaded=False, model_loaded=False, graph_capture=False, production_modified=False, quality_claim=False,
        retained_outputs_unchanged=True, observations_unchanged=True, initial_prepared_reference_exact=True,
        hardware={'uuid': 'synthetic-no-GPU'}, runtime={'torch': 'synthetic-no-GPU'},
        tokenizer_files_sha256={'synthetic': 'not-an-actual-tokenizer'}, parity=[])
    for index in range(6):
        row = dict(index=index, exact=True, control={'dtype': 'synthetic'}, candidate={'dtype': 'synthetic'}, artifacts={})
        for mode in ('control', 'candidate'):
            path = tmp_path / f'FIXTURE-{index}-{mode}.npz'
            path.write_bytes(b'synthetic CPU fixture, no numerical qualification')
            row['artifacts'][mode] = evidence.sha(path)
        report['parity'].append(row)
    def save(value=report):
        for name in ('SUMMARY.json', 'STATUS.json'):
            (tmp_path / name).write_text(json.dumps(value))
        (tmp_path / 'CLEANUP.json').write_text(json.dumps({'passed': True, 'errors': []}))
        return {'backend_report': str(tmp_path / 'SUMMARY.json'),
                'backend_report_sha256': evidence.sha(tmp_path / 'SUMMARY.json')}
    return report, save


def test_evidence_scope_remains_preparation_only(qualified_shape):
    _, save = qualified_shape
    value = evidence.load_evidence(save())
    assert value['full_call_qualified'] is False
    assert value['closed_loop_quality_qualified'] is False


@pytest.mark.parametrize('key,value', [
    ('batch_size', True), ('status', 'running'), ('source_inventory_sha256', 'old'),
    ('config_sha256', 'old'), ('formal_content_sha256', 'old'), ('preflight_content_sha256', 'old'),
    ('candidate_source_sha256', 'modified'), ('accepted_for_full_call_screen', 1),
    ('decision', 'rejected_no_prepare_speedup'), ('policy_loaded', 0), ('model_loaded', True),
    ('graph_capture', True), ('production_modified', True), ('quality_claim', True),
    ('retained_outputs_unchanged', 1), ('observations_unchanged', False), ('initial_prepared_reference_exact', False),
])
def test_foreign_or_incomplete_qualification_rejected(qualified_shape, key, value):
    report, save = qualified_shape
    report[key] = value
    with pytest.raises(ValueError):evidence.load_evidence(save())


@pytest.mark.parametrize('fault', ['missing_fixture', 'duplicate_index', 'typed_parity', 'false_exact', 'raw_bytes', 'cleanup', 'cleanup_type', 'status', 'report_sha'])
def test_raw_layout_lifecycle_and_binding_tampering_rejected(qualified_shape, fault):
    report, save = qualified_shape
    if fault == 'missing_fixture':report['parity'].pop()
    elif fault == 'duplicate_index':report['parity'][5]['index']=4
    elif fault == 'typed_parity':report['parity'][0]['candidate']['dtype']='changed'
    elif fault == 'false_exact':report['parity'][0]['exact']=1
    config = save()
    root=Path(config['backend_report']).parent
    if fault == 'raw_bytes':(root/'FIXTURE-5-candidate.npz').write_bytes(b'changed')
    elif fault == 'cleanup':(root/'CLEANUP.json').write_text(json.dumps({'passed': False,'errors':['failure']}))
    elif fault == 'cleanup_type':(root/'CLEANUP.json').write_text(json.dumps({'passed': 1,'errors':[]}))
    elif fault == 'status':(root/'STATUS.json').write_text('{}')
    elif fault == 'report_sha':config['backend_report_sha256']='old'
    with pytest.raises(ValueError):evidence.load_evidence(config)


def test_factory_refuses_missing_evidence_and_unselected_evidence():
    from vvla_repro_pi05.lite.batch_pi05 import Pi05BatchEngine
    with pytest.raises(ValueError,match='measured preparation report'):
        Pi05BatchEngine.from_pretrained('/absent',tokenizer='/absent',input_preparation='coalesced_fresh_v1')
    with pytest.raises(ValueError,match='explicit coalesced'):
        Pi05BatchEngine.from_pretrained('/absent',batch_input_preparation_evidence={})
