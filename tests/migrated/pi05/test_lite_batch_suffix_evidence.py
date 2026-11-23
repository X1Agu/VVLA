"""Support receipt rejection, never fictional CPU hardware qualification."""
from copy import deepcopy
import json

import pytest

from vvla_repro_pi05.lite import batch_suffix_evidence as evidence


@pytest.mark.parametrize('fault',['none','splits8','partial','repeat','source','raw','quality','missing'])
def test_actual_split1_support_contract_and_raw_binding(tmp_path,fault):
    cases={}
    for capacity in ('576','640'):
        path=tmp_path/('CAP-'+capacity+'.npz');path.write_bytes(b'fictional CPU schema fixture, not GPU support')
        cases[capacity]=dict(status='passed',checks={name:True for name in evidence.CHECKS},raw_path=path.name,raw_sha256=evidence.sha(path))
    report=dict(status='complete',implementation=evidence.VERSION,batch=32,selected_splits=1,
        policy_loaded=False,policy_quality_passed=False,source_sha256=evidence.source_hashes(),
        cleanup=dict(graphs_retired=True,settings_restored=True),cases=cases)
    if fault=='splits8':report['selected_splits']=8
    elif fault=='partial':report['cases'].pop('576')
    elif fault=='repeat':report['cases']['640']['checks']['repeat_exact']=False
    elif fault=='source':report['source_sha256'][evidence.SOURCES[0]]='f'*64
    elif fault=='raw':(tmp_path/'CAP-640.npz').write_bytes(b'changed')
    elif fault=='quality':report['policy_quality_passed']=True
    elif fault=='missing':report['cases']['576']['checks'].pop('all32_written')
    path=tmp_path/'BACKEND.json';path.write_text(json.dumps(report))
    config=dict(backend_report=str(path),backend_report_sha256=evidence.sha(path))
    if fault=='none':assert evidence.read_support(config)==report
    else:
        with pytest.raises(ValueError):evidence.read_support(config)
