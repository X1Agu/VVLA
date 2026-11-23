"""Pure source/geometry/byte-evidence contracts; fixtures are fictional."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from vision_fp8_fixtures import make_bundle

PATH=Path(__file__).resolve().parents[1]/'src/vvla_repro_pi05/lite/vision_fp8_evidence.py'
SPEC=importlib.util.spec_from_file_location('_vision_fp8_evidence_cpu',PATH)
e=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(e)


def test_actual_schema_original_bytes_and_precise_runtime_binding():
    bundle=make_bundle()
    assert e.validate_backend_bytes(bundle['raw'],bundle['record'],bundle['record']['hardware'])==bundle['report']
    with pytest.raises(ValueError,match='original report'):
        e.validate_backend_bytes(bundle['raw']+b' ',bundle['record'])
    changed=deepcopy(bundle['record']); changed['runtime']['torch']='2.13.1+cu130'
    with pytest.raises(ValueError): e.validate_contract_record(changed)


@pytest.mark.parametrize('fault',['missing_shape','compile','refresh','retirement','bias','tile','primitive_source'])
def test_report_mutation_does_not_become_valid_by_rehashing(fault):
    bundle=make_bundle(); report=bundle['report']; record=bundle['record']
    if fault=='missing_shape': report['rows'].pop()
    elif fault=='compile': report['rows'][0]['execution']['compiled_exact']=False
    elif fault=='refresh': report['rows'][1]['execution']['same_pointer_refresh_exact']=False
    elif fault=='retirement': report['graph_retired']=False
    elif fault=='bias': report['rows'][0]['analytic']['bias_cancellation_exact']=False
    elif fault=='tile': report['rows'][1]['high_precision']['tiles'][0]['quantized_operand_math']={}
    elif fault=='primitive_source': report['core_implementation_manifest']['source_files_sha256']['vision_fp8_native.py']='f'*64
    raw=json.dumps(report).encode(); record['report_sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises((ValueError,KeyError)): e.validate_backend_bytes(raw,record)


@pytest.mark.parametrize('value',[None,{},dict(backend_report='relative',backend_report_sha256='a'*64),
    dict(backend_report='/a',backend_report_sha256='a'*64,extra=True)])
def test_evidence_requires_exact_explicit_config(value):
    with pytest.raises(ValueError): e.validate_evidence_config(value)


def test_manifest_requires_exact54_order_bias_scale_and_explicit_empty_camera_boundary():
    from vision_fp8_fixtures import make_extension
    extension=make_extension()
    assert e.validate_vision_manifest(extension) is extension
    for field in ('name','original_bias','execution_bias','weight_scale'):
        bad=deepcopy(extension)
        if field=='name': bad['sites'][0][field]='vision.0.fc2'
        else: bad['sites'][0][field]['dtype']='float32' if field in ('original_bias','execution_bias') else 'bfloat16'
        with pytest.raises(ValueError): e.validate_vision_manifest(bad)
    bad=deepcopy(extension); del bad['sites'][0]['execution_bias']
    with pytest.raises(ValueError): e.validate_vision_manifest(bad)
    bad=deepcopy(extension); bad['sites'][0]['execution_bias']['content_sha256']='f'*64
    with pytest.raises(ValueError): e.validate_vision_manifest(bad)
    bad=deepcopy(extension); bad['dispatch']='BF16 fallback for any unsupported shape'
    with pytest.raises(ValueError): e.validate_vision_manifest(bad)
