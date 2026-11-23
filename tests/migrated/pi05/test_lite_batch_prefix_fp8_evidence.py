"""Pure synthetic schema tests, never executed CUDA/backend evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[1]/'src/vvla_repro_pi05/lite/batch_prefix_fp8_evidence.py'
_spec=importlib.util.spec_from_file_location('_b32_prefix_pure_evidence',PATH)
e=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e)


def source_fixture():
    values={name:'a'*64 for name in e.SOURCE_FILES}
    values[e.PACKAGE_PREFIX+'lite/prefix_fp8.py']=e.SHARED_B1_OWNER_SHA256
    return values


def hardware():
    return dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
        uuid='GPU-00000000-0000-0000-0000-000000000000',driver_version='unit',memory_mib=1,hostname='synthetic-unit')


def report_fixture():
    sources=source_fixture(); files={name:'b'*64 for name in (*e.DECLARATIONS,*e.BINARIES)}
    rows=[]
    for m in e.ROWS:
        for k,n in e.GEOMETRIES:
            rows.append(dict(shape=dict(batch=32,capacity=m//32,m=m,k=k,n=n),
                checks={name:True for name in e.CHECKS},
                hashes={name:'c'*64 for name in ('input','weight','quantized_input','activation_scale','output','refreshed_output')},
                sample_arrays=dict(path=f'RAW-{m}-{k}-{n}.npz',sha256='f'*64),
                quantized_fp32_oracle=dict(passed=True,atol=.03125,rtol=.015625,violations=0,max_abs=.001,rmse=.0001),
                timing_ms={name:[1.]*10 for name in ('eager_conversion_inclusive','compiled_graph_conversion_inclusive')}))
    return dict(schema=1,implementation=e.VERSION,status='complete',diagnostic_only=True,
        synthetic_activations=True,policy_loaded=False,quality_passed=False,origin_b1=deepcopy(e.ORIGIN_B1),
        runtime=deepcopy(e.RUNTIME),hardware=hardware(),source_files_sha256=sources,torch_files_sha256=files,
        native_manifest=e.expected_native_manifest(sources),rows=rows,
        cleanup=dict(graphs_retired=True,settings_restored=True,errors=[]))


def validate_report(value):
    original=report_fixture()
    return e.validate_report(value,actual_hardware=original['hardware'],actual_sources=original['source_files_sha256'],
        actual_torch_files=original['torch_files_sha256'],actual_native_manifest=original['native_manifest'])


def backend_fixture():
    r=report_fixture()
    return dict(report_sha256='d'*64,hardware=r['hardware'],runtime=r['runtime'],
        source_files_sha256=r['source_files_sha256'],torch_files_sha256=r['torch_files_sha256'],
        native_manifest=r['native_manifest'],executed_rows_sha256=e.digest(r['rows']))


def manifest_fixture():
    backend=backend_fixture(); sites=[]
    for index in range(51):
        projection=('gate_proj','up_proj','down_proj')[index%3]
        shape=[2048,16384] if index%3==2 else [16384,2048]
        record=lambda dtype,shape:dict(shape=shape,stride=[shape[1],1],dtype=dtype,content_sha256='e'*64)
        scale=record('float32',[shape[0],1]); scale['stride']=[1,1]
        sites.append(dict(name=f'prefix.{index//3}.{projection}',
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{index//3}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear',bias=None,original_weight=record('bfloat16',shape),
            quantized_weight=record('float8_e4m3fn',shape),weight_scale=scale))
    return dict(schema=1,precision='mixed_bf16_fp8_prefix',implementation='pi05-b32-prefix-rowwise-fp8-v1',
        checkpoint_sha256=e.CHECKPOINT_SHA256,scope='prefix_mlp_layers_0_16',batch=32,
        per_slot_capacities=list(e.CAPACITIES),flattened_rows=list(e.ROWS),origin_b1=deepcopy(e.ORIGIN_B1),
        backend=backend,precision_boundaries=deepcopy(e.PRECISION_BOUNDARIES),
        nonlinear_boundary='independent BF16 gate/up -> gelu_tanh_mul -> BF16 down input',
        lifecycle=dict(version='pi05-b32-prefix-fp8-live-owner-v1',
            source_sha256=backend['source_files_sha256'][e.PACKAGE_PREFIX+'lite/batch_prefix_fp8.py'],enabled=True,
            owner='Pi05NativeGraphRunner',original_parameters_preserved=True,
            content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
            live_checks=e.LIVE_CHECKS,fallback_permitted=False,
            serving_scope='ordinary_prepare_full_live_checks;private_sealed_engine_cold_and_explicit_full_audit'),sites=sites)


def test_config_origin_is_provenance_not_a_b32_baseline_and_b1_fields_rejected():
    value=dict(backend_report='/fictional/test-only.json',backend_report_sha256='a'*64,origin_b1=deepcopy(e.ORIGIN_B1))
    assert e.validate_config(value) is value
    for change in ({'bf16_control':{}},{'reference_kind':'flashrt_bf16'},{'backend_report_sha256':'not-sha'},
                   {'origin_b1':{}},{'backend_report':''}):
        with pytest.raises(ValueError): e.validate_config({**value,**change})


def test_all_four_new_shapes_source_and_binary_bindings_are_mandatory():
    report=report_fixture(); assert validate_report(report) is report
    changes=[lambda x:x['rows'].pop(),lambda x:x['rows'].append(x['rows'][0]),
        lambda x:x['rows'][0]['shape'].update(m=576),lambda x:x['rows'][0]['shape'].update(batch=1),
        lambda x:x['rows'][0]['shape'].update(m=18432.),lambda x:x['rows'][0]['shape'].update(k=1024),
        lambda x:x['source_files_sha256'].update({e.SOURCE_FILES[0]:'0'*64}),
        lambda x:x['torch_files_sha256'].update({e.BINARIES[0]:'0'*64}),
        lambda x:x['hardware'].update(hostname='other'),lambda x:x['runtime'].update(capability=[8,9]),
        lambda x:x['native_manifest'].update(supported_capacities=[576,640]),
        lambda x:x['native_manifest'].update(fallback_permitted=True)]
    for mutate in changes:
        bad=deepcopy(report); mutate(bad)
        with pytest.raises(ValueError): validate_report(bad)


@pytest.mark.parametrize('field',e.CHECKS)
def test_actual_eager_compile_capture_refresh_ownership_and_retirement_all_required(field):
    report=report_fixture(); report['rows'][0]['checks'][field]=False
    with pytest.raises(ValueError): validate_report(report)


@pytest.mark.parametrize('mutate',[
    lambda x:x.update(policy_loaded=True),lambda x:x.update(quality_passed=True),
    lambda x:x.update(diagnostic_only=False),lambda x:x['cleanup'].update(graphs_retired=False),
    lambda x:x['cleanup'].update(settings_restored=False),lambda x:x['cleanup']['errors'].append('failed'),
    lambda x:x['rows'][0]['quantized_fp32_oracle'].update(violations=1),
    lambda x:x['rows'][0]['quantized_fp32_oracle'].update(atol=1.),
    lambda x:x['rows'][0]['quantized_fp32_oracle'].update(max_abs=float('nan')),
    lambda x:x['rows'][0]['timing_ms']['eager_conversion_inclusive'].pop(),
    lambda x:x['rows'][0]['timing_ms']['compiled_graph_conversion_inclusive'].__setitem__(0,0.),
])
def test_diagnostic_scope_oracles_and_raw_timings_cannot_be_relabelled(mutate):
    report=report_fixture(); mutate(report)
    with pytest.raises(ValueError): validate_report(report)


def test_exact_ordered51_sites_and_nonprefix_precision_declaration():
    manifest=manifest_fixture(); assert e.validate_manifest(manifest) is manifest
    changes=[lambda x:x['sites'].reverse(),lambda x:x['sites'].pop(),lambda x:x.update(batch=1),
        lambda x:x['sites'][0]['original_weight'].update(dtype='float8_e4m3fn'),
        lambda x:x['sites'][0]['weight_scale'].update(shape=[1,16384]),
        lambda x:x['sites'][0].update(bias=True),lambda x:x['precision_boundaries'].update(vision='fp8'),
        lambda x:x['lifecycle'].update(original_parameters_preserved=False),
        lambda x:x['lifecycle'].update(source_sha256='0'*64),
        lambda x:x['backend']['native_manifest'].update(use_fast_accum=True),
        lambda x:x['backend']['source_files_sha256'].update({e.PACKAGE_PREFIX+'lite/prefix_fp8.py':'0'*64})]
    for mutate in changes:
        bad=deepcopy(manifest); mutate(bad)
        with pytest.raises(ValueError): e.validate_manifest(bad)


def test_source_map_rehashes_real_package_probe_and_shared_winning_b1_bytes():
    actual=e.source_hashes()
    assert set(actual)==set(e.SOURCE_FILES)
    assert actual[e.PACKAGE_PREFIX+'lite/prefix_fp8.py']==e.SHARED_B1_OWNER_SHA256
