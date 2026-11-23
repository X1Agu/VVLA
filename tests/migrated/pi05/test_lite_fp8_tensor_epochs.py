"""Real host-only native reads, plus explicit meta-owner differential checks."""
from types import SimpleNamespace
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import fp8_tensor_epochs as bulk,prefix_fp8 as prefix_owner
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy,_split_lite_overrides,_validate_compilation_options
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
from test_lite_prefix_fp8 import rig,evidence as prefix_evidence,prepare as prefix_prepare
from test_lite_vision_fp8 import setup,install,prepare


@pytest.fixture(scope='module')
def native():
    before=torch.cuda.is_initialized()
    extension,manifest=bulk.load_native_epochs()
    assert torch.cuda.is_initialized()==before
    assert not manifest['with_cuda'] and not manifest['retained_tensor_impl']
    assert not manifest['signature_cache'] and not manifest['inventory_cache']
    assert 'inference_cache_pi05.py' in manifest['source_sha256']
    assert bulk.current_manifest(extension,manifest)==manifest
    return extension


def identities(tensors):return tuple(tensor_identity(t) for t in tensors)


def test_exact_tensor_parameter_alias_offset_and_inference_versions(native):
    original=torch.arange(20,dtype=torch.float32)
    with torch.inference_mode(): inference=torch.ones(3)
    values=(original,original[3:],nn.Parameter(torch.ones(2)),inference,original)
    expected=identities(values)
    assert expected[0][1]!=expected[1][1] and expected[3][2] is None
    assert native.validate_tensor_epochs(values,expected) is None
    native.validate_tensor_epochs(values,expected)
    original.add_(1)
    with pytest.raises(RuntimeError,match='epoch changed'):native.validate_tensor_epochs(values,expected)


@pytest.mark.parametrize('field',range(6))
def test_each_expected_identity_field_rejects(native,field):
    value=torch.ones(3);expected=list(tensor_identity(value))
    expected[field]={0:expected[0]+1,1:expected[1]+1,2:expected[2]+1,
                     3:torch.float64,4:torch.device('cuda:0'),5:(4,)}[field]
    with pytest.raises(RuntimeError):native.validate_tensor_epochs((value,),(tuple(expected),))


@pytest.mark.parametrize('field',['data_ptr','_version','dtype','device','shape'])
def test_identity_shadow_and_late_live_error_not_hidden_by_early_mismatch(native,field):
    first,late=torch.ones(3),torch.ones(2)
    expected=identities((first,late));first.add_(1)
    late.__dict__[field]=object()
    with pytest.raises(TypeError,match='shadowed'):native.validate_tensor_epochs((first,late),expected)


@pytest.mark.parametrize('fault',['tensors_list','expected_list','empty','length','row_length',
    'negative','overflow','bool','shape_list','shape_equality_callback','device_equality_callback'])
def test_exact_schema_rejects_without_python_equality(native,fault):
    class UnexpectedEquality:
        def __eq__(self,other):raise AssertionError('Python equality must not execute')
    value=torch.ones(3);tensors=(value,);expected=identities(tensors)
    row=list(expected[0])
    if fault=='tensors_list':tensors=list(tensors)
    elif fault=='expected_list':expected=list(expected)
    elif fault=='empty':tensors=expected=()
    elif fault=='length':expected=()
    elif fault=='row_length':expected=((1,2),)
    else:
        if fault=='negative':row[0]=-1
        elif fault=='overflow':row[1]=1<<100
        elif fault=='bool':row[0]=True
        elif fault=='shape_list':row[5]=[3]
        elif fault=='shape_equality_callback':row[5]=(UnexpectedEquality(),)
        elif fault=='device_equality_callback':row[4]=UnexpectedEquality()
        expected=(tuple(row),)
    with pytest.raises((TypeError,ValueError,OverflowError)):native.validate_tensor_epochs(tensors,expected)


def test_tensor_subclass_and_ordinary_none_version_reject(native):
    class Other(torch.Tensor):pass
    subclass=torch.ones(3).as_subclass(Other)
    with pytest.raises(TypeError,match='exact'):native.validate_tensor_epochs((subclass,),identities((subclass,)))
    value=torch.ones(3);row=list(tensor_identity(value));row[2]=None
    with pytest.raises(RuntimeError):native.validate_tensor_epochs((value,),(tuple(row),))


def test_hot_all423_reads_preserve_public_independent_validation(setup,native,monkeypatch):
    state=setup;extension=install(state,native_epochs=native);prepare(state)
    assert len(state.base._native_epoch_validation[2])==153
    assert len(extension._native_epoch_validation[2])==270
    prepare(state) # Same generation still performs fresh live reads.
    with torch.no_grad():extension.sites[-1].execution_bias.add_(1)
    with pytest.raises(RuntimeError,match='epoch changed'):prepare(state)
    with pytest.raises(RuntimeError,match='epoch changed'):extension.sites[-1].validate()


def test_prefix_only_all153_reads_without_vision_owner(rig,native):
    handle=prefix_owner.install_prefix_fp8(rig.model,prefix_evidence())
    bulk.enable_owner(handle,native)
    runner=Pi05NativeGraphRunner(rig.model,compile_sample_actions=True)
    handle.bind_runner(runner)
    try:
        assert len(handle._native_epoch_validation[2])==153
        assert not hasattr(rig.model,'_lite_vision_fp8_installation')
        prefix_prepare(handle,runner)
        prefix_prepare(handle,runner)
        with torch.no_grad():handle.sites[-1].weight.add_(1)
        with pytest.raises(RuntimeError,match='epoch (changed|invalidated)'):prefix_prepare(handle,runner)
        with pytest.raises(RuntimeError,match='epoch changed'):handle.sites[-1].validate()
    finally:
        runner.close()
        handle.uninstall()


@pytest.mark.parametrize('scope',['prefix','vision'])
def test_native_call_binding_or_original_epoch_mutation_rejects(setup,native,scope):
    state=setup;extension=install(state,native_epochs=native);prepare(state)
    owner=state.base if scope=='prefix' else extension
    with torch.no_grad():owner.sites[0].weight.add_(1)
    with pytest.raises(RuntimeError,match='epoch (changed|invalidated)'):prepare(state)


@pytest.mark.parametrize('scope',['prefix','vision'])
@pytest.mark.parametrize('fault',['owner','callable','expected_record'])
def test_host_native_binding_and_expected_record_are_immutable(setup,native,scope,fault,monkeypatch):
    state=setup;extension=install(state,native_epochs=native);prepare(state)
    owner=state.base if scope=='prefix' else extension
    if fault=='owner':owner._native_epoch_validation=tuple(list(owner._native_epoch_validation))
    elif fault=='callable':monkeypatch.setattr(native,'validate_tensor_epochs',lambda *args:None)
    else:owner.sites[-1]._identities=((),)
    with pytest.raises(RuntimeError,match='binding changed|identity record changed'):prepare(state)


def test_native_host_option_is_default_off_and_tightly_scoped():
    assert _split_lite_overrides({'lite_fp8_host_epochs':False})==_split_lite_overrides({})
    for value in (1,0,None,'true'):
        with pytest.raises(ValueError,match='boolean'):_split_lite_overrides({'lite_fp8_host_epochs':value})
    options=_split_lite_overrides(dict(lite_prefix_fp8=True,lite_vision_fp8=True,lite_fp8_host_epochs=True,
        lite_native_lifecycle=True,lite_native_lifecycle_variant='v3',lite_compile_sample_actions=True,
        lite_batched_vision=True,lite_vision_fc2_cublaslt=False))[1]
    _validate_compilation_options(options,True)
    prefix_only=dict(options,vision_fp8=False,vision_fc2_cublaslt=True)
    _validate_compilation_options(prefix_only,True)
    for key in ('prefix_fp8','native_lifecycle','compile_sample_actions'):
        changed=dict(options);changed[key]=False
        with pytest.raises(ValueError):_validate_compilation_options(changed,True)
        changed=dict(prefix_only);changed[key]=False
        with pytest.raises(ValueError):_validate_compilation_options(changed,True)
    for config in (options,prefix_only):
        with pytest.raises(ValueError):_validate_compilation_options(config,False)
        for change in (dict(native_lifecycle_variant='v2'),dict(prefix_fp8_inventory=True),dict(prefix_fp8_gelu=True)):
            with pytest.raises(ValueError):_validate_compilation_options(dict(config,**change),True)


@pytest.mark.parametrize('field',['module_name','compiler','torch_version','source_sha256',
    'extra_cflags','with_cuda','inventory_cache'])
def test_declared_manifest_cannot_replace_actual_cold_build(native,field):
    manifest=json.loads(bulk._COLD_BUILDS[id(native)][1])
    changed=deepcopy(manifest)
    if field=='source_sha256':changed[field]['fp8_tensor_epochs.cpp']='f'*64
    elif field=='extra_cflags':changed[field]=['-O0']
    elif field in ('with_cuda','inventory_cache'):changed[field]=True
    else:changed[field]='fictional identity'
    with pytest.raises(RuntimeError,match='build identity changed'):bulk.current_manifest(native,changed)


@pytest.mark.parametrize('vision',[False,True])
@pytest.mark.parametrize('fault',['declared_false','declared_missing','missing_manifest',
    'one_owner_disabled','configured_false','manifest_runtime_tamper','vision_scope_flip','vision_count_tamper',
    'phantom_or_missing_vision_owner'])
def test_policy_metadata_requires_real_bindings_and_immutable_build(native,fault,vision):
    # Isolated declaration gate with fictional owner handles; no model loaded.
    policy=Pi05LitePolicy.__new__(Pi05LitePolicy)
    manifest=json.loads(bulk._COLD_BUILDS[id(native)][1])
    policy._fp8_host_epochs_option=True
    policy._fp8_host_epochs_vision_option=vision
    policy._lite_optimizations=dict(fp8_host_epochs=True,fp8_host_epoch_manifest=dict(core=manifest,
        prefix_tensors=153,vision_tensors=270 if vision else 0,scope='hot prepare only; standalone validation unchanged'))
    owners=[]
    for count in (153,270):
        binding=(native,native.validate_tensor_epochs,(None,)*count,())
        owners.append(SimpleNamespace(_native_epoch_validation=binding,_native_epoch_binding=binding))
    policy._prefix_fp8_installation=owners[0]
    policy._vision_fp8_installation=owners[1] if vision else None
    policy._validate_host_epochs_metadata()
    if fault=='declared_false':policy._lite_optimizations['fp8_host_epochs']=False
    elif fault=='declared_missing':del policy._lite_optimizations['fp8_host_epochs']
    elif fault=='missing_manifest':del policy._lite_optimizations['fp8_host_epoch_manifest']
    elif fault=='one_owner_disabled':owners[0]._native_epoch_validation=owners[0]._native_epoch_binding=None
    elif fault=='configured_false':policy._fp8_host_epochs_option=False
    elif fault=='manifest_runtime_tamper':policy._lite_optimizations['fp8_host_epoch_manifest']['core']['torch_version']='fake'
    elif fault=='vision_scope_flip':policy._fp8_host_epochs_vision_option=not vision
    elif fault=='vision_count_tamper':policy._lite_optimizations['fp8_host_epoch_manifest']['vision_tensors']=0 if vision else 270
    elif fault=='phantom_or_missing_vision_owner':policy._vision_fp8_installation=None if vision else owners[1]
    with pytest.raises(RuntimeError):policy._validate_host_epochs_metadata()
