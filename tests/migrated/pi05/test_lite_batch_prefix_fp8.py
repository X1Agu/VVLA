"""Explicit meta-tensor lifecycle tests; fictional reports are never usable.

Only CPU tests patch packing/runtime evidence. The real backend probe must use
unmodified live CUDA originals, every new M/K/N, actual binaries and cleanup.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import batch_prefix_fp8 as owner
from vvla_repro_pi05.lite import batch_prefix_fp8_evidence as evidence
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from test_lite_prefix_fp8 import MetaPrefix, prepare
from test_lite_batch_prefix_fp8_evidence import backend_fixture


def configuration():
    return dict(backend_report='/fictional/unit-only.json',backend_report_sha256='a'*64,
                origin_b1=deepcopy(evidence.ORIGIN_B1))


@pytest.fixture
def rig(monkeypatch):
    def geometry(module,device,shape,*,installing=False):
        assert type(module) is nn.Linear and device.type=='meta'
        assert module.weight.dtype==torch.bfloat16 and tuple(module.weight.shape)==shape
        assert module.bias is None and not module.training
        if installing: assert 'forward' not in module.__dict__
    def pack(weight,keepalive):
        return torch.empty_like(weight,dtype=torch.float8_e4m3fn),torch.empty((weight.shape[0],1),device='meta')
    monkeypatch.setattr(owner.base,'_validate_module',geometry)
    monkeypatch.setattr(owner.base,'_pack_weight',pack)
    monkeypatch.setattr(owner.base,'_tensor_record',lambda value:dict(shape=list(value.shape),stride=list(value.stride()),
        dtype=str(value.dtype).removeprefix('torch.'),content_sha256='c'*64))
    backend=backend_fixture(); backend['source_files_sha256']=evidence.source_hashes()
    backend['native_manifest']=evidence.expected_native_manifest(backend['source_files_sha256'])
    raw=b'explicit fictional CPU unit evidence; not a production backend report'
    monkeypatch.setattr(owner,'_load_evidence',lambda config,device:(raw,deepcopy(backend)))
    drains=[]; monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:drains.append('drained'))
    return SimpleNamespace(model=MetaPrefix(),backend=backend,raw=raw,drains=drains)


def installed(rig):
    handle=owner.install_batch_prefix_fp8(rig.model,configuration())
    runner=Pi05NativeGraphRunner(rig.model,compile_sample_actions=True)
    handle.bind_runner(runner)
    return handle,runner


def test_all51_original_parameters_and_final_kv_only_layer_are_preserved(rig):
    before=tuple((name,id(value)) for name,value in rig.model.named_parameters())
    keys=tuple(rig.model.state_dict()); handle,runner=installed(rig)
    try:
        prepare(handle,runner)
        manifest=handle.validate_precision_manifest(); evidence.validate_manifest(manifest)
        assert manifest['batch']==32 and manifest['flattened_rows']==[18432,20480]
        assert len(manifest['sites'])==51 and manifest['sites'][-1]['name']=='prefix.16.down_proj'
        assert manifest['origin_b1']==evidence.ORIGIN_B1
        assert manifest['precision_boundaries']==evidence.PRECISION_BOUNDARIES
        assert before==tuple((name,id(value)) for name,value in rig.model.named_parameters())
        assert keys==tuple(rig.model.state_dict())
        last=owner.base._layers(rig.model)[17].mlp
        assert all('forward' not in getattr(last,name).__dict__ for name in owner.base.PROJECTIONS)
        manifest['sites'][0]['name']='changed-copy'
        assert handle.manifest()['sites'][0]['name']=='prefix.0.gate_proj'
    finally:
        runner.close(); handle.uninstall()
    assert not hasattr(rig.model,owner.INSTALLATION)


@pytest.mark.parametrize('mutation',('source_inplace','packed_inplace','scale_inplace','packed_replace',
    'module_replace','site_order','forward','other_forward','other_precision','cold_identity_tuple'))
def test_live_epoch_geometry_topology_and_original_precision_still_fail_closed(rig,mutation):
    handle,runner=installed(rig); prepare(handle,runner); site=handle.sites[0]
    packed=tuple(id(value.quantized) for value in handle.sites)
    try:
        with torch.no_grad():
            if mutation=='source_inplace': site.weight.add_(1)
            elif mutation=='packed_inplace': site.quantized.copy_(torch.empty_like(site.quantized))
            elif mutation=='scale_inplace': site.scale.add_(1)
            elif mutation=='packed_replace': site.quantized=torch.empty_like(site.quantized)
            elif mutation=='module_replace': owner.base._layers(rig.model)[0].mlp.gate_proj=nn.Linear(2048,16384,bias=False,device='meta',dtype=torch.bfloat16).eval()
            elif mutation=='site_order': handle.sites=tuple(reversed(handle.sites))
            elif mutation=='forward': site.module.forward=lambda x:x
            elif mutation=='other_forward': rig.model.other.forward=lambda x:x
            elif mutation=='other_precision': rig.model.other.to(dtype=torch.bfloat16)
            elif mutation=='cold_identity_tuple': site._identities=tuple(list(site._identities))
        with pytest.raises((RuntimeError,ValueError,AssertionError)): prepare(handle,runner)
        if mutation not in ('packed_replace','site_order'):
            assert packed==tuple(id(value.quantized) for value in handle.sites)
    finally:
        runner.close(); handle.uninstall()


def test_lock_unique_runner_current_generation_and_retirement_remain_required(rig):
    handle,runner=installed(rig)
    try:
        with pytest.raises(RuntimeError,match='unique locked'): handle.prepare(0)
        prepare(handle,runner)
        second=Pi05NativeGraphRunner(rig.model,compile_sample_actions=True)
        try:
            with pytest.raises(RuntimeError,match='unique locked'): prepare(handle,runner)
        finally: second.close()
        with runner._call_lock:
            with pytest.raises(RuntimeError,match='stale'): handle.compilation_members(handle._generation+1)
        runner._device=torch.device('meta')
        with pytest.raises(RuntimeError,match='Drain native'): handle.uninstall()
        runner.clear()
    finally:
        runner.close(); handle.uninstall()


def test_explicit_invalidation_never_silently_repacks_even_equal_content(rig):
    handle,runner=installed(rig)
    try:
        prepare(handle,runner); before=tuple(id(site.quantized) for site in handle.sites)
        rig.model.invalidate_inference_caches()
        with pytest.raises(RuntimeError,match='epoch invalidated'): prepare(handle,runner)
        assert before==tuple(id(site.quantized) for site in handle.sites)
    finally:
        runner.close(); handle.uninstall()


@pytest.mark.parametrize('extension',owner._UNSUPPORTED)
def test_b1_extensions_cannot_implicitly_enable_unqualified_paths(rig,extension):
    setattr(rig.model,extension,object())
    with pytest.raises(ValueError,match='does not accept'): owner.install_batch_prefix_fp8(rig.model,configuration())
    assert not hasattr(rig.model,owner.INSTALLATION)


@pytest.mark.parametrize('capacity',(576,640))
def test_forward_keeps_exact_b32_shape_and_flattens_once_for_actual_primitive(rig,monkeypatch,capacity):
    handle,runner=installed(rig); prepare(handle,runner); site=handle.sites[0]; calls=[]
    def linear(value,qw,sw):
        calls.append((tuple(value.shape),qw is site.quantized,sw is site.scale))
        return torch.empty((value.shape[0],site.shape[0]),device='meta',dtype=torch.bfloat16),None,None
    monkeypatch.setattr(owner.native,'scaled_linear',linear)
    # Direct site metadata test only; production prepare rejects such rebinding.
    monkeypatch.setattr(torch,'is_autocast_enabled',lambda device:False)
    try:
        with torch.no_grad(),runner._call_lock:
            x=torch.empty((32,capacity,2048),device='meta',dtype=torch.bfloat16)
            output=site.call(site.module,x)
            assert output.shape==(32,capacity,16384) and calls==[((32*capacity,2048),True,True)]
            with pytest.raises(ValueError,match='no B1 fallback'): site.call(site.module,x[:1])
            monkeypatch.setattr(torch.compiler,'is_compiling',lambda:True)
            with pytest.raises(RuntimeError,match='prepared owner scope'): site.call(site.module,x)
            site._prepared_compile_token=object()
            assert site.call(site.module,x).shape==output.shape
            site._prepared_compile_token=None
    finally:
        runner.close(); handle.uninstall()


def test_cold_manifest_and_backend_declaration_tampering_reject(rig):
    handle,runner=installed(rig)
    try:
        prepare(handle,runner); handle._manifest['batch']=1
        with pytest.raises(RuntimeError,match='manifest declaration changed'): handle.manifest()
    finally:
        runner.close(); handle.uninstall()


def test_partial_pack_cleanup_drains_before_drop_and_keeps_failed_owner_for_retry(rig,monkeypatch):
    original=owner._Site.initialize
    def fail(site):
        if site.index==2:
            site.owner._pending.append(torch.empty(1,device='meta'))
            raise RuntimeError('injected packing')
        original(site)
    monkeypatch.setattr(owner._Site,'initialize',fail)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:(_ for _ in ()).throw(RuntimeError('drain failed')))
    with pytest.raises(RuntimeError,match='injected packing') as failure:
        owner.install_batch_prefix_fp8(rig.model,configuration())
    handle=getattr(rig.model,owner.INSTALLATION)
    assert handle.enabled and handle._pending and len(handle._installed_sites)==3
    assert any('retained owner' in note for note in failure.value.__notes__)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    handle.uninstall()
    assert not hasattr(rig.model,owner.INSTALLATION) and not handle._pending


def test_bad_report_hash_fails_before_any_runtime_or_tensor_work(tmp_path,monkeypatch):
    config=configuration(); path=tmp_path/'backend.json'; path.write_text('{}'); config['backend_report']=str(path)
    monkeypatch.setattr(owner.native,'validate_runtime',lambda device:pytest.fail('must not touch runtime'))
    with pytest.raises(ValueError,match='original bytes'): owner._load_evidence(config,torch.device('cuda'))
