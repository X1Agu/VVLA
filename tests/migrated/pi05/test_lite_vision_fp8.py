"""Remote CPU/meta ownership contracts. No real arithmetic/quality claim.

Packing/hardware are explicit fakes; actual two-shape CUDA support is separate.
Root schedules these tests; this source has not been executed locally.
"""
from copy import deepcopy
from types import MethodType, SimpleNamespace
import threading

import pytest
import torch
from torch import nn

from test_lite_prefix_fp8 import rig, evidence as base_evidence
from vision_fp8_fixtures import make_bundle
from vvla_repro_pi05.lite import prefix_fp8 as base
from vvla_repro_pi05.lite import vision_fp8 as vision
from vvla_repro_pi05.lite.batched_vision import BatchedVisionInstallation, _batched_embed_prefix
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.lite.native_lifecycle import NativeLifecycleCollector, NativeLifecycleSignatureInstallation
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1152,4304,device='meta',dtype=torch.bfloat16)
        self.fc2=nn.Linear(4304,1152,device='meta',dtype=torch.bfloat16)
        self.activation_fn=nn.GELU(approximate='tanh')
    def forward(self,x): return self.fc2(self.activation_fn(self.fc1(x)))


@pytest.fixture
def setup(rig,monkeypatch):
    model=rig.model
    with torch.device('meta'):
        tower=model.paligemma_with_expert.paligemma.model.vision_tower=nn.Module()
        tower.encoder=nn.Module()
        tower.encoder.layers=nn.ModuleList()
        for _ in range(27):
            layer=nn.Module(); layer.mlp=VisionMLP(); tower.encoder.layers.append(layer)
    model.eval()
    delattr(model,'_graph_runners') # Match actual lazy native registry.
    model._lite_batched_vision_installation=BatchedVisionInstallation(model,(3,(1,3,224,224),256,2048))
    model.embed_prefix=MethodType(_batched_embed_prefix,model)
    bundle=make_bundle(); record=bundle['record']
    backend=dict(quantizer_variant=base.native.VARIANT,
        quantizer_files_sha256=base.native.implementation_manifest()['quantizer_files_sha256'],
        hardware=deepcopy(record['hardware']),installed_binary_sha256=deepcopy(record['installed_binary_sha256']),
        torch_version=record['runtime']['torch'],torch_git_version=record['runtime']['torch_git'],
        cuda_version=record['runtime']['cuda'],capability=[12,0],compile_options=dict(vision.primitive.COMPILE_OPTIONS))
    monkeypatch.setattr(base,'_load_evidence',lambda config,device:(rig.raw,deepcopy(backend)))
    monkeypatch.setattr(vision,'_load_evidence',lambda *args:(bundle['raw'],deepcopy(record)))
    def geometry(module,device,shape,*,installing=False):
        assert type(module) is nn.Linear and module.weight.dtype == torch.bfloat16
        assert tuple(module.weight.shape)==shape and tuple(module.bias.shape)==(shape[0],)
        assert device.type=='meta' and not module.training
        if installing: assert 'forward' not in module.__dict__
    monkeypatch.setattr(vision,'_geometry',geometry)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    state=SimpleNamespace(model=model,base=base.install_prefix_fp8(model,base_evidence()),
        bundle=bundle,runner=None,extension=None,drains=rig.drains)
    yield state
    if state.runner is not None: state.runner.close()
    handle=getattr(model,vision.INSTALLATION,None)
    if handle is not None and handle.enabled: handle.uninstall()
    if state.base.enabled: state.base.uninstall()


def install(state,native_epochs=None):
    state.extension=vision.install_vision_fp8(state.model,state.bundle['config'])
    if native_epochs is not None:
        from vvla_repro_pi05.lite.fp8_tensor_epochs import enable_owner
        enable_owner(state.base,native_epochs)
        enable_owner(state.extension,native_epochs)
    native=NativeLifecycleSignatureInstallation(state.model,NativeLifecycleCollector(None,dict(collector_variant='v3',fictional=True)))
    state.model._lite_native_lifecycle_signature_installation=native
    state.runner=Pi05NativeGraphRunner(state.model,compile_sample_actions=True)
    state.base.bind_runner(state.runner); state.extension.bind_runner(state.runner)
    return state.extension


def prepare(state):
    generation=state.model._inference_cache.refresh(tuple(tensor_identity(x) for x in state.model.parameters()))
    with state.runner._call_lock:
        state.base.prepare(generation)
        state.extension.prepare(generation)
    return generation


def test_cold_lazy_install_all54_original_parameters_activation_and_manifest(setup):
    state=setup
    original=state.base.manifest()
    parameters=tuple((n,id(t)) for n,t in state.model.named_parameters())
    activations=tuple(id(layer.mlp.activation_fn) for layer in vision._layers(state.model))
    extension=vision.install_vision_fp8(state.model,state.bundle['config'])
    assert not hasattr(state.model,'_graph_runners') and len(extension.sites)==54
    # Constructor rollback may occur before native collector/runner exists.
    extension.uninstall()
    assert not hasattr(state.model,'_graph_runners') and state.base.manifest()==original
    extension=install(state); prepare(state)
    manifest=state.base.validate_precision_manifest()
    assert manifest['vision_fp8']==extension.manifest()
    assert manifest['sites']==original['sites'] and manifest['backend']==original['backend']
    assert manifest['precision_boundaries']['vision']==vision.VISION_BOUNDARY
    assert tuple((n,id(t)) for n,t in state.model.named_parameters())==parameters
    assert tuple(id(layer.mlp.activation_fn) for layer in vision._layers(state.model))==activations
    assert all(not s.quantized.is_inference() and not s.scale.is_inference()
               and not s.execution_bias.is_inference() and not s.execution_bias.requires_grad for s in extension.sites)
    assert extension.raw_report==state.bundle['raw']


@pytest.mark.parametrize('mutation',['original_weight','bias','packed','scale','execution_bias','execution_bias_rebind',
    'execution_bias_grad','weight_rebind','bias_rebind',
    'activation','site_order','foreign_forward','old_lt','batching'])
def test_all_original_private_bias_activation_and_topology_checks_reject(setup,mutation,monkeypatch):
    state=setup; extension=install(state); prepare(state); site=extension.sites[0]
    original_sites=extension.sites; original_act=site.layer.mlp.activation_fn
    try:
        with torch.no_grad():
            if mutation in ('original_weight','bias','packed','scale','execution_bias'):
                tensor={'original_weight':site.weight,'bias':site.bias,'packed':site.quantized,
                        'scale':site.scale,'execution_bias':site.execution_bias}[mutation]
                tensor.copy_(torch.empty_like(tensor))
            elif mutation=='execution_bias_rebind': site.execution_bias=site.execution_bias.clone()
            elif mutation=='execution_bias_grad': site.execution_bias.requires_grad_(True)
            elif mutation=='weight_rebind': site.module.weight=nn.Parameter(torch.empty_like(site.weight))
            elif mutation=='bias_rebind': site.module.bias=nn.Parameter(torch.empty_like(site.bias))
            elif mutation=='activation': site.layer.mlp.activation_fn=nn.GELU(approximate='tanh').eval()
            elif mutation=='site_order': extension.sites=tuple(reversed(extension.sites))
            elif mutation=='foreign_forward': site.module.forward=lambda x:x
            elif mutation=='old_lt': state.model._lite_vision_fc2_cublaslt_installation=object()
            elif mutation=='batching': state.model._lite_batched_vision_installation.enabled=False
        with pytest.raises((RuntimeError,ValueError,AssertionError)): prepare(state)
        assert extension.enabled and len(extension._installed_sites)==54
    finally:
        extension.sites=original_sites
        site.layer.mlp.activation_fn=original_act
        if mutation=='foreign_forward': site.module.forward=site.forward
        if mutation=='old_lt': delattr(state.model,'_lite_vision_fc2_cublaslt_installation')
        state.model._lite_batched_vision_installation.enabled=True


def test_explicit_invalidation_and_same_pointer_update_never_repack_live_epoch(setup):
    state=setup; extension=install(state); prepare(state)
    packed=tuple(id(site.quantized) for site in extension.sites)
    state.model.invalidate_inference_caches()
    with pytest.raises(RuntimeError,match='epoch invalidated'): prepare(state)
    assert tuple(id(site.quantized) for site in extension.sites)==packed


def test_m256_is_rejected_without_explicit_cold_blank_scope_and_m512_uses_owned_bias(setup,monkeypatch):
    state=setup
    calls=[]
    # Production installation rejects meta devices. This meta-only ownership
    # fixture also needs an explicit autocast model: Torch has no meta autocast
    # dispatch key. Preserve the real implementation for every supported device.
    original_autocast=torch.is_autocast_enabled
    meta_autocast={'enabled':False}
    def autocast_enabled(device_type=None):
        if device_type=='meta': return meta_autocast['enabled']
        return original_autocast() if device_type is None else original_autocast(device_type)
    monkeypatch.setattr(torch,'is_autocast_enabled',autocast_enabled)
    def op(x,qw,sw,bias):
        calls.append((x,qw,sw,bias))
        return torch.empty((512,qw.shape[0]),device='meta',dtype=torch.bfloat16),None,None
    monkeypatch.setattr(vision.primitive,'scaled_linear',op)
    extension=install(state); prepare(state); site=extension.sites[0]
    with torch.inference_mode(), state.runner._call_lock:
        with pytest.raises(RuntimeError,match='cold empty-camera'):
            site.module(torch.empty(1,256,1152,device='meta',dtype=torch.bfloat16))
        result=site.module(torch.empty(2,256,1152,device='meta',dtype=torch.bfloat16))
        assert result.shape==(2,256,4304) and len(calls)==1
        assert calls[0][1] is site.quantized and calls[0][2] is site.scale
        assert calls[0][3] is site.execution_bias and not calls[0][3].requires_grad
        assert type(site.bias) is nn.Parameter and site.bias.requires_grad
        assert calls[0][3].dtype == site.bias.dtype and calls[0][3].device == site.bias.device
        assert calls[0][3].shape == site.bias.shape and calls[0][3].stride() == site.bias.stride()
        meta_autocast['enabled']=True
        try:
            with pytest.raises(ValueError,match='inference geometry'):
                site.module(torch.empty(2,256,1152,device='meta',dtype=torch.bfloat16))
            assert len(calls)==1
        finally: meta_autocast['enabled']=False
        # The trusted cold scope preserves the original Linear boundary.
        extension._empty_thread=threading.get_ident()
        try:
            result=site.module(torch.empty(1,256,1152,device='meta',dtype=torch.bfloat16))
            assert result.shape==(1,256,4304) and len(calls)==1
        finally: extension._empty_thread=None


@pytest.mark.parametrize('grad_mode',[torch.no_grad,torch.inference_mode],ids=['no_grad','inference_mode'])
def test_bias_owned_clone_is_exact_independent_and_original_still_guarded(monkeypatch,grad_mode):
    # Real CPU storage/version evidence; only CUDA geometry and GEMM are faked.
    # No model load, quantization, numerical approximation, or GPU claim.
    module=nn.Linear(1152,4304,device='meta',dtype=torch.bfloat16).eval()
    module.weight=nn.Parameter(torch.empty(4304,1152,dtype=torch.bfloat16))
    module.bias=nn.Parameter(torch.arange(4304,dtype=torch.float32).to(torch.bfloat16))
    owner=SimpleNamespace(assert_execution_owner=lambda:None)
    site=vision._Site(owner,0,None,module)
    site.quantized=torch.empty(4304,1152,device='meta',dtype=torch.float8_e4m3fn)
    site.scale=torch.empty(4304,1,dtype=torch.float32)
    site.execution_bias=site.bias.detach().clone()
    site._packed=(site.quantized,site.scale,site.execution_bias)
    site._identities=tuple(tensor_identity(t) for t in (site.weight,site.bias,*site._packed))
    module.forward=site.forward
    def cpu_geometry(actual,device,shape,*,installing=False):
        assert actual is module and device.type=='cpu' and shape==(4304,1152)
        assert type(actual.weight) is nn.Parameter and type(actual.bias) is nn.Parameter
        assert actual.weight.dtype==actual.bias.dtype==torch.bfloat16
        assert actual.weight.is_contiguous() and actual.bias.is_contiguous()
    monkeypatch.setattr(vision,'_geometry',cpu_geometry)
    original=module.bias; identity=tensor_identity(original); values=original.detach().clone()
    received=[]
    def primitive(x,qw,sw,bias):
        # This is the production primitive's relevant rejection, without
        # relaxing it merely because autograd is disabled at the call site.
        assert not bias.requires_grad and bias.grad_fn is None
        assert bias is not original and type(bias) is torch.Tensor
        assert bias is site.execution_bias
        assert bias.data_ptr()!=original.data_ptr()
        assert bias.untyped_storage().data_ptr()!=original.untyped_storage().data_ptr()
        assert bias.storage_offset()==original.storage_offset() and bias.stride()==original.stride()
        assert bias.dtype==original.dtype and bias.device==original.device
        assert torch.equal(bias,values)
        received.append(bias)
        return torch.empty(512,4304,dtype=torch.bfloat16),None,None
    monkeypatch.setattr(vision.primitive,'scaled_linear',primitive)
    with grad_mode():
        result=module(torch.empty(2,256,1152,dtype=torch.bfloat16))
    assert result.shape==(2,256,4304) and len(received)==1
    assert module.bias is site.bias is original and original.requires_grad
    assert tensor_identity(original)==identity
    site.validate()
    with torch.no_grad(): original.add_(1)
    assert received[0]._version==0 and original._version==identity[2]+1
    with pytest.raises(RuntimeError,match='immutable packing epoch'): site.validate()
    module.bias=nn.Parameter(original.detach().clone())
    with pytest.raises(RuntimeError,match='ownership changed'): site.validate()


def test_actual_fullgraph_owner_trace_passes_owned_bias_and_keeps_parameter(setup,monkeypatch):
    # Real Dynamo fullgraph trace on meta tensors, with a strict fake GEMM.
    # This checks the owner path used by compilation, not just a mocked
    # is_compiling branch. CUDA primitive arithmetic remains separately tested.
    state=setup; graphs=[]
    original_autocast=torch.is_autocast_enabled
    def autocast_enabled(device_type=None):
        if device_type=='meta': return False
        return original_autocast() if device_type is None else original_autocast(device_type)
    monkeypatch.setattr(torch,'is_autocast_enabled',autocast_enabled)
    def primitive(x,qw,sw,bias):
        if bias.requires_grad: raise ValueError('inference bias must not require gradients')
        # Use the execution bias in the graph.
        return bias.view(1,-1).expand(512,-1).clone(),None,None
    monkeypatch.setattr(vision.primitive,'scaled_linear',primitive)
    extension=install(state); prepare(state); site=extension.sites[0]
    original=site.bias; identity=tensor_identity(original)
    def backend(graph,example_inputs):
        graphs.append(graph)
        return graph.forward
    compiled=torch.compile(site.module,backend=backend,fullgraph=True,dynamic=False)
    site._prepared_compile_token=object()
    try:
        with torch.inference_mode(),state.runner._call_lock:
            result=compiled(torch.empty(2,256,1152,device='meta',dtype=torch.bfloat16))
            again=compiled(torch.empty(2,256,1152,device='meta',dtype=torch.bfloat16))
        assert result.shape==again.shape==(2,256,4304)
        assert not result.requires_grad and not again.requires_grad and len(graphs)==1
        assert not any(node.op=='call_method' and node.target=='detach' for node in graphs[0].graph.nodes)
        assert site.module.bias is site.bias is original and original.requires_grad
        assert tensor_identity(original)==identity
        site.validate()
    finally:
        site._prepared_compile_token=None


def test_empty_camera_scope_checks_actual_blank_mask_and_clears_after_failure(setup):
    state=setup; extension=install(state); prepare(state)
    def fail(image):
        assert extension._empty_thread==threading.get_ident()
        raise RuntimeError('injected native blank encoder failure')
    state.model.paligemma_with_expert.embed_image=fail
    with torch.inference_mode(), state.runner._call_lock:
        for image,mask in ((torch.zeros(1,3,224,224),torch.tensor([False])),
                           (-torch.ones(1,3,224,224),torch.tensor([True]))):
            with pytest.raises(ValueError,match='cold masked'): extension.encode_empty_camera(image,mask)
        with pytest.raises(RuntimeError,match='injected native'):
            extension.encode_empty_camera(-torch.ones(1,3,224,224),torch.tensor([False]))
    assert extension._empty_thread is None


def test_compile_tokens_share_original51_and_vision54_and_always_clear(setup):
    from vvla_repro_pi05.lite.compilation import _PreparedCompiledRun, _packed_handles
    state=setup; extension=install(state); generation=prepare(state)
    context={'token':('vision-test',generation)}; state.runner._token=context['token']
    members=(*state.base.sites,*extension.sites)
    def call():
        token=members[0]._prepared_compile_token
        assert token is not None and all(s._prepared_compile_token is token for s in members)
        raise RuntimeError('injected compiled call')
    scope=_PreparedCompiledRun(state.model,call,_packed_handles(state.model),owner=state.runner,context=context)
    with state.runner._call_lock, pytest.raises(RuntimeError,match='injected compiled'): scope()
    assert all(s._prepared_compile_token is None for s in members)
    scope.retire()
    with state.runner._call_lock, pytest.raises(RuntimeError,match='retired'): scope()


def test_pending_graph_and_failed_drain_preserve_handles_then_cleanup_order(setup):
    state=setup; extension=install(state); prepare(state)
    class Event:
        fail=True
        def synchronize(self):
            if self.fail: raise RuntimeError('injected drain failure')
    event=state.runner._event=Event()
    with pytest.raises(RuntimeError,match='drain failure'): state.runner.close()
    with pytest.raises(RuntimeError,match='Drain native'): extension.uninstall()
    with pytest.raises(RuntimeError,match='Uninstall vision'): state.base.uninstall()
    assert extension.enabled and extension.sites[0].module.forward is extension.sites[0].forward
    event.fail=False; state.runner.clear(); extension.uninstall(); state.base.uninstall()


def test_partial_pack_failure_and_drain_failure_retain_reachable_owner(setup,monkeypatch):
    state=setup; original=vision._Site.initialize
    def fail(site):
        if site.index==2:
            site.owner._pending.append(torch.empty(1,device='meta'))
            raise RuntimeError('injected packing failure')
        original(site)
    def bad_drain(*args): raise RuntimeError('injected CUDA drain failure')
    with monkeypatch.context() as change:
        change.setattr(vision._Site,'initialize',fail)
        change.setattr(torch.cuda,'synchronize',bad_drain)
        with pytest.raises(RuntimeError,match='packing failure'):
            vision.install_vision_fp8(state.model,state.bundle['config'])
    handle=getattr(state.model,vision.INSTALLATION)
    assert handle.enabled and len(handle._installed_sites)==3 and handle._pending
    handle.uninstall()
    assert not hasattr(state.model,vision.INSTALLATION)


def test_existing_owner_or_bound_graph_prevents_cold_install(setup):
    state=setup
    state.model._lite_vision_fc2_cublaslt_installation=object()
    with pytest.raises(ValueError,match='no old'): vision.install_vision_fp8(state.model,state.bundle['config'])
    delattr(state.model,'_lite_vision_fc2_cublaslt_installation')
    state.runner=Pi05NativeGraphRunner(state.model,compile_sample_actions=True)
    with pytest.raises(ValueError,match='unbound'): vision.install_vision_fp8(state.model,state.bundle['config'])
