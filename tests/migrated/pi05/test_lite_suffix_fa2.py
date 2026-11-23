"""CPU contracts for public FA2 suffix dispatch; no GPU execution."""
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite.attention import Pi05FlashAttention, prepare_pi05_static_metadata, prepare_pi05_attention_metadata
import vvla_repro_pi05.lite.suffix_fa2 as candidate


def inputs(prefix=7, eager=False):
    q=torch.zeros((1,8,50,256),dtype=torch.bfloat16)
    # Noncontiguous token strides match packed QKV projection views.
    pk,pv=[torch.arange(prefix*512,dtype=torch.float32).reshape(1,1,prefix,512).bfloat16()[...,:256] for _ in range(2)]
    packed=torch.randn((1,50,2560),dtype=torch.bfloat16)
    sk,sv=[x.view(1,50,1,256).transpose(1,2) for x in packed.split((2048,256,256),-1)[1:]]
    queries=torch.ones((1,50),dtype=torch.bool)
    keys=torch.cat(((torch.arange(prefix)%3!=1)[None],queries),1)
    if eager:
        mask=torch.where(keys[:,None,None,:].expand(1,1,50,-1),0.,-2.3819763e38)
        meta=prepare_pi05_attention_metadata(mask,prefix+50)
    else:
        meta=prepare_pi05_static_metadata(queries,keys,all_queries_valid=True)
    return (q,pk,pv,sk,sv,meta),keys


@pytest.mark.parametrize('eager',[False,True])
def test_pack_is_exact_original_mapping_and_new_storage_on_each_invocation(eager):
    args,keys=inputs(eager=eager);_,pk,pv,sk,sv,m=args
    got=candidate._pack_joint_pages(pk,pv,sk,sv,m.extra['k_indices'],57)
    for actual,prefix,suffix in zip(got,(pk,pv),(sk,sv),strict=True):
        expected=torch.cat((prefix,suffix),2).transpose(1,2).flatten(0,1).index_select(0,m.extra['k_indices'])
        torch.testing.assert_close(actual[0,:len(expected)],expected,rtol=0,atol=0)
        assert actual.shape==(1,256,1,256) and actual.is_contiguous()
        assert torch.count_nonzero(actual[0,len(expected):])==0
    repeat=candidate._pack_joint_pages(pk,pv,sk,sv,m.extra['k_indices'],57)
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(got,repeat,strict=True))


@pytest.mark.parametrize('splits',[True,0,3,16,4.0])
def test_options_fail_closed(splits):
    with pytest.raises(ValueError,match='split count'):
        candidate.Pi05FA2SplitSuffixAttention(splits=splits)


def test_no_cpu_fallback_no_training_no_mutated_recipe():
    backend=candidate.Pi05FA2SplitSuffixAttention()
    args,_=inputs()
    assert not list(backend.parameters()) and not list(backend.buffers())
    with pytest.raises(ValueError,match='without gradients'):
        backend.forward_split(*args)
    with torch.no_grad(),pytest.raises(ValueError,match='requires CUDA'):
        backend.forward_split(*args)
    backend._splits=2
    with torch.no_grad(),pytest.raises(RuntimeError,match='recipe modified'):
        backend.forward_split(*args)
    with pytest.raises(RuntimeError,match='before core KV'):
        backend(*args)


@pytest.mark.parametrize('splits',[8,10])
def test_forward_consumes_device_metadata_views_without_host_data_reads(monkeypatch,splits):
    backend=candidate.Pi05FA2SplitSuffixAttention(splits=splits)
    original=candidate._validate_call
    monkeypatch.setattr(candidate,'_validate_call',partial(original,require_cuda=False))
    calls=[]
    def fake(q,k,v,cuq,cuk,maxq,maxk,**kw):
        calls.append((q,k,v,cuq,cuk,maxq,maxk,kw)); return q.clone()
    backend._kernel=backend._installed_kernel=fake
    args,_=inputs(); metadata=args[-1].extra
    def forbidden(*a,**kw):raise AssertionError('Host tensor read')
    with monkeypatch.context() as patch,torch.no_grad():
        for name in ('cpu','item','numpy','__bool__'):patch.setattr(torch.Tensor,name,forbidden)
        out=backend.forward_split(*args)
    assert out.shape==(1,8,50,256) and out.dtype==torch.bfloat16
    _,k,v,cuq,cuk,maxq,maxk,kw=calls[0]
    assert k.shape==v.shape==(1,256,1,256) and (maxq,maxk)==(50,57)
    assert cuq is metadata['cu_q'] and cuk is metadata['cu_k']
    assert kw['block_table'].data_ptr()==cuq.data_ptr() and kw['block_table'].shape==(1,1)
    assert kw['seqused_k'].data_ptr()==cuk[1:].data_ptr()
    assert kw['num_splits']==splits and kw['scale']==1/16 and kw['enable_gqa']


def test_fullgraph_cpu_tracing_preserves_refresh_without_recompile(monkeypatch):
    backend=candidate.Pi05FA2SplitSuffixAttention(splits=8)
    monkeypatch.setattr(candidate,'_validate_call',partial(candidate._validate_call,require_cuda=False))
    # Substitute only the GPU attention leaf; compilation still sees all real
    # guards/shape operations/cat/gather/pad and metadata view construction.
    def leaf(q,k,v,cuq,cuk,maxq,maxk,**kw):return q+k[0,:1]
    backend._kernel=backend._installed_kernel=leaf
    frames=[]
    def compiler(gm,example_inputs):frames.append(gm);return gm.forward
    compiled=torch.compile(backend.forward_split,backend=compiler,fullgraph=True,dynamic=False)
    args,keys=inputs();meta=args[-1]
    with torch.no_grad():
        before=compiled(*args).clone()
        keys[0,0]=False
        new=prepare_pi05_static_metadata(torch.ones((1,50),dtype=torch.bool),keys,all_queries_valid=True)
        for key,value in new.extra.items():
            if isinstance(value,torch.Tensor):meta.extra[key].copy_(value)
        args[1].add_(3)
        after=compiled(*args)
    assert len(frames)==1 and not torch.equal(before,after)
    assert any('index_select' in str(n.target) for n in frames[0].graph.nodes)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.action_horizon=50;self.sentinel=nn.Parameter(torch.ones(1));self.generation=0
        self.paligemma_with_expert=nn.Module();inner=self.paligemma_with_expert
        inner.prefix_attention=Pi05FlashAttention(1/16);inner.suffix_attention=Pi05FlashAttention(1/16)
        inner.gemma_expert=nn.Module();inner.gemma_expert.config=SimpleNamespace(num_attention_heads=8,num_key_value_heads=1,head_dim=256)
        self.eval()
    def invalidate_inference_caches(self,**kw):self.generation+=1


def test_install_restore_metadata_and_graph_lifecycle():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    model=Model();inner=model.paligemma_with_expert;prefix,original=inner.prefix_attention,inner.suffix_attention
    parameters={k:id(v) for k,v in model.named_parameters()};keys=tuple(model.state_dict())
    graph=Pi05NativeGraphRunner(model);graph.entries['live']=object()
    with pytest.raises(ValueError,match='Clear all live'):candidate.install_suffix_fa2(model)
    graph.clear();graph._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='active'):candidate.install_suffix_fa2(model)
    finally:graph._call_lock.release()
    handle=candidate.install_suffix_fa2(model,splits=4)
    assert inner.prefix_attention is prefix and inner.suffix_attention.splits==4
    assert parameters=={k:id(v) for k,v in model.named_parameters()} and keys==tuple(model.state_dict())
    manifest=handle.manifest();assert manifest['splits']==4 and manifest['dtype']=='bfloat16'
    with pytest.raises(ValueError,match='already installed'):candidate.install_suffix_fa2(model)
    graph.entries['live']=object()
    with pytest.raises(ValueError,match='Clear all live'):handle.uninstall()
    assert handle.manifest()==manifest
    graph.close();handle.uninstall()
    assert inner.suffix_attention is original and inner.prefix_attention is prefix and model.generation==2
    assert not hasattr(model,candidate.INSTALLATION) and not handle.enabled


def test_rejects_other_suffix_candidate_and_detects_external_mutation():
    model=Model();model.paligemma_with_expert.suffix_attention=nn.Identity()
    with pytest.raises(ValueError,match='without another candidate'):candidate.install_suffix_fa2(model)
    model=Model();handle=candidate.install_suffix_fa2(model)
    model.paligemma_with_expert.suffix_attention=nn.Identity()
    with pytest.raises(RuntimeError,match='modified externally'):handle.uninstall()


@pytest.mark.parametrize('value',[True,{}, {'splits':3}, {'splits':16}, {'splits':True}, {'splits':8,'block_q':16}, []])
def test_policy_override_rejects_malformed_candidate(value):
    from vvla_repro_pi05.lite.pi05 import _split_lite_overrides
    with pytest.raises(ValueError):_split_lite_overrides({'lite_suffix_fa2':value})


def test_policy_default_off_compile_compatibility_and_mutual_exclusion():
    from vvla_repro_pi05.lite.pi05 import _split_lite_overrides,_validate_compilation_options,Pi05LitePolicy
    core,options=_split_lite_overrides({})
    assert options['suffix_fa2'] is None
    policy=Pi05LitePolicy(Model(),SimpleNamespace(config=SimpleNamespace()),device='cpu',cuda_graph=False)
    assert policy._suffix_fa2_installation is None
    policy.close()
    config={'lite_suffix_fa2':{'splits':8},'lite_compile_sample_actions':True,'denoising_steps':10}
    core,options=_split_lite_overrides(config)
    assert core=={'denoising_steps':10} and options['suffix_fa2']=={'splits':8}
    config['lite_suffix_fa2']['splits']=2
    assert options['suffix_fa2']['splits']==8
    _validate_compilation_options(options,True)
    options['suffix_attention']={'splits':4,'block_q':16}
    with pytest.raises(ValueError,match='only one suffix'):_validate_compilation_options(options,True)


@pytest.mark.parametrize('splits',[8,10])
def test_policy_compile_constructor_installs_before_graph_and_closes_graph_first(monkeypatch,splits):
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy
    import vvla_repro_pi05.lite.graph as graphs
    events=[]
    class FakeGraph:
        def __init__(self,model,**kw):
            assert isinstance(model.paligemma_with_expert.suffix_attention,candidate.Pi05FA2SplitSuffixAttention)
            assert kw['compile_sample_actions'] is True
            self.model=model;events.append('graph_created_after_install')
        def compilation_manifest(self):return {'test_compile':True}
        def close(self):
            assert hasattr(self.model,candidate.INSTALLATION)
            events.append('graph_closed_before_uninstall')
    monkeypatch.setattr(graphs,'Pi05NativeGraphRunner',FakeGraph)
    model=Model();original=model.paligemma_with_expert.suffix_attention
    policy=Pi05LitePolicy(model,SimpleNamespace(config=SimpleNamespace()),device='cuda',cuda_graph=True,
                          compile_sample_actions=True,suffix_fa2={'splits':splits})
    assert policy._lite_optimizations['suffix_fa2_manifest']['splits']==splits
    assert policy._lite_optimizations['suffix_fa2_manifest']['default_splits']==8
    assert policy._lite_optimizations['suffix_fa2_manifest']['supported_splits']==[1,2,4,8,10]
    policy.close()
    assert model.paligemma_with_expert.suffix_attention is original and not hasattr(model,candidate.INSTALLATION)
    assert events==['graph_created_after_install','graph_closed_before_uninstall']


def test_policy_explicit_ten_is_copied_and_default_remains_eight():
    from vvla_repro_pi05.lite.pi05 import _split_lite_overrides
    source={'lite_suffix_fa2':{'splits':10}}
    _,options=_split_lite_overrides(source)
    assert options['suffix_fa2']=={'splits':10}
    source['lite_suffix_fa2']['splits']=16
    assert options['suffix_fa2']=={'splits':10}
    assert candidate.Pi05FA2SplitSuffixAttention().splits==8
    assert _split_lite_overrides({})[1]['suffix_fa2'] is None
