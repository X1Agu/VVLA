"""Root-scheduled CUDA gates: mature FA2, masks/stride, fullgraph and replay.

No local GPU execution during preparation. Isolated tolerances are numerical
screening, never closed-loop acceptance evidence.
"""
import pytest
import torch

from vvla_repro_pi05.lite.attention import prepare_pi05_static_metadata
from vvla_repro_pi05.lite.suffix_fa2 import Pi05FA2SplitSuffixAttention
from test_lite_suffix_attention_cuda import _inputs,_reference,_fa2

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA deferred to root GPU scheduler')


@pytest.mark.parametrize('splits',[1,2,4,8,10])
@pytest.mark.parametrize('case',['dense','holes_strided','eager','empty_prefix'])
def test_cuda_fa2_math_live_mapping_strides_and_repeat(splits,case,record_property):
    with torch.inference_mode():
        args,keys=_inputs(prefix=1 if case=='empty_prefix' else 576,holes=case!='dense',
                          strided=case=='holes_strided',eager=case=='eager',all_prefix_masked=case=='empty_prefix')
        # Masked input poison must never contribute, including live partial pages.
        for value in args[1:3]:
            value.masked_fill_(~keys[:,None,:value.shape[2],None],float('nan'))
        backend=Pi05FA2SplitSuffixAttention(splits=splits)
        result=backend.forward_split(*args)
        assert result.shape==(1,8,50,256) and result.dtype==torch.bfloat16 and torch.isfinite(result).all()
        assert result.transpose(1,2).is_contiguous()
        repeat=backend.forward_split(*args)
        torch.testing.assert_close(result,repeat,rtol=0,atol=0)
        fa2=_fa2(args);fp32=_reference(args,keys)
        torch.testing.assert_close(result,fa2,rtol=.02,atol=.003)
        torch.testing.assert_close(result,fp32,rtol=.02,atol=.003)
        record_property('max_abs_vs_fa2',float((result.float()-fa2.float()).abs().max()))
        record_property('max_abs_vs_fp32',float((result.float()-fp32.float()).abs().max()))


def capture(call,args):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):call(*args);call(*args)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=call(*args)
    graph.replay();torch.cuda.synchronize()
    return graph,result


@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('splits',[8,10])
def test_cuda_fullgraph_capture_owns_storage_and_refreshes_current_masks_kv(compiled,splits):
    with torch.inference_mode():
        backend=Pi05FA2SplitSuffixAttention(splits=splits)
        call=torch.compile(backend.forward_split,fullgraph=True,dynamic=False) if compiled else backend.forward_split
        args,keys=_inputs(strided=True,seed=41)
        other,otherkeys=_inputs(strided=True,seed=73)
        graphs=[]
        try:
            graph,result=capture(call,args);graphs.append(graph)
            graph2,result2=capture(call,other);graphs.append(graph2)
            assert result.data_ptr()!=result2.data_ptr()
            first,second=result.clone(),result2.clone()
            # No mutation of graph geometry or host flags, just current tensor data.
            keys[0,:576]=torch.arange(576,device='cuda')%7==0
            new=prepare_pi05_static_metadata(torch.ones((1,50),device='cuda',dtype=torch.bool),keys,all_queries_valid=True)
            for name,value in new.extra.items():
                if isinstance(value,torch.Tensor):args[-1].extra[name].copy_(value)
            args[1].add_(.25);args[2].mul_(-2);args[4].add_(1)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(result,_fa2(args),rtol=.02,atol=.003)
            assert not torch.equal(first,result)
            assert torch.equal(second,result2)
            updated=result.clone();graph.replay();torch.cuda.synchronize()
            assert torch.equal(updated,result)
            graph2.replay();torch.cuda.synchronize()
            torch.testing.assert_close(result2,_fa2(other),rtol=.02,atol=.003)
        finally:
            torch.cuda.synchronize()
            for graph in graphs:graph.reset()
