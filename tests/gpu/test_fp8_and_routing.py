"""Local mechanism gates. Random reduced-depth weights are not model quality."""
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


@pytest.mark.parametrize('outer_inference',[False,True])
def test_packed_fp8_weights_do_not_retain_autograd_intermediates(outer_inference):
    from vvla_pi05.experimental.optimized_precision import Projection
    weights=[torch.nn.Parameter(torch.randn(64,32,device='cuda',dtype=torch.bfloat16)) for _ in range(2)]
    with torch.no_grad():
        value=torch.cat(weights,dim=0).float()
        scale=(value.abs().amax().clamp_min(1e-12)/448).reshape(1)
        expected=(value/scale).clamp(-448,448).to(torch.float8_e4m3fn).t()
    saved=[]
    def pack(tensor):saved.append(tensor);return tensor
    with torch.inference_mode(outer_inference),torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
        mode=torch.is_grad_enabled()
        projection=Projection(object(),weights,name='fixture',stage='action_mlp')
        assert torch.is_grad_enabled()==mode
    assert saved==[], 'Inference weight packing retained a backward graph'
    for tensor in (projection.scale,projection.quantized):
        assert not tensor.requires_grad and tensor.grad_fn is None
        assert not tensor.is_inference(), 'Packed buffers need version counters for mutation guards'
    torch.testing.assert_close(projection.scale,scale,rtol=0,atol=0)
    torch.testing.assert_close(projection.quantized.float(),expected.float(),rtol=0,atol=0)
    with torch.no_grad():weights[0].add_(1)
    with pytest.raises(RuntimeError,match='changed'):projection.validate()


@pytest.mark.parametrize('width,intermediate',[(1024,4096),(2048,16384),(1152,4304)])
@pytest.mark.parametrize('horizon',[10,50])
def test_two_projection_native_fp8_eager_graph_shapes_and_owned_results(width,intermediate,horizon):
    from vvla_runtime.dynamic_fp8 import DynamicFP8Linear
    torch.manual_seed(154)
    up=torch.nn.Linear(width,intermediate,bias=False,device='cuda',dtype=torch.bfloat16).eval()
    down=torch.nn.Linear(intermediate,width,bias=False,device='cuda',dtype=torch.bfloat16).eval()
    quant_up,quant_down=DynamicFP8Linear(up),DynamicFP8Linear(down)
    current=torch.randn(2,horizon,width,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        def run():return quant_down(torch.nn.functional.gelu(quant_up(current)))
        reference=down(torch.nn.functional.gelu(up(current)))
        candidate=run()
        relative_rms=((candidate.float()-reference.float()).square().mean()/reference.float().square().mean()).sqrt()
        # Fixed before running this candidate; not a task/action quality gate.
        assert relative_rms.item()<.1
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run();run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=run()
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,candidate,rtol=0,atol=0)
        owned=captured.clone()
        current.mul_(.5)
        expected=run();graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,expected,rtol=0,atol=0)
        assert not torch.equal(owned,captured)
        torch.testing.assert_close(owned,candidate,rtol=0,atol=0)
        graph.reset()


def test_fp8_rejects_weight_mutation_before_next_capture():
    from vvla_runtime.dynamic_fp8 import DynamicFP8Linear
    module=DynamicFP8Linear(torch.nn.Linear(32,32,device='cuda',dtype=torch.bfloat16).eval())
    with torch.no_grad():
        module.weight.add_(.1)
        with pytest.raises(RuntimeError,match='changed'):module(torch.ones(16,32,device='cuda',dtype=torch.bfloat16))


@pytest.mark.parametrize('tokens',[11,51,408])
@pytest.mark.parametrize('normalize,scale',[(False,1.),(True,1.),(True,2.5)])
def test_posttopk_preserves_selected_ids_and_fp32_boundaries(tokens,normalize,scale):
    from vvla_lingbot.experimental.routing_setup_v1 import post_topk_setup
    generator=torch.Generator(device='cuda').manual_seed(54)
    scores=torch.rand(tokens,32,device='cuda',generator=generator)
    correction=torch.rand(32,device='cuda',generator=generator)*.1
    ids=torch.topk(scores+correction,4,dim=-1).indices
    gathered=scores.gather(1,ids)
    total=gathered.sum(-1,keepdim=True)
    norm=gathered/(total+1e-20) if normalize else gathered
    expected=norm*scale if scale!=1 else norm
    weights,actual_ids,debug=post_topk_setup(scores,ids,norm_topk_prob=normalize,
                                          routed_scaling_factor=scale,diagnostics=True)
    assert torch.equal(actual_ids,ids.to(torch.int32))
    assert torch.equal(debug['gathered'],gathered)
    if normalize:assert torch.equal(debug['sum'],total)
    assert torch.equal(debug['normalized'],norm)
    assert torch.equal(debug['scaled'],expected)
    assert torch.equal(weights,expected.bfloat16())

