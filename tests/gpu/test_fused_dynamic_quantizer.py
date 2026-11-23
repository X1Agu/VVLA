import pytest
import torch

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


@pytest.mark.parametrize('rows,width',[(10,1024),(50,4096),(716,2048)])
def test_fused_tensor_quantizer_preserves_scale_bytes_padding_and_graph(rows,width):
    from vvla_runtime.fp8_ops import quantize_tensor
    torch.manual_seed(41)
    value=torch.randn((rows,width),device='cuda',dtype=torch.bfloat16)
    value[0].zero_()
    expected_scale=value.float().abs().amax().clamp_min(1e-12).reshape(1)/448.0
    reference=(value.float()/expected_scale).clamp(-448,448).to(torch.float8_e4m3fn)
    actual,scale=quantize_tensor(value)
    torch.testing.assert_close(scale,expected_scale,rtol=0,atol=0)
    assert torch.equal(actual[:rows].view(torch.uint8),reference.view(torch.uint8))
    assert not actual[rows:].view(torch.uint8).any()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result,s=quantize_tensor(value)
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(result.view(torch.uint8),actual.view(torch.uint8))
    value.zero_();graph.replay();torch.cuda.synchronize()
    assert not result.view(torch.uint8).any()
    torch.testing.assert_close(s,torch.full_like(s,1e-12)/448,rtol=0,atol=0)
    graph.reset()


@pytest.mark.parametrize('rows',[10,50])
def test_fused_gelu_quantization_keeps_historical_bf16_rounding(rows):
    from vvla_runtime.fp8_ops import quantize_tensor,quantize_gelu
    from vvla_pi05.models.pi05.fused_mlp_pi05 import gelu_tanh_mul
    torch.manual_seed(17)
    packed=torch.randn((rows,8192),device='cuda',dtype=torch.bfloat16)
    gate,up=packed.chunk(2,dim=-1)
    expected,scale=quantize_tensor(gelu_tanh_mul(gate,up))
    actual,actual_scale=quantize_gelu(gate,up)
    assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
    torch.testing.assert_close(scale,actual_scale,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output,output_scale=quantize_gelu(gate,up)
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.uint8),output.view(torch.uint8))
    torch.testing.assert_close(scale,output_scale,rtol=0,atol=0)
    graph.reset()
