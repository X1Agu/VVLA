"""The actual historical Triton quantizer is executable on local Ada."""
import pytest
import torch

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


@pytest.mark.parametrize('rows,width',[(576,2048),(640,16384)])
def test_historical_rowwise_quantizer_bytes_and_scales_match_torch(rows,width):
    if torch.cuda.get_device_capability()==(8,9):
        from vvla_pi05.lite.prefix_fp8_down_sm89_quantizer_kernel import launch
    elif torch.cuda.get_device_capability()==(12,0):
        from vvla_pi05.lite.prefix_fp8_native_kernel import launch
    else:
        pytest.skip('Historical backend selects SM89 direct conversion or SM120 native conversion')
    torch.manual_seed(19)
    x=torch.randn((rows,width),device='cuda',dtype=torch.bfloat16)
    x[0].zero_()
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scales=torch.empty((rows,1),device='cuda',dtype=torch.float32)
    expected_scale=(x.float().abs().amax(dim=1,keepdim=True)/448).clamp_min(1e-12)
    expected=(x.float()/expected_scale).clamp(-448,448).to(torch.float8_e4m3fn)
    launch(x,q,scales)
    torch.testing.assert_close(scales,expected_scale,rtol=0,atol=0)
    assert torch.equal(q.view(torch.uint8),expected.view(torch.uint8))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch(x,q,scales)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8),expected.view(torch.uint8))
    graph.reset()
