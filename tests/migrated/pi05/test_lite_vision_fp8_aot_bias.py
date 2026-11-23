"""Real CPU Inductor/AOT regression; the CUDA primitive is replaced explicitly."""
from types import SimpleNamespace

import torch
from torch import nn

from vvla_repro_pi05.lite import vision_fp8 as vision
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity


def test_inductor_keeps_owned_nongrad_bias_instead_of_original_parameter(monkeypatch):
    module=nn.Linear(1152,4304,dtype=torch.bfloat16).eval()
    with torch.no_grad(): module.bias.copy_(torch.arange(4304).to(torch.bfloat16))
    original=module.bias
    site=vision._Site(SimpleNamespace(assert_execution_owner=lambda:None),0,None,module)
    site.quantized=torch.empty(4304,1152,dtype=torch.float8_e4m3fn)
    site.scale=torch.ones(4304,1,dtype=torch.float32)
    site.execution_bias=original.detach().clone()
    site._packed=(site.quantized,site.scale,site.execution_bias)
    site._identities=tuple(tensor_identity(t) for t in (site.weight,site.bias,*site._packed))
    module.forward=site.forward
    def cpu_geometry(actual,device,shape,*,installing=False):
        assert actual is module and device.type=='cpu' and shape==(4304,1152)
    monkeypatch.setattr(vision,'_geometry',cpu_geometry)
    calls=[]
    @torch.library.custom_op('vla_vision_bias_aot_v2::linear',mutates_args=())
    def strict_op(x:torch.Tensor,qw:torch.Tensor,sw:torch.Tensor,bias:torch.Tensor)->torch.Tensor:
        assert type(bias) is torch.Tensor and not bias.requires_grad and bias.grad_fn is None
        assert bias.untyped_storage()._cdata != original.untyped_storage()._cdata
        assert torch.equal(bias,original.detach())
        calls.append((bias.data_ptr(),bias.requires_grad))
        return bias.view(1,-1).expand(512,-1).clone()
    @strict_op.register_fake
    def fake(x,qw,sw,bias):
        assert not bias.requires_grad
        return torch.empty(512,4304,dtype=bias.dtype,device=bias.device)
    def primitive(x,qw,sw,bias): return strict_op(x,qw,sw,bias),None,None
    monkeypatch.setattr(vision.primitive,'scaled_linear',primitive)
    identity=tensor_identity(original)
    site._prepared_compile_token=object()
    compiled=torch.compile(module,backend='inductor',fullgraph=True,dynamic=False)
    try:
        with torch.inference_mode():
            for _ in range(2):
                result=compiled(torch.zeros(2,256,1152,dtype=torch.bfloat16))
                assert torch.equal(result,original.detach().view(1,1,-1).expand(2,256,-1))
        assert len(calls)==2 and all(pointer==site.execution_bias.data_ptr() and not grad for pointer,grad in calls)
        assert module.bias is original and original.requires_grad and tensor_identity(original)==identity
        site.validate()
        with torch.no_grad(): site.execution_bias.add_(1)
        try: site.validate()
        except RuntimeError as error: assert 'immutable packing epoch' in str(error)
        else: raise AssertionError('Mutated execution clone retained its old packing epoch')
    finally:
        site._prepared_compile_token=None
