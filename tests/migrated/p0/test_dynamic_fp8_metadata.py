"""Fake CUDA metadata contracts only; no actual kernel/backend qualification."""
from pathlib import Path
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'vllm-vla-plugin/src'))
from vvla_pi05.lite import dynamic_prefix_fp8_native as native


@pytest.mark.parametrize('batch',native.BATCH_SIZES)
@pytest.mark.parametrize('prefix',native.PER_SLOT_CAPACITIES)
def test_qualified_geometry_declarations_propagate_true_batch_shapes(batch,prefix):
    with FakeTensorMode():
        x=torch.empty((batch*prefix,2048),device='cuda',dtype=torch.bfloat16)
        weight=torch.empty((16384,2048),device='cuda',dtype=torch.float8_e4m3fn)
        scale=torch.empty((16384,1),device='cuda',dtype=torch.float32)
        result,qx,sx=native.scaled_linear(x,weight,scale)
        assert result.shape==(batch*prefix,16384) and result.dtype==torch.bfloat16
        assert qx.shape==x.shape and qx.dtype==torch.float8_e4m3fn
        assert sx.shape==(batch*prefix,1) and sx.dtype==torch.float32


@pytest.mark.parametrize('shape,dtype', [((3*576,2048),torch.bfloat16),
    ((576,1024),torch.bfloat16),((576,2048),torch.float16)])
def test_unqualified_geometry_and_dtype_cannot_silently_enter_backend(shape,dtype):
    with FakeTensorMode():
        x=torch.empty(shape,device='cuda',dtype=dtype)
        with pytest.raises(ValueError,match='explicit'):
            native.quantize_rows(x)


def test_cpu_and_training_are_rejected_before_any_backend_dispatch():
    with FakeTensorMode():
        for device,grad in [('cpu',False),('cuda',True)]:
            x=torch.empty((576,2048),device=device,dtype=torch.bfloat16,requires_grad=grad)
            with pytest.raises(ValueError,match='explicit'):
                # Validate before autograd's fake-CUDA wrapper. This local
                # Mac build has no CUDA autograd device guard implementation.
                native.validate_quantizer_input(x)
