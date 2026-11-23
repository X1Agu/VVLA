"""Pinned-runtime CPU/FakeTensor contracts; no executed GPU evidence here."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import batch_prefix_fp8_native as candidate
from vvla_repro_pi05.lite import batch_prefix_fp8_evidence as evidence


def tensor(shape,dtype,*,stride=None,device='cuda:0',grad=False):
    return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype,device=torch.device(device),
        stride=lambda:(shape[1],1) if stride is None else stride,requires_grad=grad)


def operands(m=18432,k=2048,n=16384):
    return [tensor((m,k),torch.float8_e4m3fn),tensor((n,k),torch.float8_e4m3fn),
            tensor((m,1),torch.float32,stride=(1,1)),tensor((n,1),torch.float32,stride=(1,1))]


def test_new_namespace_and_manifest_qualify_large_rows_without_changing_b1_kernel():
    root=Path(candidate.__file__).parent
    assert (root/'batch_prefix_fp8_native_kernel.py').read_bytes()==(root/'prefix_fp8_native_kernel.py').read_bytes()
    sources=evidence.source_hashes()
    assert candidate.implementation_manifest()==evidence.expected_native_manifest(sources)
    assert candidate.SUPPORTED_CAPACITIES==(18432,20480) and candidate.PER_SLOT_CAPACITIES==(576,640)
    assert 'batch32_' in candidate.QUANTIZE_OPERATOR and 'batch32_' in candidate.SCALED_MM_OPERATOR
    value=candidate.implementation_manifest(); value['required_runtime']['capability'][0]=8
    assert candidate.implementation_manifest()['required_runtime']['capability']==[12,0]


@pytest.mark.parametrize('m',candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n',candidate.SUPPORTED_GEOMETRIES)
def test_exact_flattened_b32_geometry_accepts_metadata_only(m,k,n):
    candidate.validate_quantizer_input(tensor((m,k),torch.bfloat16))
    candidate.validate_scaled_mm_inputs(*operands(m,k,n))


@pytest.mark.parametrize('change',[dict(shape=(576,2048)),dict(shape=(640,2048)),dict(shape=(32,2048)),
    dict(shape=(18432,1024)),dict(dtype=torch.float32),dict(ndim=3),dict(requires_grad=True),
    dict(device=torch.device('cpu')),dict(stride=lambda:(1,18432))])
def test_b1_rows_casts_devices_transposes_and_training_are_rejected(change):
    value=tensor((18432,2048),torch.bfloat16); value.__dict__.update(change)
    with pytest.raises(ValueError): candidate.validate_quantizer_input(value)


@pytest.mark.parametrize('slot,change',[(0,dict(shape=(576,2048))), (0,dict(dtype=torch.bfloat16)),
    (1,dict(shape=(8192,2048))), (1,dict(stride=lambda:(1,16384))),
    (1,dict(device=torch.device('cuda:1'))), (2,dict(dtype=torch.bfloat16)),
    (2,dict(shape=(1,18432))), (3,dict(stride=lambda:(2,1))),
    (3,dict(requires_grad=True)), (3,dict(ndim=1))])
def test_scale_axes_dtype_geometry_and_ownership_fail_closed(slot,change):
    values=operands(); values[slot].__dict__.update(change)
    with pytest.raises(ValueError): candidate.validate_scaled_mm_inputs(*values)


def test_runtime_does_not_adapt_to_other_architectures_binaries_or_autocast(monkeypatch):
    monkeypatch.setattr(torch,'__version__',candidate.REQUIRED_RUNTIME['torch'])
    monkeypatch.setattr(torch.version,'cuda',candidate.REQUIRED_RUNTIME['cuda'])
    monkeypatch.setattr(torch.version,'git_version',candidate.REQUIRED_RUNTIME['torch_git'])
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(12,0))
    monkeypatch.setattr(torch,'is_autocast_enabled',lambda device:False)
    candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,9))
    with pytest.raises(RuntimeError): candidate.validate_runtime(torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(12,0))
    monkeypatch.setattr(torch,'is_autocast_enabled',lambda device:True)
    with pytest.raises(RuntimeError): candidate.validate_runtime(torch.device('cuda:0'))


@pytest.mark.parametrize('m',candidate.SUPPORTED_CAPACITIES)
@pytest.mark.parametrize('k,n',candidate.SUPPORTED_GEOMETRIES)
def test_fake_fullgraph_has_two_new_opaque_ops_fresh_outputs_and_exact_shapes(m,k,n):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    for name in ('batch32_prefix_fp8_quantize_rows_v1','batch32_prefix_fp8_native_scaled_mm_v1'):
        schema=getattr(torch.ops.vvla_repro_pi05_lite,name).default._schema
        assert all(arg.alias_info is None for arg in schema.arguments)
        assert all(result.alias_info is None for result in schema.returns)
    with FakeTensorMode():
        x=torch.empty((m,k),device='cuda:0',dtype=torch.bfloat16)
        qw=torch.empty((n,k),device='cuda:0',dtype=torch.float8_e4m3fn)
        sw=torch.empty((n,1),device='cuda:0',dtype=torch.float32)
        y,qx,sx=candidate.scaled_linear(x,qw,sw)
        assert y.shape==(m,n) and y.stride()==(n,1) and y.dtype==torch.bfloat16
        assert qx.shape==(m,k) and qx.dtype==torch.float8_e4m3fn
        assert sx.shape==(m,1) and sx.stride()==(1,1) and sx.dtype==torch.float32
        assert len({id(value) for value in (x,qw,sw,y,qx,sx)})==6
        graph=make_fx(candidate.scaled_linear)(x,qw,sw)
    calls=[str(node.target) for node in graph.graph.nodes if node.op=='call_function']
    assert calls.count('vvla_repro_pi05_lite.batch32_prefix_fp8_quantize_rows_v1.default')==1
    assert calls.count('vvla_repro_pi05_lite.batch32_prefix_fp8_native_scaled_mm_v1.default')==1
    assert not any('aten._scaled_mm' in name for name in calls)
