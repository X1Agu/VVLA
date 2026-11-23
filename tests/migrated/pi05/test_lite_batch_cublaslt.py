"""B32 geometry/ABI declaration guards; executed GPU support is separate."""
from copy import deepcopy
import ctypes as C
import json
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vvla_repro_pi05.lite import batch_cublaslt as batch


class _API:
    def __init__(self):self.created=[];self.attributes=[];self.retired=[]
    def check(self,operation,*args):
        if operation.endswith('Create'):
            args[0]._obj.value=1024+len(self.created)
            self.created.append((operation,args[1:]))
        else:self.retired.append(operation)
    def attribute(self,operation,handle,attribute,value):
        self.attributes.append((operation,attribute,value.value))


def test_b32_bias_descriptor_uses16384_rows_and_original_column_major_epilogue():
    api=_API();bias=SimpleNamespace(shape=(1152,),dtype=torch.bfloat16,is_contiguous=lambda:True,data_ptr=lambda:4096)
    plan=batch.BatchBiasedPlan(api,bias)
    matrices=[args for operation,args in api.created if operation=='cublasLtMatrixLayoutCreate']
    assert matrices==[(14,4304,1152,4304),(14,4304,16384,4304),(14,1152,16384,1152)]
    assert plan.m==16384 and plan.k==4304 and plan.n==1152
    assert ('cublasLtMatmulDescSetAttribute',7,4)in api.attributes
    assert ('cublasLtMatmulDescSetAttribute',8,4096)in api.attributes
    assert sum(operation=='cublasLtMatrixLayoutSetAttribute'and value==0 for operation,attribute,value in api.attributes)==3
    plan.close();assert len(api.retired)==len(api.created)


def test_actual_opaque_operator_fake_shapes_preserve32_samples_without_cuda_init():
    before=torch.cuda.is_initialized()
    with FakeTensorMode():
        x=torch.empty((32,50,1024),dtype=torch.bfloat16)
        w=torch.empty((1024,8192),dtype=torch.bfloat16)
        y=batch.ffi.expert_cublaslt(x,w,1234567,8192)
        assert y.shape==(32,50,8192)and y.dtype==torch.bfloat16
        image=torch.empty((64,256,4304),dtype=torch.bfloat16)
        weight=torch.empty((1152,4304),dtype=torch.bfloat16);bias=torch.empty((1152,),dtype=torch.bfloat16)
        output=batch.biased.vision_fc2_cublaslt(image,weight,bias,1234567)
        assert output.shape==(64,256,1152)
    assert torch.cuda.is_initialized()==before


def test_new_expert_site_rejects_b1_and_accepts_true32_geometry(monkeypatch):
    monkeypatch.setattr(torch.compiler,'is_compiling',lambda:True)
    with FakeTensorMode():
        weight=torch.empty((2560,1024),dtype=torch.bfloat16)
        plan=SimpleNamespace(spec=deepcopy(batch.GEOMETRIES['qkv']),key=1234567)
        owner=SimpleNamespace(runtime=SimpleNamespace(plans={'qkv':plan}))
        site=batch._ExpertSite(owner,'qkv',lambda:weight);site._prepared_compile_token=object()
        got=site(torch.empty((32,50,1024),dtype=torch.bfloat16))
        assert got.shape==(32,50,2560)
        with pytest.raises(ValueError,match='32,50'):site(torch.empty((1,50,1024),dtype=torch.bfloat16))
        site._prepared_compile_token=None
        with pytest.raises(RuntimeError,match='compile scope'):site(torch.empty((32,50,1024),dtype=torch.bfloat16))


def _report_fixture(tmp_path):
    cases={}
    for label,geometry in batch.GEOMETRIES.items():
        raw=tmp_path/(label+'.npz');raw.write_bytes(b'fictional declaration test only')
        cases[label]=dict(selected_spec=dict(geometry,algorithm={},workspace_bytes=0),support_passed=True,
            fullgraph_passed=True,cuda_graph_passed=True,oracle_passed=True,timing_samples=40,
            raw_path=raw.name,raw_sha256=batch.ffi.file_sha(raw))
    report=dict(status='complete',implementation=batch.VERSION,batch=32,source_sha256=batch.sources(),
        library_sha256=batch.ffi.LIBRARY_SHA256,policy_quality_passed=False,
        checkpoint_sha256='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae',cases=cases)
    path=tmp_path/'BACKEND.json'
    return report,path


@pytest.mark.parametrize('fault',['none','b1geometry','batch','source','raw','partial','timings','quality'])
def test_evidence_binds_new_source_shape_fullgraph_capture_oracle_and_raw(tmp_path,fault):
    report,path=_report_fixture(tmp_path)
    if fault=='b1geometry':report['cases']['qkv']['selected_spec']['m']=50
    elif fault=='batch':report['batch']=1
    elif fault=='source':report['source_sha256']['batch_cublaslt.py']='f'*64
    elif fault=='raw':(tmp_path/'qkv.npz').write_bytes(b'changed')
    elif fault=='partial':report['cases']['vision_fc2']['cuda_graph_passed']=False
    elif fault=='timings':report['cases']['attention_out']['timing_samples']=0
    elif fault=='quality':report['policy_quality_passed']=True
    path.write_text(json.dumps(report))
    config=dict(backend_report=str(path),backend_report_sha256=batch.ffi.file_sha(path))
    if fault=='none':assert batch.load_evidence(config)==report
    else:
        with pytest.raises(ValueError):batch.load_evidence(config)
