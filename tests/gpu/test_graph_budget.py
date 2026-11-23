"""Production pool with real CUDA graphs and a two-layer surrogate workload.

This tests allocation/eviction mechanics, not LingBot model correctness.
"""
from collections import OrderedDict
import gc
import importlib.util
import json
from pathlib import Path
import sys
import threading
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vvla_runtime.memory import GraphMemoryBudget, cuda_snapshot

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


def pool_type(monkeypatch):
    # The pool itself only needs Torch. Isolate the full 6B engine constructor;
    # allocation, graph capture/replay, entry restoration and close stay real.
    stub=ModuleType('vvla_lingbot.lite_tuning.engine');stub.TunedEngine=object
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    import vvla_lingbot
    name='vvla_lingbot.lite_tuning._real_cuda_pool_test'
    spec=importlib.util.spec_from_file_location(name,Path(vvla_lingbot.__file__).parent/'lite_tuning/graph_pool.py')
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
    class Pool(module.PooledTunedEngine):
        def __init__(self,budget):
            self.weights=[torch.randn((128,128),device='cuda',dtype=torch.bfloat16)/16 for _ in range(2)]
            self._entries=OrderedDict();self._capacity=2;self._entry_lock=threading.Lock()
            self._pool_failed=self._pool_closed=False;self._evictions=0
            self._signature=self._graph=self._meta_key=None
            self.graph_diagnostics=None;self.recipe={'graph_builds':0}
            self.qwe=SimpleNamespace();self.action=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace()))])
            self._memory_budget=GraphMemoryBudget(budget,cuda_snapshot(),capture_reserve_bytes=0)
        def _state_signature(self):return tuple((id(w),w._version) for w in self.weights)
        def _prepare(self,*values):
            for name in module.ENTRY_FIELDS:setattr(self,name,None)
            self.static=[v.clone() for v in values]
            self.noise=torch.zeros_like(values[4]);self.p=int(values[3].sum().item())
            self.template=values[4].clone();self._constant_tensors=[self.template]
            self._constant_token=tuple((id(t),t.data_ptr(),t._version) for t in self._constant_tensors)
            self._signature=self._state_signature()
            for name in module.VISION_FIELDS:setattr(self.qwe,name,torch.ones(4,device='cuda'))
            expert=self.action.layers[0].mlp.experts
            expert._robby_moe_workspace={'scratch':torch.ones(64,device='cuda')}
            expert._robby_moe_workspace_key=self.p
        def _execute(self):
            return torch.relu(self.static[4]@self.weights[0])@self.weights[1]+self.noise
    return Pool


def inputs(length):
    return [torch.zeros((1,3,4),device='cuda'),torch.ones((1,3),device='cuda',dtype=torch.bool),
            torch.ones((1,length),device='cuda',dtype=torch.long),torch.ones((1,length),device='cuda',dtype=torch.bool),
            torch.ones((1,128),device='cuda',dtype=torch.bfloat16),torch.ones((1,3,3),device='cuda',dtype=torch.long)]


def call(pool,values,noise=None):
    images,im,tokens,lm,state,grid=values
    return pool(images,im,tokens,lm,state,noise=torch.zeros_like(state) if noise is None else noise,image_grid_thw=grid)


def test_actual_capture_eviction_budget_and_repeated_close(monkeypatch):
    Pool=pool_type(monkeypatch);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    baseline=torch.cuda.memory_allocated();after=[];reports=[]
    for cycle in range(3):
        pool=Pool(256*1024**2);retained=None
        for i in range(12):
            values=inputs(4+i%3);output=call(pool,values)
            with torch.no_grad():expected=torch.relu(values[4]@pool.weights[0])@pool.weights[1]
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
            if retained is None:retained=output;frozen=output.clone()
            assert len(pool._entries)<=2
            assert pool.last_call['memory']['resident_charge_bytes']<=256*1024**2
        torch.testing.assert_close(retained,frozen,rtol=0,atol=0)
        assert pool._evictions>=10
        weights=[weakref.ref(w) for w in pool.weights]
        reports.append(pool.last_call['memory']);pool.close();pool.close()
        assert not pool._entries and pool._graph is None
        del pool,retained,frozen,output,expected,values
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();after.append(torch.cuda.memory_allocated())
        assert all(ref() is None for ref in weights)
    Path('artifacts/graph-memory-cuda.json').write_text(json.dumps(dict(baseline=baseline,after_close=after,reports=reports),indent=2))
    # Torch/cuBLAS keeps a process workspace after its first GEMM. The contract
    # is release of this owner's tensors and no growth after repeated cycles,
    # not zero allocated/reserved bytes on a warmed CUDA context.
    assert max(after)-min(after)<1024**2,after


def test_oversized_capture_is_poisoned_before_replay(monkeypatch):
    values=inputs(4);noise=torch.zeros_like(values[4]);pool=pool_type(monkeypatch)(1)
    with pytest.raises(MemoryError,match='exceeds'):call(pool,values,noise)
    assert pool._pool_failed and not pool._entries
    with pytest.raises(RuntimeError,match='failed'):call(pool,inputs(4))
    pool.close();pool.close()
