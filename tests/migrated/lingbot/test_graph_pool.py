"""Exercise cache switching, storage ownership and failed-drain retention.

These use a closure-backed fake device to expose stale captured pointers. Real
model/CUDA parity, memory use and latency are separate remote qualifications.
"""
from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class Tensor:
    def __init__(self, value, dtype='torch.bfloat16'):
        self.value = np.array(value,copy=True)
        self.shape, self.dtype, self.device = self.value.shape,dtype,'cuda:0'
        self._version = 0
    def data_ptr(self): return id(self.value)
    def cpu(self): return self
    def numpy(self): return self.value
    def clone(self): return Tensor(self.value,self.dtype)
    def copy_(self, other): self.value[...] = other.value
    def zero_(self): self.value[...] = 0


class Device:
    def __init__(self):
        self.capture = None
        self.fail_drain = False
        self.events = []
    def is_current_stream_capturing(self): return False
    def synchronize(self):
        self.events.append('drain')
        if self.fail_drain: raise RuntimeError('injected drain failure')
    def current_stream(self): return SimpleNamespace(wait_stream=lambda _:None)
    def Stream(self): return self.current_stream()
    def stream(self, stream): return nullcontext()
    def CUDAGraph(self):
        device = self
        class Graph:
            fail_reset = False
            def replay(self): self.run()
            def reset(self):
                device.events.append('reset')
                if self.fail_reset: raise RuntimeError('injected reset failure')
        return Graph()
    @contextmanager
    def graph(self, graph, stream):
        self.capture = graph
        try: yield
        finally: self.capture = None


@pytest.fixture
def pool_class(monkeypatch):
    # Loading the cache alone must not import the CUDA-only inference stack.
    stub = ModuleType('vvla_lingbot.lite_tuning.engine')
    stub.TunedEngine = object
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    name = 'vvla_lingbot.lite_tuning._isolated_pool_tests'
    import vvla_lingbot
    p = Path(vvla_lingbot.__file__).parent/'lite_tuning/graph_pool.py'
    spec = importlib.util.spec_from_file_location(name,p)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    device = Device()
    monkeypatch.setattr(module.torch,'cuda',device)
    class Pool(module.PooledTunedEngine):
        def __init__(self, capacity):
            from collections import OrderedDict
            import threading
            self._entries, self._capacity = OrderedDict(),capacity
            self._entry_lock = threading.Lock()
            self._pool_failed = self._pool_closed = False
            self._evictions = 0
            self._signature = self._graph = self._meta_key = None
            self.graph_diagnostics = None
            self.recipe = {'graph_builds':0}
            self.qwe = SimpleNamespace()
            self.action = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace()))])
            self.token = object()
        def _state_signature(self): return self.token
        def _prepare(self, *vals):
            for field in module.ENTRY_FIELDS: setattr(self,field,object())
            self.static = [v.clone() for v in vals]
            self.noise = Tensor([0])
            self.p = int(vals[3].value.sum())
            self.template = Tensor([100*self.p])
            self._constant_tensors = [self.template]
            self._constant_token = tuple((id(t),t.data_ptr(),t._version) for t in self._constant_tensors)
            self._signature = self.token
            for field in module.VISION_FIELDS: setattr(self.qwe,field,Tensor([self.p]))
            e = self.action.layers[0].mlp.experts
            e._robby_moe_workspace = {'value':Tensor([self.p])}
            e._robby_moe_workspace_key = self.p
        def _execute(self):
            static,noise,const = self.static,self.noise,self.template
            vision = self.qwe.pos_embeds
            ws = self.action.layers[0].mlp.experts._robby_moe_workspace
            output = Tensor([0.])
            def execute():
                output.value[...] = static[0].value.sum()+noise.value.sum()+const.value.sum()+vision.value.sum()+ws['value'].value.sum()
            execute()
            if device.capture is not None: device.capture.run = execute
            return output
    return Pool,device


def call(pool,key,value=1,noise=7):
    return pool(Tensor([value]),Tensor([[True]],'torch.bool'),Tensor([[11,22]],'torch.int64'),
        Tensor([[True,key=='B']],'torch.bool'),Tensor([0]),noise=Tensor([noise]),
        image_grid_thw=Tensor([[[1,16,16]]],'torch.int64'))


def test_a_a_b_a_restores_captured_constants_workspace_and_owned_outputs(pool_class):
    Pool,_ = pool_class
    p = Pool(2)
    a = call(p,'A')
    old = a.value.copy()
    assert call(p,'A',value=9).value == 118
    assert not p.last_call['graph_build']
    assert call(p,'B').value == 212
    assert p.action.layers[0].mlp.experts._robby_moe_workspace_key == 2
    assert call(p,'A',noise=13).value == 116
    assert p.action.layers[0].mlp.experts._robby_moe_workspace_key == 1
    assert p.recipe['graph_builds'] == 2 and len(p._entries) == 2
    assert np.array_equal(a.value,old)
    p.close()


def test_eviction_rebuilds_without_reusing_evicted_graph_storage(pool_class):
    Pool,device = pool_class
    p = Pool(1)
    for key in ['A','B','A']:
        call(p,key)
    assert p.recipe['graph_builds'] == 3 and p._evictions == 2
    assert device.events.count('reset') == 2
    p.close()
    assert device.events.count('reset') == 3


def test_inactive_constant_mutation_is_rejected_before_target_replay(pool_class):
    Pool,_ = pool_class
    p = Pool(2)
    call(p,'A'); target = next(iter(p._entries.values()))
    call(p,'B')
    target.fields['template']._version += 1
    with pytest.raises(RuntimeError,match='metadata'):
        call(p,'A')
    assert p._pool_failed and len(p._entries) == 2
    p.close()


def test_failed_drain_and_reset_preserve_owners_for_close_retry(pool_class):
    Pool,device = pool_class
    p = Pool(2)
    call(p,'A');call(p,'B')
    entries = list(p._entries.values())
    device.fail_drain = True
    with pytest.raises(RuntimeError,match='drain'):
        p.close()
    assert list(p._entries.values()) == entries
    device.fail_drain = False
    entries[0].fields['_graph'].fail_reset = True
    with pytest.raises(RuntimeError,match='reset'):
        p.close()
    assert list(p._entries.values()) == entries
    entries[0].fields['_graph'].fail_reset = False
    p.close()
    assert not p._entries and p._pool_closed


def test_failed_eviction_never_replays_and_keeps_previous_entry(pool_class):
    Pool,device = pool_class
    p = Pool(1)
    call(p,'A'); entry = next(iter(p._entries.values()))
    device.fail_drain = True
    with pytest.raises(RuntimeError,match='drain'):
        call(p,'B')
    assert next(iter(p._entries.values())) is entry and p._pool_failed
    device.fail_drain = False
    with pytest.raises(RuntimeError,match='failed'):
        call(p,'A')
    p.close()
