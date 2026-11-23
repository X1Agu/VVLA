from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vvla_lingbot import private_lite as private
from vvla_lingbot.lite_tuning.state_guard import FrozenStateGuard


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(3),requires_grad=False)
        self.config=SimpleNamespace(to_dict=lambda:{'fixed':True})


class Engine:
    def __init__(self,flow):self.flow,self._frozen_guard=flow,FrozenStateGuard(flow)
    def _state_signature(self):return self._frozen_guard.check()


class Adapter:
    def __init__(self,config):
        flow=Flow()
        self._runtime=SimpleNamespace(closed=False,policy=SimpleNamespace(vla=SimpleNamespace(model=flow)),
            engine=Engine(flow))
        self._closed=self._failed=self.fail_close=False
    def _check_live(self):pass
    @contextmanager
    def _exclusive(self):yield
    def metadata(self):return {'recipe_sha256':'baseline','recipe':{'precision':'bf16'}}
    def predict(self,raw,noise):
        self._runtime.engine._frozen_guard.check()
        return {'actions':list(noise)}
    def infer(self,raw,seed):return self.predict(raw,[seed])
    def close(self):
        if self.fail_close:raise RuntimeError('injected drain failure')
        self._closed=True;self._runtime.closed=True


@pytest.fixture
def build(monkeypatch):
    created=[]
    def factory(config):
        adapter=Adapter(config);created.append(adapter);return adapter
    monkeypatch.setattr(private,'LingBotLitePolicy',factory)
    return created


def test_private_facade_does_not_export_model_or_accept_external_mutation(build):
    owner=private.factory({})
    assert not hasattr(owner,'__dict__') and not hasattr(owner,'_runtime')
    with pytest.raises(AttributeError):owner.model=object()
    assert owner.predict({},[1,2])=={'actions':[1,2]}
    meta=owner.metadata();meta['private_owner']['guard']='caller modification'
    assert owner.metadata()['private_owner']['guard']=='private_generation_and_ownership'
    owner.close()
    with pytest.raises(RuntimeError,match='closed'):
        owner.predict({},[1])


def test_explicit_full_audit_detects_unsupported_private_weight_mutation(build):
    owner=private.factory({});adapter=build[0]
    with torch.no_grad():adapter._runtime.engine.flow.weight.add_(1)
    with pytest.raises(RuntimeError,match='tensor'):
        owner.metadata()
    owner.close()


def test_owner_detachment_detected_by_cheap_serving_check(build):
    owner=private.factory({});adapter=build[0]
    adapter._runtime.policy.vla.model=Flow()
    with pytest.raises(RuntimeError,match='detached'):
        owner.predict({},[1])
    owner.close()


def test_failed_close_keeps_guard_epoch_live_for_drain_retry(build):
    owner=private.factory({});adapter=build[0]
    adapter.fail_close=True
    with pytest.raises(RuntimeError,match='drain'):owner.close()
    assert not adapter._closed
    adapter.fail_close=False;owner.close();owner.close()
    assert adapter._closed


def test_private_full_guard_control_still_checks_weight_versions_each_request(build):
    owner=private.factory({},sealed_guard=False);adapter=build[0]
    assert owner.metadata()['private_owner']['guard']=='full_per_request'
    with torch.no_grad():adapter._runtime.engine.flow.weight.add_(1)
    with pytest.raises(RuntimeError,match='tensor'):
        owner.predict({},[1])
    owner.close()


def test_successful_close_releases_the_cold_snapshot_while_facade_is_retained(build):
    import gc,weakref
    owner=private.factory({});adapter=build[0]
    model=weakref.ref(adapter._runtime.engine.flow)
    original=adapter.close
    def close():original();adapter._runtime=None
    adapter.close=close
    owner.close();gc.collect()
    assert model() is None
