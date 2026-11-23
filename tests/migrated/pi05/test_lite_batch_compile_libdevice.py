"""Cold compiler owner CPU mechanics; fictional evidence is never GPU support."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

PATH=Path(__file__).resolve().parents[1]/'src/vvla_repro_pi05/lite/batch_compile_libdevice.py'
spec=importlib.util.spec_from_file_location('batch_compile_libdevice_cpu_contract',PATH)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


@pytest.fixture
def fixture(monkeypatch):
    proof=dict(library=dict(path='/test-only/toolkit-libdevice',sha256=owner.LIBDEVICE_SHA256))
    class Config:
        emulate_precision_casts=False
        eager_numerics=SimpleNamespace(use_pytorch_libdevice=False)
        @contextmanager
        def patch(self,options):
            assert options==owner.OPTIONS
            previous=self.emulate_precision_casts
            self.emulate_precision_casts=options['emulate_precision_casts']
            try:yield
            finally:self.emulate_precision_casts=previous
    config=Config();knobs=SimpleNamespace(libdevice_path=None);calls=[]
    def helper():
        calls.append((config.emulate_precision_casts,config.eager_numerics.use_pytorch_libdevice))
        assert config.emulate_precision_casts is True
        monkeypatch.setenv('TRITON_LIBDEVICE_PATH',proof['library']['path'])
        knobs.libdevice_path=proof['library']['path']
    tasks=SimpleNamespace(_set_triton_libdevice_path=helper)
    compilation=SimpleNamespace(INDUCTOR_OPTIONS=deepcopy(owner.OPTIONS))
    monkeypatch.setattr(owner,'load_qualification',lambda evidence:deepcopy(proof))
    runtime=dict(original_compile_options=deepcopy(owner.OPTIONS),test_only='fictional CPU runtime')
    monkeypatch.setattr(owner,'_runtime',lambda policy,evidence:deepcopy(runtime))
    monkeypatch.setattr(owner,'_bindings',lambda:(config,tasks,knobs,compilation))
    monkeypatch.delenv('TRITON_LIBDEVICE_PATH',raising=False)
    class Policy:
        _closed=False
        device=SimpleNamespace(type='cuda')
        graph_runner=SimpleNamespace(_closed=False,compile_sample_actions=True)
        _lite_optimizations={'compile_sample_actions':True}
        _lock=threading.Lock()
    policy=Policy()
    value=owner.BatchCompileLibdeviceOwner(policy,{'test_only':'fictional CPU report'})
    return policy,value,config,tasks,knobs,compilation,calls,runtime,proof


def test_construct_and_manifest_do_not_call_official_helper_or_mutate_original_options(fixture):
    policy,value,config,tasks,knobs,compilation,calls,runtime,proof=fixture
    assert calls==[] and value.diagnostics()=={'initialized':False}
    assert config.emulate_precision_casts is False and knobs.libdevice_path is None
    assert compilation.INDUCTOR_OPTIONS==owner.OPTIONS
    before=value.manifest();before['actual']['original_compile_options'].clear()
    assert value.manifest()['actual']['original_compile_options']==owner.OPTIONS
    assert value.audit(policy)==value.manifest()


def test_once_inside_request_lock_original_options_and_stable_metadata(fixture):
    policy,value,config,tasks,knobs,compilation,calls,*_=fixture
    manifest=value.manifest()
    with pytest.raises(RuntimeError,match='request lock'):value.ensure_initialized(policy)
    with policy._lock:
        value.ensure_initialized(policy);value.ensure_initialized(policy)
    assert calls==[(True,False)] and config.emulate_precision_casts is False
    assert value.audit(policy)==manifest
    diag=value.diagnostics();assert diag['initialized']is True and diag['initialization_cost_inside_request']is True
    assert diag['global_precision_before']==diag['global_precision_after']
    diag['initialized']=False;assert value.diagnostics()['initialized']is True


@pytest.mark.parametrize('fault',['helper','options_replace','options_value','options_type','patch','env','knob','runner_closed','policy_closed','foreign_policy','compile_route','global_precision'])
def test_cheap_guard_rejects_changed_current_owner_math_or_library(fixture,monkeypatch,fault):
    policy,value,config,tasks,knobs,compilation,calls,*_=fixture
    other=policy
    if fault=='helper':tasks._set_triton_libdevice_path=lambda:None
    elif fault=='options_replace':compilation.INDUCTOR_OPTIONS=dict(compilation.INDUCTOR_OPTIONS)
    elif fault=='options_value':compilation.INDUCTOR_OPTIONS['use_fast_math']=True
    elif fault=='options_type':compilation.INDUCTOR_OPTIONS['use_fast_math']=0
    elif fault=='patch':config.patch=lambda *args:None
    elif fault=='env':monkeypatch.setenv('TRITON_LIBDEVICE_PATH','/foreign')
    elif fault=='knob':knobs.libdevice_path='/foreign'
    elif fault=='runner_closed':policy.graph_runner=SimpleNamespace(_closed=True)
    elif fault=='policy_closed':policy._closed=True
    elif fault=='compile_route':policy.graph_runner.compile_sample_actions=False
    elif fault=='global_precision':config.emulate_precision_casts=True
    else:
        class Other:pass
        other=Other()
    with pytest.raises(RuntimeError):value.ensure_initialized(other)
    assert calls==[]


@pytest.mark.parametrize('fault',['runtime','proof'])
def test_full_audit_rejects_current_source_or_report_change(fixture,fault):
    policy,value,config,tasks,knobs,compilation,calls,runtime,proof=fixture
    if fault=='runtime':runtime['changed']='actual binary or source'
    else:proof['changed']='qualification'
    with pytest.raises(RuntimeError):value.audit(policy)


def test_failed_initialization_can_retire_and_never_claims_initialized(fixture,monkeypatch):
    policy,value,config,tasks,knobs,compilation,calls,*_=fixture
    # Keep the original callable identity; the real helper's supported failure
    # can be emulated by making its required precision context fail to enter.
    @contextmanager
    def fail(options):raise RuntimeError('context failure');yield
    config.patch=fail
    with pytest.raises(RuntimeError):
        with policy._lock:value.ensure_initialized(policy)
    assert value.diagnostics()['initialized']is False
    value.close();value.close()
    with pytest.raises(RuntimeError):value.audit(policy)


@pytest.mark.parametrize('bad',[None,{}, {'backend_report':'missing','backend_report_sha256':'f'*64},
    {'backend_report':'missing','backend_report_sha256':owner.REPORT_SHA256,'extra':True}])
def test_foreign_or_incomplete_qualification_cannot_enable_production(bad):
    with pytest.raises(ValueError):owner.load_qualification(bad)
