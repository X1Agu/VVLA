"""Actual factory/engine/owner CPU plumbing with explicitly fictional CUDA policy.

The real batch engine, owner, processor, lock, metadata and close methods run.
Only checkpoint loading, GPU provenance, cold shared collector and CUDA noise
allocation are CPU fixtures. This is not model/SM89 numerical qualification.
"""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_lite_batch_l4_compile_libdevice import fixture as owner_fixture, owner
from test_lite_batch_l4_pi05 import Policy, observations
from vvla_repro_pi05.lite import batch_l4_pi05 as batch


@pytest.fixture
def integration(owner_fixture, monkeypatch):
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy
    f = owner_fixture
    # Ensure the real factory imports the exact owner class under contract.
    monkeypatch.setitem(sys.modules, 'vvla_repro_pi05.lite.batch_l4_compile_libdevice', owner)
    policy = Policy(16); policy.device = torch.device('cuda'); policy._calls = 0
    policy.model = torch.nn.Linear(3, 4, dtype=torch.bfloat16)
    policy.config = SimpleNamespace(chunk_size=50, num_inference_steps=10,
        max_action_dim=32, action_dim=7, execution_recipe={'prefix_mlp': 'none'})
    original = batch._origin()
    import json
    original_config = json.loads(open(original['identity_path']).read())['metadata']['resolved_config']
    policy._provenance = dict(checkpoint_sha256=batch.CHECKPOINT_SHA256, resolved_config=deepcopy(original_config))
    policy._lite_optimizations = {'compile_sample_actions': True}
    policy.graph_runner._closed = False; policy.graph_runner.entries = {}
    policy.graph_runner.compile_sample_actions = True
    policy.metadata = lambda: dict(recipe_sha256='fictional-cpu-base-recipe', checkpoint_sha256=batch.CHECKPOINT_SHA256)
    events = []; fail = dict(policy=False, state=False, seal=False)
    def close():
        events.append('policy_close')
        if fail['policy']: raise RuntimeError('CPU policy close failure')
        policy.graph_runner._closed = True; policy._closed = True
    policy.close = close
    def load(*args, **kwargs):
        events.append('load_policy'); return policy
    monkeypatch.setattr(Pi05LitePolicy, 'from_pretrained', staticmethod(load))
    monkeypatch.setattr(batch, '_hardware', lambda device: {'fixture': 'CPU only, no GPU qualification'})
    monkeypatch.setattr(batch, '_install_batch_recipe', lambda policy, selected, size: events.append('install_recipe'))
    def noise(self, actual, seeds):
        events.append('noise')
        return torch.cat([torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed)) for seed in seeds])
    monkeypatch.setattr(batch.Pi05L4BatchEngine, '_noise', noise)
    from vvla_repro_pi05.lite import batch_checkpoint_engine
    class State:
        def __init__(self, actual, capability):
            assert actual is policy and policy._lock.locked() and policy.graph_runner._call_lock.locked()
            events.append('seal')
            if fail['seal']: raise RuntimeError('CPU seal failure')
        def audit(self):
            assert policy._lock.locked() and policy.graph_runner._call_lock.locked()
            events.append('audit_state')
        def _assert_live(self):
            assert policy._lock.locked()
        def detach(self, actual, runner):
            assert actual is policy and policy._lock.locked() and runner._call_lock.locked()
            events.append('detach_state')
            if fail['state']: raise RuntimeError('CPU state detach failure')
    monkeypatch.setattr(batch_checkpoint_engine, 'BatchSealedModelState', State)
    evidence = dict(test_only='fictional CPU qualification, never deploy')
    def create(**kwargs):
        return batch.Pi05L4BatchEngine.from_pretrained('fictional-checkpoint', batch_size=16,
            compile_libdevice_cache=True, batch_l4_compile_libdevice_evidence=evidence, **kwargs)
    return SimpleNamespace(**vars(f), actual_policy=policy, events=events, fail=fail,
        evidence=evidence, create=create)


@pytest.mark.parametrize('kwargs', [dict(compile_libdevice_cache=1), dict(compile_libdevice_cache=None),
    dict(batch_l4_compile_libdevice_evidence={}), dict(compile_libdevice_cache=True, batch_size=32),
    dict(compile_libdevice_cache=True, batch_size=8), dict(compile_libdevice_cache=True, batch_size=16,
        batch_l4_compile_libdevice_evidence={})])
def test_factory_flag_geometry_and_foreign_evidence_fail_before_gpu(monkeypatch, kwargs):
    def gpu(*args): raise AssertionError('GPU discovery must not be reached')
    monkeypatch.setattr(batch, '_hardware', gpu)
    # Use the genuine cold qualification reader, not the lifecycle fake.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('unmocked_cold_qualification', owner.__file__)
    real = module_from_spec(spec); spec.loader.exec_module(real)
    monkeypatch.setitem(sys.modules, 'vvla_repro_pi05.lite.batch_l4_compile_libdevice', real)
    with pytest.raises(ValueError): batch.Pi05L4BatchEngine.from_pretrained('unused', **kwargs)


def test_cold_factory_and_prepared_fixture_do_not_initialize_or_warm(integration):
    f = integration; engine = f.create()
    assert f.calls == [] and f.actual_policy.graph_runner.calls == 0
    manifest = engine.metadata()
    assert f.calls == [] and manifest['precision'] == 'bf16'
    opt = manifest['lite_optimizations']['batch_l4_compile_libdevice_cache']
    assert opt['manifest']['default_enabled'] is False and opt['manifest']['batch_size'] == 16
    prepared = engine.prepared_fixture_batch(observations(16), list(range(16)))
    assert prepared['noise_fp32'].dtype == np.float32 and prepared['noise_fp32'].shape == (16, 50, 32)
    assert f.calls == [] and f.actual_policy.graph_runner.calls == 0
    engine.close()


def test_first_infer_runs_owner_after_fresh_prepare_noise_inside_original_timer(integration, monkeypatch):
    f = integration; engine = f.create(); policy = f.actual_policy
    before = engine.metadata(); real_build = engine._Pi05L4BatchEngine__processor.build_model_inputs
    # Trace the real call chain; no first-request fixture invokes the helper.
    def build(*args, **kwargs):
        f.events.append('prepare'); return real_build(*args, **kwargs)
    monkeypatch.setattr(engine._Pi05L4BatchEngine__processor, 'build_model_inputs', build)
    original_helper = f.tasks._set_triton_libdevice_path
    original_patch = f.config.patch
    def on_helper():
        assert f.events[-2:] == ['prepare', 'noise'] and policy._lock.locked()
        f.events.append('helper')
    f.behavior['on_helper'] = on_helper
    original_run = type(policy.graph_runner).__call__
    def graph(runner, **kwargs):
        assert f.events[-1] == ('helper' if runner.calls == 0 else 'noise')
        f.events.append('graph'); return original_run(runner, **kwargs)
    monkeypatch.setattr(type(policy.graph_runner), '__call__', graph)
    f.events.clear(); call_count_before = len(f.calls)
    seeds = list(range(16)); obs = observations(16)
    engine.reset_batch([f'cpu-episode/{i}' for i in range(16)])
    first = engine.infer_batch(obs, seeds)
    assert f.events == ['prepare', 'noise', 'helper', 'graph']
    assert f.calls[call_count_before:] == [(True, False)]
    assert first['cold'] is True and policy.graph_runner.calls == 1
    diag = first['diagnostics']['compile_libdevice_cache']
    assert diag['initialization_cost_inside_request'] is True
    assert first['diagnostics']['host_submit_ms'] >= diag['initialization_ms']
    assert first['diagnostics']['total_ms'] >= first['diagnostics']['host_submit_ms']
    retained = first['actions'].copy()
    second = engine.infer_batch(obs, seeds)
    assert len(f.calls) == call_count_before + 1 and second['cold'] is False
    np.testing.assert_array_equal(second['actions'], retained)
    changed = engine.infer_batch(observations(16, changed=True), seeds)['actions']
    np.testing.assert_array_equal(changed[:-1], retained[:-1]); assert not np.array_equal(changed[-1], retained[-1])
    np.testing.assert_array_equal(first['actions'], retained)
    assert engine.metadata() == before
    assert original_helper is f.tasks._set_triton_libdevice_path and original_patch.__func__ is f.config.patch.__func__
    engine.close()


@pytest.mark.parametrize('failure', ['state', 'policy'])
def test_cleanup_failure_retains_real_owner_until_successful_retry(integration, failure):
    f = integration; engine = f.create(); actual_owner = engine._Pi05L4BatchEngine__compile_owner
    f.fail[failure] = True
    with pytest.raises(RuntimeError, match='failure'): engine.close()
    assert actual_owner._closed is False and actual_owner._anchors
    assert actual_owner._policy() is f.actual_policy and f.actual_policy._closed is False
    f.fail[failure] = False
    engine.close(); engine.close()
    assert actual_owner._closed is True and actual_owner._anchors == ()
    assert actual_owner._policy() is None and f.actual_policy._closed is True


def test_factory_seal_failure_keeps_original_exception_and_unretired_owner(integration):
    f = integration; f.fail.update(seal=True, policy=True)
    with pytest.raises(RuntimeError, match='CPU seal failure') as raised: f.create()
    error = raised.value; actual = error.l4_batch_compile_owner
    assert error.l4_batch_policy_owner is f.actual_policy and actual is not None
    assert actual._closed is False and actual._anchors and actual._policy() is f.actual_policy
    assert any('CPU policy close failure' in note for note in error.__notes__)
    f.fail['policy'] = False; error.l4_batch_policy_owner.close(); actual.close()
    assert actual._closed is True and actual._anchors == ()


def test_normal_factory_error_retires_policy_before_closing_compiler_owner(integration, monkeypatch):
    f = integration; f.fail['seal'] = True; closed = []
    original = owner.L4BatchCompileLibdeviceOwner.close
    def close(actual):
        assert f.actual_policy._closed is True
        closed.append(actual); return original(actual)
    monkeypatch.setattr(owner.L4BatchCompileLibdeviceOwner, 'close', close)
    with pytest.raises(RuntimeError, match='CPU seal failure'): f.create()
    assert len(closed) == 1 and closed[0]._anchors == ()


def test_metadata_uses_existing_locks_and_keeps_recipe_stable(integration):
    f = integration; engine = f.create()
    first = engine.metadata(); second = engine.metadata()
    assert first == second and f.calls == []
    f.actual_policy._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='active L4 request'): engine.metadata()
    finally: f.actual_policy._lock.release()
    f.actual_policy.graph_runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='active L4 native runner'): engine.metadata()
        assert not f.actual_policy._lock.locked()
    finally: f.actual_policy.graph_runner._call_lock.release()
    assert engine.metadata() == first
    engine.close()
