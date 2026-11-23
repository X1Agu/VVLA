"""Colab CPU tests: real policy construction with injected late failures.

These exercise actual QKV/MLP/Lt ownership and restoration using the test-only
CPU Lt runtime; no CUDA init, checkpoint load or native extension build occurs.
"""
from types import SimpleNamespace

import pytest

from vvla_repro_pi05.lite import expert_cublaslt as lt
from vvla_repro_pi05.lite import cublaslt_runtime as rt
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy
from test_lite_expert_cublaslt import SmallExpert, FakeRuntime


@pytest.mark.parametrize('failure_stage,variant', [('manifest', 'v1'), ('native_lifecycle', 'v1'),
                                                  ('native_lifecycle', 'v2'), ('native_install', 'v2'),
                                                  ('bind_runner', 'v1')])
def test_late_policy_constructor_failure_retires_lt_and_restores_original_modules(monkeypatch, failure_stage, variant):
    model = SmallExpert()
    processor = SimpleNamespace(config=SimpleNamespace())
    original_parameters = tuple(id(value) for value in model.parameters())
    original_state_keys = tuple(model.state_dict())
    previous_plans = set(rt._PLANS)
    runtimes = []

    class RecordingRuntime(FakeRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runtimes.append(self)

        def manifest(self):
            if failure_stage == 'manifest':
                raise RuntimeError('injected manifest failure after Lt installation')
            return super().manifest()

    monkeypatch.setattr(lt, 'ExpertLtRuntime', RecordingRuntime)
    if failure_stage in ('native_lifecycle', 'native_install'):
        from vvla_repro_pi05.lite import native_lifecycle
        def fail_native(*, variant):
            assert variant == expected_variant
            raise RuntimeError('injected native lifecycle load failure')
        expected_variant = variant
        if failure_stage == 'native_lifecycle':
            monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', fail_native)
        else:
            sentinel = object()
            def loaded(*, variant):
                assert variant == expected_variant
                return sentinel
            def fail_install(model, *, collector):
                assert collector is sentinel
                raise RuntimeError('injected native collector installation failure')
            monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', loaded)
            monkeypatch.setattr(native_lifecycle, 'install_native_lifecycle_signature', fail_install)
    if failure_stage == 'bind_runner':
        def fail_bind(*args, **kwargs):
            raise RuntimeError('injected runner binding failure')
        monkeypatch.setattr(lt.ExpertCublasLtInstallation, 'bind_runner', fail_bind)

    policy = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        policy.__init__(model, processor, device='cuda', cuda_graph=True,
                        fused_expert_qkv=True, fused_expert_gelu=True,
                        native_lifecycle=failure_stage in ('native_lifecycle', 'native_install'),
                        native_lifecycle_variant=variant,
                        expert_cublaslt={'profile': 'l4_exact_three', 'include_down': False})
    assert policy._closed
    assert len(runtimes) == 1 and runtimes[0].closed
    assert all(plan.closed for plan in runtimes[0].plans.values())
    assert set(rt._PLANS) == previous_plans
    assert not hasattr(model, lt.INSTALLATION)
    assert not hasattr(model, '_lite_expert_qkv_installation')
    assert not hasattr(model, '_lite_expert_mlp_installation')
    assert 'forward' not in model.layer.self_attn.o_proj.__dict__
    assert 'forward' not in model.layer.mlp.__dict__
    assert tuple(id(value) for value in model.parameters()) == original_parameters
    assert tuple(model.state_dict()) == original_state_keys
    if failure_stage == 'bind_runner':
        assert policy.graph_runner._closed and not policy.graph_runner.entries
    else:
        assert policy.graph_runner is None
    policy.close()  # Controller finally blocks can safely repeat cleanup.


def test_argument_validation_failure_has_no_partial_resource_state():
    model = SmallExpert()
    policy = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(ValueError, match='boolean'):
        policy.__init__(model, SimpleNamespace(config=SimpleNamespace()),
                        device='cpu', cuda_graph=False, fused_expert_qkv='yes')
    assert policy._closed and policy.graph_runner is None
    policy.close()


def test_constructor_cleanup_failure_preserves_original_exception_with_note(monkeypatch):
    model = SmallExpert()
    policy = Pi05LitePolicy.__new__(Pi05LitePolicy)
    def fail_cleanup(self):
        raise RuntimeError('injected retirement failure')
    monkeypatch.setattr(Pi05LitePolicy, 'close', fail_cleanup)
    with pytest.raises(ValueError, match='boolean') as caught:
        policy.__init__(model, SimpleNamespace(config=SimpleNamespace()),
                        device='cpu', cuda_graph=False, fused_expert_qkv='yes')
    assert any('retirement failure' in note for note in caught.value.__notes__)
