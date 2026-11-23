"""Remote CPU contracts; no CUDA calls, checkpoint or model execution.

Run only in the root-scheduled Colab queue. Native compilation is host-only and
is shared with the existing v2 lifecycle tests/probe build cache.
"""
from contextlib import contextmanager
import gc
import threading
import weakref

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import request_identity as lookup
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import install_native_lifecycle_signature, load_native_lifecycle
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity

from test_lite_expert_cublaslt import make_meta_case, close_case


@pytest.fixture(scope='module')
def collector():
    initialized = torch.cuda.is_initialized()
    result = load_native_lifecycle(variant='v2')
    assert torch.cuda.is_initialized() == initialized
    return result


@contextmanager
def case(collector, layers=1):
    model, qkv, mlp, lt, runner = make_meta_case(layers=layers)
    native = install_native_lifecycle_signature(model, collector=collector)
    handle = lookup.install_request_identity_lookup(model)
    try:
        yield model, qkv, mlp, lt, runner, native, handle
    finally:
        runner.close()
        handle.uninstall()
        native.uninstall()
        lt.uninstall()
        mlp.uninstall()
        qkv.uninstall()


def snapshot(case):
    model, _, _, _, _, native, _ = case
    state = tuple(native.signature())
    generation = model._inference_cache.refresh(state)
    return state, generation


def prepare(case, identities=None, generation=None):
    model, qkv, mlp, lt, *_ = case
    if generation is None:
        generation = model._inference_cache.generation
    kwargs = {} if identities is None else {'identity_lookup': identities}
    qkv.prepare(generation, **kwargs)
    mlp.prepare(generation, **kwargs)
    lt.prepare(generation, **kwargs)


def test_all_six_original_fields_are_same_call_collector_tuples_and_private_reads_once(collector, monkeypatch):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        assert state == tuple(collect_lifecycle_signature(c[0]))
        calls = []
        original = lookup.tensor_identity
        monkeypatch.setattr(lookup, 'tensor_identity', lambda tensor: (calls.append(id(tensor)), original(tensor))[1])
        with c[6].prepare_scope(state, generation) as identities:
            table = {entry[0]: entry for group in state[:2] for entry in group}
            for tensor in (*c[0].parameters(), *c[0].buffers()):
                result = identities.identity(tensor)
                assert result is table[id(tensor)] and len(result) == 6
                assert result == tensor_identity(tensor)
            private = torch.ones(3)
            assert identities.identity(private) == identities.identity(private) == tensor_identity(private)
            assert calls == [id(private)]
            with pytest.raises(RuntimeError, match='Registered weights'):
                identities.forget(next(c[0].parameters()))
        assert identities._registered == identities._private == {}
        with pytest.raises(RuntimeError, match='expired'):
            identities.identity(private)


@pytest.mark.parametrize('mutation', ['inplace', 'replace', 'data_pointer', 'shape', 'dtype', 'device', 'buffer'])
def test_every_request_collects_live_original_identity_fields(collector, mutation):
    # A small CPU parameter permits real version/pointer/shape mutations without
    # allocating the expert weights. It is not used by the projection fixtures.
    with case(collector) as c, c[4]._call_lock:
        model = c[0]
        model.extra = nn.Parameter(torch.ones(2, 3))
        model.register_buffer('extra_buffer', torch.ones(2))
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            before = identities.identity(model.extra)
            before_buffer = identities.identity(model.extra_buffer)
        with torch.no_grad():
            if mutation == 'inplace': model.extra.add_(1)
            elif mutation == 'replace': model.extra = nn.Parameter(model.extra.clone())
            elif mutation == 'data_pointer': model.extra.data = model.extra.data.clone()
            elif mutation == 'shape': model.extra.data = model.extra.data.view(3, 2)
            elif mutation == 'dtype': model.extra.data = model.extra.data.bfloat16()
            elif mutation == 'device': model.extra = nn.Parameter(model.extra.to('meta'))
            elif mutation == 'buffer': model.extra_buffer.add_(1)
        state, next_generation = snapshot(c)
        assert next_generation != generation
        with c[6].prepare_scope(state, next_generation) as identities:
            after = identities.identity(model.extra)
            assert after == tensor_identity(model.extra)
            if mutation == 'buffer':
                assert identities.identity(model.extra_buffer) != before_buffer
            else:
                assert after != before


def test_private_copy_forget_refresh_and_standalone_do_not_inherit_scope(collector):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
            packer, candidate = c[1]._packers[0], c[2]._candidates[0]
            before = identities.identity(packer._packed)
            # Deliberately pre-read private state before refresh. refresh must
            # forget it before its copies, then publish the current version.
            packer.refresh(identity_lookup=identities)
            candidate.refresh(identity_lookup=identities)
            assert identities.identity(packer._packed) != before
            assert identities.identity(packer._packed) == tensor_identity(packer._packed)
            assert identities.identity(candidate._packed) == tensor_identity(candidate._packed)
            with torch.no_grad():
                packer._packed.add_(1)
            # No identity_lookup argument => always live even inside a scope.
            with pytest.raises(RuntimeError, match='packed cache'):
                packer.assert_ready()
        with pytest.raises(RuntimeError, match='packed cache'):
            packer.assert_ready()
        # Force the supported epoch refresh and recheck every transposed cache.
        c[0].invalidate_inference_caches()
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
            for site in c[3].layers[0].sites.values():
                assert identities.identity(site._weight) == tensor_identity(site._weight)


@pytest.mark.parametrize('private', ['qkv', 'mlp', 'lt_kn'])
def test_private_mutation_is_detected_on_the_next_request_even_without_generation_change(collector, private):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
        tensor = (c[1]._packers[0]._packed if private == 'qkv' else
                  c[2]._candidates[0]._packed if private == 'mlp' else
                  c[3].layers[0].sites['gate_up_packed']._weight)
        with torch.no_grad(): tensor.add_(1)
        next_state, next_generation = snapshot(c)
        assert next_generation == generation and next_state == state
        with c[6].prepare_scope(next_state, next_generation) as identities:
            with pytest.raises(RuntimeError, match='cache'):
                prepare(c, identities)


def test_standalone_original_sources_read_live_even_inside_an_active_scope(collector):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
            with torch.no_grad():
                c[3].layers[0].layer.self_attn.q_proj.weight.add_(1)
            with pytest.raises(RuntimeError, match='QKV sources changed'):
                c[3].layers[0].assert_sources()


def test_nested_cross_model_thread_generation_owner_rejected_and_exception_clears(collector):
    with case(collector) as c, case(collector) as other:
        state, generation = snapshot(c)
        with pytest.raises(RuntimeError, match='locked native owner'):
            with c[6].prepare_scope(state, generation): pass
        with c[4]._call_lock:
            with pytest.raises(RuntimeError, match='current generation'):
                with c[6].prepare_scope(state, generation - 1): pass
            with pytest.raises(ValueError, match='signature triple'):
                with c[6].prepare_scope(state[:2], generation): pass
            with pytest.raises(LookupError, match='intentional'):
                with c[6].prepare_scope(state, generation) as identities:
                    with pytest.raises(RuntimeError, match='Nested or cross-model'):
                        with other[6].prepare_scope(*snapshot(other)): pass
                    with pytest.raises(RuntimeError, match='owner/model/generation'):
                        identities.assert_for(other[0], generation)
                    errors = []
                    def worker():
                        try: identities.identity(next(c[0].parameters()))
                        except RuntimeError as exc: errors.append(str(exc))
                    thread = threading.Thread(target=worker)
                    thread.start()
                    thread.join()
                    assert len(errors) == 1 and 'another thread' in errors[0]
                    with pytest.raises(RuntimeError, match='generation'):
                        c[1].prepare(generation - 1, identity_lookup=identities)
                    raise LookupError('intentional')
            assert getattr(lookup._LOCAL, 'scope', None) is None
            assert identities._registered == identities._private == {}
            with c[6].prepare_scope(state, generation) as fresh:
                prepare(c, fresh)


@pytest.mark.parametrize('change,error', [('hooks', 'hooks'), ('geometry', 'geometry'),
                                        ('forward', 'dispatch'), ('options', 'options')])
def test_all_geometry_behavior_and_dispatch_checks_still_run(collector, change, error):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
        calls = c[3].layers[0]
        with pytest.MonkeyPatch.context() as patch:
            if change == 'hooks': patch.setitem(calls.layer.mlp.gate_proj._forward_hooks, 123, lambda *a: None)
            elif change == 'geometry': patch.setattr(calls.layer.self_attn.q_proj, 'in_features', 1)
            elif change == 'forward': patch.setattr(calls.packer, 'project', lambda x: x)
            elif change == 'options': patch.setattr(calls.candidate, '_options', (False, False))
            state, generation = snapshot(c)
            with c[6].prepare_scope(state, generation) as identities:
                # Lt owns the strongest geometry/dispatch checks. It should not
                # trust source identity equality as evidence for those fields.
                with pytest.raises((ValueError, RuntimeError), match=error):
                    c[3].prepare(generation, identity_lookup=identities)


def test_core_scope_cleanup_after_failed_generation_then_recovery(collector, monkeypatch):
    # This calls the actual core prepare hook, without executing vision/actions.
    from test_lite_expert_cublaslt import make_case
    c = make_case('g4_best_three', device='meta')
    model, qkv, mlp, lt, runner = c
    native = install_native_lifecycle_signature(model, collector=collector)
    handle = lookup.install_request_identity_lookup(model)
    args = ([torch.empty(1, 3, 2, 2, device='meta')], [torch.ones(1, dtype=torch.bool)],
            torch.ones(1, 4, dtype=torch.bool), 10)
    try:
        with runner._call_lock:
            with monkeypatch.context() as patch:
                def fail(*args, **kwargs): raise ArithmeticError('prepare interrupted')
                patch.setattr(mlp, 'prepare', fail)
                with pytest.raises(ArithmeticError, match='interrupted'):
                    model.prepare_inference_context(*args, positions=False)
            assert getattr(lookup._LOCAL, 'scope', None) is None
            context = model.prepare_inference_context(*args, positions=False)
            assert context['token'][1] == lt._generation == qkv._generation == mlp._generation
            assert not any(isinstance(value, lookup._RequestIdentities) for value in context.values())
    finally:
        runner.close()
        handle.uninstall()
        native.uninstall()
        lt.uninstall()
        mlp.uninstall()
        qkv.uninstall()


def test_scope_and_handle_do_not_retain_models_or_private_tensors(collector):
    with case(collector) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            private = torch.ones(1)
            private_ref = weakref.ref(private)
            identities.identity(private)
            del private
            assert private_ref() is None
        model_ref, runner_ref = weakref.ref(c[0]), weakref.ref(c[4])
        handle = c[6]
    del c
    gc.collect()
    assert model_ref() is None and runner_ref() is None
    assert identities._model() is None and handle._model() is None


def test_install_requires_native_and_is_reversible_without_touching_parameters(collector):
    model, qkv, mlp, lt, runner = make_meta_case(layers=1)
    try:
        with pytest.raises(ValueError, match='complete native'):
            lookup.install_request_identity_lookup(model)
        native = install_native_lifecycle_signature(model, collector=collector)
        original = tuple(id(tensor) for tensor in model.parameters())
        handle = lookup.install_request_identity_lookup(model)
        before = handle.manifest()
        assert handle.enabled and before['cross_request_cache'] is False
        assert before['retained_tensor_or_model_references'] is False
        with pytest.raises(ValueError, match='already installed'):
            lookup.install_request_identity_lookup(model)
        assert tuple(id(tensor) for tensor in model.parameters()) == original
        handle.uninstall()
        assert not handle.enabled and not hasattr(model, lookup.INSTALLATION)
        native.uninstall()
    finally:
        close_case(qkv, mlp, lt, runner)


def test_18_layer_warm_request_reads_54_private_identities_and_keeps_90_geometry_checks(collector, monkeypatch):
    from collections import Counter
    counts = Counter()
    with case(collector, layers=18) as c, c[4]._call_lock:
        state, generation = snapshot(c)
        with c[6].prepare_scope(state, generation) as identities:
            prepare(c, identities)
        def wrap(owner, name, key):
            original = getattr(owner, name)
            def counted(*args, **kwargs):
                counts[key] += 1
                return original(*args, **kwargs)
            monkeypatch.setattr(owner, name, counted)
        wrap(lookup, 'tensor_identity', 'private_identity')
        for calls in c[3].layers:
            wrap(calls.packer, 'validate_geometry', 'qkv_geometry')
            wrap(calls.candidate, 'validate', 'mlp_geometry')
            for site in calls.sites.values():
                wrap(site, 'validate', 'site_geometry')
        for request in (1, 2):
            # A new full native collect precedes every scope, even unchanged.
            state, generation = snapshot(c)
            with c[6].prepare_scope(state, generation) as identities:
                prepare(c, identities)
            assert counts == dict(private_identity=54 * request, qkv_geometry=18 * request,
                                  mlp_geometry=18 * request, site_geometry=54 * request)
