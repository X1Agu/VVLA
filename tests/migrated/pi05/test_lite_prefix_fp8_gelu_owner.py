"""Remote CPU ownership contracts with explicit meta weights/fake packing.

These tests do not claim arithmetic or real-policy quality. The independent
primitive CUDA contracts and subsequent real preflight/closed loop cover those.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from prefix_fp8_gelu_fixtures import REPO, make_backend, make_evidence
from test_lite_prefix_fp8 import evidence as base_evidence, prepare, rig
from vvla_repro_pi05.lite import prefix_fp8 as base
from vvla_repro_pi05.lite import prefix_fp8_gelu_owner as fusion
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner


@pytest.fixture
def setup(rig, tmp_path, monkeypatch):
    fictional = make_evidence(tmp_path, source_root=REPO)
    backend = make_backend(fictional['record'])
    monkeypatch.setattr(base, '_load_evidence', lambda config, device: (rig.raw, deepcopy(backend)))
    monkeypatch.setattr(base, '_hardware', lambda device: deepcopy(fictional['hardware']))
    real_load = fusion.load_evidence
    monkeypatch.setattr(fusion, 'load_evidence', lambda config, hardware:
                        real_load(config, hardware, fictional['source_root']))
    handle = base.install_prefix_fp8(rig.model, base_evidence())
    state = SimpleNamespace(model=rig.model, base=handle, fixture=fictional, runner=None, extension=None)
    yield state
    if state.runner is not None: state.runner.close()
    extension = getattr(state.model, fusion.INSTALLATION, None)
    if extension is not None and extension.enabled:
        # Individual mutation tests explicitly restore the injected foreign
        # forwards before their teardown. Production must never overwrite one.
        extension.uninstall()
    if state.base.enabled: state.base.uninstall()


def install(state):
    state.extension = fusion.install_prefix_fp8_gelu(state.model, state.fixture['config'])
    state.runner = Pi05NativeGraphRunner(state.model, compile_sample_actions=True)
    state.base.bind_runner(state.runner)
    return state.extension


def test_real_meta_model_preserves_lazy_registry_through_cold_install_and_retirement(setup):
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vvla_repro_pi05.models.pi05.precision_pi05 import ExecutionRecipe, install_execution_recipe

    state = setup
    state.base.uninstall()
    # Use the actual model constructor and BF16 recipe, not MetaPrefix's eager
    # WeakSet. Only weight packing/hardware evidence use the fixture's fakes.
    with torch.device('meta'):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend='torch_fa2', prefix_kv_only=True))
        set_bfloat16_layout(model)
    model.eval()
    install_execution_recipe(model, ExecutionRecipe(fuse_prefix_mlp=True))
    assert not hasattr(model, '_graph_runners')
    state.model = model
    state.base = base.install_prefix_fp8(model, base_evidence())
    assert not hasattr(model, '_graph_runners')
    packed = tuple((id(site.quantized), id(site.scale)) for site in state.base.sites)
    state.extension = fusion.install_prefix_fp8_gelu(model, state.fixture['config'])
    assert not hasattr(model, '_graph_runners')
    # Constructor rollback before any runner exists must also preserve absence.
    state.extension.uninstall()
    assert not hasattr(model, '_graph_runners')
    assert all('forward' not in layer.mlp.__dict__ for layer in base._layers(model))

    extension = install(state)
    assert tuple(model._graph_runners) == (state.runner,)
    assert state.runner._device is None and not state.runner.entries
    prepare(state.base, state.runner)
    assert state.base.validate_precision_manifest()['gelu_fusion'] == extension.manifest()
    assert tuple((id(site.quantized), id(site.scale)) for site in state.base.sites) == packed
    state.runner.close()
    extension.uninstall()
    # close retires the runner without eagerly removing its live WeakSet entry.
    assert tuple(model._graph_runners) == (state.runner,) and state.runner._closed
    assert all('forward' not in layer.mlp.__dict__ for layer in base._layers(model))


def test_nonempty_registry_is_rejected_even_before_base_runner_binding(setup):
    state = setup
    state.runner = Pi05NativeGraphRunner(state.model, compile_sample_actions=True)
    assert state.base._runner is None and state.base._generation is None
    registry = state.model._graph_runners
    with pytest.raises(ValueError, match='unbound'):
        fusion.install_prefix_fp8_gelu(state.model, state.fixture['config'])
    assert state.model._graph_runners is registry and tuple(registry) == (state.runner,)
    assert not hasattr(state.model, fusion.INSTALLATION)
    assert all('forward' not in layer.mlp.__dict__ for layer in base._layers(state.model))


def test_default_base_is_unchanged_and_fusion_preserves_all51_packed_original_state(setup):
    state = setup
    original_manifest = state.base.manifest()
    original_parameters = tuple((name, id(value)) for name, value in state.model.named_parameters())
    original_state = tuple(state.model.state_dict())
    packed = tuple((id(site.weight), id(site.quantized), id(site.scale)) for site in state.base.sites)
    assert 'gelu_fusion' not in original_manifest
    extension = install(state)
    assert len(extension.members) == 17
    assert tuple((id(site.weight), id(site.quantized), id(site.scale)) for site in state.base.sites) == packed
    assert tuple((name, id(value)) for name, value in state.model.named_parameters()) == original_parameters
    assert tuple(state.model.state_dict()) == original_state
    assert 'forward' not in base._layers(state.model)[17].mlp.__dict__
    prepare(state.base, state.runner)
    manifest = state.base.validate_precision_manifest()
    assert manifest['gelu_fusion']['contracts']['files_sha256'] == state.fixture['record']['files_sha256']
    assert {k: v for k, v in manifest.items() if k != 'gelu_fusion'} == original_manifest
    assert extension.raw_evidence == state.fixture['blobs']
    manifest['gelu_fusion']['modules'].pop()
    assert len(state.base.manifest()['gelu_fusion']['modules']) == 17
    state.runner.close()
    extension.uninstall()
    assert state.base.manifest() == original_manifest
    assert all('forward' not in layer.mlp.__dict__ for layer in base._layers(state.model))


@pytest.mark.parametrize('mutation', ('foreign_mlp_forward', 'nonprefix_forward', 'member_order',
                                     'packed_version', 'original_weight_version', 'kernel_binding'))
def test_existing_complete_live_checks_and_owned_mlp_binding_remain_enforced(setup, monkeypatch, mutation):
    state = setup; extension = install(state)
    prepare(state.base, state.runner)
    first = extension.members[0]
    original_members = extension.members
    foreign = None
    if mutation == 'foreign_mlp_forward':
        foreign = first.module
        foreign.forward = lambda value: value
    elif mutation == 'nonprefix_forward':
        foreign = state.model.other
        foreign.forward = lambda value: value
    elif mutation == 'member_order': extension.members = tuple(reversed(extension.members))
    elif mutation == 'kernel_binding': monkeypatch.setattr(fusion.kernel, 'launch', lambda *args: None)
    else:
        with torch.no_grad():
            tensor = state.base.sites[0].scale if mutation == 'packed_version' else state.base.sites[0].weight
            tensor.add_(1)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            prepare(state.base, state.runner)
    finally:
        extension.members = original_members
        if foreign is first.module: foreign.forward = first.forward
        elif foreign is not None: delattr(foreign, 'forward')


def test_extension_and_base_compilation_tokens_share_locked_generation_and_retire(setup):
    from vvla_repro_pi05.lite.compilation import _PreparedCompiledRun, _packed_handles
    state = setup; extension = install(state)
    prepare(state.base, state.runner)
    generation = state.base._generation
    with pytest.raises(RuntimeError, match='unique locked'):
        extension.compilation_members(generation)
    context = {'token': ('gelu-owner-contract', generation)}
    state.runner._token = context['token']
    members = (*state.base.sites, *extension.members)
    calls = []
    def compiled():
        tokens = tuple(member._prepared_compile_token for member in members)
        assert tokens[0] is not None and all(token is tokens[0] for token in tokens)
        calls.append('same-scope')
        raise RuntimeError('injected compile execution failure')
    scope = _PreparedCompiledRun(state.model, compiled, _packed_handles(state.model),
                                 owner=state.runner, context=context)
    with state.runner._call_lock:
        with pytest.raises(RuntimeError, match='injected compile'):
            scope()
        with pytest.raises(RuntimeError, match='stale'):
            extension.compilation_members(generation + 1)
    assert calls == ['same-scope']
    assert all(member._prepared_compile_token is None for member in members)
    scope.retire()
    with state.runner._call_lock:
        with pytest.raises(RuntimeError, match='retired'): scope()


def test_graph_retirement_precedes_forward_restore_and_base_storage_release(setup):
    state = setup; extension = install(state)
    prepare(state.base, state.runner)
    state.runner._device = torch.device('meta')
    with pytest.raises(RuntimeError, match='Drain native graph'): extension.uninstall()
    with pytest.raises(RuntimeError, match='Uninstall prefix GELU'): state.base.uninstall()
    assert extension.enabled and state.base.enabled
    state.runner.clear()
    extension.uninstall()
    assert not hasattr(state.model, fusion.INSTALLATION)
    state.base.uninstall()
    assert not hasattr(state.model, base.INSTALLATION)


def test_external_forward_is_not_overwritten_during_cleanup(setup):
    state = setup; extension = install(state)
    member = extension.members[0]
    foreign = lambda value: value
    member.module.forward = foreign
    state.runner.close()
    with pytest.raises(RuntimeError, match='externally'): extension.uninstall()
    assert extension.enabled and member.module.forward is foreign
    member.module.forward = member.forward
    extension.uninstall()


def test_partial_install_failure_rolls_back_only_fusion_without_touching_packed_state(setup, monkeypatch):
    state = setup
    delattr(state.model, '_graph_runners')
    packed = tuple((id(site.quantized), id(site.scale)) for site in state.base.sites)
    real_validate = fusion._MLPMember.validate
    def fail(member):
        if member.index == 8: raise RuntimeError('injected member publication failure')
        real_validate(member)
    monkeypatch.setattr(fusion._MLPMember, 'validate', fail)
    with pytest.raises(RuntimeError, match='injected member'):
        fusion.install_prefix_fp8_gelu(state.model, state.fixture['config'])
    assert not hasattr(state.model, fusion.INSTALLATION)
    assert not hasattr(state.model, '_graph_runners')
    assert state.base.enabled and len(base._selected_modules(state.model)) == 51
    assert all('forward' not in layer.mlp.__dict__ for layer in base._layers(state.model))
    assert tuple((id(site.quantized), id(site.scale)) for site in state.base.sites) == packed


def test_published_epoch_invalidation_still_requires_reinstall_and_new_identity(setup):
    state = setup; extension = install(state)
    prepare(state.base, state.runner)
    state.model.invalidate_inference_caches()
    with pytest.raises(RuntimeError, match='epoch invalidated'): prepare(state.base, state.runner)
    assert extension.enabled


def test_fusion_cannot_be_installed_after_runner_binding(setup):
    state = setup
    state.runner = Pi05NativeGraphRunner(state.model, compile_sample_actions=True)
    state.base.bind_runner(state.runner)
    with pytest.raises(ValueError, match='unbound'):
        fusion.install_prefix_fp8_gelu(state.model, state.fixture['config'])
    assert not hasattr(state.model, fusion.INSTALLATION)


def test_forward_consumes_original_gate_up_and_down_packed_storage_without_cache(setup, monkeypatch):
    state = setup
    calls = []
    def projection(x, qw, sw):
        calls.append(('projection', x, qw, sw))
        return torch.empty((x.shape[0], qw.shape[0]), dtype=torch.bfloat16, device='meta'), None, None
    def down(gate, up, qw, sw):
        calls.append(('down', gate, up, qw, sw))
        return torch.empty((gate.shape[0], 2048), dtype=torch.bfloat16, device='meta'), None, None
    monkeypatch.setattr(base.native, 'scaled_linear', projection)
    monkeypatch.setattr(fusion.primitive, 'scaled_down', down)
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda *args: False)
    # The fake operators belong to a fresh fictional cold epoch, never a
    # mutation of an already published base operator binding.
    state.base.uninstall()
    state.base = base.install_prefix_fp8(state.model, base_evidence())
    extension = install(state)
    prepare(state.base, state.runner)
    member = extension.members[0]
    with state.runner._call_lock, torch.no_grad():
        a = member.module(torch.empty((1, 576, 2048), dtype=torch.bfloat16, device='meta'))
        b = member.module(torch.empty((1, 576, 2048), dtype=torch.bfloat16, device='meta'))
    assert a.shape == b.shape == (1, 576, 2048) and a.dtype == b.dtype == torch.bfloat16
    assert a is not b and len(calls) == 6
    for offset in (0, 3):
        assert [item[0] for item in calls[offset:offset + 3]] == ['projection', 'projection', 'down']
        assert calls[offset][2] is member.sites[0].quantized
        assert calls[offset + 1][2] is member.sites[1].quantized
        assert calls[offset + 2][3] is member.sites[2].quantized
        assert calls[offset + 2][4] is member.sites[2].scale
