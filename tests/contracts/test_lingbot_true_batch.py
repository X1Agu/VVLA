"""CPU contracts only; no checkpoint, native library or B>1 qualification."""
import importlib.util
from pathlib import Path
import sys
import threading
import types

import numpy as np
import pytest
import torch
from torch import nn

from vvla_lingbot.experimental import true_batch_v1 as candidate


class Config:
    def __init__(self):
        self.nested = {'value': [1, 2]}


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.first = nn.Linear(3, 4, bias=False)
        self.second = nn.Linear(3, 4, bias=False)
        self.second.weight = self.first.weight
        self.register_buffer('metadata', torch.arange(4))
        self.experts = nn.Module()
        self.experts._robby_moe_workspace = None
        self.eval()


def test_shared_parameters_independent_mutable_views():
    flow = Flow()
    state = torch.random.get_rng_state().clone()
    a, receipt = candidate.clone_parameter_shared_view(flow)
    b, _ = candidate.clone_parameter_shared_view(flow)
    assert torch.equal(state, torch.random.get_rng_state())
    assert a.first.weight is b.second.weight is flow.first.weight
    assert a.first is not b.first and a.experts is not b.experts
    assert receipt['checkpoint_tensor_copies'] == 0
    a.metadata.add_(10)
    a.config.nested['value'].append(3)
    a.experts._robby_moe_workspace = {'counts': torch.zeros(3)}
    assert torch.equal(flow.metadata, torch.arange(4)) and torch.equal(b.metadata, flow.metadata)
    assert flow.config.nested == b.config.nested == {'value': [1, 2]}
    assert flow.experts._robby_moe_workspace is b.experts._robby_moe_workspace is None
    # Shared checkpoint mutation is visible through every view, never a stale copy.
    version = flow.first.weight._version
    with torch.no_grad():
        a.first.weight.add_(1)
    assert b.first.weight._version == flow.first.weight._version == version + 1


@pytest.mark.parametrize('mutation', ['training', 'workspace', 'vision'])
def test_used_or_training_flow_rejected(mutation):
    flow = Flow()
    if mutation == 'training':
        flow.train()
    elif mutation == 'workspace':
        flow.experts._robby_moe_workspace = {'counts': torch.zeros(1)}
    else:
        flow.qwenvl_with_expert = nn.Module()
        flow.qwenvl_with_expert.pos_embeds = torch.ones(1)
        flow.qwenvl_with_expert.eval()
    with pytest.raises(RuntimeError):
        candidate.clone_parameter_shared_view(flow)


def test_row_noise_permutation_split_continuity_and_global_rng():
    state = torch.random.get_rng_state().clone()
    stream = candidate.RowNoise('cpu')
    for identity, seed in [('a', 7), ('b', 11), ('c', 13), ('d', 17)]:
        stream.reset(identity, seed)
    first = stream.draw(['a', 'b', 'c', 'd'], h=2, width=3)
    second = stream.draw(['b', 'a'], h=2, width=3)
    for i, (identity, seed) in enumerate([('a', 7), ('b', 11), ('c', 13), ('d', 17)]):
        control = torch.Generator().manual_seed(seed)
        expected0 = torch.randn((1, 2, 3), dtype=torch.bfloat16, generator=control)
        assert torch.equal(first[i:i+1], expected0)
        if identity in ('a', 'b'):
            expected1 = torch.randn((1, 2, 3), dtype=torch.bfloat16, generator=control)
            assert torch.equal(second[['b', 'a'].index(identity):['b', 'a'].index(identity)+1], expected1)
    assert stream.calls == {'a': 2, 'b': 2, 'c': 1, 'd': 1}
    assert torch.equal(state, torch.random.get_rng_state())
    stream.reset('a', 7)
    assert torch.equal(stream.draw(['a'], h=2, width=3), first[:1])
    for invalid in [['a', 'a'], ['missing'], ['a', 'b', 'c']]:
        with pytest.raises(ValueError):
            stream.draw(invalid)


def test_official_apply_collate_and_owned_unapply_calls():
    class Policy:
        def __init__(self):
            self.applied = self.collated = 0

        def _prepare_model_input(self, observation):
            self.applied += 1
            return dict(images=torch.full((3, 2, 2), observation['value']),
                        img_masks=torch.ones(3, dtype=torch.bool),
                        lang_tokens=torch.arange(4), lang_masks=torch.ones(4, dtype=torch.bool),
                        state=torch.full((55,), observation['value']),
                        image_grid_thw=torch.ones((3, 3), dtype=torch.int64))

        def _pad_and_stack_tensors(self, values):
            self.collated += 1
            return torch.stack(values)

        def _unapply_batched_actions(self, applied, actions):
            return {'action': actions.numpy()[:, :, :14]}

    policy = Policy()
    applied, values = candidate.prepare_real_rows(policy, [{'value': 1.0}, {'value': 2.0}], device='cpu')
    assert policy.applied == 2 and policy.collated == 6
    assert all(value.shape[0] == 2 for value in values)
    assert values[0].dtype == torch.bfloat16 and values[2].dtype == torch.int64
    source = torch.ones((2, 50, 55))
    output = candidate.unapply_owned_rows(policy, applied, source)
    source.zero_()
    assert output['action'].shape == (2, 50, 14) and output['action'].dtype == np.float32
    assert output['action'].flags.owndata and np.all(output['action'] == 1)
    class CPUFakeEngine:
        device = 'cpu'
        flow = types.SimpleNamespace(config=types.SimpleNamespace(n_action_steps=50, max_action_dim=55))
        last_call = {'actual_tensor_batch': 2, 'graph_replay': False}

        def __call__(self, image, im, tokens, lm, state, *, noise, image_grid_thw):
            return noise

    raw = candidate.RawTrueBatchPolicy(policy, CPUFakeEngine())
    for identity, seed in [('a', 7), ('b', 11)]:
        raw.reset(identity, seed)
    observations = [{'value': 1.0}, {'value': 2.0}]
    result = raw.infer(observations, ['a', 'b'], expected_calls=[0, 0], record_noise=True)
    actual_noise = raw.last_noise.clone()
    for identity, seed in [('a', 7), ('b', 11)]:
        raw.reset(identity, seed)
    explicit = raw.infer(observations, ['a', 'b'], explicit_noise=actual_noise, expected_calls=[0, 0])
    assert np.array_equal(result['action'], explicit['action'])
    assert raw.noise.calls == {'a': 0, 'b': 0} and raw.calls == {'a': 1, 'b': 1}
    with pytest.raises(ValueError, match='request order'):
        raw.infer(observations, ['a', 'b'], expected_calls=[0, 0])
    with pytest.raises(ValueError):
        candidate.prepare_real_rows(policy, [{'value': 1.0}] * 3, device='cpu')


def test_prepare_override_keeps_inherited_math_and_rejects_invalid_rows():
    # The archived test exercised AST expansion. That mechanism is intentionally
    # gone; verify the explicit override and the retained inherited preparation.
    from vvla_lingbot.batch_engine import BatchedEngine
    engine_type=candidate._engine_type()
    assert engine_type._prepare is BatchedEngine._prepare
    owner=engine_type.__new__(engine_type)
    owner._p2_fixed_batch=2
    owner.flow=types.SimpleNamespace(state_proj=types.SimpleNamespace(in_features=55))
    with pytest.raises(ValueError,match='exact fixed B'):
        owner._validate_batch_inputs(*[torch.zeros(1,3) for _ in range(6)])


def test_close_drain_failure_keeps_all_execution_references(monkeypatch):
    pool = candidate.TrueBatchEnginePool.__new__(candidate.TrueBatchEnginePool)
    graph = types.SimpleNamespace(reset=lambda: None)
    engine = types.SimpleNamespace(_graph=graph)
    pool._engines, pool._views = {2: engine}, {2: object()}
    pool._lock, pool.closed, pool.device = threading.Lock(), False, 'cuda:0'
    pool.flow, pool._source_guard = object(), object()
    def failed_drain(device):
        raise RuntimeError('injected drain failure')
    monkeypatch.setattr(torch.cuda, 'synchronize', failed_drain)
    with pytest.raises(RuntimeError, match='injected drain failure'):
        pool.close()
    assert pool._engines[2] is engine and pool._views and pool.flow is not None and not pool.closed
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    pool.close()
    assert pool.closed and not pool._engines and not pool._views and pool.flow is None
    pool.close()


def test_real_tail_partition_preserves_every_row_without_dummy():
    for size in (1, 2, 3, 4, 7, 8, 9, 15, 16, 23):
        observations = [{'real_index': i} for i in range(size)]
        identities = [str(i) for i in range(size)]
        batches = list(candidate.split_real_batches(observations, identities))
        assert all(len(rows) == len(ids) and len(rows) in (1, 2, 4, 8) for rows, ids in batches)
        assert [row for rows, ids in batches for row in rows] == observations
        assert [identity for rows, ids in batches for identity in ids] == identities
    with pytest.raises(ValueError):
        list(candidate.split_real_batches([{}, {}], ['a', None]))
