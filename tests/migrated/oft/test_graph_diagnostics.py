import json

import torch

from vvla_runtime.execution.graph import TensorGraphRunner
from vvla_runtime.execution.graph_diagnostics import (
    GraphDiagnostics, describe_lingbot_key, describe_oft_key,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_key_revisit_is_distinguished_from_first_seen_and_hit(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = GraphDiagnostics(path, model='test')
    for key, outcome in [('A', 'capture'), ('A', 'hit'), ('B', 'capture'), ('A', 'capture'), ('B', 'hit')]:
        log.observe(key, lambda: {'shape': key}, outcome=outcome)
    events = read(path)
    assert len(events) == 4
    assert events[1]['reason'] == 'new_metadata_key'
    assert events[2]['seen_before'] and events[2]['reason'] == 'previously_seen_key'
    assert events[3]['outcome'] == 'hit'
    assert events[3]['cumulative_counts'] == {'capture': 3, 'hit': 2}
    assert events[1]['changed_components'] == ['shape']


def test_unchanged_warm_hit_does_not_describe_or_write(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = GraphDiagnostics(path, model='test')
    key = ('a',)
    digest = log.observe(key, lambda: {'shape': [1]}, outcome='capture')
    before = path.read_bytes()
    def forbidden():
        raise AssertionError('No warm-path encoding or copies')
    for _ in range(20):
        assert log.observe(key, forbidden, outcome='hit') == digest
    assert path.read_bytes() == before
    assert log.counts['hit'] == 20


def test_lingbot_mask_content_and_shape_are_visible_but_observations_are_not(tmp_path):
    specs = tuple(((1, 51), 'torch.bfloat16', 'cuda:0') for _ in range(6))
    first = specs + (b'camera', b'language', b'grid')
    second = specs + (b'camera', b'changed-mask', b'grid')
    path = tmp_path / 'events.jsonl'
    log = GraphDiagnostics(path, model='lingbot')
    for key in [first, second]:
        log.observe(key, lambda: describe_lingbot_key(key), outcome='capture')
    events = read(path)
    assert events[1]['changed_components'] == ['language_mask_sha256']
    assert events[0]['key']['images']['shape'] == [1, 51]
    assert 'changed-mask' not in path.read_text()
    assert len(events[0]['key']['grid_sha256']) == 64


def test_oft_generation_change_is_explicit(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = GraphDiagnostics(path, model='openvla-oft')
    for generation in [1, 2]:
        key = ('revision', generation, ('x',), (((1, 3), torch.float32, torch.device('cpu')),))
        log.observe(key, lambda: describe_oft_key(key), outcome='eager')
    assert read(path)[1]['reason'] == 'model_state_or_revision_changed'


def test_bounded_history_does_not_claim_new_key_is_definitely_unseen(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = GraphDiagnostics(path, model='test', history_limit=1)
    for key in ['A', 'B', 'C']:
        log.observe(key, lambda: {'shape': key}, outcome='capacity_fallback')
    last = read(path)[-1]
    assert not last['history_complete']
    assert last['reason'] == 'key_not_in_bounded_history'
    assert len(log.seen) == 1


def test_real_runner_preserves_outputs_and_does_not_key_input_values(tmp_path):
    path = tmp_path / 'events.jsonl'
    runner = TensorGraphRunner(lambda x: x * 2, revision='test', diagnostics_path=path)
    for x in [torch.ones(2, 3), torch.full((2, 3), 7.), torch.ones(1, 3), torch.ones(2, 3)]:
        assert torch.equal(runner(x=x), x * 2)
    events = read(path)
    assert len(events) == 3
    assert events[-1]['seen_before']
    assert events[-1]['outcome'] == 'eager'
    runner.close()


def test_new_session_does_not_claim_old_history(tmp_path):
    path = tmp_path / 'events.jsonl'
    for _ in range(2):
        log = GraphDiagnostics(path, model='test')
        log.observe('A', lambda: {'shape': [1]}, outcome='capture')
    events = read(path)
    assert events[0]['session'] != events[1]['session']
    assert not events[1]['seen_before']

