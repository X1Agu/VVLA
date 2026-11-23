"""Diagnostic contracts prepared for remote CPU / scheduled CUDA execution only."""
import ast
from contextlib import nullcontext
import gc
import importlib.util
import inspect
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from test_lite_gpu_image_prepare import processor, observation, exact_inputs

PROFILING = Path(__file__).resolve().parents[2] / 'benchmarks/pi05-lite-bf16-20261105/profiling'


@pytest.fixture(scope='module')
def helper():
    spec = importlib.util.spec_from_file_location('pinned_input_staging_test', PROFILING / 'pinned_input_staging.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStream:
    def __init__(self, transport, name):
        self.transport, self.name = transport, name

    def synchronize(self):
        ops = self.transport
        ops.log.append(('stream_sync', self.name))
        if ops.fail_sync:
            ops.fail_sync -= 1
            raise RuntimeError('injected stream synchronization failure')
        ops.complete(self)


class FakeEvent:
    def __init__(self, transport):
        self.transport, self.stream, self.limit = transport, None, -1

    def record(self, stream):
        ops = self.transport
        ops.log.append(('record', stream.name))
        if ops.fail_record:
            ops.fail_record -= 1
            raise RuntimeError('injected event record failure')
        self.stream, self.limit = stream, len(ops.jobs)

    def query(self):
        ops = self.transport
        ops.log.append(('query', self.stream.name))
        if ops.fail_query:
            ops.fail_query -= 1
            raise RuntimeError('injected query failure')
        return all(job['done'] for job in ops.jobs[:self.limit] if job['stream'] is self.stream)

    def synchronize(self):
        ops = self.transport
        ops.log.append(('event_sync', self.stream.name))
        if ops.fail_event_sync:
            ops.fail_event_sync -= 1
            raise RuntimeError('injected event synchronization failure')
        ops.complete(self.stream, self.limit)


class FakeTransfers:
    """Defer the actual CPU byte copy until the simulated DMA fence completes.

    Reusing a source without retirement will corrupt a prior destination, so
    these tests exercise lifetime behavior rather than an eager clone mock.
    """
    def __init__(self):
        self.log, self.jobs, self.allocations = [], [], []
        self.fail_record = self.fail_sync = self.fail_upload = self.fail_query = self.fail_event_sync = self.fail_event = 0
        self.current = FakeStream(self, 'A')

    def allocate(self, shape):
        value = torch.empty(shape, dtype=torch.uint8)
        self.allocations.append(value)
        return value

    def event(self):
        if self.fail_event:
            self.fail_event -= 1
            raise RuntimeError('injected event allocation failure')
        return FakeEvent(self)

    def current_stream(self, device):
        return self.current

    def upload(self, source, device):
        self.log.append(('upload', self.current.name))
        destination = torch.empty_like(source)
        self.jobs.append(dict(source=source, destination=destination, stream=self.current, done=False))
        if self.fail_upload:
            self.fail_upload -= 1
            raise RuntimeError('injected upload failure after queueing')
        return destination

    def complete(self, stream, limit=None):
        for job in self.jobs[:limit]:
            if job['stream'] is stream and not job['done']:
                job['destination'].copy_(job['source'])
                job['done'] = True


def test_slot_waits_before_full_source_overwrite_and_keeps_prior_output_owned(helper):
    ops = FakeTransfers()
    slot = helper.TransferSlot((7,), ops)
    first = slot.upload(torch.arange(7, dtype=torch.uint8), 'cuda')
    second = slot.upload(torch.full((7,), 255, dtype=torch.uint8), 'cuda')
    assert torch.equal(first, torch.arange(7, dtype=torch.uint8))
    assert ('event_sync', 'A') in ops.log
    assert ops.log.index(('event_sync', 'A')) < len(ops.log)-2
    assert first.data_ptr() != second.data_ptr() and len(ops.allocations) == 1
    slot.close()
    assert torch.equal(second, torch.full((7,), 255, dtype=torch.uint8))
    assert torch.equal(first, torch.arange(7, dtype=torch.uint8))
    assert slot.buffer is None and slot.closed
    slot.close()
    with pytest.raises(RuntimeError, match='closed'):
        slot.upload(torch.zeros(7, dtype=torch.uint8), 'cuda')


def test_completed_slot_uses_query_without_sync_and_cross_stream_retirement(helper):
    ops = FakeTransfers()
    slot = helper.TransferSlot((3,), ops)
    slot.upload(torch.tensor([1, 2, 3], dtype=torch.uint8), 'cuda')
    ops.complete(ops.current)
    slot.upload(torch.tensor([4, 5, 6], dtype=torch.uint8), 'cuda')
    assert not any(kind == 'event_sync' for kind, _ in ops.log)
    old = ops.current
    ops.current = FakeStream(ops, 'B')
    third = slot.upload(torch.tensor([7, 8, 9], dtype=torch.uint8), 'cuda')
    assert ('event_sync', old.name) in ops.log and slot.stream is ops.current
    slot.close()
    assert torch.equal(third, torch.tensor([7, 8, 9], dtype=torch.uint8))


@pytest.mark.parametrize('failure', ['fail_record', 'fail_upload'])
def test_queue_or_record_error_syncs_and_raises_but_allows_safe_next_request(helper, failure):
    ops = FakeTransfers()
    setattr(ops, failure, 1)
    slot = helper.TransferSlot((3,), ops)
    with pytest.raises(RuntimeError, match='injected'):
        slot.upload(torch.tensor([1, 2, 3], dtype=torch.uint8), 'cuda')
    assert ops.jobs[0]['done'] and not slot.pending and not slot.poisoned
    assert slot.fallback_syncs == 1 and ('stream_sync', 'A') in ops.log
    later = slot.upload(torch.tensor([9, 8, 7], dtype=torch.uint8), 'cuda')
    slot.close()
    assert torch.equal(later, torch.tensor([9, 8, 7], dtype=torch.uint8))


def test_failed_record_and_fallback_poison_retains_source_until_close_can_retire(helper):
    ops = FakeTransfers()
    ops.fail_record = ops.fail_sync = 1
    slot = helper.TransferSlot((3,), ops)
    with pytest.raises(RuntimeError, match='poisoned'):
        slot.upload(torch.tensor([1, 2, 3], dtype=torch.uint8), 'cuda')
    source = slot.buffer
    assert slot.poisoned and slot.pending and not slot.recorded
    with pytest.raises(RuntimeError, match='poisoned'):
        slot.upload(torch.tensor([9, 9, 9], dtype=torch.uint8), 'cuda')
    assert torch.equal(source, torch.tensor([1, 2, 3], dtype=torch.uint8))
    ops.fail_sync = 1
    with pytest.raises(RuntimeError, match='synchronization'):
        slot.close()
    assert slot.buffer is source and not slot.closed
    slot.close()
    assert slot.closed and slot.buffer is None and ops.jobs[0]['done']


@pytest.mark.parametrize('failure', ['fail_query', 'fail_event_sync'])
def test_retirement_error_never_overwrites_pending_source(helper, failure):
    ops = FakeTransfers()
    slot = helper.TransferSlot((2,), ops)
    slot.upload(torch.tensor([4, 5], dtype=torch.uint8), 'cuda')
    setattr(ops, failure, 1)
    with pytest.raises(RuntimeError, match='injected'):
        slot.upload(torch.tensor([8, 9], dtype=torch.uint8), 'cuda')
    assert slot.poisoned and torch.equal(slot.buffer, torch.tensor([4, 5], dtype=torch.uint8))
    slot.close()


def test_event_creation_precedes_dma_and_invalid_geometry_never_queues(helper):
    ops = FakeTransfers()
    slot = helper.TransferSlot((3,), ops)
    for source in (torch.zeros(2, dtype=torch.uint8), torch.zeros(3), np.zeros(3, np.uint8)):
        with pytest.raises(ValueError, match='geometry'):
            slot.upload(source, 'cuda')
    ops.fail_event = 1
    with pytest.raises(RuntimeError, match='allocation'):
        slot.upload(torch.zeros(3, dtype=torch.uint8), 'cuda')
    assert not ops.jobs and not slot.pending
    slot.close()


def test_cpu_processor_fallback_is_exact_counted_and_never_selected(helper):
    proc, obs = processor(), observation()
    before = proc.build_model_inputs(obs, return_host_layout=True)
    ops = FakeTransfers()
    handle = helper.install_pinned_preparation(proc, transport=ops)
    counters = handle.snapshot()
    try:
        actual = proc.build_model_inputs(obs, return_host_layout=True)
        exact_inputs(actual, before)
        assert handle.snapshot() == dict(image_selected=0, metadata_selected=0, image_fallback=2, metadata_fallback=1)
        with pytest.raises(ValueError, match='selected timing'):
            helper.assert_selected(counters, handle.snapshot())
        assert not ops.allocations and not ops.jobs
    finally:
        handle.close()


def test_hooks_two_image_slots_full_metadata_bytes_and_fresh_destinations(helper, monkeypatch):
    proc = processor(device='cuda')
    ops = FakeTransfers()
    handle = helper.install_pinned_preparation(proc, transport=ops)
    # CPU transport simulates CUDA DMA; arithmetic independently locked by AST
    # and deferred true-CUDA parity, rather than pretending to run the kernel.
    monkeypatch.setattr(helper, 'resize_uploaded', lambda raw, height, width: raw)
    values = [np.full((256, 256, 3), value, np.uint8) for value in (3, 7, 11)]
    images = [handle.image(value, 224, 224, 'cuda') for value in values]
    assert len(ops.allocations) == 2 and handle.images[0].uploads == 2 and handle.images[1].uploads == 1
    first = handle.metadata([True, True, False], list(range(200)), [True]*200, 'cuda')
    second = handle.metadata([False, True, False], [-1]*200, [False]*200, 'cuda')
    assert len(ops.allocations) == 3 and handle.payload.buffer.numel() == 1808
    # Padding is rewritten even if corrupted between completed transfers.
    ops.complete(ops.current)
    handle.payload.buffer[3:8].fill_(255)
    third = handle.metadata([True, False, False], [0]*200, [True]*200, 'cuda')
    assert torch.equal(handle.payload.buffer[3:8], torch.zeros(5, dtype=torch.uint8))
    handle.close()
    for actual, value in zip(images, (3, 7, 11), strict=True):
        assert torch.equal(actual, torch.full_like(actual, value))
    assert first[1].tolist() == [list(range(200))] and second[1].tolist() == [[-1]*200]
    assert third[1].tolist() == [[0]*200]
    assert len({item[1].untyped_storage().data_ptr() for item in (first, second, third)}) == 3


def test_owner_guards_lock_close_and_released_processor_retirement(helper):
    proc = processor()
    ops = FakeTransfers()
    handle = helper.install_pinned_preparation(proc, transport=ops)
    with pytest.raises(ValueError, match='unoccupied'):
        helper.install_pinned_preparation(proc)
    with handle._lock, pytest.raises(RuntimeError, match='active'):
        handle.close()
    with handle._lock, pytest.raises(RuntimeError, match='another'):
        handle.metadata([True]*3, [0]*200, [True]*200, 'cpu')
    proc.metadata_prepare_hook_version = 'changed'
    with pytest.raises(RuntimeError, match='dispatch'):
        handle.metadata([True]*3, [0]*200, [True]*200, 'cpu')
    proc.metadata_prepare_hook_version = 'per_request_v1'
    with pytest.raises(ValueError, match='target'):
        handle.metadata([True]*3, [0]*200, [True]*200, 'cuda')
    original = getattr(proc, helper.IMAGE_HOOK)
    setattr(proc, helper.IMAGE_HOOK, object())
    with pytest.raises(RuntimeError, match='dispatch'):
        handle.close()
    setattr(proc, helper.IMAGE_HOOK, original)
    ref = weakref.ref(proc)
    del proc
    gc.collect()
    assert ref() is None  # Handle owns no processor/model reference.
    handle.close()
    assert handle._closed


def test_poisoned_owner_rejects_all_hooks_and_close_retires(helper):
    proc = processor(device='cuda')
    ops = FakeTransfers()
    ops.fail_record = ops.fail_sync = 1
    handle = helper.install_pinned_preparation(proc, transport=ops)
    with pytest.raises(RuntimeError, match='poisoned'):
        handle.metadata([True]*3, [0]*200, [True]*200, 'cuda')
    with pytest.raises(RuntimeError, match='poisoned'):
        handle.image(np.zeros((256, 256, 3), np.uint8), 224, 224, 'cuda')
    handle.close()
    assert not hasattr(proc, helper.INSTALLATION)


def test_close_attempts_every_slot_and_retains_failed_installation_for_retry(helper, monkeypatch):
    proc = processor(device='cuda')
    ops = FakeTransfers()
    handle = helper.install_pinned_preparation(proc, transport=ops)
    monkeypatch.setattr(helper, 'resize_uploaded', lambda raw, height, width: raw)
    for value in (1, 2):
        handle.image(np.full((256, 256, 3), value, np.uint8), 224, 224, 'cuda')
    handle.metadata([True, True, False], [0]*200, [True]*200, 'cuda')
    ops.fail_event_sync = 1
    with pytest.raises(RuntimeError, match='retirement retry'):
        handle.close()
    assert not handle.images[0].closed and handle.images[0].buffer is not None
    assert handle.images[1].closed and handle.payload.closed
    assert getattr(proc, helper.INSTALLATION) is handle and not handle._closed
    with pytest.raises(RuntimeError, match='poisoned'):
        handle.metadata([True]*3, [0]*200, [True]*200, 'cuda')
    handle.close()
    assert handle._closed and not hasattr(proc, helper.INSTALLATION)


def test_arithmetic_tail_is_exact_production_ast(helper):
    from vvla_repro_pi05.lite.gpu_image_prepare import prepare_uint8_image
    def tail(function):
        body = ast.parse(inspect.getsource(function)).body[0].body
        begin = next(index for index, statement in enumerate(body)
                     if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
                     and statement.targets[0].id == 'tensor')
        return [ast.dump(statement, include_attributes=False) for statement in body[begin:]]
    assert tail(helper.resize_uploaded) == tail(prepare_uint8_image)


def test_selected_timing_requires_exact_delta_not_historical_fallback_total(helper):
    before = dict(image_selected=8, metadata_selected=4, image_fallback=2, metadata_fallback=1)
    after = dict(image_selected=10, metadata_selected=5, image_fallback=2, metadata_fallback=1)
    helper.assert_selected(before, after)
    for field in before:
        changed = dict(after)
        changed[field] += 1
        with pytest.raises(ValueError, match='selected timing'):
            helper.assert_selected(before, changed)


def test_probe_identical_start_end_fences_and_separate_host_ready_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(PROFILING))
    import probe_pinned_input_staging as probe
    from types import SimpleNamespace
    monkeypatch.setattr(probe, 'WARMUP', 2)
    monkeypatch.setattr(probe, 'SAMPLES', 3)
    log = []
    class Event:
        def record(self): log.append('record')
        def synchronize(self): log.append('event_sync')
        def elapsed_time(self, other): return .25
    fake_torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(
        synchronize=lambda: log.append('device_sync'), Event=lambda **kw: Event()))
    class Processor:
        def __init__(self, handle): self.handle = handle
        def build_model_inputs(self, observation, return_host_layout):
            log.append('prepare')
            if self.handle:
                self.handle.counts['image_selected'] += 2
                self.handle.counts['metadata_selected'] += 1
            return 'prepared'
    class Handle:
        def __init__(self):
            self.counts = dict(image_selected=0, metadata_selected=0, image_fallback=0, metadata_fallback=0)
        def snapshot(self): return dict(self.counts)
    ticks = iter(range(0, 100000000, 1000000))
    monkeypatch.setattr(probe.time, 'perf_counter_ns', lambda: next(ticks))
    monkeypatch.setattr(probe, 'input_signature', lambda value: (log.append('parity'), 'reference')[1])
    baseline = probe.measure(fake_torch, Processor(None), {}, 'reference')
    baseline_log = list(log)
    log.clear()
    handle = Handle()
    candidate = probe.measure(fake_torch, Processor(handle), {}, 'reference', handle)
    assert log == baseline_log
    assert baseline['host_return_p50_ms'] == candidate['host_return_p50_ms'] == 1.0
    assert baseline['ready_p50_ms'] == candidate['ready_p50_ms'] == 2.0
    assert log[-6:] == ['device_sync', 'record', 'prepare', 'record', 'event_sync', 'parity']
    rows = [dict(name=name, **value) for name, value in zip(probe.ORDER, (baseline, candidate, baseline), strict=True)]
    summary = probe.summarize(rows)
    assert summary['ready_p50_ms']['conservative_savings_ms'] == 0
    rows[1]['ready_p50_ms'] = float('nan')
    with pytest.raises(ValueError, match='duration'):
        probe.summarize(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only scheduled remote owner executes CUDA')
def test_cuda_full_processor_burst_streams_freshness_and_pending_close(helper, monkeypatch):
    monkeypatch.syspath_prepend(str(PROFILING))
    from probe_pinned_input_staging import contract_observations, exercise_contracts, close_contract
    from probe_packed_input_metadata import input_signature
    from vvla_repro_pi05.lite.gpu_image_prepare import install_gpu_uint8_prepare
    from vvla_repro_pi05.lite.packed_input_metadata import install_packed_input_metadata
    from test_lite_packed_input_metadata import DynamicTokenizer
    proc = processor(device='cuda')
    proc.config.tokenizer_max_length = 200
    proc.tokenizer = DynamicTokenizer()
    cases, keys = contract_observations(observation())
    with torch.inference_mode():
        image = install_gpu_uint8_prepare(proc)
        packed = install_packed_input_metadata(proc)
        try:
            references = [input_signature(proc.build_model_inputs(obs, return_host_layout=True)) for obs in cases]
        finally:
            packed.close()
            image.close()
        handle = helper.install_pinned_preparation(proc)
        try:
            report = exercise_contracts(torch, proc, handle, cases, references, keys)
            assert report['passed']
            assert all(slot.buffer.is_pinned() for slot in (*handle.images, handle.payload))
            assert sum(slot.buffer.numel() for slot in (*handle.images, handle.payload)) == 395024
            assert close_contract(torch, proc, handle, cases[0], references[0])['passed']
        finally:
            handle.close()
