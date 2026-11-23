"""Regression for a feasible B8 hidden behind an infeasible seventh prefix."""
import asyncio

import pytest

from vvla.errors import AdmissionRejected
from vvla.serve import DynamicBatcher, DynamicBatchServer, Request, scheduling_policy


COST_MS = {1: 27, 2: 37, 4: 52, 8: 87, 16: 157}


def request(arm, *, start=0., group=("A",), slo=.100, valid=None, hard=1.):
    return Request(f"arm-{arm}", {"arm": arm}, arm + 100, start, hard, group, slo, valid)


class FakeOwner:
    def __init__(self):
        self.now = 0.
        self.batches = []

    def infer_requests(self, rows):
        assert len({r.compatibility for r in rows}) == 1
        self.batches.append([r.request_id for r in rows])
        self.now += COST_MS[len(rows)] / 1000
        return [(r.request_id, r.observation["arm"], r.seed) for r in rows]


async def run(rows, *, cohort=True, cancel=None, after_window=None, capacity=32, wait_ms=2.):
    owner = FakeOwner()
    policy = scheduling_policy(dict(name="slack-edf", service_ms=COST_MS, margin_ms=0,
                                    safety_factor=1, cohort_admission=cohort))
    batcher = DynamicBatcher(owner, supported=tuple(COST_MS), max_batch=16, wait_ms=wait_ms,
                             policy=policy, queue_capacity=capacity, clock=lambda: owner.now)
    await batcher.start()
    try:
        tasks = [asyncio.create_task(batcher.submit(r)) for r in rows]
        # All submissions run under the same condition before the owner wakes up.
        await asyncio.sleep(0)
        if cancel is not None:
            tasks[cancel].cancel()
            # Cancelling the task also cancels its queued future immediately.
        if after_window is not None:
            owner.now = after_window.admitted_at
            tasks.append(asyncio.create_task(batcher.submit(after_window)))
        async with batcher.condition:
            owner.now = max(owner.now, wait_ms/1000)
            batcher.condition.notify_all()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        submitted = rows if after_window is None else [*rows, after_window]
        for r, result in zip(submitted, results, strict=True):
            if isinstance(result, dict):
                assert result["result"] == (r.request_id, r.observation["arm"], r.seed)
                assert result["scheduler"]["deadline_met"]
                if r.valid_until_at is not None:
                    assert result["scheduler"]["action_valid"]
                else:
                    assert result["scheduler"]["slo_met"]
        assert not batcher.pending and not batcher._admission_cutoffs
        assert batcher.total_offered == len(tasks)
        return owner.batches, results, batcher.total_rejected
    finally:
        await batcher.close()


@pytest.mark.parametrize("cohort,expected", [(False, [4, 2]), (True, [8])])
def test_collection_changes_the_synchronized_prefix_only_with_the_option(cohort, expected):
    batches, result, rejected = asyncio.run(run([request(i) for i in range(8)], cohort=cohort))
    assert list(map(len, batches)) == expected
    assert [i for i, r in enumerate(result) if isinstance(r, AdmissionRejected)] == ([] if cohort else [6, 7])
    assert rejected == (0 if cohort else 2)


@pytest.mark.parametrize("rotation", range(8))
def test_rotated_arms_and_missing_eighth_use_only_real_members(rotation):
    arms = [(rotation + n) % 8 for n in range(8)]
    rows = [request(a) for a in arms]
    batches, _, rejected = asyncio.run(run(rows))
    assert batches == [[r.request_id for r in rows]] and rejected == 0
    batches, result, rejected = asyncio.run(run(rows[:7]))
    assert list(map(len, batches)) == [4, 2] and rejected == 1
    assert isinstance(result[6], AdmissionRejected)
    assert result[6].event["submitted_to_model"] is False


def test_one_arm_and_two_incompatible_groups():
    batches, _, rejected = asyncio.run(run([request(7)]))
    assert batches == [["arm-7"]] and rejected == 0
    # Exactly seven compatible rows and an eighth with a different layout.
    rows = [request(i) for i in range(7)] + [request(7, group=("B",), slo=.25)]
    batches, result, rejected = asyncio.run(run(rows))
    assert list(map(len, batches)) == [4, 2, 1] and rejected == 1
    assert isinstance(result[6], AdmissionRejected)
    assert batches[-1] == ["arm-7"]


def test_expiry_and_soft_recovery_use_each_members_own_contract():
    tight = [request(i) for i in range(6)] + [request(6, valid=.020), request(7)]
    batches, result, rejected = asyncio.run(run(tight))
    assert list(map(len, batches)) == [4, 2] and rejected == 2
    assert all(isinstance(result[i], AdmissionRejected) for i in (6,7))
    # The previous buffer is already empty, but a new chunk can still be used.
    rows = [request(i, slo=0., valid=.1) for i in range(8)]
    batches, result, _ = asyncio.run(run(rows))
    assert all(r["scheduler"]["action_valid"] and not r["scheduler"]["slo_met"] for r in result if isinstance(r, dict))
    assert list(map(len, batches)) == [4, 2]


def test_tighter_arms_keep_priority_within_a_compatible_collection():
    rows = [request(i, slo=.2) for i in range(6)] + [request(6, slo=.060), request(7, slo=.060)]
    batches, _, rejected = asyncio.run(run(rows))
    assert rejected == 0 and list(map(len, batches)) == [4, 4]
    assert batches[0][:2] == ["arm-6", "arm-7"]


def test_public_service_routes_eight_independent_observations(monkeypatch):
    from types import SimpleNamespace
    from vvla.serve import server as server_module

    class Owner(FakeOwner):
        def capabilities(self): return {"batch_sizes": list(COST_MS)}
        def close(self): pass

    async def exercise():
        owner = Owner()
        monkeypatch.setattr(server_module, "time", SimpleNamespace(perf_counter=lambda: owner.now))
        policy = scheduling_policy(dict(name="slack-edf", service_ms=COST_MS, safety_factor=1,
                                        margin_ms=0, cohort_admission=True))
        server = DynamicBatchServer(lambda: owner, policy=policy, target_slo_ms=100, max_batch=8)
        await server.start()
        server.batcher.clock = lambda: owner.now
        try:
            responses = await asyncio.gather(*(server.infer({"arm":i}, seed=100+i, request_id=f"arm-{i}")
                                               for i in range(8)))
            assert all(r["result"] == (f"arm-{i}",i,100+i) for i,r in enumerate(responses))
            assert owner.batches == [[f"arm-{i}" for i in range(8)]]
            assert server.scheduling_metadata()["policy"]["cohort_admission"] is True
        finally:
            await server.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [0, 6, 7])
def test_cancel_before_dispatch_never_supplies_a_lane_or_steals_a_result(cancel):
    batches, result, _ = asyncio.run(run([request(i) for i in range(8)], cancel=cancel))
    assert isinstance(result[cancel], asyncio.CancelledError)
    assert list(map(len, batches)) == [4, 2]
    assert f"arm-{cancel}" not in [rid for group in batches for rid in group]


def test_window_end_capacity_and_zero_wait_bound_provisional_admission():
    rows = [request(i) for i in range(7)]
    batches, result, rejected = asyncio.run(run(rows, after_window=request(7, start=.003)))
    assert list(map(len, batches)) == [4, 2] and rejected == 2
    assert all(isinstance(result[i], AdmissionRejected) for i in (6,7))
    batches, result, rejected = asyncio.run(run([*rows,request(7)], capacity=7))
    assert list(map(len, batches)) == [4, 2] and isinstance(result[7], OverflowError)
    assert rejected == 1
    batches, _, rejected = asyncio.run(run([*rows,request(7)], wait_ms=0))
    assert list(map(len, batches)) == [4, 2] and rejected == 2
    with pytest.raises(ValueError, match="boolean"):
        scheduling_policy(dict(name="slack-edf", service_ms=COST_MS, cohort_admission="yes"))


def test_short_budget_uses_admission_horizon_and_only_measured_small_widths():
    options=dict(name='slack-edf',service_ms=COST_MS,safety_factor=1.,margin_ms=0.,cohort_admission=True)
    original=scheduling_policy(options)
    guarded=scheduling_policy(dict(options,short_budget={'budget_ms':50,'effective_ms':{1:23.7,2:32.3,4:45.9}}))
    fast=request(0,slo=.05);slow=request(1,slo=.1)
    assert guarded.service_cost(1,[fast])==pytest.approx(.0237)
    assert guarded.service_cost(2,[slow,fast])==pytest.approx(.0323)
    assert guarded.service_cost(8,[fast,slow])==original.cost(8)
    assert guarded.service_cost(1,[slow])==original.cost(1)
    assert guarded.service_cost(1,[request(3,slo=None)])==original.cost(1)
    assert guarded.predict_finish([],fast,.024,[1,2,4,8,16],.002) is not None
    assert original.predict_finish([],fast,.024,[1,2,4,8,16],.002) is None
    slow_rows=[request(i,slo=.100) for i in range(8)]
    assert guarded.predict_finish(slow_rows[:-1],slow_rows[-1],.008,[1,2,4,8,16],.002)==original.predict_finish(slow_rows[:-1],slow_rows[-1],.008,[1,2,4,8,16],.002)
    assert guarded.select(slow_rows,.053,[1,2,4,8,16])==original.select(slow_rows,.053,[1,2,4,8,16])


@pytest.mark.parametrize('setup', [
    {'budget_ms':0,'effective_ms':{1:23}}, {'budget_ms':50,'effective_ms':{3:15}},
    {'budget_ms':50,'effective_ms':{1:80}}, {'budget_ms':50,'effective_ms':{'1':float('nan')}}])
def test_short_budget_rejects_unusable_cost_tables(setup):
    with pytest.raises(ValueError):
        scheduling_policy(dict(name='slack-edf',service_ms=COST_MS,short_budget=setup))
