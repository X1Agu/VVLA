from vvla_runtime.memory import GraphMemoryBudget, MemorySnapshot
import pytest


def snap(allocated, reserved, peak=None):
    return MemorySnapshot(allocated,reserved,allocated if peak is None else peak,10000-reserved,10000)


def test_budget_accounts_alias_storage_allocator_and_shared_weights_separately():
    budget=GraphMemoryBudget(400,snap(1000,2000),capture_reserve_bytes=100)
    assert budget.observe(snap(1100,2300,1600),120)==300
    assert budget.fits(100) and not budget.fits(101)
    report=budget.report()
    assert report['shared_baseline']['allocated']==1000
    assert report['capture_peak_allocated_delta_bytes']==600
    assert report['temporary_peak_is_hard_capped'] is False
    assert budget.observe(snap(1050,2050),75)==75


def test_budget_never_silently_accepts_negative_or_oversized_reserve():
    with pytest.raises(ValueError):GraphMemoryBudget(-1,snap(0,0))
    with pytest.raises(ValueError):GraphMemoryBudget(100,snap(0,0),capture_reserve_bytes=101)
