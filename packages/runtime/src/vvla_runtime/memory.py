"""Explicit CUDA accounting for serialized graph owners.

Resident admission is bounded after capture. CUDA itself may allocate a temporary
peak before a capture finishes; reserve is an admission estimate, not an allocator
hard cap. Consumers must fail/retire an over-budget candidate before replay.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    allocated: int
    reserved: int
    peak_allocated: int
    free: int
    total: int


def cuda_snapshot(device=None):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return MemorySnapshot(torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
                          torch.cuda.max_memory_allocated(device), free, total)


def tensor_storage_bytes(values):
    """Deduplicate actual CUDA storages, including aliases/views."""
    import torch
    seen_objects, storages = set(), {}
    def visit(value):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value: visit(child)
    visit(values)
    return sum(storages.values())


class GraphMemoryBudget:
    def __init__(self, budget_bytes, baseline, *, capture_reserve_bytes=64*1024*1024):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError("graph_memory_budget_bytes must be positive")
        if type(capture_reserve_bytes) is not int or capture_reserve_bytes < 0:
            raise ValueError("capture_reserve_bytes must be nonnegative")
        if capture_reserve_bytes > budget_bytes:
            raise ValueError("Capture reserve exceeds graph budget")
        self.limit = budget_bytes
        self.reserve = capture_reserve_bytes
        self.baseline = baseline
        self.last = baseline
        self.resident = 0
        self.tensor_bytes = 0
        self.peak = 0

    def observe(self, snapshot, owned_tensor_bytes):
        self.last = snapshot
        self.tensor_bytes = owned_tensor_bytes
        # Allocator reservation includes graph private pools and fragmentation.
        # Device-wide free memory is diagnostic: other processes can change it.
        self.resident = max(owned_tensor_bytes, snapshot.reserved-self.baseline.reserved,
                            snapshot.allocated-self.baseline.allocated, 0)
        self.peak = max(self.peak, snapshot.peak_allocated-self.baseline.allocated)
        return self.resident

    def fits(self, extra=0):
        return self.resident + extra <= self.limit

    def report(self):
        return {"budget_bytes": self.limit, "resident_charge_bytes": self.resident,
                "owned_tensor_bytes": self.tensor_bytes, "capture_reserve_estimate_bytes": self.reserve,
                "capture_peak_allocated_delta_bytes": self.peak,
                "peak_scope": "process allocator high-water mark; may include work before this owner",
                "shared_baseline": asdict(self.baseline), "current": asdict(self.last),
                "device_wide_used_delta_bytes": self.baseline.free-self.last.free,
                "scope": "serialized owner; PyTorch allocator and owned graph tensors",
                "temporary_peak_is_hard_capped": False,
                "external_cuda_allocations_included": False}

