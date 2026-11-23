# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small inference-only caches; tensors referenced by graphs are never mutated.

Ordinary in-place parameter updates are detected by tensor versions. Explicit
loaders and callers performing unsupported ``.data`` writes must invalidate.
The graph runner uses the generation to retire captured stale constants too.
"""

from collections import OrderedDict

from torch import Tensor, is_inference


def tensor_identity(tensor):
    # Processor-owned exact inference tensors have no version counter. Avoid
    # raising an exception for that normal case, without caching any identity
    # field or changing the attribute/exception behavior of Tensor subclasses.
    if type(tensor) is Tensor and is_inference(tensor):
        version = None
    else:
        try:
            version = tensor._version
        except RuntimeError:
            version = None
    return (id(tensor), tensor.data_ptr(), version, tensor.dtype, tensor.device, tuple(tensor.shape))


class Pi05InferenceCache:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.generation = 0
        self.signature = None
        self.empty_features = OrderedDict()
        self.time_schedules = OrderedDict()

    def clear(self):
        self.generation += 1
        self.signature = None
        self.empty_features.clear()
        self.time_schedules.clear()

    def refresh(self, signature):
        if self.signature != signature:
            self.clear()
            self.signature = signature
        return self.generation

    def get_or_create(self, entries, key, factory):
        if key not in entries:
            value = factory()
            if len(entries) >= self.capacity:
                entries.popitem(last=False)
            entries[key] = value
        entries.move_to_end(key)
        return entries[key]


def copy_position_context(target, source):
    """Refresh request-dependent tensors without changing captured addresses."""
    if target is None or source is None:
        if target is not source:
            raise ValueError("Position optimization changed without a new graph")
        return
    if target.keys() != source.keys():
        raise ValueError("Position context fields changed")
    for key in target:
        old, new = target[key], source[key]
        if isinstance(old, tuple):
            if len(old) != len(new):
                raise ValueError("RoPE context geometry changed")
            pairs = zip(old, new, strict=True)
        else:
            pairs = [(old, new)]
        for dst, src in pairs:
            if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                raise ValueError("Position tensor geometry changed")
            dst.copy_(src)
