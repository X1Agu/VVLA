"""Remote CPU contracts for the exact-Tensor inference-version fast path."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.models.pi05 import inference_cache_pi05 as cache
from vvla_repro_pi05.lite import layout


def legacy_identity(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (id(tensor), tensor.data_ptr(), version, tensor.dtype, tensor.device, tuple(tensor.shape))


@pytest.mark.parametrize('inference', [False, True])
@pytest.mark.parametrize('mutation', ['contents', 'shape', 'storage', 'dtype', 'device', 'replacement'])
def test_all_six_fields_remain_live_for_ordinary_and_inference_tensors(inference, mutation):
    context = torch.inference_mode if inference else torch.no_grad
    with context():
        tensor = torch.ones(2, 3)
        before = cache.tensor_identity(tensor)
        assert before == legacy_identity(tensor)
        if mutation == 'contents': tensor.add_(1)
        elif mutation == 'shape': tensor.resize_(3, 2)
        elif mutation == 'storage': tensor.data = tensor.clone()
        elif mutation == 'dtype': tensor.data = tensor.bfloat16()
        elif mutation == 'device': tensor = tensor.to('meta')
        elif mutation == 'replacement': tensor = tensor.clone()
        after = cache.tensor_identity(tensor)
        assert after == legacy_identity(tensor)
        assert len(after) == 6
        if inference:
            assert after[2] is None
        if inference and mutation == 'contents':
            # The existing contract has no value hash/version for inference
            # tensors; do not invent a new guarantee for content-only writes.
            assert after == before
        else:
            assert after != before


def test_exact_inference_tensor_avoids_runtime_exception_but_normal_version_is_read():
    with torch.inference_mode():
        inference = torch.ones(1)
    ordinary = torch.ones(1)
    exceptions = []
    codes = {legacy_identity.__code__, cache.tensor_identity.__code__}
    previous = sys.gettrace()
    def trace(frame, event, arg):
        if frame.f_code in codes and event == 'exception' and arg[0] is RuntimeError:
            exceptions.append(frame.f_code)
        return trace
    try:
        sys.settrace(trace)
        assert cache.tensor_identity(inference) == legacy_identity(inference)
        before = cache.tensor_identity(ordinary)
        ordinary.add_(1)
        assert cache.tensor_identity(ordinary)[2] == before[2] + 1
    finally:
        sys.settrace(previous)
    assert exceptions == [legacy_identity.__code__]


class CustomVersionTensor(torch.Tensor):
    version_reads = 0
    @property
    def _version(self):
        type(self).version_reads += 1
        return 731


class RuntimeVersionTensor(torch.Tensor):
    @property
    def _version(self):
        raise RuntimeError('custom version unavailable')


class InvalidVersionTensor(torch.Tensor):
    @property
    def _version(self):
        raise ValueError('custom non-runtime failure')


@pytest.mark.parametrize('subclass', [CustomVersionTensor, RuntimeVersionTensor, InvalidVersionTensor])
def test_tensor_subclass_attribute_and_exception_behavior_is_unchanged(monkeypatch, subclass):
    tensor = torch.ones(1).as_subclass(subclass)
    def forbidden(_): raise AssertionError('Subclass must not call inference predicate')
    monkeypatch.setattr(cache, 'is_inference', forbidden)
    if subclass is InvalidVersionTensor:
        for function in (legacy_identity, cache.tensor_identity):
            with pytest.raises(ValueError, match='custom non-runtime'):
                function(tensor)
    else:
        assert cache.tensor_identity(tensor) == legacy_identity(tensor)
        if subclass is CustomVersionTensor:
            assert subclass.version_reads >= 2


def test_parameters_keep_live_version_without_calling_inference_predicate(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    def forbidden(_): raise AssertionError('Parameter must keep the original path')
    monkeypatch.setattr(cache, 'is_inference', forbidden)
    before = cache.tensor_identity(parameter)
    with torch.no_grad(): parameter.add_(1)
    after = cache.tensor_identity(parameter)
    assert after == legacy_identity(parameter) and after[2] == before[2] + 1


@pytest.mark.parametrize('inference', [False, True])
def test_three_camera_host_layout_still_performs_21_complete_identity_reads(monkeypatch, inference):
    model = SimpleNamespace(
        action_horizon=50, config=SimpleNamespace(prefix_pack_bucket_size=64),
        optimization_flags=lambda: (True, False, True, True, True),
        paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(image_size=224, patch_size=14)))))
    with torch.inference_mode() if inference else nullcontext():
        images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
        cameras = (True, True, False)
        tokens = tuple(index < 13 for index in range(200))
        masks = [torch.tensor([present]) for present in cameras]
        lang_masks = torch.tensor([tokens])
        calls = []
        def identity(tensor):
            result = cache.tensor_identity(tensor)
            assert result == legacy_identity(tensor)
            calls.append(id(tensor))
            return result
        monkeypatch.setattr(layout, 'tensor_identity', identity)
        host = layout.HostMaskLayout.from_processor(cameras, tokens, images, masks, lang_masks)
        layout.HostLayoutCache().get(model, host, images, masks, lang_masks)
        assert len(calls) == 21
        assert all(calls.count(id(tensor)) == 3 for tensor in (*images, *masks, lang_masks))
        with pytest.raises(ValueError, match='does not belong'):
            host.validate(images, masks, lang_masks.clone())
