"""Remote CPU/CUDA contracts for fresh aligned metadata packing."""
import numpy as np
import pytest
import torch

from test_lite_gpu_image_prepare import processor, observation, exact_inputs
from vvla_repro_pi05.lite.packed_input_metadata import (
    HOOK, INSTALLATION, build_cpu_payload, metadata_views, payload_layout, prepare_metadata,
    install_packed_input_metadata, PackedInputMetadata,
)


@pytest.mark.parametrize('cameras,length', [(1, 1), (3, 200), (8, 17), (9, 2)])
def test_payload_exact_native_types_alignment_nonoverlap_and_ownership(cameras, length):
    camera_values = [index % 2 == 0 for index in range(cameras)]
    ids = [-(2**63), 2**63-1] + list(range(max(0, length-2))) if length > 1 else [257151]
    masks = [index % 3 != 1 for index in range(length)]
    payload, layout = build_cpu_payload(camera_values, ids, masks)
    image_masks, tokens, language = metadata_views(payload, layout)
    assert payload.data_ptr() % 8 == tokens.data_ptr() % 8 == 0
    assert tokens.dtype == torch.int64 and tokens.shape == (1, length)
    assert tokens.tolist() == [ids] and language.tolist() == [masks]
    assert [item.item() for item in image_masks] == camera_values
    regions = [*image_masks, tokens, language]
    addresses = [(item.data_ptr(), item.data_ptr() + item.numel() * item.element_size()) for item in regions]
    for index, (start, end) in enumerate(addresses):
        for other_start, other_end in addresses[index+1:]:
            assert end <= other_start or other_end <= start
    assert len({item.untyped_storage().data_ptr() for item in regions}) == 1
    original = payload.clone()
    image_masks[0].logical_not_()
    assert torch.equal(tokens, torch.tensor([ids])) and language.tolist() == [masks]
    fresh, _ = build_cpu_payload(camera_values, ids, masks)
    assert fresh.data_ptr() != payload.data_ptr() and torch.equal(fresh, original)
    assert layout['token_offset'] % 8 == 0 and layout['bytes'] == layout['mask_offset'] + length


@pytest.mark.parametrize('cameras,tokens', [(0, 2), (True, 2), (3, 0), (3, 1.5)])
def test_invalid_payload_geometry_fails_early(cameras, tokens):
    with pytest.raises(ValueError):
        payload_layout(cameras, tokens)


def test_cpu_fallback_is_original_separate_typed_constructors():
    camera, tokens, masks = prepare_metadata([True, False], [4, 9], [1, 0], 'cpu')
    assert [value.tolist() for value in camera] == [[True], [False]]
    assert tokens.tolist() == [[4, 9]] and masks.tolist() == [[True, False]]
    assert len({value.untyped_storage().data_ptr() for value in (*camera, tokens, masks)}) == 4
    with pytest.raises(ValueError, match='equally sized'):
        build_cpu_payload([True], [1, 2], [1])
    payload, layout = build_cpu_payload([True], [1, 2], [1, 0])
    with pytest.raises(ValueError, match='aligned uint8 geometry'):
        metadata_views(torch.cat([torch.zeros(1, dtype=torch.uint8), payload])[1:], layout)


class DynamicTokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        ids = [2] + [ord(value) for value in text[:max_length-1]]
        if 'short' in text:
            ids = ids[:3]
        count = len(ids)
        return dict(input_ids=ids + [0] * (max_length-count), attention_mask=[1]*count + [0]*(max_length-count))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Only scheduled remote owner executes CUDA'))])
@pytest.mark.parametrize('image_hook', [False, True])
def test_full_processor_bitwise_dynamic_prompt_missing_camera_roundtrip_and_old_outputs(device, image_hook):
    from vvla_repro_pi05.lite.gpu_image_prepare import install_gpu_uint8_prepare
    proc = processor(device=device, resolution=(16, 16))
    proc.tokenizer = DynamicTokenizer()
    image = install_gpu_uint8_prepare(proc) if image_hook else None
    sources = [observation((24, 32, 3)), observation((32, 24, 3)), observation((24, 32, 3))]
    sources[1]['prompt'] = 'short'
    sources[1]['state'] = np.zeros(8, np.float32)
    del sources[1]['observation.images.image2']
    with torch.inference_mode():
        expected = [proc.build_model_inputs(obs, return_host_layout=True) for obs in sources]
    handle = install_packed_input_metadata(proc)
    try:
        retained = []
        for obs, before in zip(sources, expected, strict=True):
            with torch.inference_mode():
                actual = proc.build_model_inputs(obs, return_host_layout=True)
            exact_inputs(actual, before)
            retained.append(actual)
        for actual, before in zip(retained, expected, strict=True):
            exact_inputs(actual, before)
        # Exposed regions may be written by a caller without corrupting any
        # other region or a subsequent request. A→B→A never reads stale bytes.
        saved_tokens, saved_masks = retained[0][2].clone(), retained[0][3].clone()
        with torch.inference_mode():
            retained[0][1][0].logical_not_()
        assert torch.equal(retained[0][2], saved_tokens) and torch.equal(retained[0][3], saved_masks)
        with torch.inference_mode():
            exact_inputs(proc.build_model_inputs(sources[0], return_host_layout=True), expected[0])
        empty = {'state': sources[0]['state'], 'prompt': 'empty'}
        with pytest.raises(ValueError, match='at least one'):
            proc.build_model_inputs(empty)
    finally:
        handle.close()
        if image is not None:
            image.close()


def test_hook_live_checks_active_call_and_idempotent_restoration():
    proc = processor(resolution=(16, 16))
    builder = proc.build_model_inputs.__func__
    handle = install_packed_input_metadata(proc)
    assert handle.manifest()['actual_selected'] == 'native_cpu'
    with pytest.raises(ValueError, match='already installed'):
        install_packed_input_metadata(proc)
    with handle._lock, pytest.raises(RuntimeError, match='active request'):
        handle.close()
    with handle._lock, pytest.raises(RuntimeError, match='another request'):
        handle([True], [2], [1], 'cpu')
    proc.metadata_prepare_hook_version = 'changed'
    with pytest.raises(RuntimeError, match='dispatch changed'):
        handle([True], [2], [1], 'cpu')
    proc.metadata_prepare_hook_version = 'per_request_v1'
    handle.close(); handle.close()
    assert not hasattr(proc, HOOK) and not hasattr(proc, INSTALLATION)
    assert proc.build_model_inputs.__func__ is builder


def test_undeclared_padding_camera_slots_keep_original_false_masks():
    proc, obs = processor(resolution=(16, 16)), observation((24, 24, 3))
    proc.config.max_cameras = 4  # Three declared features plus one padding slot.
    before = proc.build_model_inputs(obs, return_host_layout=True)
    handle = install_packed_input_metadata(proc)
    try:
        after = proc.build_model_inputs(obs, return_host_layout=True)
        exact_inputs(after, before)
        assert after[4].cameras == (True, True, False, False)
    finally:
        handle.close()


def test_policy_default_strict_option_manifest_and_failed_construction_rolls_back(monkeypatch):
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    assert _split_lite_overrides({})[1]['packed_input_metadata'] is False
    for value in (1, None, 'true'):
        with pytest.raises(ValueError, match='boolean'):
            _split_lite_overrides({'lite_packed_input_metadata': value})
    original = _policy()
    candidate = Pi05LitePolicy(original.model, original.processor, device='cuda', cuda_graph=True,
                              packed_input_metadata=True)
    handle = candidate._packed_metadata_installation
    assert candidate._lite_optimizations['packed_input_metadata_manifest'] == handle.manifest()
    candidate.graph_runner.entries['retirement'] = object()
    close = handle.close
    def close_after_graph():
        assert candidate.graph_runner._closed and not candidate.graph_runner.entries
        return close()
    monkeypatch.setattr(handle, 'close', close_after_graph)
    candidate.close()
    assert handle._closed and candidate._packed_metadata_installation is None
    def fail_manifest(self):
        raise RuntimeError('injected packed metadata manifest failure')
    monkeypatch.setattr(PackedInputMetadata, 'manifest', fail_manifest)
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        failed.__init__(original.model, original.processor, device='cpu', cuda_graph=False,
                        packed_input_metadata=True)
    assert failed._closed and failed._packed_metadata_installation is None
    assert not hasattr(original.processor, HOOK) and not hasattr(original.processor, INSTALLATION)
    original.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only scheduled remote owner executes CUDA')
def test_cuda_exactly_one_pageable_blocking_transfer(monkeypatch):
    original_to, transfers = torch.Tensor.to, []
    def checked_to(tensor, *args, **kwargs):
        transfers.append((tensor.device.type, tensor.dtype, tensor.numel(), tensor.is_pinned(), args, kwargs))
        return original_to(tensor, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'to', checked_to)
        cameras, tokens, masks = prepare_metadata([True, True, False], list(range(200)), [1]*200, 'cuda')
    assert transfers == [('cpu', torch.uint8, 1808, False, (), {'device': 'cuda', 'non_blocking': False})]
    assert [value.item() for value in cameras] == [True, True, False]
    assert tokens.tolist() == [list(range(200))] and masks.all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only scheduled remote owner executes CUDA')
def test_cuda_one_payload_views_cross_stream_lifetime_and_original_rng_state():
    camera, ids, masks = [True, True, False], list(range(200)), [1]*37 + [0]*163
    original_rng = torch.cuda.get_rng_state()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        first = prepare_metadata(camera, ids, masks, 'cuda')
        snapshot = [tensor.clone() for tensor in (*first[0], first[1], first[2])]
        stream.synchronize()  # Complete reads before consuming from another stream.
    second = prepare_metadata([False, True, False], list(reversed(ids)), masks, 'cuda')
    first_values = [*first[0], first[1], first[2]]
    assert torch.equal(first[1], torch.tensor([ids], device='cuda'))
    assert len({value.untyped_storage().data_ptr() for value in first_values}) == 1
    assert first[1].untyped_storage().data_ptr() != second[1].untyped_storage().data_ptr()
    assert all(torch.equal(value, retained) for value, retained in zip(first_values, snapshot, strict=True))
    assert torch.equal(torch.cuda.get_rng_state(), original_rng)
