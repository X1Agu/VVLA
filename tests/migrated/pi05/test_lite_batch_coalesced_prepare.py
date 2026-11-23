"""CPU golden-input tests for an unselected bulk preparation candidate.

These tests prove neither CUDA rounding parity nor policy quality/performance.
"""
import copy

import numpy as np
import pytest
import torch

from vvla_repro_pi05.lite.batch_coalesced_prepare import build_coalesced_inputs
from vvla_repro_pi05.lite.batch_processor import Pi05BatchProcessor
from vvla_repro_pi05.lite.gpu_image_prepare import install_gpu_uint8_prepare
from vvla_repro_pi05.lite.packed_input_metadata import install_packed_input_metadata
from vvla_repro_pi05.lite.processor import Pi05LiteProcessor
from vvla_repro_pi05.models.pi05.config import Pi05Config


class Tokenizer:
    def __init__(self):
        self.texts = []
    def __call__(self, text, *, max_length, **kwargs):
        self.texts.append(text)
        # Content-dependent IDs and varying masks make state/order mistakes
        # observable without requiring the external checkpoint tokenizer.
        raw = [1] + [ord(value) for value in text]
        count = min(len(raw), max_length)
        return {'input_ids': (raw+[0]*max_length)[:max_length],
                'attention_mask': [1]*count+[0]*(max_length-count)}


def processor(resolution=(8, 8)):
    config = Pi05Config(image_resolution=resolution, max_cameras=3, tokenizer_max_length=128,
        input_features={'observation.state': {'shape': [8]},
            'observation.images.front': {'shape': [3, 16, 12]},
            'observation.images.wrist': {'shape': [3, 16, 12]},
            'observation.images.empty': {'shape': [3, 8, 8]}},
        output_features={'action': {'shape': [7]}}, norm_stats={'state': {
            'mode': 'mean_std', 'mean': list(np.arange(8)/10), 'std': [1, 2, 1, 2, 1, 2, 1, 2]}})
    return Pi05LiteProcessor(config, Tokenizer(), torch.device('cpu'))


def observations(shape=(16, 12, 3)):
    rng = np.random.default_rng(881)
    return [{'images': {'observation.images.front': rng.integers(0, 256, shape, np.uint8),
        **({'observation.images.wrist': rng.integers(0, 256, shape, np.uint8)} if i%3 else {})},
        'state': np.array([-2, -1, -.5, 0, .25, .5, 1, 2], np.float32)+np.float32(i/32),
        'prompt': ' pick_cube\n'+str(i)} for i in range(32)]


def exact_inputs(reference, candidate):
    for left, right in zip(reference[:4], candidate[:4], strict=True):
        pairs = zip(left, right, strict=True) if isinstance(left, list) else [(left, right)]
        for a, b in pairs:
            assert a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
            if a.dtype == torch.float32:
                assert torch.equal(a.view(torch.int32), b.view(torch.int32))
            else:
                assert torch.equal(a, b)
    if len(reference) == 5:
        a, b = reference[4], candidate[4]
        assert a.cameras == b.cameras and a.tokens == b.tokens and a.mask_bytes == b.mask_bytes
        a.validate(reference[0], reference[1], reference[3])
        b.validate(candidate[0], candidate[1], candidate[3])


@pytest.mark.parametrize('shape,resolution', [((16, 12, 3), (8, 8)), ((8, 8, 3), (8, 8)),
    ((7, 19, 3), (13, 13)), ((19, 7, 3), (11, 11))])
@pytest.mark.parametrize('installed', [False, True])
def test_bulk_cpu_bits_slots_masks_prompts_and_fresh_ownership_match_reference(shape, resolution, installed):
    proc, obs = processor(resolution), observations(shape)
    if installed:
        install_gpu_uint8_prepare(proc)
        install_packed_input_metadata(proc)
    reference = Pi05BatchProcessor(proc, 32).build_model_inputs(obs, return_host_layout=True)
    texts = list(proc.tokenizer.texts)
    candidate = build_coalesced_inputs(proc, obs, return_host_layout=True)
    assert proc.tokenizer.texts[32:] == texts
    exact_inputs(reference, candidate)
    snapshots = [image.clone() for image in candidate[0]] + [candidate[2].clone(), candidate[3].clone()]
    changed = copy.deepcopy(obs)
    changed[7]['images']['observation.images.front'].fill(177)
    changed[7]['state'][0] = 100
    changed[7]['prompt'] = 'different current task'
    next_inputs = build_coalesced_inputs(proc, changed, return_host_layout=True)
    assert not torch.equal(candidate[0][0][7], next_inputs[0][0][7])
    assert not torch.equal(candidate[2][7], next_inputs[2][7])
    for actual, saved in zip(candidate[0]+[candidate[2], candidate[3]], snapshots, strict=True):
        assert torch.equal(actual, saved)
    assert candidate[2].untyped_storage().data_ptr() != next_inputs[2].untyped_storage().data_ptr()


@pytest.mark.parametrize('kind', ['float', 'heterogeneous', 'noncontiguous'])
def test_unhandled_domain_falls_back_and_noncontiguous_uint8_is_exact(kind):
    proc, obs = processor(), observations()
    if kind == 'float':
        obs[9]['images']['observation.images.front'] = obs[9]['images']['observation.images.front'].astype(np.float32)/255
    if kind == 'heterogeneous':
        obs[9]['images']['observation.images.front'] = np.full((9, 11, 3), 33, np.uint8)
    if kind == 'noncontiguous':
        for row in obs:
            row['images'] = {key: value[:, ::-1] for key, value in row['images'].items()}
    exact_inputs(Pi05BatchProcessor(proc, 32).build_model_inputs(obs, return_host_layout=True),
                 build_coalesced_inputs(proc, obs, return_host_layout=True))


@pytest.mark.parametrize('kind', ['foreign_hook', 'locked_owner', 'invalid_state', 'no_cameras'])
def test_foreign_dispatch_busy_owner_and_invalid_raw_inputs_rejected(kind):
    proc, obs = processor(), observations()
    owner = None
    if kind == 'foreign_hook': proc._lite_image_prepare = lambda *args: None
    if kind == 'locked_owner':
        owner = install_packed_input_metadata(proc)
        owner._lock.acquire()
    if kind == 'invalid_state': obs[11]['state'][0] = np.nan
    if kind == 'no_cameras': obs[11]['images'] = {}
    try:
        with pytest.raises((ValueError, RuntimeError)):
            build_coalesced_inputs(proc, obs, return_host_layout=True)
    finally:
        if owner is not None: owner._lock.release()


def test_autocast_cannot_silently_change_the_original_uint8_hook_fallback():
    proc, obs = processor(), observations()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='no-autocast'):
            build_coalesced_inputs(proc, obs)


def engine(mode, *, compile_owner=None):
    from collections import Counter
    import threading
    from types import SimpleNamespace
    from vvla_repro_pi05.lite import batch_pi05
    class Runner:
        def __init__(self):
            self.calls, self.stats, self.last_call = 0, Counter(), {}
        def __call__(self, **values):
            self.calls += 1
            self.last_call = {'cold': self.calls == 1}
            self.stats['calls'] += 1
            return values['noise'] + values['images'][0].mean((1, 2, 3))[:, None, None]
    policy = SimpleNamespace(processor=processor(), device=torch.device('cpu'),
        graph_runner=Runner(), _closed=False, _lock=threading.Lock(), _action_norm_cache=None)
    policy.close = lambda: setattr(policy, '_closed', True)
    return batch_pi05._wrap_policy(policy, batch_pi05._CAPABILITY,
        input_preparation=mode,compile_owner=compile_owner), policy


def test_private_modes_have_one_identical_model_call_and_original_owned_outputs():
    original, old_policy = engine('per_row')
    candidate, new_policy = engine('coalesced_fresh_v1')
    obs, seeds = observations(), list(range(32))
    actual = candidate.infer_batch(obs, seeds)
    expected = original.infer_batch(obs, seeds)
    assert actual['actions'].tobytes() == expected['actions'].tobytes()
    assert candidate.last_noise().tobytes() == original.last_noise().tobytes()
    assert old_policy.graph_runner.calls == new_policy.graph_runner.calls == 1
    saved = actual['actions'].copy()
    changed = copy.deepcopy(obs)
    changed[31]['images']['observation.images.front'] ^= np.uint8(255)
    later = candidate.infer_batch(changed, seeds)
    assert later['actions'][:31].tobytes() == saved[:31].tobytes()
    assert later['actions'][31].tobytes() != saved[31].tobytes()
    assert actual['actions'].tobytes() == saved.tobytes()
    assert actual['actions'].flags.owndata and actual['actions'].flags.c_contiguous
    with pytest.raises(AttributeError):
        candidate.input_preparation = 'per_row'
    original.close();candidate.close()


@pytest.mark.parametrize('mode', ['coalesced', '', None, True, {'mode': 'coalesced_fresh_v1'}])
def test_unknown_preparation_mode_is_rejected_before_model_load(mode):
    from vvla_repro_pi05.lite.batch_pi05 import Pi05BatchEngine
    with pytest.raises(ValueError, match='input_preparation'):
        Pi05BatchEngine.from_pretrained('/absent/checkpoint', input_preparation=mode)


@pytest.mark.parametrize('mode', ['per_row', 'coalesced_fresh_v1'])
@pytest.mark.parametrize('change', ['class_builder', 'class_output', 'instance_builder', 'single', 'output', 'batch_size'])
def test_cold_processor_anchors_reject_later_owner_or_method_replacement(mode, change, monkeypatch):
    current, policy = engine(mode)
    wrapper = current._Pi05BatchEngine__processor
    if change == 'class_builder':
        monkeypatch.setattr(type(wrapper), 'build_model_inputs', lambda *args, **kw: None)
    elif change == 'class_output':
        monkeypatch.setattr(type(wrapper), 'build_model_outputs', lambda *args, **kw: None)
    elif change == 'instance_builder':
        wrapper.build_model_inputs = lambda *args, **kw: None
    elif change == 'single':
        wrapper._single = processor()
    elif change == 'output':
        wrapper._output = processor()
    else:
        wrapper.batch_size = True
    with pytest.raises(RuntimeError, match='ownership'):
        current.infer_batch(observations(), list(range(32)))
    assert policy.graph_runner.calls == 0
    current.close()


def test_compile_initialization_occurs_after_current_preparation_before_one_graph_call():
    class Owner:
        def __init__(self):self.calls=0;self.initialized=False;self.closed=False
        def ensure_initialized(self,policy):
            assert policy._lock.locked()
            assert len(policy.processor.tokenizer.texts)==32*(self.calls+1)
            assert policy.graph_runner.calls==self.calls
            self.calls+=1;self.initialized=True
        def diagnostics(self):return {'initialized':self.initialized}
        def close(self):self.closed=True
    owner=Owner();current,policy=engine('coalesced_fresh_v1',compile_owner=owner)
    obs,seeds=observations(),list(range(32))
    first=current.infer_batch(obs,seeds)
    again=current.infer_batch(obs,seeds)
    assert owner.calls==2 and policy.graph_runner.calls==2
    assert first['actions'].tobytes()==again['actions'].tobytes()
    assert first['diagnostics']['compile_libdevice_cache']=={'initialized':True}
    current.close();current.close();assert owner.closed


def test_compile_initialization_failure_stops_before_model_and_releases_request_lock():
    class Owner:
        closed=False
        def ensure_initialized(self,policy):raise RuntimeError('initialization failed')
        def close(self):self.closed=True
    owner=Owner();current,policy=engine('per_row',compile_owner=owner)
    with pytest.raises(RuntimeError,match='initialization failed'):
        current.infer_batch(observations(),list(range(32)))
    assert policy.graph_runner.calls==0 and not policy._lock.locked()
    current.close();assert owner.closed


@pytest.mark.parametrize('value',[None,0,1,'true',{}])
def test_compile_initialization_selection_requires_bool(value):
    from vvla_repro_pi05.lite.batch_pi05 import Pi05BatchEngine
    with pytest.raises(ValueError,match='explicit immutable boolean'):
        Pi05BatchEngine.from_pretrained('/absent',compile_libdevice_cache=value)


def test_disabled_compile_method_cannot_accept_qualification_evidence():
    from vvla_repro_pi05.lite.batch_pi05 import Pi05BatchEngine
    with pytest.raises(ValueError,match='explicit compile cache method'):
        Pi05BatchEngine.from_pretrained('/absent',batch_compile_libdevice_evidence={})
