"""CPU source contracts for actual OFT capture hooks and canonical coordinates.

No checkpoint/model loading; tests are prepared for the root-owned queue only.
"""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "workers/oft/src"))
    return importlib.import_module("vvla_oft.lite_prepared")


def actual_arrays(*, padded=False, packed=False):
    # Two prompt tokens (incl. BOS/stop), 56 action placeholders, EOS.
    count, capacity, projected = 59, 64 if padded else 59, 513
    ids = np.ones((1, capacity), np.int64)
    ids[0, :2], ids[0, 58], ids[0, 59:] = [1, 29871], 2, 0
    mask = np.zeros_like(ids, dtype=bool)
    mask[:, :count] = True
    action_mask = np.zeros_like(mask)
    action_mask[:, 2:58] = True
    raw = dict(pixel_values_bf16_bits=np.arange(12 * 224 * 224, dtype=np.uint16).reshape(1, 12, 224, 224),
               proprio_bf16_bits=np.arange(8, dtype=np.uint16).reshape(1, 8),
               input_ids=ids, attention_mask=mask, action_mask=action_mask,
               action_starts=np.array([1], np.int64))
    total = projected + (count if packed else capacity)
    raw.update(pack_indices=np.arange(total, dtype=np.int64)[None, :],
               position_ids=np.arange(total, dtype=np.int64)[None, :],
               packed_valid=np.arange(total)[None, :] < projected + count,
               head_indices=projected + 1 + np.arange(56, dtype=np.int64)[None, :],
               readout_indices=projected + 1 + np.arange(56, dtype=np.int64)[None, :])
    return raw, projected


@pytest.mark.parametrize("packed", [False, True])
def test_canonical_actual_native_padding_and_packing_match_official_without_lost_tokens(prepared, packed):
    native, projected = actual_arrays(padded=True, packed=packed)
    official, _ = actual_arrays()
    a = prepared.canonicalize(native, projected_count=projected, native=True)
    b = prepared.canonicalize(official, projected_count=projected, native=False)
    assert set(a) == {"pixel_values_bf16_bits", "proprio_bf16_bits", "input_ids", "attention_mask",
                      "action_mask", "action_starts", "position_ids", "readout_indices"}
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].flags.owndata and a[key].flags.c_contiguous
    native["pixel_values_bf16_bits"][:] = 0
    assert np.any(a["pixel_values_bf16_bits"] != 0)


@pytest.mark.parametrize("fault", ["middle_mask", "action_in_padding", "mask_nonbinary", "action_shift",
                                   "position_shift", "packed_order", "head_padding", "head_shift",
                                   "image_dtype", "proprio_shape", "missing_key", "readout"])
def test_no_invalid_layout_or_missing_actual_field_can_be_canonicalized_away(prepared, fault):
    raw, projected = actual_arrays(padded=True)
    native = True
    if fault == "middle_mask":
        raw["attention_mask"][0, 1] = False
    elif fault == "action_in_padding":
        raw["action_mask"][0, 60] = True
    elif fault == "mask_nonbinary":
        raw["attention_mask"] = raw["attention_mask"].astype(np.int64)
        raw["attention_mask"][0, 1] = 2
    elif fault == "action_shift":
        raw["action_starts"][0] += 1
    elif fault == "position_shift":
        raw["position_ids"][0, 1] += 1
    elif fault == "packed_order":
        raw["pack_indices"][0, :2] = [1, 0]
    elif fault == "head_padding":
        raw["head_indices"][0, 0] = projected + 60
    elif fault == "head_shift":
        raw["head_indices"][0, 0] += 1
    elif fault == "image_dtype":
        raw["pixel_values_bf16_bits"] = raw["pixel_values_bf16_bits"].astype(np.float32)
    elif fault == "proprio_shape":
        raw["proprio_bf16_bits"] = raw["proprio_bf16_bits"].reshape(8)
    elif fault == "missing_key":
        del raw["pixel_values_bf16_bits"]
    else:
        native = False
        raw["readout_indices"][0, 0] += 1
    with pytest.raises(ValueError):
        prepared.canonicalize(raw, projected_count=projected, native=native)


@pytest.fixture
def hook_policy():
    torch = pytest.importorskip("torch")
    raw, projected = actual_arrays()
    seen = []
    class Model:
        norm_stats = {"debug": {"action": {"q01": [0.] * 7, "q99": [1.] * 7},
                                "proprio": {"q01": [0.] * 8, "q99": [1.] * 8}}}
        def _prepare_input_for_action_prediction(self, ids, mask):
            seen.append("inputs")
            return ids, mask
        def _process_action_masks(self, labels):
            seen.append("mask")
            return labels
        def _process_vision_features(self, pixels, *args):
            seen.append("vision")
            return torch.zeros((1, projected - 1, 4), dtype=torch.bfloat16)
        def _process_proprio_features(self, vision, state, *args):
            seen.append("proprio")
            return torch.zeros((1, projected, 4), dtype=torch.bfloat16)
        def _regression_or_discrete_prediction(self, *args):
            seen.append("regression")
            return np.full((8, 7), 0.25, np.float32)
    model = Model()
    # An existing instance override must be restored as that exact object.
    original_vision = model._process_vision_features
    model._process_vision_features = original_vision
    policy = SimpleNamespace(_model=model, _execution="official", _batch_size=1,
        _deps=SimpleNamespace(torch=torch), _metadata={"recipe_sha256": "1" * 64, "source_sha256": {"lite.py": "2" * 64}})
    def call(fail_at=None):
        ids, mask = model._prepare_input_for_action_prediction(torch.from_numpy(raw["input_ids"]),
                                                               torch.from_numpy(raw["attention_mask"]))
        action_mask = model._process_action_masks(torch.from_numpy(raw["action_mask"]))
        if fail_at == "mask":
            raise RuntimeError("injected original prediction failure")
        vision = model._process_vision_features(torch.from_numpy(raw["pixel_values_bf16_bits"]).view(torch.bfloat16), None, False)
        output = model._process_proprio_features(vision, torch.from_numpy(raw["proprio_bf16_bits"]).view(torch.bfloat16).reshape(8), object())
        return model._regression_or_discrete_prediction(None, action_mask, output, mask, None, projected, 1, object())
    return policy, call, seen, raw


def test_hooks_capture_actual_consumer_tensors_call_originals_and_restore_exact_bindings(prepared, hook_policy):
    policy, call, seen, raw = hook_policy
    before = dict(vars(policy._model))
    capture = prepared.PreparedCapture(policy, "debug")
    with capture:
        actions = call()
    assert seen == ["inputs", "mask", "vision", "proprio", "regression"]
    assert vars(policy._model) == before
    assert all(vars(policy._model)[name] is value for name, value in before.items())
    report = capture.finish(actions)
    assert report["metadata"]["production_prediction_executed"] and report["metadata"]["hooks_restored"]
    assert report["metadata"]["projected_count"] == 513
    expected = prepared.canonicalize(raw, projected_count=513, native=False)
    for key in expected:
        np.testing.assert_array_equal(report["arrays"][key], expected[key])
    raw["input_ids"][:] = 17
    assert report["arrays"]["input_ids"][0, 1] == 29871
    assert report["actions"].flags.owndata


def test_original_prediction_failure_restores_all_instance_bindings(prepared, hook_policy):
    policy, call, _, _ = hook_policy
    before = dict(vars(policy._model))
    capture = prepared.PreparedCapture(policy, "debug")
    with pytest.raises(RuntimeError, match="original prediction failure"):
        with capture:
            call(fail_at="mask")
    assert vars(policy._model) == before and not capture.hooks
    with pytest.raises(ValueError, match="Incomplete"):
        capture.finish(np.zeros((8, 7), np.float32))


def test_missing_official_method_rolls_back_partial_hook_installation(prepared, hook_policy, monkeypatch):
    policy, _, _, _ = hook_policy
    before = dict(vars(policy._model))
    monkeypatch.delattr(type(policy._model), "_process_action_masks")
    capture = prepared.PreparedCapture(policy, "debug")
    with pytest.raises(AttributeError):
        with capture:
            pytest.fail("incomplete hooks must not execute prediction")
    assert vars(policy._model) == before and not capture.hooks


def test_changed_norm_stats_during_capture_is_rejected(prepared, hook_policy):
    policy, call, _, _ = hook_policy
    capture = prepared.PreparedCapture(policy, "debug")
    with capture:
        actions = call()
    policy._model.norm_stats = copy.deepcopy(policy._model.norm_stats)
    policy._model.norm_stats["debug"]["action"]["q99"][0] = 2.
    with pytest.raises(ValueError, match="Normalization stats changed"):
        capture.finish(actions)


def test_restore_failure_attempts_other_bindings_and_remains_visible_for_poison(prepared, hook_policy, monkeypatch):
    policy, _, _, _ = hook_policy
    before = dict(vars(policy._model))
    base = type(policy._model)
    blocked = {"value": True}
    def guarded_delete(self, name):
        if blocked["value"] and name == "_prepare_input_for_action_prediction":
            raise RuntimeError("injected restore failure")
        object.__delattr__(self, name)
    monkeypatch.setattr(base, "__delattr__", guarded_delete, raising=False)
    capture = prepared.PreparedCapture(policy, "debug")
    with pytest.raises(RuntimeError, match="restore every"):
        with capture:
            pass
    assert [row[0] for row in capture.hooks] == ["_prepare_input_for_action_prediction"]
    assert set(vars(policy._model)) == set(before) | {"_prepare_input_for_action_prediction"}
    assert policy._model._process_vision_features is before["_process_vision_features"]
    blocked["value"] = False
    capture.__exit__(None, None, None)
    assert vars(policy._model) == before and not capture.hooks
