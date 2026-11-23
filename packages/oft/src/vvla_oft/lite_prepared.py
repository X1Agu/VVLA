"""Cold B1 snapshots of tensors actually consumed by OFT production prediction.

No alternative processor, tokenization, normalization or inference path exists
here. Official hooks call their original bindings and are always restored. The
native snapshot consumes the exact service.collate result sent to its runner.
"""
import copy
import hashlib
import json

import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def canonicalize(raw, *, projected_count, native):
    """Canonicalize only right padding and native packed position coordinates."""
    required = {"pixel_values_bf16_bits", "proprio_bf16_bits", "input_ids",
                "attention_mask", "action_mask", "action_starts"}
    if not required <= raw.keys() or type(projected_count) is not int or projected_count < 1:
        raise ValueError("Incomplete actual OFT prepared capture")
    ids, mask, actions = raw["input_ids"], raw["attention_mask"], raw["action_mask"]
    if ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Expected actual B1 int64 input_ids")
    if mask.shape != ids.shape or actions.shape != ids.shape:
        raise ValueError("Actual token mask geometry differs")
    if not np.isin(mask, [0, 1]).all() or not np.isin(actions, [0, 1]).all():
        raise ValueError("Actual token masks must be binary")
    valid = mask.astype(bool)
    count = int(valid.sum())
    if count == 0 or not valid[0, :count].all() or valid[0, count:].any():
        raise ValueError("Only verified right padding may be removed")
    if actions[0, count:].any():
        raise ValueError("Action positions cannot be removed with padding")
    starts = raw["action_starts"]
    if starts.dtype != np.int64 or starts.shape != (1,):
        raise ValueError("Actual action starts must be B1 int64")
    start = int(starts[0])
    expected_action_mask = np.zeros((1, count), dtype=bool)
    if start < 0 or start + 58 != count:
        raise ValueError("Actual shifted 56-action readout geometry differs")
    expected_action_mask[0, start + 1:start + 57] = True
    if not np.array_equal(actions[:, :count].astype(bool), expected_action_mask):
        raise ValueError("Actual action mask differs from captured readout")
    pixels, proprio = raw["pixel_values_bf16_bits"], raw["proprio_bf16_bits"]
    if pixels.dtype != np.uint16 or pixels.shape != (1, 12, 224, 224):
        raise ValueError("Expected actual two-camera BF16 processor pixels")
    if proprio.dtype != np.uint16 or proprio.shape != (1, 8):
        raise ValueError("Expected actual BF16 proprio [1,8]")
    original_readout = projected_count + start + np.arange(56, dtype=np.int64)[None, :]
    if native:
        for name in ("pack_indices", "position_ids", "packed_valid", "head_indices"):
            if name not in raw:
                raise ValueError("Missing actual native layout: " + name)
        indices, positions, packed_valid, head = (raw[name] for name in
            ("pack_indices", "position_ids", "packed_valid", "head_indices"))
        if indices.dtype != np.int64 or positions.dtype != np.int64 or head.dtype != np.int64:
            raise ValueError("Actual native indices must be int64")
        if indices.ndim != 2 or indices.shape[0] != 1 or positions.shape != indices.shape or packed_valid.shape != indices.shape:
            raise ValueError("Actual native packed geometry differs")
        if not np.isin(packed_valid, [0, 1]).all():
            raise ValueError("Actual packed mask must be binary")
        if head.shape != (1, 56) or np.any(head < 0) or np.any(head >= indices.shape[1]):
            raise ValueError("Actual native readout is out of bounds")
        packed_valid = packed_valid.astype(bool)
        if not np.take_along_axis(packed_valid, head, axis=1).all():
            raise ValueError("Actual native readout targets padding")
        restored = indices[packed_valid][None, :]
        expected_positions = np.arange(projected_count + count, dtype=np.int64)[None, :]
        if not np.array_equal(restored, expected_positions):
            raise ValueError("Packed layout removed or reordered a valid position")
        canonical_positions = positions[packed_valid][None, :]
        if not np.array_equal(canonical_positions, restored):
            raise ValueError("Actual native rotary position differs from original position")
        if not np.array_equal(np.take_along_axis(indices, head, axis=1), original_readout):
            raise ValueError("Actual native readout changed original action positions")
    else:
        # The pinned official regression calls Llama with position_ids=None;
        # original valid positions are implicit consecutive positions. Readout
        # bounds come from actual NUM_PATCHES / NUM_PROMPT_TOKENS arguments.
        canonical_positions = np.arange(projected_count + count, dtype=np.int64)[None, :]
        if not np.array_equal(raw.get("readout_indices"), original_readout):
            raise ValueError("Actual official readout differs")
    arrays = {"pixel_values_bf16_bits": pixels, "proprio_bf16_bits": proprio,
              "input_ids": ids[:, :count], "attention_mask": valid[:, :count],
              "action_mask": actions[:, :count].astype(bool), "action_starts": starts,
              "position_ids": canonical_positions, "readout_indices": original_readout}
    return {name: np.array(value, copy=True, order="C") for name, value in arrays.items()}


class PreparedCapture:
    """Request-local hooks. Caller owns the policy lock and executes one predict."""

    def __init__(self, policy, norm_key):
        self.policy, self.norm_key = policy, norm_key
        self.raw, self.hooks, self.counts = {}, [], {}
        self.projected_count = None
        self.norm = copy.deepcopy(policy._model.norm_stats[norm_key])
        self.native_path = policy._execution == "native"

    def _array(self, tensor, *, bf16=False):
        torch = self.policy._deps.torch
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Prepared capture requires actual Torch tensors")
        if bf16:
            if tensor.dtype != torch.bfloat16:
                raise ValueError("Prepared capture must observe actual BF16 consumer input")
            tensor = tensor.view(torch.uint16)
        return tensor.detach().contiguous().cpu().numpy().copy()

    def _once(self, name, value):
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.counts[name] != 1:
            raise ValueError("Ambiguous repeated official prepared capture: " + name)
        self.raw[name] = value

    def _hook(self, name, intercept):
        model = self.policy._model
        original = getattr(model, name)
        had_instance_value = name in vars(model)
        previous = vars(model).get(name)
        def wrapped(*args, **kwargs):
            return intercept(original, *args, **kwargs)
        self.hooks.append((name, had_instance_value, previous))
        setattr(model, name, wrapped)

    def __enter__(self):
        if self.native_path:
            return self
        if self.policy._execution != "official" or self.policy._batch_size != 1:
            raise ValueError("Actual official fixture capture supports the upstream B1 path")
        try:
            def inputs(original, *args, **kwargs):
                result = original(*args, **kwargs)
                self._once("input_ids", self._array(result[0]))
                self._once("attention_mask", self._array(result[1]))
                return result
            def masks(original, *args, **kwargs):
                result = original(*args, **kwargs)
                self._once("action_mask", self._array(result))
                return result
            def vision(original, pixels, *args, **kwargs):
                self._once("pixel_values_bf16_bits", self._array(pixels, bf16=True))
                return original(pixels, *args, **kwargs)
            def proprio(original, projected, state, *args, **kwargs):
                self._once("proprio_bf16_bits", self._array(state, bf16=True).reshape(1, 8))
                result = original(projected, state, *args, **kwargs)
                self.projected_count = int(result.shape[1])
                return result
            def regression(original, *args, **kwargs):
                # Exact positional signature in the locked upstream source.
                if kwargs or len(args) != 8:
                    raise ValueError("Locked official regression signature changed")
                patches, start = args[5], args[6]
                if patches != self.projected_count or type(start) is not int:
                    raise ValueError("Actual official projected/readout geometry differs")
                self._once("action_starts", np.array([start], dtype=np.int64))
                self._once("readout_indices", patches + start + np.arange(56, dtype=np.int64)[None, :])
                return original(*args, **kwargs)
            for name, hook in (("_prepare_input_for_action_prediction", inputs),
                ("_process_action_masks", masks), ("_process_vision_features", vision),
                ("_process_proprio_features", proprio), ("_regression_or_discrete_prediction", regression)):
                self._hook(name, hook)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def native(self, tensors):
        if not self.native_path or self.raw:
            raise ValueError("Unexpected/repeated native prepared capture")
        for name in ("input_ids", "attention_mask", "action_mask", "action_starts",
                     "pack_indices", "position_ids", "packed_valid", "head_indices"):
            self.raw[name] = self._array(tensors[name])
        self.raw["pixel_values_bf16_bits"] = self._array(tensors["pixel_values"], bf16=True)
        self.raw["proprio_bf16_bits"] = self._array(tensors["proprio"], bf16=True)
        self.projected_count = self.policy._adapter.projected_count

    def __exit__(self, exc_type, exc, traceback):
        # Reverse exact bindings, including absence of an instance override.
        failed, errors = [], []
        for name, existed, previous in reversed(self.hooks):
            try:
                if existed:
                    setattr(self.policy._model, name, previous)
                else:
                    delattr(self.policy._model, name)
            except BaseException as error:
                failed.append((name, existed, previous))
                errors.append(error)
        self.hooks = list(reversed(failed))
        if errors:
            failure = RuntimeError("Unable to restore every official prepared capture binding")
            for error in errors:
                failure.add_note(repr(error))
            raise failure from exc

    def finish(self, actions):
        if self.hooks:
            raise RuntimeError("Prepared hooks must be restored before publishing evidence")
        if _digest(self.policy._model.norm_stats[self.norm_key]) != _digest(self.norm):
            raise ValueError("Normalization stats changed during the actual fixture call")
        arrays = canonicalize(self.raw, projected_count=self.projected_count, native=self.native_path)
        return {"arrays": arrays,
                "raw_arrays": {name: np.array(value, copy=True, order="C") for name, value in self.raw.items()},
                "actions": np.array(actions, dtype=np.float32, copy=True, order="C"),
                "metadata": {"schema": 1, "production_prediction_executed": True,
                    "diagnostic_only": True, "hooks_restored": True,
                    "capture_path": "actual_service_collate_to_native_runner" if self.native_path else "actual_upstream_B1_prediction_hooks",
                    "norm_key": self.norm_key, "norm_stats_sha256": _digest(self.norm),
                    "projected_count": self.projected_count,
                    "canonicalization": "remove verified right padding; restore actual packed original positions/readout",
                    "official_positions": "implicit consecutive Llama positions; actual regression bounds" if not self.native_path else None,
                    "recipe_sha256": self.policy._metadata["recipe_sha256"],
                    "source_sha256": copy.deepcopy(self.policy._metadata["source_sha256"])}}
