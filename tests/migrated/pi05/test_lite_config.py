"""Regression for the actual LIBERO checkpoint config retrieved on PRO6000."""

from copy import deepcopy
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from vvla_repro_pi05.lite.pi05 import load_lite_config


# Exact config captured in local-state/g4/probe-ssh.json on 2026-11-05.
_LIBERO = json.loads(r'''{
  "type": "pi05", "n_obs_steps": 1,
  "input_features": {
    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.state": {"type": "STATE", "shape": [8]},
    "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 224, 224]}
  },
  "output_features": {"action": {"type": "ACTION", "shape": [7]}},
  "device": "cuda", "use_amp": false, "push_to_hub": true,
  "repo_id": "pepijn223/pi05_libero_31_9_non_q", "private": null,
  "tags": null, "license": null, "pretrained_path": "lerobot/pi05_libero",
  "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
  "dtype": "bfloat16", "chunk_size": 50, "n_action_steps": 50,
  "max_state_dim": 32, "max_action_dim": 32, "num_inference_steps": 10,
  "time_sampling_beta_alpha": 1.5, "time_sampling_beta_beta": 1.0,
  "time_sampling_scale": 0.999, "time_sampling_offset": 0.001,
  "min_period": 0.004, "max_period": 4.0,
  "image_resolution": [224, 224], "empty_cameras": 1, "tokenizer_max_length": 200,
  "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
  "gradient_checkpointing": true, "compile_model": true, "compile_mode": "max-autotune",
  "optimizer_lr": 2.5e-05, "optimizer_betas": [0.9, 0.95], "optimizer_eps": 1e-08,
  "optimizer_weight_decay": 0.01, "optimizer_grad_clip_norm": 1.0,
  "scheduler_warmup_steps": 1000, "scheduler_decay_steps": 6000, "scheduler_decay_lr": 2.5e-06
}''')


def _write(path, raw):
    data = json.dumps(raw, indent=2).encode()
    (path / "config.json").write_bytes(data)
    return data


def _sidecar(path, *, state_width=8):
    (path / "policy_preprocessor.json").write_text(json.dumps({"steps": [{
        "registry_name": "normalizer_processor", "state_file": "stats.safetensors",
        "config": {"norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}},
    }]}))
    save_file({
        "observation.state.mean": torch.zeros(state_width), "observation.state.std": torch.ones(state_width),
        "action.mean": torch.arange(7, dtype=torch.float32), "action.std": torch.ones(7),
    }, str(path / "stats.safetensors"))


def test_materialized_empty_camera_config_is_consumed_once_without_checkpoint_mutation(tmp_path):
    original = _write(tmp_path, _LIBERO)
    _sidecar(tmp_path)
    config, audit = load_lite_config(tmp_path)
    assert config.image_feature_keys == [
        "observation.images.image", "observation.images.image2", "observation.images.empty_camera_0",
    ]
    assert (config.state_dim, config.action_dim, config.chunk_size) == (8, 7, 50)
    assert config.norm_stats["state"]["mode"] == config.norm_stats["action"]["mode"] == "mean_std"
    assert config.norm_stats["action"]["mean"] == list(range(7))
    assert (tmp_path / "config.json").read_bytes() == original
    assert audit["original_config_sha256"] == hashlib.sha256(original).hexdigest()
    assert audit["checkpoint_files_modified"] is False
    assert audit["schema_normalization"][0]["original"] == 1
    assert audit["schema_normalization"][0]["loader_value"] == 0


@pytest.mark.parametrize("count", [True, -1, 1.5, "1", None])
def test_invalid_camera_counts_fail_closed(tmp_path, count):
    raw = deepcopy(_LIBERO)
    raw["empty_cameras"] = count
    _write(tmp_path, raw)
    with pytest.raises(ValueError, match="empty_cameras"):
        load_lite_config(tmp_path)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "type", "keys", "capacity"])
def test_inconsistent_explicit_camera_slots_fail_closed(tmp_path, fault):
    raw = deepcopy(_LIBERO)
    key = "observation.images.empty_camera_0"
    if fault == "missing":
        del raw["input_features"][key]
    elif fault == "extra":
        raw["input_features"]["observation.images.empty_camera_1"] = deepcopy(raw["input_features"][key])
    elif fault == "shape":
        raw["input_features"][key]["shape"] = [3, 256, 256]
    elif fault == "type":
        raw["input_features"][key]["type"] = "STATE"
    elif fault == "keys":
        raw["image_feature_keys"] = ["observation.images.image", "observation.images.image2"]
    else:
        raw["max_cameras"] = 2
    _write(tmp_path, raw)
    with pytest.raises(ValueError):
        load_lite_config(tmp_path)


def test_normalizer_width_and_other_unsupported_capabilities_remain_validated(tmp_path):
    _write(tmp_path, _LIBERO)
    _sidecar(tmp_path, state_width=7)
    with pytest.raises(ValueError, match="expected 8"):
        load_lite_config(tmp_path)
    raw = deepcopy(_LIBERO)
    raw["use_visual_memory"] = True
    _write(tmp_path, raw)
    with pytest.raises(ValueError, match="use_visual_memory"):
        load_lite_config(tmp_path)
