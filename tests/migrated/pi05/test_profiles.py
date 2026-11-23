import json
from pathlib import Path

import pytest

from vvla_repro_pi05.profiles import apply_profile


def deploy():
    return {"stages": [{"model_config": {}, "custom_pipeline_args": {}}]}


def test_blackwell_profile_matches_frozen_activation_contract():
    frozen = json.loads((Path(__file__).parents[1] / "provenance/pi05-frozen-profile.json").read_text())
    value = deploy()
    apply_profile(value, "blackwell-nvfp4", artifact="/artifact", artifact_sha256="a" * 64)
    stage = value["stages"][0]
    model = stage["model_config"]
    assert model["execution_recipe"] == frozen["base_recipe_for_packed_artifact_load"]
    assert all(model[key] == expected for key, expected in frozen["model_flags"].items())
    assert stage["custom_pipeline_args"]["pi05_pointwise_fusions"] == frozen["pointwise_fusions_after_artifact_load"]
    assert model["precision_hardware_fallback"] == "error"


@pytest.mark.parametrize("kwargs", [{}, {"artifact": "x"}, {"artifact_sha256": "a" * 64},
                                    {"artifact": "x", "calibration": "y", "artifact_sha256": "a" * 64}])
def test_blackwell_rejects_ambiguous_or_unverified_assets(kwargs):
    with pytest.raises(ValueError):
        apply_profile(deploy(), "blackwell-nvfp4", **kwargs)


def test_bf16_does_not_accept_packed_fp4_assets():
    with pytest.raises(ValueError):
        apply_profile(deploy(), "bf16", artifact="x")


def test_fp8_prefix_keeps_the_paired_attention_and_vision_settings():
    value = deploy()
    value["stages"][0]["model_config"].update(attention_backend="eager", vision_bf16=False)
    apply_profile(value, "fp8-prefix")
    model = value["stages"][0]["model_config"]
    assert model["attention_backend"] == "eager" and model["vision_bf16"] is False
    assert model["execution_recipe"] == {"prefix_mlp": "fp8"}
    assert model["precision_hardware_fallback"] == "error"
    with pytest.raises(ValueError):
        apply_profile(deploy(), "fp8-prefix", artifact="fp4")


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_full_profile_retains_portable_fusions_and_excludes_sm120_only_quantization(precision):
    value = deploy()
    apply_profile(value, "full-" + precision)
    stage = value["stages"][0]
    model = stage["model_config"]
    for flag in ["cache_empty_camera_features", "precompute_time_and_positions", "prefix_kv_only", "pack_prefix", "vision_bf16"]:
        assert model[flag] is True
    recipe = model["execution_recipe"]
    assert recipe["prefix_mlp"] == ("fp8" if precision == "fp8" else "none")
    assert recipe["vision_mlp"] == recipe["expert_mlp"] == "none"
    assert recipe["fuse_prefix_down_quant"] is False and recipe["fuse_gate_up"] is False
    assert recipe["fuse_prefix_rmsnorm"] and recipe["prefix_rmsnorm_reduction"] == "torch"
    assert recipe["fuse_prefix_mlp"] and recipe["fuse_vision_residual_cast"]
    assert all(stage["custom_pipeline_args"]["pi05_pointwise_fusions"].values())
    assert model["precision_hardware_fallback"] == "error"
    with pytest.raises(ValueError):
        apply_profile(deploy(), "full-" + precision, artifact="fp4")
