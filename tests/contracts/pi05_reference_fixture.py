"""CPU contracts for the real processor and standalone policy boundary."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from vvla_pi05.lite.attention import prepare_pi05_attention_metadata, prepare_pi05_static_metadata
from vvla_pi05.lite.pi05 import Pi05LitePolicy
from vvla_pi05.lite.processor import Pi05LiteProcessor, resize_unit_then_siglip
from vvla_pi05.models.pi05.config import Pi05Config
from vvla_pi05.models.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE, default_weight_loader
from vvla_pi05.models.pi05.processor_pi05 import Pi05Processor


class _Tokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        self.last_text = text
        return {"input_ids": [1] * max_length, "attention_mask": [1] * max_length}


class _Model(torch.nn.Module):
    action_horizon = 2
    action_dim = 4

    def __init__(self):
        super().__init__()
        self.register_buffer("output", torch.zeros(1, 2, 4))

    def sample_actions(self, *, images, image_masks, lang_tokens, lang_masks, noise, num_steps):
        self.output.copy_(noise)
        return self.output


def _policy():
    config = Pi05Config(
        chunk_size=2, n_action_steps=2, max_action_dim=4, max_state_dim=4,
        image_resolution=(4, 4), max_cameras=2, tokenizer_max_length=5,
        input_features={"observation.state": {"shape": [3]}, "observation.images.front": {"shape": [3, 4, 4]}},
        output_features={"action": {"shape": [3]}},
        use_relative_actions=True, relative_exclude_joints=["gripper"],
        action_feature_names=["x", "y", "gripper"],
        norm_stats={
            "state": {"mode": "mean_std", "mean": [0, 0, 0], "std": [10, 10, 10]},
            "action": {"mode": "mean_std", "mean": [10, 20, 30], "std": [2, 3, 4]},
        },
    )
    processor = Pi05LiteProcessor(config, _Tokenizer(), torch.device("cpu"))
    return Pi05LitePolicy(_Model(), processor, device="cpu", cuda_graph=False)


def _obs():
    return {
        "observation.images.front": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.array([1, 2, 3], dtype=np.float32), "prompt": "pick_cube",
    }


def test_raw_to_physical_order_and_owned_output():
    policy = _policy()
    noise = np.ones((1, 2, 4), dtype=np.float32)
    first = policy.predict(_obs(), noise)
    # inverse norm [12,23,34], THEN relative -> absolute [13,25,34].
    np.testing.assert_array_equal(first["actions"], [[13, 25, 34], [13, 25, 34]])
    assert first["actions"].dtype == np.float32 and first["actions"].flags.owndata
    assert first["cold"]
    before = first["actions"].copy()
    second = policy.predict(_obs(), noise * 2)
    assert not second["cold"]
    np.testing.assert_array_equal(first["actions"], before)
    np.testing.assert_array_equal(noise, np.ones_like(noise))
    assert "pick cube" in policy.processor.tokenizer.last_text
    assert second["diagnostics"]["total_ms"] >= second["diagnostics"]["host_prepare_ms"]


def test_seed_explicit_noise_equivalence_without_global_rng_changes():
    policy = _policy()
    before = torch.random.get_rng_state().clone()
    first = policy.infer(_obs(), 1234)
    noise = torch.randn((1, 2, 4), generator=torch.Generator().manual_seed(1234)).numpy()
    explicit = policy.predict(_obs(), noise)
    np.testing.assert_array_equal(first["actions"], explicit["actions"])
    assert torch.equal(before, torch.random.get_rng_state())
    policy.reset("another_episode")
    np.testing.assert_array_equal(policy.infer(_obs(), 1234)["actions"], first["actions"])


@pytest.mark.parametrize("noise", [np.zeros((2, 4), np.float32), np.zeros((1, 2, 4), np.float64), np.full((1, 2, 4), np.nan, np.float32)])
def test_reject_invalid_explicit_noise(noise):
    with pytest.raises(ValueError, match="noise"):
        _policy().predict(_obs(), noise)


def test_policy_lock_and_close_fail_closed():
    policy = _policy()
    policy._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="serialized"):
            policy.infer(_obs(), 1)
        with pytest.raises(RuntimeError, match="active"):
            policy.reset()
    finally:
        policy._lock.release()
    policy.close()
    with pytest.raises(RuntimeError, match="closed"):
        policy.infer(_obs(), 1)


def test_standalone_imports_never_touch_serving_runtime():
    source = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import importlib.abc, sys
sys.path.insert(0, {str(source)!r})
class RejectServing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'vllm', 'vllm_omni'}}:
            raise AssertionError('unexpected serving import: ' + fullname)
sys.meta_path.insert(0, RejectServing())
from vvla_pi05.lite import Pi05LitePolicy
from vvla_pi05.lite import attention, graph, dtype
attention.make_pi05_flash_attention(8, 1, 256, backend='torch_fa2')
from vvla_pi05.models.pi05 import modeling_pi05, processor_pi05
from vvla_pi05.models.pi05.config import Pi05Config
import torch
with torch.device('meta'):
    model = modeling_pi05.Pi05ForActionPrediction(Pi05Config(attention_backend='torch_fa2'))
assert model.paligemma_with_expert.prefix_attention.backend == 'torch_fa2'
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_attention_metadata_retains_row_order_and_mask_semantics():
    q = torch.tensor([[True, False, True], [False, True, False]])
    k = torch.tensor([[False, True, True, False], [True, False, True, True]])
    fixed = prepare_pi05_static_metadata(q, k).extra
    assert fixed["q_indices"].tolist() == [0, 2, 4, 1, 3, 5]
    assert fixed["cu_q"].tolist() == [0, 2, 3]
    assert fixed["cu_k"].tolist() == [0, 2, 5]
    visible = q[:, :, None] & k[:, None, :]
    mask = torch.where(visible[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
    eager = prepare_pi05_attention_metadata(mask, 4).extra
    assert eager["q_indices"].tolist() == [0, 2, 4]
    assert eager["k_indices"].tolist() == fixed["k_indices"][:5].tolist()
    mask[0, 0, 0, 2] = OPENPI_ATTENTION_MASK_VALUE
    with pytest.raises(ValueError, match="separable"):
        prepare_pi05_attention_metadata(mask, 4)


def test_loader_casts_original_weights_and_rejects_broadcasting():
    target = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    default_weight_loader(target, source)
    assert torch.equal(target, source.to(torch.bfloat16))
    with pytest.raises(ValueError, match="shape"):
        default_weight_loader(target, source[0])


def test_canonical_resize_precedes_normalization_and_keeps_native_padding():
    from vvla_pi05.models.pi05.processor_pi05 import Pi05ImageProcessor

    image = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    expected = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    expected = torch.nn.functional.interpolate(expected, (224, 224), mode="bilinear", align_corners=False).clamp(0, 1)
    expected = expected * 2.0 - 1.0
    actual = resize_unit_then_siglip(image, 224, 224, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # This fixture distinguishes the operation ordering at model BF16 precision.
    historical = Pi05ImageProcessor(224).preprocess_single(image)
    assert (actual.bfloat16() != historical.bfloat16()).any()
    rectangular = resize_unit_then_siglip(np.full((4, 8, 3), 255, np.uint8), 8, 8, torch.device("cpu"))
    assert torch.equal(rectangular[:, :, :2], torch.full((1, 3, 2, 8), -1.0))
    assert torch.equal(rectangular[:, :, 2:6], torch.ones((1, 3, 4, 8)))


def test_canonical_processor_keeps_shared_state_tokens_and_empty_masks():
    policy = _policy()
    baseline = Pi05Processor(policy.config, _Tokenizer(), torch.device("cpu"))
    images, image_masks, tokens, masks = policy.processor.build_model_inputs(_obs())
    reference = baseline.build_model_inputs(_obs())
    assert [bool(mask.item()) for mask in image_masks] == [True, False]
    assert torch.equal(images[1], torch.full_like(images[1], -1))
    assert torch.equal(tokens, reference[2]) and torch.equal(masks, reference[3])
    assert baseline.tokenizer.last_text == policy.processor.tokenizer.last_text


def test_preflight_and_saved_fixture_hash_real_model_boundary():
    import hashlib

    policy = _policy()
    policy.config.vision_bf16 = True
    noise = np.ones((1, 2, 4), dtype=np.float32)
    report = policy.preflight(_obs(), noise, repeat=2)
    fixture = policy.prepared_fixture(_obs(), noise)
    assert report["passed"] and report["deterministic_replays"] == 2
    assert report["metadata"]["sampler_dtype"] == "float32"
    assert report["metadata"]["image_preprocess_version"] == "lerobot_resize_unit_then_siglip_v1"
    for field, array in (("images", fixture["image_0_bf16_bits"]), ("tokens", fixture["tokens"]),
                         ("masks", fixture["masks"]), ("noise_fp32", fixture["noise_fp32"]),
                         ("noise_bf16", fixture["noise_bf16_bits"])):
        digest = report["prepared_sha256"][field]
        if field == "images":
            digest = digest[0]
        assert hashlib.sha256(array.tobytes()).hexdigest() == digest
    assert fixture["image_0_bf16_bits"].dtype == np.uint16
    np.testing.assert_array_equal(fixture["noise_fp32"], noise)


def test_metadata_stays_stable_across_process_diagnostics_and_inference():
    policy = _policy()
    baseline = policy.metadata()
    policy.load_seconds = 123.456
    policy.reset("ep0")
    policy.infer(_obs(), 42)
    assert policy.metadata() == baseline
    assert "load_seconds" not in baseline


def test_full_bf16_layout_checkpoint_and_recipe_integration_on_meta():
    """Exercise production module types/guards without multi-GB CPU allocation."""
    from vvla_pi05.lite.dtype import set_bfloat16_layout
    from vvla_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vvla_pi05.models.pi05.precision_pi05 import (
        install_execution_recipe, set_pointwise_fusions, validate_execution_recipe,
    )
    from vvla_pi05.profiles import apply_profile

    deploy = {"stages": [{"model_config": {}, "custom_pipeline_args": {}}]}
    apply_profile(deploy, "full-bf16")
    stage = deploy["stages"][0]
    config = Pi05Config(**stage["model_config"], attention_backend="torch_fa2")
    with torch.device("meta"):
        model = Pi05ForActionPrediction(config)
        set_bfloat16_layout(model)
        # Strict loading exercises the same names/casts/invalidation hooks as
        # real safetensors, with shape-only tensors instead of real parameters.
        model.load_weights(("model." + name, param.detach()) for name, param in model.named_parameters())
        model.eval()
        install_execution_recipe(model, config.execution_recipe)
        set_pointwise_fusions(model, **stage["custom_pipeline_args"]["pi05_pointwise_fusions"])
        validate_execution_recipe(model)
    assert model._execution_state["status"] == "ready"
    assert model._execution_state["recipe"].fuse_vision_residual_cast
    assert model._execution_state["recipe"].fuse_rope
    assert model._execution_state["recipe"].fuse_prefix_residual_square
    vision = model.paligemma_with_expert.paligemma.model.vision_tower
    assert type(vision.encoder.layers[0]).__name__ == "VisionResidualCastLayer"
    assert type(vision.encoder.layers[0].layer_norm1).__name__ == "_Pi05VisionLayerNormFP32"
    assert vision.encoder.layers[0].layer_norm1.weight.dtype == torch.float32
