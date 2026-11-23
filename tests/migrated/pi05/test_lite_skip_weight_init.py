"""CPU evidence for skipped construction, strict loading and stable identity."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from vvla_repro_pi05.lite import pi05 as lite
from vvla_repro_pi05.models.pi05 import modeling_pi05 as core
from vvla_repro_pi05.models.pi05.config import Pi05Config


@pytest.fixture(scope="module")
def probe():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" /
            "pi05-lite-bf16-20261105" / "audit" / "skip_weight_init_probe.py")
    spec = importlib.util.spec_from_file_location("skip_weight_init_cpu_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_constructors_buffers_strict_load_and_action_parity(probe):
    report = probe.run_audit()
    assert report["passed"] and report["device"] == "cpu"
    baseline = report["modes"]["normal"]
    candidate = report["modes"]["lite_skip_weight_init"]
    assert baseline["actual_tensor_write_calls"]["normal_"] > 0
    assert baseline["actual_tensor_write_calls"]["uniform_"] > 0
    assert candidate["actual_tensor_write_calls"].get("normal_", 0) == 0
    assert candidate["actual_tensor_write_calls"].get("uniform_", 0) == 0
    assert len(candidate["buffers"]) == 6
    assert candidate["buffers"] == baseline["buffers"]
    assert candidate["pali_embedding_lm_head_same_parameter"]
    assert candidate["unique_parameter_elements"] == baseline["unique_parameter_elements"]
    assert candidate["all_parameters_finite_after_strict_load"]
    assert candidate["all_reference_parameters_equal_after_strict_load"]
    assert candidate["output_bitwise_equal_reference"]
    assert "time_mlp_out.bias" in candidate["strict_missing_parameter_raises"]


@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_loader_option_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="lite_skip_weight_init must be a boolean"):
        lite._split_lite_overrides({"lite_skip_weight_init": value})


def test_option_defaults_off_and_does_not_enter_model_config():
    assert lite._split_lite_overrides({})[1]["skip_weight_init"] is False
    config, options = lite._split_lite_overrides({"lite_skip_weight_init": True, "vision_bf16": True})
    assert config == {"vision_bf16": True}
    assert options["skip_weight_init"] is True
    assert options["native_lifecycle_variant"] == "v1"
    assert all(not value for name, value in options.items()
               if name not in ("skip_weight_init", "native_lifecycle_variant"))


def test_metadata_reads_actual_blas_preference_without_setting_it(monkeypatch):
    from test_lite_pi05 import _policy

    calls = []

    def preference(*args):
        calls.append(args)
        return "_BlasBackend.Cublaslt"

    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", preference)
    assert _policy().metadata()["cuda_blas_preference"] == "_BlasBackend.Cublaslt"
    assert calls == [()]
    monkeypatch.delattr(torch.backends.cuda, "preferred_blas_library")
    assert _policy().metadata()["cuda_blas_preference"] == "unavailable"


def test_constructor_failure_restores_global_initializers_and_releases_lock(monkeypatch):
    original_torch = torch.nn.init.normal_
    original_hf = PreTrainedModel.init_weights

    def fail(config):
        assert lite._MODEL_CONSTRUCTION_LOCK.locked()
        assert torch.nn.init.normal_ is not original_torch
        assert PreTrainedModel.init_weights is not original_hf
        raise RuntimeError("constructor failure")

    monkeypatch.setattr(core, "Pi05ForActionPrediction", fail)
    with pytest.raises(RuntimeError, match="constructor failure"):
        lite._construct_pi05_model(None, skip_weight_init=True)
    assert torch.nn.init.normal_ is original_torch
    assert PreTrainedModel.init_weights is original_hf
    assert not lite._MODEL_CONSTRUCTION_LOCK.locked()


def test_unaudited_transformers_release_fails_before_construction(monkeypatch):
    monkeypatch.setattr(lite.package_metadata, "version", lambda name: "5.20.0")
    monkeypatch.setattr(core, "Pi05ForActionPrediction", lambda config: pytest.fail("must not construct"))
    with pytest.raises(RuntimeError, match="audited transformers==5.19.0"):
        lite._construct_pi05_model(None, skip_weight_init=True)
    assert not lite._MODEL_CONSTRUCTION_LOCK.locked()


class _Tokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        return {"input_ids": [1] * max_length, "attention_mask": [1] * max_length}

    def get_vocab(self):
        return {"<test>": 1}


def test_real_loader_metadata_recipe_parity_and_incomplete_checkpoint(probe, tmp_path, monkeypatch):
    config = Pi05Config(
        chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8,
        num_inference_steps=2, image_resolution=(28, 28), tokenizer_max_length=8,
        input_features={"observation.state": {"shape": [8]},
                        "observation.images.front": {"shape": [3, 28, 28]}},
        output_features={"action": {"shape": [8]}},
        norm_stats={name: {"mode": "mean_std", "mean": [0] * 8, "std": [1] * 8}
                    for name in ("state", "action")},
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _Tokenizer())
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"), probe.small_dimensions():
            reference = core.Pi05ForActionPrediction(config)
            weights = {name: param.detach().clone() for name, param in reference.named_parameters()}
            save_file(weights, str(tmp_path / "model.safetensors"))
            policies = []
            for enabled in (False, True):
                policies.append(lite.Pi05LitePolicy.from_pretrained(
                    tmp_path, tokenizer=str(tmp_path), device="cpu", cuda_graph=False,
                    attention_backend="eager", profile="bf16",
                    config_overrides={"lite_skip_weight_init": enabled},
                ))
            ordinary, skipped = policies
            ordinary_metadata, skipped_metadata = ordinary.metadata(), skipped.metadata()
            assert not ordinary_metadata["lite_optimizations"]["skip_weight_init"]
            assert skipped_metadata["lite_optimizations"]["skip_weight_init"]
            assert skipped_metadata["weight_initialization"] == {
                "skipped": True, "context": "transformers.initialization.no_init_weights",
                "paligemma_tying": "explicit_after_constructor", "strict_checkpoint_load": True,
            }
            assert ordinary_metadata["recipe_sha256"] != skipped_metadata["recipe_sha256"]
            for field in ("checkpoint_sha256", "preprocessing_sha256", "resolved_config"):
                assert ordinary_metadata[field] == skipped_metadata[field]
            pali = skipped.model.paligemma_with_expert.paligemma
            assert pali.lm_head.weight is pali.model.language_model.embed_tokens.weight
            assert all(torch.isfinite(param).all() for param in skipped.model.parameters())
            assert dict(ordinary.model.named_parameters()).keys() == dict(skipped.model.named_parameters()).keys()
            for name, param in ordinary.model.named_parameters():
                assert torch.equal(param, dict(skipped.model.named_parameters())[name])
            for name, buffer in ordinary.model.named_buffers():
                assert torch.equal(buffer, dict(skipped.model.named_buffers())[name])
            observation = {"observation.images.front": np.zeros((28, 28, 3), np.uint8),
                           "state": np.zeros(8, np.float32), "prompt": "test"}
            noise = np.ones((1, 4, 8), np.float32)
            expected = ordinary.predict(observation, noise)["actions"]
            result = skipped.predict(observation, noise)["actions"]
            np.testing.assert_array_equal(result, expected)
            assert result.flags.owndata
            skipped.load_seconds = 123
            skipped.reset("another episode")
            assert skipped.metadata() == skipped_metadata
            for policy in policies:
                policy.close()

            del weights["time_mlp_out.bias"]
            save_file(weights, str(tmp_path / "model.safetensors"))
            with pytest.raises(RuntimeError, match="time_mlp_out.bias"):
                lite.Pi05LitePolicy.from_pretrained(
                    tmp_path, tokenizer=str(tmp_path), device="cpu", cuda_graph=False,
                    attention_backend="eager", profile="bf16",
                    config_overrides={"lite_skip_weight_init": True},
                )
    finally:
        torch.set_num_threads(previous_threads)
