"""Exact postprocess/cache semantics, with transfer evidence deferred to CUDA."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from vvla_repro_pi05.lite.action_norm_cache import ActionNormDeviceCache
from vvla_repro_pi05.models.pi05.processor_pi05 import NormStats
from test_lite_pi05 import _obs, _policy
from test_lite_skip_weight_init import probe


@pytest.mark.parametrize("value", [1, "true", None])
def test_policy_rejects_nonboolean_action_cache_before_model_allocation(value, monkeypatch, tmp_path):
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Nonboolean cache option reached model allocation")
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    with pytest.raises(ValueError, match="boolean"):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, config_overrides={"lite_action_norm_cache": value})


def test_policy_default_off_predict_parity_provenance_owned_outputs_and_close_order(monkeypatch):
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    from test_lite_pi05 import _Model

    core, options = _split_lite_overrides({})
    assert core == {} and options["action_norm_cache"] is False
    baseline = _policy()
    assert baseline._action_norm_cache is None
    assert baseline.metadata()["lite_optimizations"]["action_norm_cache"] is False
    assert "action_norm_cache_manifest" not in baseline.metadata()["lite_optimizations"]
    candidate = Pi05LitePolicy(_Model(), baseline.processor, device="cpu", cuda_graph=False,
                               action_norm_cache=True, provenance={"checkpoint_sha256": "a" * 64})
    cache = candidate._action_norm_cache
    metadata = candidate.metadata()
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["lite_optimizations"]["action_norm_cache"] is True
    assert metadata["lite_optimizations"]["action_norm_cache_manifest"] == cache.manifest()
    noise = np.ones((1, 2, 4), np.float32)
    first = candidate.predict(_obs(), noise)["actions"]
    np.testing.assert_array_equal(first, baseline.predict(_obs(), noise)["actions"])
    assert first.flags.owndata and first.dtype == np.float32
    saved = first.copy()
    baseline.processor._action_norm.lower.numpy()[0] += 2
    np.testing.assert_array_equal(candidate.predict(_obs(), noise * 2)["actions"],
                                  baseline.predict(_obs(), noise * 2)["actions"])
    np.testing.assert_array_equal(first, saved)
    assert cache.refreshes == 2 and candidate.metadata() == metadata
    events = []
    class Graph:
        def close(self):
            events.append("graph")
    candidate.graph_runner = Graph()
    original_close = cache.close
    def close_cache():
        assert events == ["graph"]
        events.append("cache")
        original_close()
    monkeypatch.setattr(cache, "close", close_cache)
    candidate.close()
    assert events == ["graph", "cache"] and cache._closed
    assert candidate._action_norm_cache is None
    candidate.close()
    with pytest.raises(RuntimeError, match="closed"):
        candidate.predict(_obs(), noise)
    baseline.close()


def test_real_loader_action_cache_recipe_changes_without_checkpoint_or_input_provenance_changes(probe, tmp_path, monkeypatch):
    from dataclasses import asdict
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from test_lite_skip_weight_init import _Tokenizer
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

    config = Pi05Config(chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8,
                        num_inference_steps=2, image_resolution=(28, 28), tokenizer_max_length=8,
                        input_features={"observation.state": {"shape": [8]},
                                        "observation.images.front": {"shape": [3, 28, 28]}},
                        output_features={"action": {"shape": [8]}},
                        norm_stats={name: {"mode": "mean_std", "mean": [0] * 8, "std": [1] * 8}
                                    for name in ("state", "action")})
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _Tokenizer())
    old_threads = torch.get_num_threads()
    policies = []
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"), probe.small_dimensions():
            reference = Pi05ForActionPrediction(config)
            save_file({name: parameter.detach().clone() for name, parameter in reference.named_parameters()},
                      str(tmp_path / "model.safetensors"))
            for enabled in (False, True):
                policies.append(pi05.Pi05LitePolicy.from_pretrained(
                    tmp_path, tokenizer=str(tmp_path), device="cpu", cuda_graph=False,
                    attention_backend="eager", profile="bf16",
                    config_overrides={"lite_action_norm_cache": enabled}))
            baseline, cached = policies
            before, after = baseline.metadata(), cached.metadata()
            assert before["recipe_sha256"] != after["recipe_sha256"]
            for field in ("checkpoint_sha256", "preprocessing_sha256", "resolved_config", "source_sha256"):
                assert before[field] == after[field]
            assert after["lite_optimizations"]["action_norm_cache_manifest"] == cached._action_norm_cache.manifest()
            observation = {"observation.images.front": np.zeros((28, 28, 3), np.uint8),
                           "state": np.zeros(8, np.float32), "prompt": "pick object"}
            noise = torch.randn(1, 4, 8).numpy()
            expected = baseline.predict(observation, noise)["actions"]
            actual = cached.predict(observation, noise)["actions"]
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.owndata and cached.metadata() == after
    finally:
        for policy in policies:
            policy.close()
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("mode", ["mean_std", "min_max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("relative", [False, True])
def test_exact_original_arithmetic_dtype_tail_relative_order_and_owned_boundary(mode, dtype, relative):
    processor = _policy().processor
    processor._action_norm = replace(processor._action_norm, mode=mode)
    processor.relative_actions.enabled = relative
    actions = torch.randn(1, 2, 4, dtype=dtype)
    before = actions.clone()
    stats = processor._action_norm
    cache = ActionNormDeviceCache(processor)
    manifest = cache.manifest()
    expected = processor.build_model_outputs(actions, _obs())
    actual = np.array(cache.build_model_outputs(actions, _obs()), copy=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.flags.owndata
    assert cache.refreshes == 1
    assert processor._action_norm is stats
    assert cache._stats.lower.data_ptr() != stats.lower.data_ptr()
    torch.testing.assert_close(actions, before, rtol=0, atol=0)
    owned = actual.copy()
    cache.build_model_outputs(actions * 2, _obs())
    assert cache.refreshes == 1 and cache.manifest() == manifest
    np.testing.assert_array_equal(actual, owned)
    cache.close()


def test_stats_inplace_data_numpy_alias_replacement_and_inference_tensor_refresh():
    processor = _policy().processor
    cache = ActionNormDeviceCache(processor)
    actions = torch.randn(1, 2, 4)
    snapshots = []
    def verify(refreshes):
        actual = cache.build_model_outputs(actions, _obs())
        np.testing.assert_array_equal(actual, processor.build_model_outputs(actions, _obs()))
        assert cache.refreshes == refreshes
        snapshots.append((cache._stats, cache._stats.lower.clone()))
    verify(1)
    processor._action_norm.lower.add_(0.5)
    verify(2)
    processor._action_norm.upper.data.mul_(0.5)
    verify(3)
    processor._action_norm.lower.numpy()[1] = -12
    verify(4)
    # Equivalent values under new tensor objects require no upload.
    processor._action_norm = NormStats("mean_std", processor._action_norm.lower.clone(), processor._action_norm.upper.clone())
    verify(4)
    with torch.inference_mode():
        processor._action_norm = NormStats("mean_std", torch.tensor([1., 2., 3.]), torch.tensor([2., 2., 2.]))
        verify(5)
        processor._action_norm.lower.add_(1)
        verify(6)
    for old_stats, owned in snapshots:
        torch.testing.assert_close(old_stats.lower, owned, rtol=0, atol=0)
    cache.close()


def test_target_dtype_mode_none_and_nonstandard_statistics_preserve_original_path():
    processor = _policy().processor
    cache = ActionNormDeviceCache(processor)
    actions = torch.randn(1, 2, 4)
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        value = actions.to(dtype)
        np.testing.assert_array_equal(cache.build_model_outputs(value, _obs()), processor.build_model_outputs(value, _obs()))
        assert cache._stats.lower.dtype == dtype
    assert cache.refreshes == 3
    processor._action_norm = replace(processor._action_norm, mode="min_max")
    np.testing.assert_array_equal(cache.build_model_outputs(actions, _obs()), processor.build_model_outputs(actions, _obs()))
    assert cache.refreshes == 4
    processor._action_norm = None
    np.testing.assert_array_equal(cache.build_model_outputs(actions, _obs()), processor.build_model_outputs(actions, _obs()))
    assert cache._stats is None and cache._key is None
    # A noncampaign dtype uses native conversion each time instead of an
    # implicit normalization contract change or a D2H source inspection.
    processor._action_norm = NormStats("mean_std", torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    np.testing.assert_array_equal(cache.build_model_outputs(actions, _obs()), processor.build_model_outputs(actions, _obs()))
    assert cache._stats is None and cache.refreshes == 4
    cache.close()


def test_input_images_tokens_masks_state_noise_and_processor_methods_are_unchanged():
    processor = _policy().processor
    raw = _obs()
    noise = torch.randn(1, 2, 4)
    noise_before, state_before = noise.clone(), raw["state"].copy()
    before = processor.build_model_inputs(raw)
    methods = processor.build_model_inputs.__func__, processor.build_model_outputs.__func__
    cache = ActionNormDeviceCache(processor)
    cache.build_model_outputs(noise, raw)
    after = processor.build_model_inputs(raw)
    for old_group, new_group in zip(before, after, strict=True):
        old_group = old_group if isinstance(old_group, list) else [old_group]
        new_group = new_group if isinstance(new_group, list) else [new_group]
        for old, new in zip(old_group, new_group, strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            if old.dtype.is_floating_point:
                torch.testing.assert_close(old.bfloat16(), new.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(noise, noise_before, rtol=0, atol=0)
    np.testing.assert_array_equal(raw["state"], state_before)
    cache.close()
    assert methods == (processor.build_model_inputs.__func__, processor.build_model_outputs.__func__)


def test_close_concurrency_foreign_processor_and_exception_recovery():
    processor = _policy().processor
    cache = ActionNormDeviceCache(processor)
    actions = torch.ones(1, 2, 4)
    cache._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="request"):
            cache.build_model_outputs(actions, _obs())
        with pytest.raises(RuntimeError, match="request"):
            cache.close()
    finally:
        cache._lock.release()
    processor.build_model_outputs = lambda *args: np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError, match="original"):
        ActionNormDeviceCache(processor)
    with pytest.raises(RuntimeError, match="changed"):
        cache.build_model_outputs(actions, _obs())
    del processor.build_model_outputs
    # A failed request must release the helper lock and remain retryable.
    cache.build_model_outputs(actions, _obs())
    cache.close()
    cache.close()
    assert cache._stats is None
    with pytest.raises(RuntimeError, match="closed"):
        cache.build_model_outputs(actions, _obs())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Action stats transfers deferred to GPU queue owner")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_exact_outputs_and_warm_stats_uploads_eliminated(dtype, record_property):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Uploads(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.stats_uploads = 0
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if function is torch.ops.aten._to_copy.default:
                target = kwargs.get("device")
                if (args[0].device.type == "cpu" and tuple(args[0].shape) == (3,)
                        and target is not None and torch.device(target).type == "cuda"):
                    self.stats_uploads += 1
            return function(*args, **kwargs)

    processor = _policy().processor
    processor.relative_actions.enabled = False
    cache = ActionNormDeviceCache(processor)
    actions = torch.randn(1, 2, 4, device="cuda", dtype=dtype)
    counts = []
    owned = None
    for index in range(3):
        if index == 2:
            processor._action_norm.lower.numpy()[0] += 0.5
        expected = processor.build_model_outputs(actions, _obs())
        with Uploads() as evidence:
            actual = np.array(cache.build_model_outputs(actions, _obs()), copy=True)
        counts.append(evidence.stats_uploads)
        np.testing.assert_array_equal(actual, expected)
        if index == 0:
            owned, snapshot = actual, actual.copy()
    assert counts == [2, 0, 2]
    np.testing.assert_array_equal(owned, snapshot)
    record_property("stats_uploads_cold_warm_mutated", json.dumps(counts))
    # CPU and another CUDA stream remain independent destination lifecycles.
    cpu_actions = actions.cpu()
    np.testing.assert_array_equal(cache.build_model_outputs(cpu_actions, _obs()), processor.build_model_outputs(cpu_actions, _obs()))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        np.testing.assert_array_equal(cache.build_model_outputs(actions, _obs()), processor.build_model_outputs(actions, _obs()))
    cache.close()
