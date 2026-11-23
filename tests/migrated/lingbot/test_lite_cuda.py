"""Deferred real-checkpoint contracts. Root runs one mode per fresh GPU process.

No synthetic result is a model-quality claim. Enable explicitly with
LINGBOT_LITE_RUN_CUDA=1, LINGBOT_LITE_CONFIG and LINGBOT_LITE_FIXTURE.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vvla_lingbot.lite import factory


pytestmark = pytest.mark.skipif(os.environ.get("LINGBOT_LITE_RUN_CUDA") != "1",
                                reason="Explicit root-controlled serial CUDA queue only")


def load_fixture(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].item() if archive[key].ndim == 0 else archive[key].copy()
                for key in archive.files}


def bytes_of(raw):
    return {key: value.tobytes() if isinstance(value, np.ndarray) else value
            for key, value in raw.items()}


def test_episode_generator_three_consecutive_bf16_draws_match_seed_once_global():
    """RNG-only contract: task ID seeds the stream; raw task is an instruction.

    This does not establish saved-episode replay or closed-loop task quality.
    No model is loaded by this test.
    """
    import torch
    condition_task, sim_seed = "place_cans_plasticbox", 7
    policy_seed = int.from_bytes(
        hashlib.sha256(f"lingbot-robotwin-v1/{condition_task}/{sim_seed}".encode()).digest()[:4],
        "big",
    )
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    shape = (1, 50, 55)
    with torch.random.fork_rng(devices=[device_index]):
        # Historical behavior: seed exactly once, then consume consecutive draws.
        torch.cuda.manual_seed(policy_seed)
        expected = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)]
        # Harness behavior: one external Generator, never reseeded inside the loop.
        episode_generator = torch.Generator(device=device).manual_seed(policy_seed)
        actual = [torch.randn(shape, device=device, dtype=torch.bfloat16,
                              generator=episode_generator) for _ in range(3)]
        expected_bytes = [value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                          for value in expected]
        actual_bytes = [value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        for value in actual]
        assert actual_bytes == expected_bytes
        assert len(set(actual_bytes)) == 3, "Episode noise must advance rather than repeat call zero"
        # The public infer(seed) protocol deliberately starts a fresh generator.
        fresh_generator = torch.Generator(device=device).manual_seed(policy_seed)
        fresh = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=fresh_generator)
        assert fresh.cpu().contiguous().view(torch.uint8).numpy().tobytes() == expected_bytes[0]


def test_real_b1_seed_noise_official_boundary_stream_reset_and_close(monkeypatch):
    import torch
    config = json.loads(Path(os.environ["LINGBOT_LITE_CONFIG"]).read_bytes())
    observation = load_fixture(os.environ["LINGBOT_LITE_FIXTURE"])
    raw_before = bytes_of(observation)
    adapter = factory(config)
    try:
        metadata = adapter.metadata()
        with torch.cuda.device(config.get("device", "cuda:0")):
            first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(first_stream):
                first = adapter.infer(observation, 200042)
            noise = adapter.initial_noise()
            # This is call-zero / per-call seed parity only. The separate episode
            # test above checks the continuous seed-once stream across three draws.
            device_index = torch.cuda.current_device()
            with torch.random.fork_rng(devices=[device_index]):
                torch.cuda.manual_seed(200042)
                legacy_noise = torch.randn((1, 50, 55), device="cuda", dtype=torch.bfloat16)
                assert legacy_noise.cpu().view(torch.uint16).numpy().tobytes() == noise.tobytes()
            first_bytes = first["actions"].tobytes()
            with torch.cuda.stream(second_stream):
                repeated = adapter.infer(observation, 200042)
                fixed = adapter.predict(observation, noise)
            assert repeated["actions"].tobytes() == first_bytes
            assert fixed["actions"].tobytes() == first_bytes
            assert adapter.initial_noise().tobytes() == noise.tobytes()
            assert first["cold"] and not repeated["cold"] and not fixed["cold"]
            assert first["actions"].dtype == np.float32 and first["actions"].shape == (50, 14)
            assert all(row["actions"].flags.owndata for row in (first, repeated, fixed))
            assert not np.shares_memory(first["actions"], fixed["actions"])

            # Untimed evidence uses the same production path. The explicit
            # noise call must neither redraw nor advance an episode Generator.
            episode_generator = torch.Generator(device=config.get("device", "cuda:0")).manual_seed(200042)
            explicit = torch.randn((1, 50, 55), device=config.get("device", "cuda:0"),
                                   dtype=torch.bfloat16, generator=episode_generator)
            generator_state = episode_generator.get_state().clone()
            global_state = torch.cuda.get_rng_state().clone()
            policy = adapter._runtime.policy
            feature = policy.vla.feature_transform
            slots = [(obj, key, key in vars(obj), vars(obj).get(key)) for obj, key in (
                (feature, "apply"), (feature, "unapply"),
                (policy, "_prepare_model_input"), (policy.vla, "sample_actions_batch"))]
            with torch.cuda.stream(first_stream):
                prepared = adapter.prepared_fixture(observation, explicit)
            with torch.cuda.stream(second_stream):
                prepared_again = adapter.prepared_fixture(observation, noise)
                after_fixture = adapter.predict(observation, noise)
            assert torch.equal(episode_generator.get_state(), generator_state)
            assert torch.equal(torch.cuda.get_rng_state(), global_state)
            assert prepared["actions"].tobytes() == prepared_again["actions"].tobytes() == first_bytes
            assert after_fixture["actions"].tobytes() == first_bytes and not after_fixture["cold"]
            assert prepared["metadata"]["hooks_restored"]
            assert prepared["metadata"]["production_prediction_executed"]
            assert prepared["arrays"]["noise_bf16_bits"].tobytes() == noise.tobytes()
            assert prepared["metadata"]["sampler_input_keys"] == [
                "images", "img_masks", "lang_tokens", "lang_masks", "state", "noise", "image_grid_thw"]
            assert prepared["arrays"]["state_bf16_bits"].shape == (1, policy.config.max_state_dim)
            for key in prepared["arrays"]:
                assert prepared["arrays"][key].tobytes() == prepared_again["arrays"][key].tobytes()
                assert prepared["arrays"][key].flags.owndata
                assert not np.shares_memory(prepared["arrays"][key], prepared_again["arrays"][key])
            for stage in ("apply", "prepared", "collated", "unapply"):
                for key in ("action_is_pad", "joint_mask", "state_joint_mask", "action_joint_mask"):
                    assert f"{stage}.{key}" in prepared["arrays"]
            assert prepared["arrays"]["unapply.state"].dtype == np.float32
            assert prepared["arrays"]["prepared.state_bf16_bits"].dtype == np.uint16
            for obj, key, had_slot, old_slot in slots:
                assert (key in vars(obj)) == had_slot
                if had_slot:
                    assert vars(obj)[key] is old_slot
            assert adapter.metadata() == metadata

            # Independent invocation of the original policy CPU preparation and
            # unapply around the same actual core, with owned initial BF16 noise.
            # This is a boundary/adapter parity check, not cross-framework parity.
            with adapter._exclusive(), torch.inference_mode(False), torch.no_grad():
                adapter._check_live()
                adapter._handoff()
                policy = adapter._runtime.policy
                bridge = policy.sample_actions_fn
                source_noise = torch.from_numpy(noise.copy()).view(torch.bfloat16).to(config.get("device", "cuda:0"))

                def original_sample(*args, **kwargs):
                    return adapter._core_sample(*args, noise=source_noise.clone(), **kwargs)

                try:
                    policy.sample_actions_fn = original_sample
                    original = np.array(policy.infer(dict(observation))["action"], dtype=np.float32, copy=True)
                    torch.cuda.synchronize()
                finally:
                    policy.sample_actions_fn = bridge
                assert original.tobytes() == first_bytes

            adapter.reset("new-episode")
            after_reset = adapter.infer(observation, 200042)
            assert after_reset["actions"].tobytes() == first_bytes
            changed = adapter.infer(observation, 200043)
            assert changed["actions"].tobytes() != first_bytes
            assert first["actions"].tobytes() == first_bytes
            assert bytes_of(observation) == raw_before
            assert adapter.metadata() == metadata
            if adapter._runtime.engine is not None:
                engine = adapter._runtime.engine
                assert engine.audit is None and engine.graph_diagnostics is None
                assert engine.recipe["experimental_tuning"]["variant"] == "combined"
                assert engine.last_call["graph_replay"] and not engine.last_call["graph_build"]
                engine.route_audit()  # Untimed real dynamic-MoE route ownership check.

            # Inject a reported drain failure without a GPU fault. close must not
            # drop a live owner or lease even when all earlier work happened to finish.
            runtime = adapter._runtime
            original_sync = torch.cuda.synchronize
            failed = [False]

            def fail_once(*args, **kwargs):
                if not failed[0]:
                    failed[0] = True
                    raise RuntimeError("injected close drain failure")
                return original_sync(*args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(torch.cuda, "synchronize", fail_once)
                with pytest.raises(RuntimeError, match="injected close"):
                    adapter.close()
                assert adapter._runtime is runtime and not adapter._closed
                adapter.close()
            assert adapter._closed
    finally:
        adapter.close()


def test_base_engine_graph_failure_retains_graph_before_outer_drain():
    """Tiny CUDA execution, actual Engine.__call__; no model allocation here."""
    import torch
    from vvla_lingbot.engine import Engine
    engine = Engine.__new__(Engine)
    engine._signature = engine._meta_key = engine._graph = None
    engine.recipe = {"graph_builds": 0}
    engine.audit = None
    device = "cuda"
    values = [torch.ones((1, 1), device=device, dtype=torch.bfloat16),
              torch.ones((1, 3), device=device, dtype=torch.bool),
              torch.ones((1, 2), device=device, dtype=torch.int64),
              torch.ones((1, 2), device=device, dtype=torch.bool),
              torch.ones((1, 55), device=device, dtype=torch.bfloat16),
              torch.ones((3, 3), device=device, dtype=torch.int64)]

    def prepare(*inputs):
        engine.static = [value.clone() for value in inputs]
        engine.noise = torch.zeros((1, 50, 55), device=device, dtype=torch.bfloat16)

    engine._prepare = prepare
    engine._execute = lambda: engine.noise + engine.static[4].unsqueeze(1)
    try:
        first = engine(*values[:5], image_grid_thw=values[5], noise=torch.zeros((1, 50, 55), device=device, dtype=torch.bfloat16))
        torch.cuda.synchronize()
        assert first.shape == (1, 50, 55) and engine._graph is not None
        saved_graph = engine._graph
        changed_mask = values[3].clone()
        changed_mask[0, 1] = False
        sync = torch.cuda.synchronize

        def failed_sync(*args, **kwargs):
            raise RuntimeError("injected rebuild drain failure")

        # The graph must still be owned when rebuild's first synchronization fails.
        from unittest.mock import patch
        with patch.object(torch.cuda, "synchronize", failed_sync):
            with pytest.raises(RuntimeError, match="rebuild drain"):
                engine(values[0], values[1], values[2], changed_mask, values[4], image_grid_thw=values[5])
        assert engine._graph is saved_graph
        sync()
    finally:
        # Failed synchronization intentionally does not clear/release the graph.
        # The successful final drain is the ownership boundary for this tiny test.
        torch.cuda.synchronize()
        if engine._graph is not None:
            engine._graph.reset()
            engine._graph = None
