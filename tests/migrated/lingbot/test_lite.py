"""CPU/mocked adapter contracts. These do not load LingBot or execute CUDA."""
from contextlib import nullcontext
import copy
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vvla_lingbot import lite


def raw():
    return {"task": "place_cans_plasticbox", "observation.state": np.arange(14, dtype=np.float32),
            **{name: np.zeros((12, 16, 3), np.uint8) for name in lite.CAMERAS}}


class FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def record(self, stream):
        self.calls.append("record")


def bare_adapter(monkeypatch):
    adapter = lite.LingBotLitePolicy.__new__(lite.LingBotLitePolicy)
    calls = []
    stream = SimpleNamespace(wait_event=lambda event: calls.append("wait"))
    cuda = SimpleNamespace(device=lambda device: nullcontext(), current_stream=lambda device: stream,
                           is_current_stream_capturing=lambda: False,
                           synchronize=lambda device: calls.append("drain"))
    torch = SimpleNamespace(cuda=cuda, inference_mode=lambda flag: nullcontext(), no_grad=nullcontext)
    counter_module = ModuleType("torch._dynamo.utils")
    counter_module.counters = {"stats": {"unique_graphs": 0}}
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", counter_module)
    adapter._lock = threading.Lock()
    adapter._closed = adapter._failed = False
    adapter._event_recorded = False
    adapter._event = FakeEvent(calls)
    adapter._active_request = adapter._last_noise = adapter._core_sample = None
    adapter._requests, adapter._episode_id = 0, None
    adapter._metadata = {"fixed": {"nested": 1}}
    adapter._check_live = lambda: calls.append("live")
    adapter._freeze_policy_configuration = lambda: calls.append("freeze")
    output = np.arange(700, dtype=np.float32).reshape(50, 14)

    def infer(observation):
        calls.append("infer")
        adapter._active_request["used"] = True
        return {"action": output}

    adapter._runtime = SimpleNamespace(torch=torch, device="cuda:0", engine=None,
                                       policy=SimpleNamespace(infer=infer),
                                       reset_robot=lambda: calls.append("reset"),
                                       close=lambda: calls.append("close"))
    return adapter, calls, output


def test_raw_contract_keeps_task_pixels_and_state_dtype():
    policy = lite.LingBotLitePolicy.__new__(lite.LingBotLitePolicy)
    for dtype in (np.float32, np.float64):
        observation = raw()
        observation["observation.state"] = observation["observation.state"].astype(dtype)
        before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
        policy._validate_raw(observation)
        assert observation["task"] == before["task"]
        for name in (*lite.CAMERAS, "observation.state"):
            assert observation[name].dtype == before[name].dtype
            assert observation[name].tobytes() == before[name].tobytes()
    for key in ("reset", "batch", "path_to_pi_model"):
        with pytest.raises(ValueError, match="only"):
            policy._validate_raw({**raw(), key: True})
    observation = raw()
    del observation[lite.CAMERAS[0]]
    with pytest.raises(ValueError):
        policy._validate_raw(observation)
    observation = raw()
    observation[lite.CAMERAS[0]] = observation[lite.CAMERAS[0]].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        policy._validate_raw(observation)


def test_full_call_returns_owned_cpu_actions_and_handoffs_every_replay(monkeypatch):
    adapter, calls, source = bare_adapter(monkeypatch)
    first = adapter.infer(raw(), 123)
    assert first["cold"] is True and first["actions"].flags.owndata
    assert first["actions"].dtype == np.float32 and first["actions"].shape == (50, 14)
    saved = first["actions"].copy()
    source.fill(-5)
    second = adapter.infer(raw(), 124)
    assert second["cold"] is False
    assert first["actions"].tobytes() == saved.tobytes()
    assert calls == ["live", "infer", "record", "live", "wait", "infer", "record"]
    assert not np.shares_memory(first["actions"], second["actions"])
    adapter.reset("episode-2")
    assert calls[-5:] == ["live", "wait", "drain", "reset", "freeze"]
    assert adapter._episode_id == "episode-2"
    metadata = adapter.metadata()
    metadata["fixed"]["nested"] = 99
    assert adapter.metadata()["fixed"]["nested"] == 1
    adapter.close()
    assert calls[-1] == "close"


@pytest.mark.parametrize("method", ["infer", "predict", "prepared_fixture", "reset", "close", "metadata"])
def test_no_operation_can_bypass_active_request_lock(monkeypatch, method):
    adapter, _, _ = bare_adapter(monkeypatch)
    adapter._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="one infer/reset/close"):
            getattr(adapter, method)(raw(), 1) if method in ("infer", "predict", "prepared_fixture") else getattr(adapter, method)()
    finally:
        adapter._lock.release()
        adapter.close()


def test_failure_keeps_inputs_and_owner_until_retryable_close(monkeypatch):
    adapter, calls, _ = bare_adapter(monkeypatch)
    runtime = adapter._runtime

    def fail(observation):
        adapter._active_request["owned_noise"] = object()
        raise RuntimeError("sampler failed")

    runtime.policy.infer = fail
    with pytest.raises(RuntimeError, match="sampler failed"):
        adapter.infer(raw(), 4)
    pending = adapter._active_request
    assert pending["owned_noise"] is not None and adapter._failed
    for operation in (lambda: adapter.infer(raw(), 5), lambda: adapter.reset("next")):
        with pytest.raises(RuntimeError, match="prior request failed"):
            operation()
    runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        adapter.close()
    assert adapter._runtime is runtime and adapter._active_request is pending and not adapter._closed
    runtime.close = lambda: calls.append("drain succeeded")
    adapter.close()
    assert adapter._closed and adapter._active_request is None and adapter._runtime is None
    adapter.close()


@pytest.mark.parametrize("seed", [-1, 2 ** 32, 1.0, True, "12"])
def test_seed_is_an_exact_uint32(seed):
    with pytest.raises(ValueError, match="uint32"):
        lite._uint32(seed)


def test_bf16_explicit_noise_is_owned_and_seeded_protocol_keeps_initial_bytes():
    # CPU torch only: exercise exact dtype/noise ownership without a model/CUDA.
    import torch
    adapter = lite.LingBotLitePolicy.__new__(lite.LingBotLitePolicy)
    adapter._runtime = SimpleNamespace(torch=torch, policy=SimpleNamespace(config=SimpleNamespace(max_state_dim=55)))
    calls = []

    def core(*args, noise, **kwargs):
        calls.append(noise.clone())
        noise.add_(torch.tensor(1, dtype=torch.bfloat16))
        return noise

    adapter._core_sample = core
    state = torch.zeros((1, 55), dtype=torch.bfloat16)
    caller = torch.randn(lite.NOISE_SHAPE, dtype=torch.bfloat16)
    before = caller.view(torch.uint8).clone()
    adapter._active_request = {"seed": None, "noise": caller, "used": False}
    adapter._sample(None, None, None, None, state)
    assert torch.equal(before, caller.view(torch.uint8))
    assert torch.equal(adapter._last_noise.view(torch.uint8), before)
    assert adapter._last_noise.data_ptr() != caller.data_ptr()
    for _ in range(2):
        adapter._active_request = {"seed": 42, "noise": None, "used": False}
        adapter._sample(None, None, None, None, state)
    assert torch.equal(calls[-1].view(torch.uint8), calls[-2].view(torch.uint8))
    with pytest.raises(RuntimeError, match="Unexpected sampler"):
        adapter._sample(None, None, None, None, state)
    with pytest.raises(ValueError, match="never silently rounded"):
        adapter._explicit_noise(torch.zeros(lite.NOISE_SHAPE, dtype=torch.float32))
    bits = caller.view(torch.uint16).numpy().copy()
    restored = adapter._explicit_noise(bits)
    assert torch.equal(restored.view(torch.uint8), before)
    restored.add_(1)
    assert bits.tobytes() == before.numpy().tobytes()


def test_live_check_rejects_math_autocast_and_object_configuration_changes(monkeypatch):
    adapter = lite.LingBotLitePolicy.__new__(lite.LingBotLitePolicy)
    torch = SimpleNamespace(is_autocast_enabled=lambda device: False)
    feature = object()
    adapter._runtime = SimpleNamespace(torch=torch, policy=SimpleNamespace(vla=SimpleNamespace(feature_transform=feature)))
    adapter._math = {"tf32": False, "blas": "Cublas"}
    monkeypatch.setattr(lite, "math_settings", lambda t: dict(adapter._math))
    obj = object()
    adapter._policy_objects, adapter._policy_values = (obj,), '{"value":false}'
    adapter._preprocessing_state = ("exact cpu values",)
    monkeypatch.setattr(lite, "preprocessing_state", lambda feature: adapter._preprocessing_state)
    adapter._policy_configuration = lambda: ((obj,), '{"value":false}')
    checked = []
    adapter._official_guard = SimpleNamespace(check=lambda: checked.append(True))
    adapter._check_live()
    assert checked == [True]
    monkeypatch.setattr(lite, "math_settings", lambda t: {"tf32": True, "blas": "Cublas"})
    with pytest.raises(RuntimeError, match="Numerical dispatch"):
        adapter._check_live()
    monkeypatch.setattr(lite, "math_settings", lambda t: dict(adapter._math))
    torch.is_autocast_enabled = lambda device: device == "cuda"
    with pytest.raises(RuntimeError, match="autocast"):
        adapter._check_live()
    torch.is_autocast_enabled = lambda device: False
    adapter._policy_configuration = lambda: ((object(),), '{"value":false}')
    with pytest.raises(RuntimeError, match="outside locked reset"):
        adapter._check_live()
    adapter._policy_configuration = lambda: ((obj,), '{"value":0}')
    with pytest.raises(RuntimeError, match="outside locked reset"):
        adapter._check_live()
    adapter._policy_configuration = lambda: ((obj,), '{"value":false}')
    monkeypatch.setattr(lite, "preprocessing_state", lambda feature: ("changed cpu value",))
    with pytest.raises(RuntimeError, match="CPU values or mappings"):
        adapter._check_live()


def test_normalizer_bytes_signed_zero_and_mapping_mutations_are_live():
    class Normalizer:
        def normalize(self, value):
            return value

        def unnormalize(self, value):
            return value

    class Feature:
        def apply(self, value):
            return value

        def unapply(self, value):
            return value

    normalizer = Normalizer()
    normalizer.norm_stats = {"action.arm.position": {"mean": np.zeros((50, 12), np.float64)}}
    normalizer.norm_type = {"action.arm.position": "bounds_99_woclip"}
    feature = Feature()
    feature.normalizer = normalizer
    feature.key_mapping = {"arm": {"start": 0, "end": 6}}
    feature.key_reverse_mapping = {"action": [{"target_key": "arm", "end": 6}]}
    feature.action_subtract_state = {"arm": False}
    snapshot = lite.preprocessing_state(feature)
    assert snapshot == lite.preprocessing_state(feature)
    normalizer.norm_stats["action.arm.position"]["mean"][0, 0] = -0.0
    assert snapshot != lite.preprocessing_state(feature)  # raw bytes, not value equality
    normalizer.norm_stats["action.arm.position"]["mean"][0, 0] = 0.0
    assert snapshot == lite.preprocessing_state(feature)
    normalizer.norm_type["action.arm.position"] = "identity"
    assert snapshot != lite.preprocessing_state(feature)
    normalizer.norm_type["action.arm.position"] = "bounds_99_woclip"
    feature.key_mapping["arm"]["end"] = 7
    assert snapshot != lite.preprocessing_state(feature)
    feature.key_mapping["arm"]["end"] = 6
    feature.action_subtract_state["arm"] = 0  # even bool/int substitutions differ
    assert snapshot != lite.preprocessing_state(feature)


def test_official_guard_detects_in_place_parameters_registration_and_config():
    import torch
    from vvla_lingbot.lite_tuning.state_guard import FrozenStateGuard

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.config = SimpleNamespace(to_dict=lambda: {"steps": 10})

    for change in (lambda m: m.linear.weight.add_(1),
                   lambda m: setattr(m, "linear", torch.nn.Linear(3, 2)),
                   lambda m: setattr(m.config, "to_dict", lambda: {"steps": 11})):
        model = Model()
        guard = FrozenStateGuard(model)
        guard.check()
        with torch.no_grad():
            change(model)
        with pytest.raises(RuntimeError, match="changed"):
            guard.check()


def prepared_adapter(monkeypatch):
    """Small CPU boundary fixture, with the actual Lite _run/_sample implementation."""
    import torch
    adapter, calls, _ = bare_adapter(monkeypatch)
    proxy = adapter._runtime.torch
    for name in ("Tensor", "bfloat16", "uint16", "from_numpy", "isfinite"):
        setattr(proxy, name, getattr(torch, name))

    class Feature:
        def __init__(self):
            self.states, self.actions = ["arm"], ["arm"]
            self.key_mapping = {"arm": {"origin_keys": "observation.state"}}
            self.key_reverse_mapping = {"action": [{"target_key": "arm"}]}
            self.action_subtract_state, self.action_relative_type = {"arm": False}, {"arm": "none"}
            self.return_item_befor_padding = False
            self.feature_config = SimpleNamespace(joints=["arm"], joints_max_dim={"arm": 55})
            self.normalizer = SimpleNamespace(norm_type={"arm": "bounds"})

        def apply(self, observation, policy_eval=False):
            assert policy_eval is True
            state = torch.zeros(55, dtype=torch.float32)
            state[:14] = observation["observation.state"].to(torch.float32)
            self.applied = {
                "images": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2),
                "img_masks": torch.tensor([True, True, False]),
                "lang_tokens": torch.tensor([4, 3, 2], dtype=torch.int64),
                "lang_masks": torch.tensor([True, True, False]), "state": state,
                "actions": torch.zeros(50, 55), "action_is_pad": torch.zeros(50),
                "joint_mask": torch.ones(50, 55, dtype=torch.bool),
                "state_joint_mask": torch.arange(55) < 14,
                "action_joint_mask": torch.arange(55) < 14,
                "image_grid_thw": torch.tensor([[1, 2, 3]] * 3, dtype=torch.int64),
            }
            return self.applied

        def unapply(self, item):
            return {"action": item["actions"][:, :14] + item["state"][:14]}

    feature = Feature()

    class VLA:
        feature_transform = feature

        def sample_actions_batch(self, observation):
            # Preserve distinct actual collate and sampler dtypes/layouts.
            result = adapter._sample(
                observation["images"].to(torch.bfloat16), observation["img_masks"],
                observation["lang_tokens"], observation["lang_masks"], observation["state"],
                image_grid_thw=observation["image_grid_thw"],
            )
            observation["state"] = observation["state"].to(torch.float32)
            return result.to(torch.float32)

    class Policy:
        vla = VLA()
        config = SimpleNamespace(max_state_dim=55)

        def _prepare_model_input(self, observation):
            values = dict(observation)
            values["observation.state"] = torch.from_numpy(values["observation.state"])
            values = self.vla.feature_transform.apply(values, policy_eval=True)
            values["state"] = values["state"].to(torch.bfloat16)
            return values

        def infer(self, observation):
            transformed = self._prepare_model_input(observation)
            batch = {key: value.unsqueeze(0) for key, value in transformed.items()}
            actions = self.vla.sample_actions_batch(batch)
            single = dict(transformed, actions=actions[0], state=transformed["state"].to(torch.float32))
            return self.vla.feature_transform.unapply(single)

    policy = Policy()
    # Also exercise restoration of an existing instance-level bound method.
    policy._prepare_model_input = policy._prepare_model_input
    adapter._runtime.policy = policy
    adapter._metadata.update(source={"norm_stats_sha256": "a" * 64}, preprocessing_sha256="b" * 64)
    observed = []

    def core(images, img_masks, lang_tokens, lang_masks, state, *, noise, **kwargs):
        observed.append({"images": images.clone(), "state": state.clone(), "noise": noise.clone(),
                         "image_grid_thw": kwargs["image_grid_thw"].clone()})
        noise.add_(torch.tensor(0.5, dtype=torch.bfloat16))
        return noise

    adapter._core_sample = core
    return adapter, calls, feature, observed


def test_prepared_fixture_observes_actual_boundaries_restores_hooks_and_owns_bytes(monkeypatch):
    import torch
    adapter, calls, feature, observed = prepared_adapter(monkeypatch)
    observation = raw()
    observation["task"] = "Put the cans in the plastic box."
    noise = torch.arange(2750).reshape(lite.NOISE_SHAPE).to(torch.bfloat16)
    original_noise = noise.view(torch.uint16).numpy().copy()
    original_bound = adapter._runtime.policy._prepare_model_input
    baseline = adapter.predict(observation, noise)
    fixture = adapter.prepared_fixture(observation, noise)
    arrays, metadata = fixture["arrays"], fixture["metadata"]
    assert set(fixture) == {"arrays", "raw_arrays", "metadata", "actions"}
    assert metadata["production_prediction_executed"] and metadata["hooks_restored"]
    assert metadata["instruction"] == observation["task"]
    assert metadata["stage_order"] == ["apply", "prepared", "collated", "sampler", "sampler_output", "unapply", "physical"]
    assert arrays["noise_bf16_bits"].tobytes() == original_noise.tobytes()
    assert noise.view(torch.uint16).numpy().tobytes() == original_noise.tobytes()
    for key in ("images", "state", "noise"):
        assert arrays[key + "_bf16_bits"].tobytes() == observed[-1][key].view(torch.uint16).contiguous().numpy().tobytes()
    assert arrays["image_grid_thw"].tobytes() == observed[-1]["image_grid_thw"].numpy().tobytes()
    assert arrays["apply.state"].dtype == np.float32
    assert arrays["prepared.state_bf16_bits"].dtype == np.uint16
    assert arrays["unapply.state"].dtype == np.float32
    assert metadata["tensor_layouts"]["apply.images"]["stride"] == [12, 1, 4]
    assert metadata["tensor_layouts"]["noise_bf16_bits"]["dtype"] == "torch.bfloat16"
    for stage in ("apply", "prepared", "collated", "unapply"):
        for field in ("joint_mask", "action_is_pad", "state_joint_mask", "action_joint_mask"):
            assert f"{stage}.{field}" in arrays
    assert "joint_mask" not in metadata["sampler_input_keys"]
    assert fixture["actions"].tobytes() == baseline["actions"].tobytes()
    assert fixture["actions"].flags.owndata
    assert all(array.flags.owndata for array in (*arrays.values(), *fixture["raw_arrays"].values()))
    assert not {"apply", "unapply"} & set(vars(feature))
    assert "sample_actions_batch" not in vars(adapter._runtime.policy.vla)
    assert adapter._runtime.policy._prepare_model_input is original_bound
    saved = {key: value.tobytes() for key, value in arrays.items()}
    repeated = adapter.prepared_fixture(observation, original_noise)
    assert {key: value.tobytes() for key, value in repeated["arrays"].items()} == saved
    after = adapter.predict(observation, original_noise)
    assert not after["cold"] and after["actions"].tobytes() == baseline["actions"].tobytes()
    feature.applied["images"].fill_(-99)
    observation["observation.state"].fill(-99)
    assert {key: value.tobytes() for key, value in arrays.items()} == saved
    assert fixture["raw_arrays"]["observation.state"][0] == 0
    adapter.close()


def test_prepared_fixture_failure_restores_and_poison_retains_pending_transfer(monkeypatch):
    import torch
    adapter, calls, feature, _ = prepared_adapter(monkeypatch)
    source = object()
    capture = lite._PreparedFixture.value

    def fail(self, key, value):
        if key == "noise":
            self.pending.append(source)  # emulate a failing asynchronous copy/drain
            raise RuntimeError("injected fixture copy failure")
        return capture(self, key, value)

    monkeypatch.setattr(lite._PreparedFixture, "value", fail)
    with pytest.raises(RuntimeError, match="fixture copy") as caught:
        adapter.prepared_fixture(raw(), torch.zeros(lite.NOISE_SHAPE, dtype=torch.bfloat16))
    assert caught.value.lingbot_adapter is adapter and adapter._failed
    recorder = adapter._active_request["fixture"]
    assert recorder.pending == [source] and recorder.hooks == []
    assert not {"apply", "unapply"} & set(vars(feature))
    with pytest.raises(RuntimeError, match="prior request failed"):
        adapter.predict(raw(), torch.zeros(lite.NOISE_SHAPE, dtype=torch.bfloat16))
    adapter._runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        adapter.close()
    assert adapter._active_request["fixture"] is recorder and recorder.pending == [source]
    adapter._runtime.close = lambda: calls.append("close")
    adapter.close()
    assert adapter._active_request is None


def test_prepared_fixture_restore_failure_is_retryable_only_after_drain(monkeypatch):
    import torch
    adapter, calls, feature, _ = prepared_adapter(monkeypatch)
    real_restore = lite._PreparedFixture.restore

    def failed_restore(self):
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(lite._PreparedFixture, "restore", failed_restore)
    with pytest.raises(RuntimeError, match="restore failure"):
        adapter.prepared_fixture(raw(), torch.zeros(lite.NOISE_SHAPE, dtype=torch.bfloat16))
    recorder = adapter._active_request["fixture"]
    assert recorder.hooks and "apply" in vars(feature)
    monkeypatch.setattr(lite._PreparedFixture, "restore", real_restore)
    adapter._runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        adapter.close()
    assert recorder.hooks and "apply" in vars(feature)
    adapter._runtime.close = lambda: calls.append("drain succeeded")
    adapter.close()
    assert calls[-1] == "drain succeeded" and not recorder.hooks and "apply" not in vars(feature)


def test_prepared_fixture_partial_install_and_missing_masks_fail_without_fake_evidence(monkeypatch):
    import torch
    adapter, _, feature, _ = prepared_adapter(monkeypatch)
    noise = torch.zeros(lite.NOISE_SHAPE, dtype=torch.bfloat16)
    real_hook = lite._PreparedFixture._hook

    def fail_third(self, owner, name, **kwargs):
        if name == "sample_actions_batch":
            raise RuntimeError("injected partial install")
        return real_hook(self, owner, name, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(lite._PreparedFixture, "_hook", fail_third)
        with pytest.raises(RuntimeError, match="partial install"):
            adapter.prepared_fixture(raw(), noise)
    assert adapter._failed and adapter._active_request["fixture"].hooks == []
    assert not {"apply", "unapply"} & set(vars(feature))
    adapter.close()

    adapter, _, feature, _ = prepared_adapter(monkeypatch)
    original = feature.apply

    def missing_mask(*args, **kwargs):
        result = original(*args, **kwargs)
        del result["action_joint_mask"]
        return result

    feature.apply = missing_mask
    with pytest.raises(RuntimeError, match="Missing actual apply"):
        adapter.prepared_fixture(raw(), noise)
    assert feature.apply is missing_mask and adapter._failed
    assert adapter._active_request["fixture"].hooks == []
    adapter.close()
