"""Direct B1 LingBot raw CPU -> owned physical CPU actions, without a server.

Both execution modes retain the official FeatureTransform and BF16 sampler
boundaries. CUDA/ML dependencies are imported only during runtime construction.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import struct
import threading
import time

from .runtime import LiteRuntimeConfig, build_managed_runtime, digest

CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
           "observation.images.cam_right_wrist")
NOISE_SHAPE = (1, 50, 55)
_PREPARED_FIELDS = ("images", "img_masks", "state", "lang_tokens", "lang_masks",
                    "actions", "action_is_pad", "joint_mask", "state_joint_mask",
                    "action_joint_mask")


class _PreparedFixture:
    """Untimed observation of the real Python boundaries, never a processor.

    Only instance methods outside the compiled sampler are temporarily wrapped.
    The request owns this recorder (including incomplete transfers and hooks)
    until success, or until a failed owner's close has drained the device.
    """

    def __init__(self, adapter, observation):
        import numpy as np
        self.adapter = adapter
        self.torch = adapter._runtime.torch
        self.arrays = {}
        self.raw_arrays = {key: np.array(observation[key], copy=True, order="C")
                           for key in (*CAMERAS, "observation.state")}
        self.layouts, self.values, self.unavailable = {}, {}, {}
        self.stages, self.sampler_keys, self.hooks = [], [], []
        self.pending = []
        self.instruction = observation["task"]
        self.raw_layouts = {key: {"dtype": str(observation[key].dtype),
                                 "shape": list(observation[key].shape),
                                 "strides_bytes": list(observation[key].strides)}
                            for key in self.raw_arrays}

    def value(self, key, value):
        import numpy as np
        torch = self.torch
        if isinstance(value, torch.Tensor):
            name = key + ("_bf16_bits" if value.dtype == torch.bfloat16 else "")
            if name in self.arrays:
                raise RuntimeError(f"Repeated fixture tensor: {name}")
            # Own the source before a potentially failing transfer. The request
            # retains pending sources if CUDA synchronization/copy fails.
            self.pending.append(value)
            layout = {"dtype": str(value.dtype), "shape": list(value.shape),
                      "stride": list(value.stride()), "device": str(value.device),
                      "storage": "bf16_uint16_bits" if value.dtype == torch.bfloat16 else "native_numpy"}
            cpu = value.detach().cpu().contiguous()
            self.pending.append(cpu)
            if value.dtype == torch.bfloat16:
                cpu = cpu.view(torch.uint16)
            self.arrays[name] = cpu.numpy().copy(order="C")
            layout["array_dtype"] = str(self.arrays[name].dtype)
            self.layouts[name] = layout
            self.pending.clear()  # blocking CPU copy completed
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject or key in self.arrays:
                raise RuntimeError(f"Unsupported or repeated fixture array: {key}")
            self.arrays[key] = np.array(value, copy=True, order="C")
            self.layouts[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                 "strides_bytes": list(value.strides), "device": "cpu",
                                 "storage": "native_numpy", "array_dtype": str(value.dtype)}
        elif isinstance(value, dict):
            for name, item in value.items():
                if type(name) is not str:
                    raise TypeError("Fixture tensor dictionaries require string keys")
                self.value(f"{key}.{name}", item)
        elif isinstance(value, (tuple, list)):
            self.values[key + ".container"] = type(value).__name__
            for index, item in enumerate(value):
                self.value(f"{key}.{index}", item)
        elif value is None or type(value) in (str, bool, int, float):
            self.values[key] = value
        else:
            self.unavailable[key] = f"Not exported: {type(value).__module__}.{type(value).__qualname__}"

    def observe(self, stage, value):
        if stage in self.stages:
            raise RuntimeError(f"B1 fixture observed {stage} more than once")
        self.stages.append(stage)
        if stage in ("apply", "prepared", "collated", "unapply"):
            if not isinstance(value, dict) or not set(_PREPARED_FIELDS) <= set(value):
                raise RuntimeError(f"Missing actual {stage} FeatureTransform fields")
            if any(not isinstance(value[name], self.torch.Tensor) for name in _PREPARED_FIELDS):
                raise RuntimeError(f"Non-tensor actual {stage} FeatureTransform fields")
        self.value(stage, value)

    def sampler(self, images, img_masks, lang_tokens, lang_masks, state, noise, kwargs):
        if "sampler" in self.stages:
            raise RuntimeError("B1 fixture observed sampler more than once")
        self.stages.append("sampler")
        values = dict(images=images, img_masks=img_masks, lang_tokens=lang_tokens,
                      lang_masks=lang_masks, state=state, noise=noise, **kwargs)
        self.sampler_keys = list(values)
        for name in ("images", "state", "noise"):
            if not isinstance(values[name], self.torch.Tensor) or values[name].dtype != self.torch.bfloat16:
                raise RuntimeError(f"Actual sampler {name} must retain BF16")
        for name, value in values.items():
            self.value(name, value)

    def _hook(self, owner, name, *, before=None, after=None):
        original = getattr(owner, name)
        had_slot = name in vars(owner)
        old_slot = vars(owner).get(name)

        def observed(*args, **kwargs):
            if before:
                self.observe(before, args[0] if args else kwargs["observation" if before == "collated" else "item"])
            result = original(*args, **kwargs)
            if after:
                self.observe(after, result)
            return result

        # Retain the restoration record even when an unusual __setattr__ fails.
        self.hooks.append((owner, name, had_slot, old_slot, observed))
        setattr(owner, name, observed)

    def install(self):
        policy = self.adapter._runtime.policy
        feature = policy.vla.feature_transform
        self._hook(feature, "apply", after="apply")
        self._hook(policy, "_prepare_model_input", after="prepared")
        self._hook(policy.vla, "sample_actions_batch", before="collated")
        self._hook(feature, "unapply", before="unapply", after="physical")

    def restore(self):
        errors, retained = [], []
        for record in reversed(self.hooks):
            owner, name, had_slot, old_slot, observed = record
            try:
                if had_slot:
                    setattr(owner, name, old_slot)
                elif name in vars(owner):
                    delattr(owner, name)
                if (name in vars(owner)) != had_slot or (had_slot and vars(owner)[name] is not old_slot):
                    raise RuntimeError(f"Could not restore fixture hook {name}")
            except BaseException as error:
                retained.append(record)
                errors.append(error)
        self.hooks = list(reversed(retained))
        if errors:
            error = RuntimeError("Prepared fixture hook restoration failed; close must retry")
            for failure in errors:
                error.add_note(repr(failure))
            raise error

    def finish(self, result):
        expected = ["apply", "prepared", "collated", "sampler", "sampler_output", "unapply", "physical"]
        if self.hooks or self.pending or self.stages != expected or "physical.action" not in self.arrays:
            raise RuntimeError("Incomplete real prepared fixture or unrestored hooks")
        policy = self.adapter._runtime.policy
        feature = policy.vla.feature_transform
        # These are actual live CPU semantics, already protected by _check_live.
        semantics = {name: copy.deepcopy(getattr(feature, name)) for name in
                     ("states", "actions", "key_mapping", "key_reverse_mapping",
                      "action_subtract_state", "action_relative_type", "return_item_befor_padding")}
        semantics.update(joints=list(feature.feature_config.joints),
                         joints_max_dim=dict(feature.feature_config.joints_max_dim),
                         norm_type=copy.deepcopy(feature.normalizer.norm_type))
        semantics = json.loads(json.dumps(semantics, allow_nan=False))
        source = self.adapter._metadata["source"]
        metadata = {"schema": 1, "production_prediction_executed": True, "hooks_restored": True,
                    "instruction": self.instruction, "task_semantics": "frozen_unseen_instruction",
                    "sampler_input_keys": self.sampler_keys, "tensor_layouts": self.layouts,
                    "raw_layouts": self.raw_layouts, "values": self.values,
                    "stage_order": self.stages, "required_stage_fields": list(_PREPARED_FIELDS),
                    "norm_binding": {"norm_stats_sha256": source["norm_stats_sha256"],
                                     "preprocessing_sha256": self.adapter._metadata["preprocessing_sha256"]},
                    "unapply_binding": {"implementation": "FeatureTransform.unapply (actual invocation)",
                                        "state_boundary": "official BF16 roundtrip then FP32",
                                        "semantics": semantics},
                    "cold": result["cold"], "diagnostics": copy.deepcopy(result["diagnostics"]),
                    "not_provided": {"processor_internal_intermediates": "Only actual method boundaries are observed; no reconstructed resize/normalization/tokenizer internals",
                                     "compiled_sampler_internals": "Sampler call tensors/output only; no internal layer/graph tensor equivalence claim",
                                     **self.unavailable},
                    "noise_protocol": "explicit fixed BF16 noise; no RNG draw or episode-generator advancement",
                    "timing_scope": "untimed diagnostic; extra owned CPU copies are not benchmark latency"}
        return {"arrays": self.arrays, "raw_arrays": self.raw_arrays,
                "metadata": metadata, "actions": result["actions"]}


def math_settings(torch):
    """Read actual global numerical dispatch state without changing it."""
    cuda = torch.backends.cuda
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": cuda.matmul.allow_tf32,
            "matmul_allow_bf16_reduced_precision_reduction": cuda.matmul.allow_bf16_reduced_precision_reduction,
            "matmul_allow_fp16_reduced_precision_reduction": cuda.matmul.allow_fp16_reduced_precision_reduction,
            "preferred_blas_library": str(cuda.preferred_blas_library()),
            "flash_sdp": cuda.flash_sdp_enabled(), "math_sdp": cuda.math_sdp_enabled(),
            "mem_efficient_sdp": cuda.mem_efficient_sdp_enabled(),
            "cudnn_sdp": cuda.cudnn_sdp_enabled(),
            "math_sdp_allow_fp16_bf16_reduction": cuda.fp16_bf16_reduction_math_sdp_allowed(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(torch.get_default_device())}


def _uint32(seed):
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be a uint32 integer")
    return seed


def preprocessing_state(feature):
    """Exact owned CPU transformation state; no lossy NumPy repr/string hashes."""
    import numpy as np

    def freeze(value):
        kind = type(value)
        if kind in (type(None), bool, int, str, bytes):
            return kind, value
        if kind is float:
            return kind, struct.pack("!d", value)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise RuntimeError("Object arrays are unsupported in fixed preprocessing state")
            return (kind, id(value), value.dtype.str, value.shape, value.strides,
                    value.__array_interface__["data"][0], value.tobytes(order="C"))
        if isinstance(value, np.generic):
            return kind, value.dtype.str, value.tobytes()
        if isinstance(value, dict):
            return kind, id(value), tuple((freeze(key), freeze(item)) for key, item in value.items())
        if kind in (list, tuple):
            return kind, id(value), tuple(freeze(item) for item in value)
        if kind in (set, frozenset):
            return kind, id(value), frozenset(freeze(item) for item in value)
        if hasattr(value, "__dict__") and not callable(value):
            return kind, id(value), freeze(vars(value))
        raise RuntimeError(f"Unsupported live preprocessing state type: {kind.__qualname__}")

    excluded = {"model_config", "tokenizer", "processor", "normalizer"}
    feature_fields = tuple((name, freeze(value)) for name, value in vars(feature).items()
                           if name not in excluded)
    normalizer = feature.normalizer
    return (type(feature), type(feature).apply, type(feature).unapply, feature_fields,
            type(normalizer), type(normalizer).normalize, type(normalizer).unnormalize,
            freeze(vars(normalizer)))


class LingBotLitePolicy:
    def __init__(self, config):
        self._lock = threading.Lock()
        self._runtime = None
        self._failed = False
        self._closed = False
        self._event = None
        self._event_recorded = False
        self._active_request = None
        self._requests = 0
        self._episode_id = None
        self._last_noise = None
        self._core_sample = None
        self._metadata = None
        self._official_guard = None
        self.config = LiteRuntimeConfig.from_mapping(config.to_mapping() if isinstance(config, LiteRuntimeConfig) else config)
        try:
            self._runtime = build_managed_runtime(self.config)
            self._initialize()
        except BaseException as error:
            # build_managed_runtime already attaches its partially constructed
            # owner when it cannot drain. Later adapter failures use this owner.
            error.lingbot_adapter = self
            if self._runtime is None:
                self._runtime = getattr(error, "lingbot_runtime", None)
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"Lite cleanup remains pending; retry error.lingbot_adapter.close(): {cleanup!r}")
            raise

    def _initialize(self):
        runtime = self._runtime
        policy = runtime.policy
        torch = runtime.torch
        if policy.config.n_action_steps != 50 or policy.config.max_action_dim != 55 or policy.config.num_steps != 10:
            raise ValueError("This Lite entrypoint requires H50 / 10 steps / padded action width55")
        if not policy.use_bf16 or policy.use_fp32 or not policy.chunk_ret or policy.use_length != 50:
            raise ValueError("Unexpected official policy precision/chunk configuration")
        features = policy.vla.feature_transform.org_features
        if set(features["images"]) != set(CAMERAS) or set(features["states"]) != {"observation.state"} or set(features["actions"]) != {"action"}:
            raise ValueError("Robot features differ from the pinned three-camera [50,14] contract")
        if not policy.config.use_cache or policy.config.action_fp32:
            raise ValueError("Expected cached BF16 action expert")
        self._core_sample = policy.sample_actions_fn
        self._sample_bridge = self._sample
        policy.sample_actions_fn = self._sample_bridge
        if runtime.engine is None:
            from .lite_tuning.state_guard import FrozenStateGuard
            self._official_guard = FrozenStateGuard(policy.vla.model)
        self._math = math_settings(torch)
        if self._math["default_dtype"] != "torch.float32" or self._math["default_device"] != "cpu":
            raise ValueError("Official preprocessing requires the FP32 / CPU tensor defaults")
        self._freeze_policy_configuration()
        with torch.cuda.device(runtime.device):
            self._event = torch.cuda.Event()
        from .diffusion import sampling_clock
        clock_dtype = torch.bfloat16 if self.config.denoise_precision == 'bf16' else torch.float32
        dt, time_values = sampling_clock(policy.config.num_steps, clock_dtype)
        times = [float(t) for t in time_values]
        if runtime.engine is not None:
            recipe = copy.deepcopy(runtime.engine.recipe)
            recipe.pop("graph_builds", None)
            if self.config.mode=='framework-bf16':
                if recipe["experimental_tuning"]["enabled"] != ["guard", "moe_grid", "rope", "router_cache"]:
                    raise ValueError("The complete historical combined recipe is required")
            elif (recipe['mode']!=self.config.mode or 'experimental_tuning' in recipe
                    or len(recipe['quantized_linears'])!=3*len(runtime.engine.vlm.layers)):
                raise ValueError('Historical VLM precision requires all dense MLP projections and the original engine')
            if runtime.engine.audit is not None or getattr(runtime.engine,'graph_diagnostics',None) is not None:
                raise ValueError("Lite warm calls must not write recipe/request/graph log files")
        else:
            recipe = {"mode": "official-bf16", "use_compile": True,
                      "compile_calls": ["qwenvl_with_expert", "sample_actions"],
                      "torch_compile_options": "official defaults",
                      "precompute_grid_thw": bool(policy.config.precompute_grid_thw)}
        precision=('mixed_bf16_'+self.config.mode.removeprefix('framework-')+'_vlm'
                   if self.config.mode in ('framework-fp8','framework-nvfp4') else 'bf16')
        if self.config.denoise_precision != 'bf16':
            precision = 'bf16+' + self.config.denoise_precision
            if recipe['denoise_time_fp32'] != times:
                raise RuntimeError('Loaded graph engine and advertised clock differ')
        recipe.update(batch_size=1, precision=precision, horizon=50,
                      timestep_dtype=str(clock_dtype).removeprefix("torch."), timestep_update="repeated_addition",
                      timestep_dt=float(dt), timestep_values=times,
                      noise_dtype="bfloat16", noise_shape=list(NOISE_SHAPE),
                      noise_algorithm="torch.randn CUDA local Generator manual_seed uint32",
                      task_semantics="Original RoboTwin task string/key; unchanged chat-template or BOS/newline tokenization; no task-to-instruction lookup",
                      state_pre_unapply="BF16 roundtrip then FP32 (official)",
                      raw_to_action="official prepare/apply/collate/sample/unapply; owned np.float32 copy",
                      math_settings=dict(self._math))
        source = copy.deepcopy(runtime.sources)
        preprocessing = {k: source[k] for k in ("official_source_sha256", "training_config_sha256",
                         "robot_config_sha256", "norm_stats_sha256", "qwen_metadata_sha256")}
        self._metadata = {"schema": 1, "framework": "lingbot-lite",
                          "mode": self.config.mode, "precision": precision, "batch_size": 1,
                          "attention_backend": ("vllm_fa2" if runtime.engine is not None else
                                                getattr(runtime.policy.config, "attention_implementation", "upstream")),
                          "horizon": 50, "action_shape": [50, 14],
                          "action_space": "physical_robotwin", "noise_shape": list(NOISE_SHAPE),
                          "checkpoint_sha256": source["checkpoint_sha256"],
                          "preprocessing_sha256": digest(preprocessing),
                          "recipe_sha256": digest(recipe), "recipe": recipe,
                          "source": source, "runtime": dict(runtime.actual_runtime),
                          "quality_evidence": "none; direct adapter awaiting external validation"}

    def _policy_configuration(self):
        policy = self._runtime.policy
        feature = policy.vla.feature_transform
        identity = (policy, policy.vla, policy.config, policy.data_config, policy.processor,
                    feature, feature.normalizer, feature.model_config, feature.tokenizer,
                    feature.processor, policy.sample_actions_fn, self._runtime.engine)
        values = {"config": policy.config.to_dict(), "data": vars(policy.data_config),
                  "robot": policy.robot_config, "action_key": sorted(policy.action_key),
                  "use_bf16": policy.use_bf16, "use_fp32": policy.use_fp32,
                  "use_compile": policy.use_compile, "chunk_ret": policy.chunk_ret,
                  "use_length": policy.use_length}
        return identity, json.dumps(values, sort_keys=True, default=str, allow_nan=False)

    def _freeze_policy_configuration(self):
        self._policy_objects, self._policy_values = self._policy_configuration()
        self._preprocessing_state = preprocessing_state(self._runtime.policy.vla.feature_transform)

    def _check_live(self):
        torch = self._runtime.torch
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("External autocast is unsupported for this fixed BF16 recipe")
        if math_settings(torch) != self._math:
            raise RuntimeError("Numerical dispatch settings changed; construct a fresh runtime")
        objects, values = self._policy_configuration()
        if len(objects) != len(self._policy_objects) or any(a is not b for a, b in zip(objects, self._policy_objects)) or values != self._policy_values:
            raise RuntimeError("Policy/preprocessor/configuration changed outside locked reset")
        if preprocessing_state(self._runtime.policy.vla.feature_transform) != self._preprocessing_state:
            raise RuntimeError("FeatureTransform/normalizer CPU values or mappings changed")
        if self._official_guard is not None:
            self._official_guard.check()

    @contextmanager
    def _exclusive(self, *, closing=False):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("LingBot Lite permits only one infer/reset/close operation at a time")
        try:
            if not closing:
                if self._closed:
                    raise RuntimeError("LingBot Lite is closed")
                if self._failed:
                    raise RuntimeError("A prior request failed; close this owner before constructing another")
            yield
        finally:
            self._lock.release()

    def _handoff(self):
        torch = self._runtime.torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("External CUDA graph capture is unsupported")
        if self._event_recorded:
            torch.cuda.current_stream(self._runtime.device).wait_event(self._event)

    def _sample(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        # Called only from official sample_actions_batch while _exclusive is held.
        request = self._active_request
        if request is None or request["used"] or noise is not None:
            raise RuntimeError("Unexpected sampler invocation outside a single Lite request")
        request["used"] = True
        torch = self._runtime.torch
        if state.dtype != torch.bfloat16 or tuple(state.shape) != (1, self._runtime.policy.config.max_state_dim):
            raise ValueError("Official prepared state lost its B1 BF16/configured padded-width boundary")
        if request["seed"] is not None:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(request["seed"])
            owned_noise = torch.randn(NOISE_SHAPE, dtype=state.dtype, device=state.device,
                                      generator=generator)
        else:
            # Always own storage: official sample_actions updates x_t in place.
            owned_noise = request["noise"].to(device=state.device).clone()
        self._last_noise = owned_noise.detach().clone()
        self._runtime.policy._last_policy_noise = self._last_noise
        request["owned_noise"] = owned_noise
        recorder = request.get("fixture")
        if recorder is not None:
            recorder.sampler(images, img_masks, lang_tokens, lang_masks, state, owned_noise, kwargs)
        result = self._core_sample(images, img_masks, lang_tokens, lang_masks, state,
                                   noise=owned_noise, **kwargs)
        if recorder is not None:
            recorder.observe("sampler_output", result)
        return result

    def _validate_raw(self, observation):
        import numpy as np
        if type(observation) is not dict or set(observation) != {"task", "observation.state", *CAMERAS}:
            raise ValueError("raw observation must contain task, observation.state and the three RoboTwin cameras only")
        if type(observation["task"]) is not str or not observation["task"]:
            raise ValueError("task must be the original nonempty RoboTwin task key, not a rewritten instruction")
        state = observation["observation.state"]
        if not isinstance(state, np.ndarray) or state.dtype not in (np.dtype("float32"), np.dtype("float64")) or state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must be finite CPU NumPy FP32/FP64[14]; official preprocessing owns its casts")
        for name in CAMERAS:
            array = observation[name]
            if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
                raise ValueError(f"{name} must be a CPU uint8 HWC RGB image")

    def _explicit_noise(self, value):
        import numpy as np
        torch = self._runtime.torch
        if isinstance(value, np.ndarray):
            if value.dtype != np.uint16 or value.shape != NOISE_SHAPE:
                raise ValueError("NumPy explicit noise must contain BF16 raw bits as uint16[1,50,55]")
            value = torch.from_numpy(np.array(value, copy=True, order="C")).view(torch.bfloat16)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or tuple(value.shape) != NOISE_SHAPE:
            raise ValueError("Explicit noise must be BF16[1,50,55], never silently rounded FP32")
        # No GPU finite reduction added to the warm path. Caller-provided noise
        # is validated here, outside model submission but inside the full timer.
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Explicit noise must be finite")
        return value

    def _run(self, observation, *, seed=None, noise=None, prepared_fixture=False):
        with self._exclusive():
            self._validate_raw(observation)
            runtime = self._runtime
            torch = runtime.torch
            with torch.cuda.device(runtime.device), torch.inference_mode(False), torch.no_grad():
                request = {"seed": seed, "noise": None, "used": False}
                self._active_request = request
                begin = time.perf_counter()
                try:
                    self._check_live()
                    self._handoff()
                    if noise is not None:
                        request["noise"] = self._explicit_noise(noise)
                    recorder = None
                    if prepared_fixture:
                        recorder = _PreparedFixture(self, observation)
                        request["fixture"] = recorder
                    from torch._dynamo.utils import counters
                    before = counters["stats"]["unique_graphs"]
                    primary = None
                    try:
                        if recorder is not None:
                            recorder.install()
                        result = runtime.policy.infer(dict(observation))
                    except BaseException as error:
                        primary = error
                        raise
                    finally:
                        if recorder is not None:
                            try:
                                recorder.restore()
                            except BaseException as cleanup:
                                if primary is None:
                                    raise
                                primary.add_note(f"Prepared fixture hooks require close retry: {cleanup!r}")
                    if recorder is not None:
                        self._check_live()
                    import numpy as np
                    actions = np.array(result["action"], dtype=np.float32, order="C", copy=True)
                    if not request["used"] or actions.shape != (50, 14) or not np.isfinite(actions).all():
                        raise RuntimeError("Unexpected non-finite or incomplete physical action chunk")
                    delta = counters["stats"]["unique_graphs"] - before
                    diagnostics = dict(runtime.engine.last_call) if runtime.engine else {
                        "graph_build": False, "capture_ms": 0.0, "graph_replay": False}
                    cold = self._requests == 0 or bool(diagnostics["graph_build"]) or delta > 0
                    diagnostics.update(compile_graphs_delta=delta, first_forward=self._requests == 0,
                                       adapter_infer_ms=(time.perf_counter() - begin) * 1000,
                                       episode_id=self._episode_id)
                    result = {"actions": actions, "cold": cold, "diagnostics": diagnostics}
                    if recorder is not None:
                        result = recorder.finish(result)
                    self._event.record(torch.cuda.current_stream(runtime.device))
                    self._event_recorded = True
                    self._requests += 1
                    self._active_request = None
                    return result
                except BaseException as error:
                    # Keep explicit request/noise/model/graph ownership until a
                    # successful close drain. No more infer/reset GPU submissions.
                    self._failed = True
                    error.lingbot_adapter = self
                    raise

    def infer(self, raw_observation, seed):
        return self._run(raw_observation, seed=_uint32(seed))

    def predict(self, raw_observation, noise):
        if noise is None:
            raise ValueError("predict requires explicit BF16 noise")
        return self._run(raw_observation, noise=noise)

    def prepared_fixture(self, raw_observation, explicit_noise):
        """Untimed real predict path plus owned boundary tensors and actions.

        Fixed noise never draws from or advances an external episode Generator.
        This uses predict's production _run, including its lock, live checks,
        event handoff and failure poisoning; no processor is reimplemented.
        """
        if explicit_noise is None:
            raise ValueError("prepared_fixture requires explicit BF16 noise")
        return self._run(raw_observation, noise=explicit_noise, prepared_fixture=True)

    def initial_noise(self):
        """Untimed fixture helper: owned CPU uint16 bytes of the last BF16 draw."""
        with self._exclusive():
            if self._last_noise is None:
                raise RuntimeError("No successful noise draw is available")
            torch = self._runtime.torch
            with torch.cuda.device(self._runtime.device):
                self._handoff()
                return self._last_noise.detach().cpu().contiguous().view(torch.uint16).numpy().copy()

    def reset(self, episode_id=None):
        if episode_id is not None and type(episode_id) not in (str, int):
            raise TypeError("episode_id must be a string, integer or None")
        with self._exclusive():
            runtime = self._runtime
            try:
                with runtime.torch.cuda.device(runtime.device):
                    self._check_live()
                    self._handoff()
                    # Reset reconstructs FeatureTransform. Retire all earlier work
                    # before releasing old preprocessing/state references.
                    runtime.torch.cuda.synchronize(runtime.device)
                    runtime.reset_robot()
                    self._freeze_policy_configuration()
                self._episode_id = episode_id
            except BaseException as error:
                self._failed = True
                error.lingbot_adapter = self
                raise

    def metadata(self):
        with self._exclusive():
            return copy.deepcopy(self._metadata)

    def close(self):
        with self._exclusive(closing=True):
            if self._closed:
                return
            if self._runtime is not None:
                try:
                    self._runtime.close()
                except BaseException as error:
                    error.lingbot_adapter = self
                    raise
            # A failed transfer/restore retains its recorder and hooked objects
            # until the runtime drain succeeds. Retry restoration before release.
            if self._active_request is not None:
                recorder = self._active_request.get("fixture")
                if recorder is not None:
                    try:
                        recorder.restore()
                    except BaseException as error:
                        error.lingbot_adapter = self
                        raise
            # Every closure/input/output/event remains reachable until drain and
            # graph.reset both succeeded. No empty_cache/global Dynamo reset.
            self._active_request = self._last_noise = self._core_sample = None
            self._official_guard = self._sample_bridge = None
            self._policy_objects = None
            self._preprocessing_state = None
            self._event = self._runtime = None
            self._closed = True


def factory(config):
    return LingBotLitePolicy(config)
