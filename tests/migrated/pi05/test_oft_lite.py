"""Direct OFT boundary contracts using fake loaders/executors; no ML import.

These test dispatch, precision-boundary ownership and failure retirement, not
numerical equivalence of the actual CUDA model. Run in the root-owned queue.
"""
from contextlib import contextmanager, nullcontext
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def lite(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "workers/oft/src"))
    module = importlib.import_module("vvla_oft.lite")
    monkeypatch.setattr(module, "_PROCESS_ORIGINS", {})
    return module


def observation(value=1, key="a"):
    return {"full_image": np.full((256, 256, 3), value, np.uint8),
            "wrist_image": np.full((256, 256, 3), value + 1, np.uint8),
            "state": np.arange(8, dtype=np.float32) + value,
            "prompt": "Pick Block " + str(value), "norm_key": key}


@pytest.fixture
def backend(lite, monkeypatch, tmp_path):
    from vvla_runtime.execution.requests import RequestLedger

    events = []
    state = SimpleNamespace(inference=False, autocast=False, sync_failure=False,
                            runner_failure=False, collate_failure=False,
                            close_failure=False, construction_failure=None,
                            official_output=np.full((8, 7), 0.25, np.float64),
                            official_failure=False, prepared=[], collated=[], produced=[],
                            adapters=[], runners=[], load_count=0)

    @contextmanager
    def inference_mode(enabled=True):
        old = state.inference
        state.inference = enabled
        try:
            yield
        finally:
            state.inference = old

    def synchronize(device):
        events.append(("synchronize", device))
        if state.sync_failure:
            raise RuntimeError("injected synchronize failure")

    torch = SimpleNamespace(__version__="2.13.0+cu130", inference_mode=inference_mode,
        no_grad=nullcontext, is_autocast_enabled=lambda device=None: state.autocast,
        get_float32_matmul_precision=lambda: "highest", get_default_dtype=lambda: "torch.float32",
        are_deterministic_algorithms_enabled=lambda: False,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                             synchronize=synchronize),
        backends=SimpleNamespace(cuda=SimpleNamespace(
            matmul=SimpleNamespace(allow_tf32=False, allow_bf16_reduced_precision_reduction=True,
                                  allow_fp16_reduced_precision_reduction=True),
            preferred_blas_library=lambda: "_BlasBackend.Cublas", flash_sdp_enabled=lambda: True,
            mem_efficient_sdp_enabled=lambda: True, math_sdp_enabled=lambda: True,
            cudnn_sdp_enabled=lambda: True),
            cudnn=SimpleNamespace(allow_tf32=False, enabled=True, benchmark=False, deterministic=False)))
    model = SimpleNamespace(norm_stats={"a": {}, "b": {}},language_model=SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=32),model=SimpleNamespace(layers=[object() for _ in range(32)])))
    head, proprio, processor = object(), object(), object()

    def official(cfg, actual_model, actual_processor, obs, prompt, **kwargs):
        assert state.inference is True
        assert actual_model is model and actual_processor is processor
        assert kwargs == {"action_head": head, "proprio_projector": proprio}
        assert vars(cfg) == dict(num_images_in_input=2, use_proprio=True,
                                  unnorm_key=obs["norm_key"], center_crop=True)
        events.append(("official", prompt, obs["norm_key"]))
        obs["state"][:] = -99
        obs["full_image"][:] = 0
        if state.official_failure:
            raise RuntimeError("injected official failure")
        return state.official_output

    utils = SimpleNamespace(get_vla_action=official)

    def load(source, checkpoint, *, device, resource_publisher):
        state.load_count += 1
        assert not state.inference
        events.append(("load", device))
        if state.construction_failure == "loader_unpublished":
            raise RuntimeError("loader constructor failed before publication")
        resource_publisher("model", model)
        if state.construction_failure == "loader_published":
            raise RuntimeError("loader constructor failed after publication")
        for name, value in (("head", head), ("proprio", proprio), ("processor", processor), ("utils", utils)):
            resource_publisher(name, value)
        return model, head, proprio, processor, utils, {"sha256": "a" * 64, "functions": ["official"]}

    class Adapter:
        def __init__(self, actual_model, actual_head, actual_proprio, *args, **kwargs):
            assert not state.inference
            assert (actual_model, actual_head, actual_proprio) == (model, head, proprio)
            self.args, self.kwargs = args, kwargs
            self.native_path = os.environ.get("VLA_VLLM_NATIVE_PATH")
            state.adapters.append(self)
            if state.construction_failure == "adapter":
                raise RuntimeError("adapter constructor failed")

        def eval(self):
            return self

        def prepare_inputs(self, tensors, *, device, validate):
            assert not state.inference and validate is False and device == "cuda:0"
            events.append(("native_prepare",))
            return tensors

    class Runner:
        def __init__(self, adapter, **kwargs):
            self.adapter, self.kwargs = adapter, kwargs
            self.forward = adapter
            self.state_contract = kwargs.get('state_contract')
            self.stats = {}
            state.runners.append(self)
            if state.construction_failure == "runner":
                raise RuntimeError("runner constructor failed")

        def close(self):
            events.append(("runner_close",))
            if state.close_failure:
                raise RuntimeError("injected runner close failure")

    class Service:
        def __init__(self, actual_model, actual_processor, actual_utils, runner,
                     *, capacity, batch_size, prepare_inputs):
            assert (actual_model, actual_processor, actual_utils) == (model, processor, utils)
            assert capacity == 128
            self.ledger = RequestLedger()
            self.prepare_inputs, self.batch_size = prepare_inputs, batch_size
            if state.construction_failure == "service":
                raise RuntimeError("service constructor failed")

        def prepare_row(self, obs, key):
            assert not state.inference
            state.prepared.append((obs, key))
            return (int(obs["full_image"][0, 0, 0]), key)

        def collate(self, rows, *, pad):
            state.collated.append((list(rows), pad))
            if state.collate_failure:
                raise RuntimeError("injected collate failure")
            tensors = {"rows": list(rows), "pixel_values": np.zeros((len(rows), 1))}
            if self.prepare_inputs:
                tensors = self.prepare_inputs(tensors)
            return tensors, len(rows)

        def predict_prepared(self, tickets, tensors, count, keys):
            self.ledger.validate_batch(tickets)
            if state.runner_failure:
                raise RuntimeError("injected model failure")
            values = [np.full((8, 7), value + (100 if key == "b" else 0), np.float64)
                      for value, key in tensors["rows"]]
            state.produced.extend(values)
            assert count == len(tickets) == len(keys)
            return dict(zip(tickets, values, strict=True))

    reference = SimpleNamespace(load_reference=load, CHECKPOINT_REVISION="c" * 40,
        REPOSITORY_REVISION="d" * 40, TRANSFORMERS_REVISION="e" * 40)
    deps = SimpleNamespace(torch=torch, transformers=SimpleNamespace(__version__="4.40.1"),
        reference=reference, official=Adapter, native=Adapter, runner=Runner, service=Service)
    monkeypatch.setattr(lite, "_dependencies", lambda: deps)
    monkeypatch.setattr(lite, "_pin_runtime_origins", lambda *args, **kwargs:
                        {"pinned": {"source": str(tmp_path)}, "observed_modules": {}})
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "dataset_statistics.json").write_text('{"a": {}, "b": {}}')

    owners = []
    def make(**kwargs):
        try:
            owner = lite.OFTLitePolicy(tmp_path, tmp_path, **kwargs)
        except BaseException as exc:
            if hasattr(exc, "oft_lite_owner"):
                owners.append(exc.oft_lite_owner)
            raise
        owners.append(owner)
        return owner

    yield SimpleNamespace(state=state, events=events, deps=deps, model=model, make=make)
    state.close_failure = state.sync_failure = False
    for owner in owners:
        owner.close()


@pytest.mark.parametrize("options", [
    {"execution": "unknown"}, {"batch_size": True}, {"batch_size": 0}, {"batch_size": 33},
    {"cuda_graph": "false"}, {"native_options": []}, {"native_options": {"backend": "sdpa"}},
    {"execution": "native", "native_options": {"mlp_precision": "fp8"}},
    {"execution": "native", "native_options": {"fuse_qkv": "false"}},
    {"execution": "native", "native_options": {"backend": "unknown"}},
    {"execution": "native", "batch_size": 2, "native_options": {"reference_compatible": True}},
    {"execution": "native", "cuda_graph": True, "native_options": {"reference_compatible": True}},
    {"execution": "native", "native_path": "/unused"}, {"norm_key": ""},
    {"native_state_comparator": True}, {"native_state_comparator": 1},
    {"execution": "native", "native_state_comparator": None},
    {"execution": "native", "native_state_comparator": "false"},
])
def test_invalid_configuration_before_dependencies(lite, monkeypatch, options):
    monkeypatch.setattr(lite, "_dependencies", lambda: pytest.fail("dependencies must not load"))
    with pytest.raises(ValueError):
        lite.OFTLitePolicy("unused", "unused", **options)


def test_native_state_default_off_never_loads_and_preserves_recipe(backend, lite, monkeypatch):
    monkeypatch.setattr(lite, '_native_state_dependencies',
                        lambda: pytest.fail('disabled comparator must not import/build'))
    for execution in ('official', 'native'):
        policy = backend.make(execution=execution)
        metadata = policy.metadata()
        assert 'native_state_comparator' not in metadata
        assert set(metadata['recipe']) == {'execution', 'batch_size', 'cuda_graph',
                                           'native_options', 'native_path', 'math_state'}
        if policy._runner is not None:
            assert 'state_contract' not in policy._runner.kwargs
        policy.predict(observation())
        policy.close()


@pytest.fixture
def native_state_double(backend, lite, monkeypatch):
    # Constructor/ownership integration only. Real C++ live-field contracts are
    # in test_oft_native_state.py; this is not numerical model evidence.
    manifest = {'version': 'oft-native-live-state-comparator-v1', 'binary_sha256': 'a'*64}
    helper = SimpleNamespace(manifest=lambda: copy.deepcopy(manifest))
    calls = []
    def load(*, verbose):
        assert verbose is False and not backend.state.inference
        assert len(backend.state.adapters) == 1 and not backend.state.runners
        calls.append('load')
        return helper
    def contract(adapter, *, native_comparator):
        assert adapter is backend.state.adapters[0] and native_comparator is helper
        calls.append('contract')
        return SimpleNamespace(model=adapter, native_comparator=helper)
    monkeypatch.setattr(lite, '_native_state_dependencies',
                        lambda: SimpleNamespace(load=load, contract=contract))
    return SimpleNamespace(helper=helper, manifest=manifest, calls=calls)


def test_native_state_cold_injection_manifest_and_no_request_build(backend, lite, native_state_double):
    state = native_state_double
    policy = backend.make(execution='native', native_state_comparator=True)
    assert state.calls == ['load', 'contract']
    assert policy._runner.state_contract is policy._native_state_contract
    assert policy._runner.state_contract.native_comparator is state.helper
    metadata = policy.metadata()
    assert metadata['native_state_comparator'] == state.manifest
    assert metadata['recipe']['native_state_comparator'] == {
        'version': state.manifest['version'], 'manifest_sha256': lite._digest(state.manifest)}
    assert metadata['recipe_sha256'] == lite._digest(metadata['recipe'])
    policy.predict(observation()); policy.predict(observation(2))
    assert state.calls == ['load', 'contract']
    assert policy.metadata() == metadata
    policy.close()
    assert backend.events[-2:] == [('runner_close',), ('synchronize', 'cuda:0')]
    assert policy._native_state_comparator is policy._native_state_contract is None


@pytest.mark.parametrize('change', ['contract', 'comparator', 'contract_model', 'runner_forward'])
def test_native_state_live_binding_cannot_change_after_identity(backend, native_state_double, change):
    policy = backend.make(execution='native', native_state_comparator=True)
    if change == 'contract':
        policy._runner.state_contract = object()
    elif change == 'comparator':
        policy._native_state_contract.native_comparator = None
    elif change == 'contract_model':
        policy._native_state_contract.model = object()
    else:
        policy._runner.forward = object()
    with pytest.raises(RuntimeError, match='binding changed'):
        policy.predict(observation())
    assert not backend.state.prepared


def test_native_state_failed_close_retains_helper_and_contract(backend, native_state_double):
    policy = backend.make(execution='native', native_state_comparator=True)
    contract = policy._native_state_contract
    backend.state.close_failure = True
    with pytest.raises(RuntimeError, match='runner close'):
        policy.close()
    assert policy._native_state_comparator is native_state_double.helper
    assert policy._native_state_contract is contract and policy._model is backend.model
    backend.state.close_failure = False
    policy.close()
    assert policy._native_state_contract is policy._native_state_comparator is None


@pytest.mark.parametrize('drain_failure', [False, True])
def test_native_state_build_failure_never_silently_falls_back(backend, lite, monkeypatch, drain_failure):
    def fail(**kwargs): raise RuntimeError('native build rejected')
    monkeypatch.setattr(lite, '_native_state_dependencies', lambda: SimpleNamespace(load=fail))
    backend.state.sync_failure = drain_failure
    with pytest.raises(RuntimeError, match='native build rejected') as error:
        backend.make(execution='native', native_state_comparator=True)
    assert not backend.state.runners and not backend.state.prepared
    if drain_failure:
        assert error.value.oft_lite_owner._model is backend.model
        assert error.value.oft_lite_owner._lease_held
        backend.state.sync_failure = False
        error.value.oft_lite_owner.close()
    policy = backend.make(execution='native')
    policy.close()


def test_source_inventory_binds_native_cpp_bytes(lite):
    import hashlib
    import vvla_runtime
    root = Path(vvla_runtime.__file__).parent
    path = 'execution/csrc/native_state.cpp'
    assert lite._source_manifest()['vvla_runtime/'+path] == hashlib.sha256((root/path).read_bytes()).hexdigest()


@pytest.mark.parametrize("runtime", ["torch", "transformers", "cuda", "tf32", "autocast"])
def test_environment_mismatch_before_loader(backend, runtime):
    deps = backend.deps
    if runtime == "torch":
        deps.torch.__version__ = "2.10.0"
    elif runtime == "transformers":
        deps.transformers.__version__ = "5.19.0"
    elif runtime == "cuda":
        deps.torch.cuda.is_available = lambda: False
    elif runtime == "tf32":
        deps.torch.backends.cuda.matmul.allow_tf32 = True
    else:
        backend.state.autocast = True
    with pytest.raises(RuntimeError):
        backend.make()
    assert backend.state.load_count == 0


def test_selected_visible_device_is_forwarded_to_loader(backend):
    # Replaces the old rejection of every device except cuda:0.
    backend.deps.torch.cuda.current_device = lambda: 1
    policy=backend.make()
    assert policy._device=='cuda:1'
    policy.close()


def test_official_b1_preserves_input_and_checkpoint_gripper_owned_output(backend):
    policy = backend.make(norm_key="a")
    obs = observation()
    obs.pop("norm_key")
    original = copy.deepcopy(obs)
    value = policy.predict(obs)
    assert value.shape == (8, 7) and value.dtype == np.float32 and value.flags.owndata
    assert value.flags.c_contiguous and np.all(value == np.float32(0.25))
    for name in ("state", "full_image", "wrist_image"):
        np.testing.assert_array_equal(obs[name], original[name])
    assert not backend.state.adapters and not backend.state.prepared
    backend.state.official_output[:] = 0.75
    again = policy.predict(obs)
    assert np.all(value == 0.25) and np.all(again == 0.75)
    assert not np.shares_memory(value, again)
    metadata = policy.metadata()
    assert metadata["action_space"] == "physical_oft_checkpoint"
    assert metadata["gripper"]["transform_applied"] is False
    assert metadata["capacity"] is None and metadata["weights_content_verified"] is False
    policy.close()
    assert backend.events[-1] == ("synchronize", "cuda:0")


@pytest.mark.parametrize("execution", ["official", "native"])
def test_batch_order_mixed_norm_keys_tail_rows_and_ownership(backend, execution):
    policy = backend.make(execution=execution, batch_size=4)
    obs = [observation(3, "b"), observation(7, "a")]
    value = policy.predict_batch(obs)
    assert value.shape == (2, 8, 7) and value.dtype == np.float32 and value.flags.owndata
    assert np.all(value[0] == 103) and np.all(value[1] == 7)
    assert backend.state.collated[-1] == ([(3, "b"), (7, "a")], True)
    assert not policy._service.ledger._active
    for array in backend.state.produced:
        array[:] = -999
    assert np.all(value[0] == 103) and np.all(value[1] == 7)
    reversed_value = policy.predict_batch(obs[::-1])
    np.testing.assert_array_equal(reversed_value, value[::-1])
    assert policy.predict(obs[0]).flags.owndata
    if execution == "official":
        assert backend.state.adapters[0].kwargs == {"execution": "official_batch"}
    else:
        recipe = backend.state.adapters[0].args[0]
        assert recipe.attention == "sdpa" and recipe.mlp_precision == "bf16"
        assert not recipe.fuse_qkv and not recipe.fuse_silu_mul
    policy.close()
    assert backend.events[-2:] == [("runner_close",), ("synchronize", "cuda:0")]
    assert policy._runner is None and policy._model is None


def test_explicit_native_options_restored_environment_and_unpadded_anchor(backend, monkeypatch):
    monkeypatch.setenv("VLA_VLLM_NATIVE_PATH", "/previous")
    policy = backend.make(execution="native", cuda_graph=True, native_path="/chosen",
        native_options={"backend": "vllm_fa2", "fuse_rope": True, "fuse_silu_mul": True})
    adapter = backend.state.adapters[-1]
    assert adapter.native_path == "/chosen"
    assert os.environ["VLA_VLLM_NATIVE_PATH"] == "/previous"
    recipe = adapter.args[0]
    assert recipe.pack_sequence and recipe.precompute_positions and recipe.fuse_rope
    assert backend.state.runners[-1].kwargs["use_graph"] is True
    policy.close()
    anchor = backend.make(execution="native", native_options={"reference_compatible": True})
    anchor.predict(observation())
    assert backend.state.collated[-1][1] is False
    anchor.close()


def test_explicit_mlp_fp8_has_distinct_precision_and_actual_installation_manifest(lite, backend, monkeypatch):
    adapter_type = backend.deps.native
    original = adapter_type.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.quantization_manifest = [dict(name=f"llama.{i}.{name}", category="llama_mlp",
            precision="fp8", reference_weight_device="cpu")
            for i in range(32) for name in ("gate_proj", "up_proj", "down_proj")]
    monkeypatch.setattr(adapter_type, "__init__", initialize)
    policy = backend.make(execution="native", native_path="/chosen", cuda_graph=True,
        native_options={"backend": "vllm_fa2", "mlp_precision": "fp8",
                        "fuse_silu_mul": True, "share_mlp_quant": True})
    metadata = policy.metadata()
    assert metadata["precision"] == "mixed_bf16_fp8_mlp"
    assert metadata["recipe"]["native_options"]["mlp_precision"] == "fp8"
    quant = metadata["quantization"]
    assert len(quant["manifest"]) == 96 and quant["manifest_sha256"] == lite._digest(quant["manifest"])
    quant["manifest"][0]["precision"] = "bf16"
    assert policy.metadata()["quantization"]["manifest"][0]["precision"] == "fp8"
    policy.close()


def test_incomplete_mlp_fp8_installation_cannot_claim_quantized_metadata(backend, monkeypatch):
    adapter_type = backend.deps.native
    original = adapter_type.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.quantization_manifest = []
    monkeypatch.setattr(adapter_type, "__init__", initialize)
    with pytest.raises(RuntimeError, match="every declared Llama MLP"):
        backend.make(execution="native", native_path="/chosen",
            native_options={"backend": "vllm_fa2", "mlp_precision": "fp8"})


@pytest.mark.parametrize("change", [
    {"state": np.full(8, np.nan)}, {"state": np.zeros(7)}, {"state": ["x"] * 8},
    {"full_image": np.zeros((256, 256, 3), np.float32)},
    {"wrist_image": np.zeros((0, 256, 3), np.uint8)},
    {"wrist_extra": np.zeros((256, 256, 3), np.uint8)},
    {"prompt": " "}, {"norm_key": None}, {"norm_key": "missing"},
])
def test_all_rows_validated_before_any_preprocessing(backend, change):
    policy = backend.make(execution="native", batch_size=2)
    with pytest.raises(ValueError):
        policy.predict_batch([observation(), observation() | change])
    assert not backend.state.prepared and not policy._service.ledger._active
    policy.close()


@pytest.mark.parametrize("failure", ["collate_failure", "runner_failure"])
def test_failed_request_drains_and_retires_all_tickets_without_retry(backend, failure):
    policy = backend.make(execution="native", batch_size=2)
    setattr(backend.state, failure, True)
    with pytest.raises(RuntimeError, match="injected"):
        policy.predict_batch([observation(), observation(2)])
    assert not policy._service.ledger._active
    assert backend.events[-1] == ("synchronize", "cuda:0")
    setattr(backend.state, failure, False)
    assert policy.predict(observation()).shape == (8, 7)
    policy.close()


def test_failed_sync_poison_retains_owner_until_retryable_close(backend):
    policy = backend.make(execution="native")
    backend.state.runner_failure = backend.state.sync_failure = True
    with pytest.raises(RuntimeError, match="model failure") as failure:
        policy.predict(observation())
    assert "owner retained" in failure.value.__notes__[0]
    assert not policy._service.ledger._active
    with pytest.raises(RuntimeError, match="poisoned"):
        policy.predict(observation())
    with pytest.raises(RuntimeError, match="synchronize"):
        policy.close()
    assert policy._model is backend.model and policy._runner is not None
    backend.state.sync_failure = False
    policy.close()
    assert policy._model is None


def test_failed_runner_close_never_drops_resources_and_close_is_idempotent(backend):
    policy = backend.make(execution="native")
    backend.state.close_failure = True
    with pytest.raises(RuntimeError, match="close failure"):
        policy.close()
    assert policy._model is backend.model and policy._service is not None
    backend.state.close_failure = False
    policy.close()
    count = len(backend.events)
    policy.close()
    assert len(backend.events) == count
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset("later")


@pytest.mark.parametrize("stage", ["loader_unpublished", "loader_published", "adapter", "runner", "service", "metadata"])
def test_partial_construction_drains_before_dropping_components(backend, lite, monkeypatch, stage):
    backend.state.construction_failure = stage
    if stage == "metadata":
        def fail(*args):
            raise RuntimeError("metadata constructor failed")
        monkeypatch.setattr(lite.OFTLitePolicy, "_build_metadata", fail)
    with pytest.raises(RuntimeError, match="constructor failed"):
        backend.make(execution="native")
    assert backend.events[-1] == ("synchronize", "cuda:0")
    if stage in ("service", "metadata"):
        assert backend.events[-2] == ("runner_close",)


def test_constructor_failed_retirement_returns_retryable_owner(backend):
    backend.state.construction_failure = "service"
    backend.state.close_failure = True
    with pytest.raises(RuntimeError, match="service constructor") as failure:
        backend.make(execution="native")
    owner = failure.value.oft_lite_owner
    assert owner._model is backend.model and owner._runner is not None
    backend.state.close_failure = False
    owner.close()
    assert owner._model is None


def test_unknown_configured_norm_key_rejected_before_adapter_and_drained(backend):
    with pytest.raises(ValueError, match="normalization key"):
        backend.make(execution="native", norm_key="missing")
    assert not backend.state.adapters
    assert backend.events[-1] == ("synchronize", "cuda:0")


def test_official_exception_does_not_mutate_observation_and_explicit_retry_works(backend):
    policy = backend.make()
    obs = observation()
    original = copy.deepcopy(obs)
    backend.state.official_failure = True
    with pytest.raises(RuntimeError, match="official failure"):
        policy.predict(obs)
    np.testing.assert_array_equal(obs["state"], original["state"])
    np.testing.assert_array_equal(obs["full_image"], original["full_image"])
    assert backend.events[-1] == ("synchronize", "cuda:0")
    backend.state.official_failure = False
    assert policy.predict(obs).shape == (8, 7)
    policy.close()


@pytest.mark.parametrize("batch", [[], {}, [None], [observation(), observation()]])
def test_invalid_batch_rejected_before_preprocessing(backend, batch):
    policy = backend.make(execution="native", batch_size=1)
    with pytest.raises(ValueError):
        policy.predict_batch(batch)
    assert not backend.state.prepared
    policy.close()


def test_metadata_stable_deepcopy_reset_and_serialization(backend):
    policy = backend.make(execution="native")
    original = policy.metadata()
    json.dumps(original, allow_nan=False)
    tampered = policy.metadata()
    tampered["recipe"]["native_options"]["fuse_qkv"] = True
    tampered["preprocessing"]["functions"].append("fake")
    policy.predict(observation())
    policy.reset("next")
    assert policy.metadata() == original and policy._sequence == 0
    assert not policy._service.ledger._active
    policy.close()
    assert policy.metadata() == original


def test_outer_inference_mode_does_not_change_construction_or_preparation(backend):
    with backend.deps.torch.inference_mode():
        policy = backend.make(execution="native")
        policy.predict(observation())
        assert backend.state.inference is True
        policy.close()


def test_serialized_predict_reset_close_and_late_autocast(backend):
    policy = backend.make(execution="native")
    with policy._request():
        for call in (lambda: policy.predict(observation()), lambda: policy.reset("x"), policy.close):
            with pytest.raises(RuntimeError, match="serialized|active"):
                call()
    backend.state.autocast = True
    with pytest.raises(RuntimeError, match="autocast"):
        policy.predict(observation())
    assert not backend.state.prepared
    backend.state.autocast = False
    policy.close()


@pytest.mark.parametrize("value", [np.zeros((7, 7)), np.full((8, 7), np.inf)])
def test_invalid_official_output_is_rejected_without_gripper_repair(backend, value):
    policy = backend.make()
    backend.state.official_output = value
    with pytest.raises(RuntimeError, match="invalid action"):
        policy.predict(observation())
    policy.close()


def test_process_lease_lasts_until_successful_close(backend):
    first = backend.make(execution="native")
    with pytest.raises(RuntimeError, match="one live owner"):
        backend.make()
    assert backend.state.load_count == 1
    backend.state.close_failure = True
    with pytest.raises(RuntimeError, match="close failure"):
        first.close()
    with pytest.raises(RuntimeError, match="one live owner"):
        backend.make()
    backend.state.close_failure = False
    first.close()
    second = backend.make()
    assert backend.state.load_count == 2
    second.close()


@pytest.mark.parametrize("stage", ["loader_unpublished", "loader_published"])
def test_partial_loader_sync_failure_retains_lease_even_without_published_model(backend, stage):
    backend.state.construction_failure, backend.state.sync_failure = stage, True
    with pytest.raises(RuntimeError, match="loader constructor failed") as failure:
        backend.make()
    owner = failure.value.oft_lite_owner
    assert owner._lease_held
    assert (owner._model is not None) == (stage == "loader_published")
    with pytest.raises(RuntimeError, match="one live owner"):
        backend.make()
    backend.state.sync_failure = False
    owner.close()
    assert not owner._lease_held
    backend.state.construction_failure = None
    replacement = backend.make()
    replacement.close()


@pytest.mark.parametrize("setting", ["tf32", "cudnn_tf32", "float32", "bf16_reduction", "blas", "sdpa", "determinism", "default_dtype", "device"])
def test_actual_math_changes_rejected_before_preprocessing(backend, setting):
    policy = backend.make(execution="native")
    original = policy.metadata()
    torch = backend.deps.torch
    if setting == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif setting == "cudnn_tf32":
        torch.backends.cudnn.allow_tf32 = True
    elif setting == "float32":
        torch.get_float32_matmul_precision = lambda: "high"
    elif setting == "bf16_reduction":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    elif setting == "blas":
        torch.backends.cuda.preferred_blas_library = lambda: "_BlasBackend.Cublaslt"
    elif setting == "sdpa":
        torch.backends.cuda.flash_sdp_enabled = lambda: False
    elif setting == "determinism":
        torch.are_deterministic_algorithms_enabled = lambda: True
    elif setting == "default_dtype":
        torch.get_default_dtype = lambda: "torch.float64"
    else:
        torch.cuda.current_device = lambda: 1
    with pytest.raises(RuntimeError, match="math/device state changed"):
        policy.predict(observation())
    assert not backend.state.prepared and policy.diagnostics()["counters"]["calls"] == 0
    assert policy.metadata() == original
    policy.close()


def test_diagnostics_are_actual_cumulative_and_reset_does_not_fabricate_cold(backend):
    policy = backend.make(execution="native")
    assert policy.diagnostics()["counters"]["calls"] == 0
    policy.predict(observation())
    first = policy.diagnostics()
    assert first["last_call"]["first_call"] and first["last_call"]["cold"]
    assert first["counters"]["first_calls"] == 1
    policy.reset("next")
    policy.predict(observation())
    second = policy.diagnostics()
    assert not second["last_call"]["first_call"] and not second["last_call"]["cold"]
    assert second["counters"]["calls"] == 2 and second["counters"]["cold_calls"] == 1
    backend.state.runners[0].stats.update(captures=2, replays=3, capacity_fallbacks=1)
    counters = policy.diagnostics()["counters"]
    assert (counters["captures"], counters["replays"], counters["capacity_fallbacks"]) == (2, 3, 1)
    policy.close()
    assert policy.diagnostics()["counters"] == counters
    json.dumps(policy.diagnostics(), allow_nan=False)


def test_prepared_fixture_executes_same_predict_and_updates_diagnostics(backend, monkeypatch):
    module = importlib.import_module("vvla_oft.lite_prepared")
    events = []
    class Capture:
        def __init__(self, policy, key):
            self.hooks = []
            events.append(("constructed", key))
        def __enter__(self):
            events.append(("enter",))
            return self
        def native(self, tensors):
            events.append(("actual_tensors", copy.deepcopy(tensors)))
        def __exit__(self, *args):
            events.append(("restore",))
        def finish(self, actions):
            events.append(("finish",))
            return {"actions": actions.copy(), "metadata": {"production_prediction_executed": True}}
    monkeypatch.setattr(module, "PreparedCapture", Capture)
    policy = backend.make(execution="native")
    value = policy.prepared_fixture(observation(3, "b"))
    assert np.all(value["actions"] == 103)
    assert events == [("constructed", "b"), ("enter",), ("actual_tensors", {"rows": [(3, "b")], "pixel_values": np.zeros((1, 1))}),
                      ("restore",), ("finish",)]
    counters = policy.diagnostics()["counters"]
    assert counters["calls"] == counters["successes"] == counters["prepared_fixtures"] == 1
    assert policy.diagnostics()["last_call"]["prepared_fixture"]
    policy.predict(observation())
    assert not policy.diagnostics()["last_call"]["cold"]
    policy.close()


def test_later_graph_capture_and_capacity_fallback_deltas_are_not_hidden(backend, monkeypatch):
    policy = backend.make(execution="native", cuda_graph=True)
    policy.predict(observation())
    original = policy._service.predict_prepared
    def captured(*args):
        policy._runner.stats["captures"] = 1
        return original(*args)
    monkeypatch.setattr(policy._service, "predict_prepared", captured)
    policy.predict(observation(2))
    assert policy.diagnostics()["last_call"]["cold"]
    assert policy.diagnostics()["last_call"]["graph_delta"]["captures"] == 1
    def fallback(*args):
        policy._runner.stats["capacity_fallbacks"] = 1
        return original(*args)
    monkeypatch.setattr(policy._service, "predict_prepared", fallback)
    policy.predict(observation(3))
    assert policy.diagnostics()["last_call"]["graph_delta"]["capacity_fallbacks"] == 1
    policy.close()


@pytest.mark.parametrize("failure_at", ["before_publication", "model", "head", "proprio"])
def test_legacy_loader_optional_publisher_hands_off_before_gpu_transfer(lite, monkeypatch, tmp_path,
                                                                       failure_at):
    """Execute the real loader control flow with imports/allocations stubbed."""
    import sys
    import types
    reference = importlib.import_module("vvla_oft.reference")
    source = tmp_path / "upstream"
    fork_file = tmp_path / "transformers-openvla-oft/src/transformers/models/llama/modeling_llama.py"
    fork_file.parent.mkdir(parents=True)
    fork_file.write_text("# pinned test bytes")
    (tmp_path / "dataset_statistics.json").write_text('{"debug": {}}')
    for name in ("action_head", "proprio_projector"):
        (tmp_path / (name + "--one_checkpoint.pt")).write_bytes(b"test")
    published, events = [], []
    class Model:
        def __init__(self):
            self.language_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa", hidden_size=4),
                model=SimpleNamespace(layers=[SimpleNamespace(self_attn=type("LlamaSdpaAttention", (), {})())]))
            self.vision_backbone = SimpleNamespace(set_num_images_in_input=lambda n: None)
        def state_dict(self):
            return {}
        def to(self, device):
            events.append("model_to")
            if failure_at == "model":
                raise RuntimeError("injected model GPU transfer failure")
            return self
        def eval(self):
            return self
    model = Model()
    class Component:
        def __init__(self, name):
            self.name = name
        def to(self, **kwargs):
            events.append(self.name + "_to")
            if failure_at == self.name:
                raise RuntimeError("injected " + self.name + " GPU transfer failure")
            return self
        def eval(self):
            return self
    head, proprio = Component("head"), Component("proprio")
    class OpenVLA:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if failure_at == "before_publication":
                raise RuntimeError("injected internal allocation failure")
            return model, {}
    modules = {}
    for name in ("torch", "transformers", "transformers.models", "transformers.models.llama",
                 "prismatic", "prismatic.extern", "prismatic.extern.hf", "prismatic.models", "prismatic.vla"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules["torch"].bfloat16 = "bf16"
    modules["transformers"].AutoTokenizer = object()
    modules["transformers.models.llama"].modeling_llama = SimpleNamespace(__file__=str(fork_file))
    leaves = {
        "prismatic.extern.hf.configuration_prismatic": {"OpenVLAConfig": SimpleNamespace(from_pretrained=lambda path: SimpleNamespace())},
        "prismatic.extern.hf.modeling_prismatic": {"OpenVLAForActionPrediction": OpenVLA},
        "prismatic.extern.hf.processing_prismatic": {"PrismaticProcessor": object, "PrismaticImageProcessor": object},
        "prismatic.models.action_heads": {"L1RegressionActionHead": lambda **kwargs: head},
        "prismatic.models.projectors": {"ProprioProjector": lambda **kwargs: proprio},
        "prismatic.vla.constants": {"NUM_ACTIONS_CHUNK": 8, "ACTION_DIM": 7, "PROPRIO_DIM": 8},
    }
    for name, attributes in leaves.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(reference, "configure_upstream", lambda root: None)
    monkeypatch.setattr(reference.subprocess, "check_output", lambda *args, **kwargs: reference.TRANSFORMERS_REVISION)
    with pytest.raises(RuntimeError, match="injected"):
        reference.load_reference(source, tmp_path, resource_publisher=lambda name, value: published.append((name, value)))
    count = {"before_publication": 0, "model": 1, "head": 2, "proprio": 3}[failure_at]
    assert published == [("model", model), ("head", head), ("proprio", proprio)][:count]
    assert events == ["model_to", "head_to", "proprio_to"][:count]
    # The old call signature still reaches exactly the same transfer boundary.
    with pytest.raises(RuntimeError, match="injected"):
        reference.load_reference(source, tmp_path)


@pytest.fixture
def origins(lite, monkeypatch, tmp_path):
    source, native = tmp_path / "openvla-oft", tmp_path / "native"
    fork = tmp_path / "transformers-openvla-oft/src/transformers/models/llama/modeling_llama.py"
    installed = tmp_path / "installed/transformers/models/llama/modeling_llama.py"
    for path in (fork, installed):
        path.parent.mkdir(parents=True)
        path.write_text("# identical pinned fork bytes\n")
    flash = lambda: None
    modules = {
        "prismatic": SimpleNamespace(__path__=[str(source / "prismatic")]),
        "prismatic.extern.hf.modeling_prismatic": SimpleNamespace(__file__=str(source / "prismatic/extern/hf/modeling_prismatic.py")),
        "transformers.models.llama.modeling_llama": SimpleNamespace(__file__=str(installed)),
    }
    def add_native():
        modules.update({
            "vllm": SimpleNamespace(__file__=str(native / "vllm/__init__.py"), __version__="0.32.0"),
            "vllm._C_stable_libtorch": SimpleNamespace(__file__=str(native / "vllm/_C_stable_libtorch.so")),
            "vllm.vllm_flash_attn": SimpleNamespace(__file__=str(native / "vllm/vllm_flash_attn/__init__.py"),
                                                  flash_attn_varlen_func=flash),
            "vvla_runtime.execution.native": SimpleNamespace(_loaded=True, _flash=flash),
        })
    monkeypatch.setattr(lite, "_origin_modules", lambda: modules)
    return SimpleNamespace(source=source, native=native, fork=fork, installed=installed,
                           modules=modules, add_native=add_native, root=tmp_path)


def test_same_source_origin_pin_is_reusable_but_different_source_requires_process(lite, origins):
    first = lite._pin_runtime_origins(origins.source, None, native_requested=False, require_loaded=True)
    second = lite._pin_runtime_origins(origins.source, None, native_requested=False, require_loaded=True)
    assert first == second
    assert first["pinned"]["llama_module"]["path"] == str(origins.installed)
    with pytest.raises(RuntimeError, match="already pinned for source"):
        lite._pin_runtime_origins(origins.root / "other-source", None, native_requested=False)


def test_preimported_namespace_outside_requested_root_is_rejected_before_pin(lite, origins):
    origins.modules["prismatic"].__path__ = [str(origins.root / "wrong/prismatic")]
    with pytest.raises(RuntimeError, match="different origin: prismatic"):
        lite._pin_runtime_origins(origins.source, None, native_requested=False)
    assert lite._PROCESS_ORIGINS == {}


@pytest.mark.parametrize("drift", ["content", "installed_path", "model_path"])
def test_later_imported_fork_or_model_origin_drift_is_rejected(lite, origins, drift):
    lite._pin_runtime_origins(origins.source, None, native_requested=False, require_loaded=True)
    if drift == "content":
        origins.installed.write_text("# different implementation\n")
    elif drift == "installed_path":
        other = origins.root / "other-installed-llama.py"
        other.write_bytes(origins.installed.read_bytes())
        origins.modules["transformers.models.llama.modeling_llama"].__file__ = str(other)
    else:
        origins.modules["prismatic.extern.hf.modeling_prismatic"].__file__ = str(origins.root / "other/model.py")
    with pytest.raises(RuntimeError, match="differs|origin"):
        lite._pin_runtime_origins(origins.source, None, native_requested=False, require_loaded=True)


@pytest.mark.parametrize("drift", ["path", "module_path", "missing_abi", "callable", "version", "loaded_flag"])
def test_native_loaded_state_cannot_silently_retarget_backend(lite, origins, drift):
    origins.add_native()
    first = lite._pin_runtime_origins(origins.source, origins.native,
        native_requested=True, require_loaded=True, require_native=True)
    assert first["pinned"]["native_root"] == str(origins.native)
    path = origins.native
    if drift == "path":
        path = origins.root / "other-native"
    elif drift == "module_path":
        origins.modules["vllm._C_stable_libtorch"].__file__ = str(origins.root / "wrong/_C.so")
    elif drift == "missing_abi":
        del origins.modules["vllm._C_stable_libtorch"]
    elif drift == "callable":
        origins.modules["vllm.vllm_flash_attn"].flash_attn_varlen_func = lambda: None
    elif drift == "version":
        origins.modules["vllm"].__version__ = "0.33.0"
    else:
        origins.modules["vvla_runtime.execution.native"]._loaded = False
    with pytest.raises(RuntimeError):
        lite._pin_runtime_origins(origins.source, path,
            native_requested=True, require_loaded=True, require_native=True)


def test_close_releases_owner_but_never_resets_process_origins(backend, lite):
    sentinel = {"source": "frozen-interpreter-source", "fork": "frozen-fork"}
    lite._PROCESS_ORIGINS.update(sentinel)
    policy = backend.make()
    policy.close()
    assert lite._PROCESS_ORIGINS == sentinel
