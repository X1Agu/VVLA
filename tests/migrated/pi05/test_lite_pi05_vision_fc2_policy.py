"""CPU-only policy integration contracts; execute in the remote test queue.

The optional installers and graph runner are injected modules. These tests do
not import the CUDA fc2 helper, create CUDA tensors, compile a model, or assume
the helper's private implementation. Kernel/descriptor contracts live in the
separate vision-fc2 helper tests.
"""

from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vvla_repro_pi05.lite import pi05
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides


class _Model(torch.nn.Module):
    action_horizon = 50
    action_dim = 32

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("sentinel", torch.arange(4))
        self._graph_runners = []


def _model_and_processor():
    config = SimpleNamespace(chunk_size=50, num_inference_steps=10,
                             action_dim=7, attention_backend="torch_fa2")
    return _Model(), SimpleNamespace(config=config)


def _inventory(model):
    return {
        "parameters": tuple((name, id(value)) for name, value in model.named_parameters()),
        "buffers": tuple((name, id(value)) for name, value in model.named_buffers()),
        "modules": tuple((name, id(value)) for name, value in model.named_modules()),
        "state_keys": tuple(model.state_dict()),
    }


def _module(monkeypatch, suffix, **members):
    name = "vvla_repro_pi05.lite." + suffix
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def _forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU policy contract attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def _forbid_optional_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled/invalid fc2 options reached optional allocation")
    _module(monkeypatch, "batched_vision", install_batched_vision=forbidden)
    _module(monkeypatch, "vision_fc2_cublaslt", install_vision_fc2_cublaslt=forbidden)
    _module(monkeypatch, "graph", Pi05NativeGraphRunner=forbidden)
    return forbidden


def _forbid_model_load(monkeypatch, tmp_path):
    from safetensors import torch as safetensors_torch
    from transformers import AutoTokenizer
    from vvla_repro_pi05.models.pi05.config import Pi05Config

    forbidden = _forbid_optional_work(monkeypatch)
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    monkeypatch.setattr(safetensors_torch, "load_file", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))


class _InjectedFailure(RuntimeError):
    pass


class _Rig:
    """Observable dependency/retirement protocol, without foreign resources."""

    def __init__(self, monkeypatch, stage=None):
        self.stage = stage
        self.failure = _InjectedFailure("injected " + str(stage))
        self.events = []
        self.batched = []
        self.fc2 = []
        self.runners = []
        rig = self

        class BatchedHandle:
            def __init__(self, model):
                self.model, self.enabled = model, True

            def manifest(self):
                rig.events.append("batched.manifest")
                return {"version": "test-batched", "slots": [0, 1]}

            def uninstall(self):
                assert self.enabled
                assert not hasattr(self.model, "_test_fc2_handle")
                assert not self.model._graph_runners
                assert self.model._test_batched_handle is self
                rig.events.append("batched.uninstall")
                self.enabled = False
                del self.model._test_batched_handle

        class Fc2Handle:
            def __init__(self, model):
                self.model, self.enabled = model, True
                self.runner = None
                self.bind_count = 0
                self.manifest_calls = 0
                self.manifest_payload = {
                    "version": "test-fc2",
                    "layers": [{"index": 0, "algorithm": {"id": 17}}],
                    "requirements": ["batched vision", "compiled native runner"],
                    "source_sha256": {"test_helper": "a" * 64},
                }

            def manifest(self):
                rig.events.append("fc2.manifest")
                self.manifest_calls += 1
                if rig.stage == "fc2_manifest":
                    raise rig.failure
                return deepcopy(self.manifest_payload)

            def bind_runner(self, runner):
                assert self.enabled and self.runner is None and self.bind_count == 0
                assert self.model._test_batched_handle.enabled
                assert self.model._graph_runners == [runner]
                assert runner.model is self.model and not runner._closed
                assert runner.options["compile_sample_actions"] is True
                rig.events.append("fc2.bind")
                self.runner = runner
                self.bind_count += 1
                # Fail after accepting ownership, so rollback must retire the
                # already-created runner before touching this returned handle.
                if rig.stage == "bind":
                    raise rig.failure

            def uninstall(self):
                assert self.enabled and self.model._test_fc2_handle is self
                assert self.model._test_batched_handle.enabled
                assert not self.model._graph_runners
                if self.runner is not None:
                    assert self.runner._closed and not self.runner.entries
                rig.events.append("fc2.uninstall")
                self.enabled = False
                del self.model._test_fc2_handle

        def install_batched(model):
            assert not model._graph_runners and not hasattr(model, "_test_batched_handle")
            rig.events.append("batched.install")
            handle = BatchedHandle(model)
            rig.batched.append(handle)
            model._test_batched_handle = handle
            return handle

        def install_fc2(model):
            assert not model._graph_runners and not hasattr(model, "_test_fc2_handle")
            assert model._test_batched_handle.enabled
            rig.events.append("fc2.install")
            handle = Fc2Handle(model)
            rig.fc2.append(handle)
            model._test_fc2_handle = handle
            return handle

        class Runner:
            def __init__(self, model, **options):
                assert model._test_fc2_handle.enabled
                assert model._test_batched_handle.enabled
                assert not model._graph_runners
                rig.events.append("runner.init")
                if rig.stage == "runner":
                    # An unreturned runner owns no resources; policy must
                    # still roll back both previously returned installers.
                    raise rig.failure
                self.model, self.options = model, options
                self._closed = False
                self.entries = {"retirement_sentinel": object()}
                model._graph_runners.append(self)
                rig.runners.append(self)

            def compilation_manifest(self):
                assert self.model._test_fc2_handle.runner is self
                rig.events.append("runner.manifest")
                if rig.stage == "compilation_manifest":
                    raise rig.failure
                return {"backend": "mock", "options": {"fullgraph": True}}

            def close(self):
                assert not self._closed
                assert self.model._test_fc2_handle.enabled
                assert self.model._test_batched_handle.enabled
                rig.events.append("runner.close")
                self.entries.clear()
                self._closed = True
                self.model._graph_runners.remove(self)

        _module(monkeypatch, "batched_vision", install_batched_vision=install_batched)
        _module(monkeypatch, "vision_fc2_cublaslt", install_vision_fc2_cublaslt=install_fc2)
        _module(monkeypatch, "graph", Pi05NativeGraphRunner=Runner)


def _enabled_options():
    return dict(device="cuda", cuda_graph=True, batched_vision=True,
                compile_sample_actions=True, vision_fc2_cublaslt=True)


def test_vision_fc2_is_default_off_without_optional_installation(monkeypatch):
    _forbid_optional_work(monkeypatch)
    assert _split_lite_overrides({})[1]["vision_fc2_cublaslt"] is False
    model, processor = _model_and_processor()
    policy = Pi05LitePolicy(model, processor, device="cpu", cuda_graph=False)
    assert policy._vision_fc2_cublaslt_installation is None
    metadata = policy.metadata()["lite_optimizations"]
    assert metadata["vision_fc2_cublaslt"] is False
    assert "vision_fc2_cublaslt_manifest" not in metadata
    policy.close()
    policy.close()


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_vision_fc2_requires_exact_boolean_before_optional_work(monkeypatch, tmp_path, value):
    _forbid_model_load(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="lite_vision_fc2_cublaslt must be a boolean"):
        _split_lite_overrides({"lite_vision_fc2_cublaslt": value})
    with pytest.raises(ValueError, match="lite_vision_fc2_cublaslt must be a boolean"):
        Pi05LitePolicy.from_pretrained(tmp_path, config_overrides={
            "lite_vision_fc2_cublaslt": value,
            "lite_batched_vision": True,
            "lite_compile_sample_actions": True,
        })
    model, processor = _model_and_processor()
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    options = _enabled_options()
    options["vision_fc2_cublaslt"] = value
    with pytest.raises(ValueError, match="vision_fc2_cublaslt must be a boolean"):
        failed.__init__(model, processor, **options)
    assert failed._closed and failed.graph_runner is None
    assert all(getattr(failed, name) is None for name in pi05._POLICY_HANDLE_FIELDS)
    failed.close()


@pytest.mark.parametrize("batched,compiled,graphs", [
    (False, True, True), (True, False, True), (True, True, False),
    (False, False, True), (False, True, False), (True, False, False),
    (False, False, False),
])
def test_dependencies_fail_before_model_weights_tokenizer_or_installers(
        monkeypatch, tmp_path, batched, compiled, graphs):
    _forbid_model_load(monkeypatch, tmp_path)
    message = "Vision fc2 cuBLASLt requires batched vision, compiled sampling and native CUDA graphs"
    with pytest.raises(ValueError, match=message):
        Pi05LitePolicy.from_pretrained(tmp_path, cuda_graph=graphs, config_overrides={
            "lite_vision_fc2_cublaslt": True,
            "lite_batched_vision": batched,
            "lite_compile_sample_actions": compiled,
        })
    model, processor = _model_and_processor()
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(ValueError, match=message):
        failed.__init__(model, processor, device="cuda" if graphs else "cpu",
                        cuda_graph=graphs, batched_vision=batched,
                        compile_sample_actions=compiled, vision_fc2_cublaslt=True)
    assert failed._closed and failed.graph_runner is None
    assert all(getattr(failed, name) is None for name in pi05._POLICY_HANDLE_FIELDS)
    failed.close()


def test_install_bind_manifest_copy_and_graph_fc2_batched_close_order(monkeypatch):
    rig = _Rig(monkeypatch)
    model, processor = _model_and_processor()
    before = _inventory(model)
    policy = Pi05LitePolicy(model, processor, **_enabled_options())
    handle, runner, batched = rig.fc2[0], rig.runners[0], rig.batched[0]
    assert policy._vision_fc2_cublaslt_installation is handle
    assert policy._vision_installation is batched and policy.graph_runner is runner
    assert rig.events == ["batched.install", "batched.manifest", "fc2.install", "fc2.manifest",
                          "runner.init", "fc2.bind", "runner.manifest"]
    assert handle.bind_count == 1 and model._graph_runners == [runner]
    assert runner.options == {"max_graphs": 16, "elide_identity_gathers": False,
                              "host_layout_cache": False, "compile_sample_actions": True}
    assert _inventory(model) == before

    first, stable = policy.metadata(), policy.metadata()
    assert first == stable
    options = first["lite_optimizations"]
    assert options["vision_fc2_cublaslt"] is True
    assert options["vision_fc2_cublaslt_manifest"] == handle.manifest_payload
    options["vision_fc2_cublaslt_manifest"]["layers"][0]["algorithm"]["id"] = -1
    options["vision_fc2_cublaslt_manifest"]["requirements"].clear()
    options["vision_fc2_cublaslt_manifest"]["source_sha256"].clear()
    options["compilation_manifest"]["options"]["fullgraph"] = False
    options["vision_fc2_cublaslt"] = False
    policy.reset("another-episode")
    assert policy.metadata() == stable
    assert handle.manifest_calls == 1 and handle.bind_count == 1

    policy.close()
    retired_events = list(rig.events)
    policy.close()
    assert rig.events == retired_events
    assert rig.events[-3:] == ["runner.close", "fc2.uninstall", "batched.uninstall"]
    assert runner._closed and not runner.entries and not model._graph_runners
    assert not handle.enabled and not batched.enabled and policy._closed
    assert policy._vision_fc2_cublaslt_installation is policy._vision_installation is None
    assert not hasattr(model, "_test_fc2_handle") and not hasattr(model, "_test_batched_handle")
    assert _inventory(model) == before


@pytest.mark.parametrize("stage", ["fc2_manifest", "runner", "bind", "compilation_manifest"])
def test_partial_constructor_failure_retires_returned_resources_in_dependency_order(monkeypatch, stage):
    rig = _Rig(monkeypatch, stage=stage)
    model, processor = _model_and_processor()
    before = _inventory(model)
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(_InjectedFailure) as caught:
        failed.__init__(model, processor, **_enabled_options())
    assert caught.value is rig.failure
    assert not getattr(caught.value, "__notes__", ())
    assert len(rig.fc2) == len(rig.batched) == 1
    assert all(not handle.enabled for handle in rig.fc2 + rig.batched)
    assert failed._closed and not model._graph_runners
    assert all(getattr(failed, name) is None for name in pi05._POLICY_HANDLE_FIELDS)
    assert not hasattr(model, "_test_fc2_handle") and not hasattr(model, "_test_batched_handle")
    if stage in ("bind", "compilation_manifest"):
        assert len(rig.runners) == 1 and failed.graph_runner is rig.runners[0]
        assert failed.graph_runner._closed and not failed.graph_runner.entries
        assert rig.fc2[0].bind_count == 1
        assert rig.events[-3:] == ["runner.close", "fc2.uninstall", "batched.uninstall"]
    else:
        assert not rig.runners and failed.graph_runner is None
        assert rig.fc2[0].bind_count == 0
        assert rig.events[-2:] == ["fc2.uninstall", "batched.uninstall"]
    assert _inventory(model) == before
    retired_events = list(rig.events)
    failed.close()
    assert rig.events == retired_events
