"""Cold ownership/config tests without importing the model or touching CUDA."""
from contextlib import nullcontext
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vvla_lingbot import runtime


def config_files(tmp_path):
    official = tmp_path / "official"
    for directory in (official / "configs/robot_configs", official / "deploy",
                      official / "lingbotvla/data/vla_data", tmp_path / "qwen",
                      tmp_path / "native", tmp_path / "deps", tmp_path / "checkpoints/a/b/hf"):
        directory.mkdir(parents=True, exist_ok=True)
    (official / "configs/robot_configs/robotwin.yaml").write_text("{}")
    (tmp_path / "norm.json").write_text("{}")
    sources = {"deploy/lingbot_vla_v2_policy.py": "# fixed policy\n",
               "lingbotvla/data/vla_data/utils.py": "# fixed transform\n"}
    for name, content in sources.items():
        (official / name).write_text(content)
    manifest = {"schema": 1, "official_commit": "a" * 40,
                "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}}
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "checkpoints/a/b/hf"
    (checkpoint / "model.safetensors").write_bytes(b"mock weights, never loaded")
    (checkpoint / "config.json").write_text('{"value":1}')
    (checkpoint / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    (checkpoint.parent.parent.parent / "lingbotvla_cli.yaml").write_text("model: {}")
    (tmp_path / "qwen/config.json").write_text("{}")
    return {"mode": "official-bf16", "checkpoint": str(checkpoint),
            "official_root": str(official), "qwen": str(tmp_path / "qwen"),
            "robot_config": str(official / "configs/robot_configs/robotwin.yaml"),
            "norm_stats": str(tmp_path / "norm.json"), "native_path": str(tmp_path / "native"),
            "python_paths": [str(tmp_path / "deps")], "source_manifest": str(manifest_path),
            "source_manifest_sha256": runtime.file_sha256(manifest_path),
            "expected_runtime": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                                 "torch": "test", "transformers": "test", "cuda": "test"}}


class FakeCuda:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def device(self, device):
        return nullcontext()

    def synchronize(self, device):
        self.calls.append("drain")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected drain failure")


class FakeGraph:
    def __init__(self, calls, failures=0):
        self.calls, self.failures = calls, failures

    def reset(self):
        self.calls.append("reset")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected graph reset failure")


def test_drain_failure_retains_graph_policy_lease_and_retry_order():
    bundle = runtime.RuntimeBundle()
    cuda = FakeCuda(failures=1)
    graph = FakeGraph(cuda.calls)
    policy, engine = object(), SimpleNamespace(_graph=graph)
    bundle.torch = SimpleNamespace(cuda=cuda)
    bundle.cuda_started, bundle.policy, bundle.engine = True, policy, engine
    try:
        with pytest.raises(RuntimeError, match="drain failure"):
            bundle.close()
        assert bundle.policy is policy and bundle.engine is engine and engine._graph is graph
        assert cuda.calls == ["drain"] and not bundle.closed
        with pytest.raises(RuntimeError, match="one active"):
            runtime.RuntimeBundle()
        bundle.close()
        assert cuda.calls == ["drain", "drain", "reset"]
        assert bundle.closed and bundle.policy is None and bundle.engine is None
        bundle.close()
        assert cuda.calls == ["drain", "drain", "reset"]
    finally:
        cuda.failures = 0
        bundle.close()


def test_graph_reset_failure_retains_owner_and_lease():
    bundle = runtime.RuntimeBundle()
    cuda = FakeCuda()
    graph = FakeGraph(cuda.calls, failures=1)
    bundle.torch = SimpleNamespace(cuda=cuda)
    bundle.cuda_started, bundle.policy = True, object()
    bundle.engine = SimpleNamespace(_graph=graph)
    try:
        with pytest.raises(RuntimeError, match="reset failure"):
            bundle.close()
        assert bundle.engine._graph is graph and bundle.policy is not None
        with pytest.raises(RuntimeError, match="one active"):
            runtime.RuntimeBundle()
    finally:
        graph.failures = 0
        bundle.close()
    assert cuda.calls == ["drain", "reset", "drain", "reset"]


@pytest.mark.parametrize('capacity', [True, 0, 65, 2.5])
def test_graph_cache_capacity_rejects_invalid_values(tmp_path, capacity):
    value = config_files(tmp_path)
    value['graph_cache_capacity'] = capacity
    with pytest.raises(ValueError,match='graph_cache_capacity'):
        runtime.LiteRuntimeConfig.from_mapping(value)


def test_official_sampler_cannot_silently_select_framework_graph_pool(tmp_path):
    value = config_files(tmp_path)
    value['graph_cache_capacity'] = 2
    with pytest.raises(ValueError,match='framework-only'):
        runtime.LiteRuntimeConfig.from_mapping(value)


def test_environment_restore_includes_import_mutation_and_failure(tmp_path, monkeypatch):
    config = runtime.LiteRuntimeConfig.from_mapping(config_files(tmp_path))
    monkeypatch.setenv("QWEN3VL_PATH", "before")
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    old_cwd, old_paths = Path.cwd(), list(sys.path)
    with pytest.raises(RuntimeError, match="import failed"):
        with runtime.local_environment(config):
            assert Path.cwd() == config.official_root
            assert os.environ["QWEN3VL_PATH"] == str(config.qwen)
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            os.environ["PYTHONHASHSEED"] = "42"
            sys.path.insert(0, "official import side effect")
            raise RuntimeError("import failed")
    assert Path.cwd() == old_cwd and sys.path == old_paths
    assert os.environ["QWEN3VL_PATH"] == "before" and "PYTHONHASHSEED" not in os.environ


def test_config_direct_instance_is_revalidated_and_versions_are_immutable(tmp_path):
    value = config_files(tmp_path)
    config = runtime.LiteRuntimeConfig.from_mapping(value)
    value["expected_runtime"]["torch"] = "mutated caller dict"
    assert config.expected_runtime["torch"] == "test"
    with pytest.raises(TypeError):
        config.expected_runtime["torch"] = "changed"
    invalid = runtime.LiteRuntimeConfig(**{**config.__dict__, "mode": "framework-fp8"})
    with pytest.raises(ValueError, match="only"):
        runtime.build_managed_runtime(invalid)
    assert not runtime._LEASE.locked()


def test_checkpoint_identity_includes_config_index_and_source_inventory(tmp_path):
    config = runtime.LiteRuntimeConfig.from_mapping(config_files(tmp_path))
    first = runtime.validate_assets(config)
    assert {"model.safetensors", "config.json", "model.safetensors.index.json"} <= first["checkpoint_files_sha256"].keys()
    (config.checkpoint / "config.json").write_text('{"value":2}')
    second = runtime.validate_assets(config)
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    assert first["checkpoint_files_sha256"]["model.safetensors"] == second["checkpoint_files_sha256"]["model.safetensors"]
    (config.official_root / "deploy/unpinned.py").write_text("# extra")
    with pytest.raises(ValueError, match="exactly cover"):
        runtime.validate_assets(config)


def test_partial_engine_owned_before_init_and_legacy_partial_preserved():
    bundle = runtime.RuntimeBundle()

    class Engine:
        def __init__(self, flow, mode, *, variant, audit):
            assert bundle.engine is self
            self.arguments = flow, mode, variant, audit
            raise ValueError("partial constructor")

    try:
        with pytest.raises(ValueError, match="partial constructor"):
            runtime._construct_engine(bundle, functools.partial(Engine, variant="combined"),
                                      "flow", "framework-bf16", audit=None)
        assert bundle.engine.arguments == ("flow", "framework-bf16", "combined", None)
    finally:
        bundle.close()


def test_constructor_primary_error_keeps_retryable_owner_and_restores_context(tmp_path, monkeypatch):
    config = runtime.LiteRuntimeConfig.from_mapping(config_files(tmp_path))
    cuda = FakeCuda(failures=1)
    fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"),
                                 cuda=cuda, inference_mode=lambda flag: nullcontext(),
                                 no_grad=nullcontext, is_autocast_enabled=lambda device: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(runtime, "_PROCESS_PIN", None)
    module = ModuleType("deploy.lingbot_vla_v2_policy")
    module.__file__ = str(config.official_root / "deploy/lingbot_vla_v2_policy.py")
    module.set_seed_everywhere = lambda seed: None

    class Policy:
        def __init__(self, *args, **kwargs):
            self.partial_owned_tensor = object()
            raise ValueError("primary model load failure")

    module.LingbotVLAv2Server = Policy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    old_cwd, old_paths = Path.cwd(), list(sys.path)
    with pytest.raises(ValueError, match="primary model load failure") as caught:
        runtime.build_managed_runtime(config)
    owner = caught.value.lingbot_runtime
    try:
        assert isinstance(owner.policy, Policy) and owner.policy.partial_owned_tensor is not None
        assert not owner.closed and runtime._LEASE.locked()
        assert Path.cwd() == old_cwd and sys.path == old_paths
        assert any("cleanup did not retire" in note for note in caught.value.__notes__)
    finally:
        owner.close()


def test_combined_package_preserves_all_seven_upstream_bytes():
    root = Path(runtime.__file__).parent / "lite_tuning"
    manifest = json.loads((root / "SOURCE.json").read_bytes())
    assert manifest["byte_identical"] is True and manifest["variant"] == "combined"
    assert set(manifest["files"]) == {"__init__.py", "engine.py", "graph_diagnostics.py",
                                       "moe.py", "rope.py", "router.py", "state_guard.py"}
    for name, record in manifest["files"].items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == record["git_blob_sha1"]


def test_actual_native_origins_unknown_load_and_changed_bytes_fail_closed(tmp_path, monkeypatch):
    value = config_files(tmp_path)
    value["mode"] = "framework-bf16"
    value["expected_runtime"]["vllm"] = "test"
    config = runtime.LiteRuntimeConfig.from_mapping(value)
    files = {"vllm": config.native_path / "vllm/__init__.py",
             "vllm._C_stable_libtorch": config.native_path / "vllm/_C_stable_libtorch.so",
             "vvla_runtime.execution.native": config.python_paths[0] / "native.py"}
    modules = {}
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source-only fake origin; never imported")
        modules[name] = SimpleNamespace(__file__=str(path))
    modules["vllm"].__path__ = [str(config.native_path / "vllm")]
    native = modules["vvla_runtime.execution.native"]
    native._loaded = False
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(modules=modules))
    monkeypatch.setattr(runtime, "_NATIVE_PIN", None)
    runtime.dependency_origins(config, before_load=True)
    native._loaded = True
    with pytest.raises(RuntimeError, match="before this verified"):
        runtime.dependency_origins(config, before_load=True)
    recorded = runtime.dependency_origins(config)
    assert set(recorded) == set(files)
    assert recorded == runtime.dependency_origins(config, before_load=True)
    files["vllm._C_stable_libtorch"].write_bytes(b"changed native library bytes")
    with pytest.raises(RuntimeError, match="bytes changed"):
        runtime.dependency_origins(config, before_load=True)


def test_namespace_paths_cannot_escape_declared_official_root(tmp_path, monkeypatch):
    root = tmp_path / "official"
    root.mkdir()
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(modules={
        "deploy": SimpleNamespace(__path__=[str(root / "deploy"), str(tmp_path / "elsewhere")])}))
    with pytest.raises(RuntimeError, match="Conflicting loaded official"):
        runtime._check_import_origins(root)


def test_native_distribution_pin_uses_explicit_target_not_interpreter_default(tmp_path, monkeypatch):
    calls = []

    def distributions(*, path):
        calls.append(path)
        return [SimpleNamespace(metadata={"Name": "vllm"}, version="0.32.0"),
                SimpleNamespace(metadata={"Name": "unrelated"}, version="1")]

    monkeypatch.setattr(runtime.importlib.metadata, "distributions", distributions)
    assert runtime._native_distribution_version(tmp_path) == "0.32.0"
    assert calls == [[str(tmp_path)]]
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda **kwargs: [])
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime._native_distribution_version(tmp_path)
