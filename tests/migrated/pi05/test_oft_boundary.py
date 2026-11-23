"""Reject malformed robot observations before any preprocessing or GPU work."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vvla_repro_pi05.oft_config import validate_runtime_options
from vvla_repro_pi05.oft_config import make_oft_config


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "workers/oft/src/vvla_oft/worker.py"
    spec = importlib.util.spec_from_file_location("oft_boundary_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = object.__new__(module.Worker)
    worker.default_key = "debug"
    worker.service = SimpleNamespace(model=SimpleNamespace(norm_stats={"debug": {}}))
    return worker


@pytest.mark.parametrize(
    "change",
    [
        {"state": np.full(8, np.nan)},
        {"state": np.full(8, np.inf)},
        {"state": np.zeros(7)},
        {"state": ["a"] * 8},
        {"full_image": np.zeros((224, 224, 3), dtype=np.float32)},
        {"wrist_image": np.zeros((224, 224), dtype=np.uint8)},
        {"prompt": ""},
        {"prompt": []},
        {"norm_key": "missing"},
        {"norm_key": None},
    ],
)
def test_invalid_observation_does_not_reach_preprocessing(worker, change):
    observation = dict(
        full_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="pick block",
    )
    observation.update(change)
    # The stub deliberately has no prepare_row: getting that far is a failure.
    with pytest.raises(ValueError):
        worker.prepare(observation)


@pytest.mark.parametrize(
    "option",
    [
        "enable_sleep_mode",
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
        "enable_prefix_caching",
        "lora_path",
        "quantization_config",
        "diffusion_offload_config",
        "kv_offload_config",
    ],
)
def test_unsupported_residency_or_model_features_are_rejected(option):
    with pytest.raises(ValueError):
        validate_runtime_options(SimpleNamespace(**{option: True}))


@pytest.mark.parametrize(
    "option",
    [
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "data_parallel_size",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
        "text_encoder_tp_size",
        "ulysses_degree",
        "ring_degree",
        "allgather_degree",
    ],
)
def test_unimplemented_parallelism_is_rejected(option):
    with pytest.raises(ValueError):
        validate_runtime_options(SimpleNamespace(parallel_config=SimpleNamespace(**{option: 2})))


def test_single_resident_replica_options_are_supported():
    validate_runtime_options(
        SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1))
    )


@pytest.fixture
def reference_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "openvla", "text_config": {"model_type": "llama"}}))
    (tmp_path / "dataset_statistics.json").write_text('{"debug": {}}')
    return dict(checkpoint=tmp_path, worker_python=tmp_path / "python", source=tmp_path,
                native_path=tmp_path, worker_log=tmp_path / "worker.log", norm_key="debug",
                action_space="test", reference_compatible=True)


@pytest.mark.parametrize("change", [{"batch": 2}, {"graph": True}, {"backend": "vllm_paged_fa2"},
                                    {"reference_compatible": "true"}, {"mlp_precision": "fp4"}])
def test_reference_mode_rejects_execution_changes(reference_config, change):
    with pytest.raises(ValueError):
        make_oft_config(**(reference_config | change))


def test_fp8_is_explicit_and_keeps_reference_attention(reference_config):
    deploy = make_oft_config(**reference_config, mlp_precision="fp8")
    stage = deploy["stages"][0]
    assert stage["max_num_seqs"] == 1 and stage["enforce_eager"]
    assert stage["custom_pipeline_args"]["mlp_precision"] == "fp8"
    assert stage["custom_pipeline_args"]["reference_compatible"] is True
    assert stage["custom_pipeline_args"]["backend"] == "sdpa"


def test_public_config_forwards_full_fp8_and_fusions(reference_config):
    flags = dict.fromkeys(('fuse_rope', 'fuse_silu_mul', 'fuse_rmsnorm', 'fuse_residual_square'), True)
    config = make_oft_config(**(reference_config | dict(reference_compatible=False,
        backend='vllm_fa2', graph=True, mlp_precision='fp8', fp8_scope='all_linear', **flags)))
    args = config['stages'][0]['custom_pipeline_args']
    assert args['fp8_scope'] == 'all_linear' and args['backend'] == 'vllm_fa2'
    assert all(args[name] for name in flags)


@pytest.mark.parametrize('change', [dict(fp8_scope='unknown'), dict(fp8_scope='all_linear'),
    dict(fuse_rope=True), dict(fuse_silu_mul=True), dict(fuse_rmsnorm='true')])
def test_invalid_precision_and_reference_fusion_config(reference_config, change):
    with pytest.raises(ValueError):
        make_oft_config(**(reference_config | change))


def test_public_selective_fp8_preserves_qkv_and_shared_activation_options(reference_config):
    deploy=make_oft_config(**(reference_config | dict(reference_compatible=False, backend='vllm_fa2',
        graph=True, mlp_precision='fp8', fuse_silu_mul=True, share_mlp_quant=True, fuse_qkv=True)))
    args=deploy['stages'][0]['custom_pipeline_args']
    assert args['fp8_scope']=='mlp' and args['share_mlp_quant'] and args['fuse_qkv']


@pytest.mark.parametrize('change', [dict(share_mlp_quant=True), dict(qkvo_layout='invalid'),
    dict(mlp_precision='fp8',fp8_scope='all_linear',fuse_qkv=True)])
def test_incompatible_public_projection_options(reference_config,change):
    with pytest.raises(ValueError):make_oft_config(**(reference_config | dict(reference_compatible=False) | change))
