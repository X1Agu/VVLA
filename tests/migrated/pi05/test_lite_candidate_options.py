"""Candidate option validation before checkpoint allocation and lifecycle ownership."""
import pytest

from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides


def test_default_and_private_suffix_recipe():
    core, options = _split_lite_overrides({})
    assert core == {} and options["suffix_attention"] is None
    assert options["fused_adarms_reduction"] is False
    source = {"splits": 4, "block_q": 16}
    _, options = _split_lite_overrides({"lite_suffix_attention": source})
    source["splits"] = 8
    assert options["suffix_attention"] == {"splits": 4, "block_q": 16}


@pytest.mark.parametrize("options", [
    {"lite_suffix_attention": True},
    {"lite_suffix_attention": {}},
    {"lite_suffix_attention": {"splits": 3, "block_q": 16}},
    {"lite_suffix_attention": {"splits": 4, "block_q": 64}},
    {"lite_suffix_attention": {"splits": True, "block_q": 16}},
    {"lite_fused_adarms_reduction": True},
    {"lite_fused_adarms_reduction": 1},
    {"lite_compile_preserve_action_linears": True},
    {"lite_compile_preserve_action_linears": 1},
    {"lite_batched_vision": "true"},
    {"lite_compile_sample_actions": True, "lite_suffix_attention": {"splits": 4, "block_q": 16}},
])
def test_invalid_options_rejected_before_model_allocation(monkeypatch, tmp_path, options):
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid candidate options reached allocation")
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    with pytest.raises(ValueError):
        Pi05LitePolicy.from_pretrained(tmp_path, config_overrides=options)


def test_policy_records_reduction_and_suffix_and_retires_graph_before_uninstall():
    from test_lite_pi05 import _policy
    import torch
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    model.eval()
    policy = Pi05LitePolicy(model, _policy().processor, device="cpu", cuda_graph=False,
                             fused_adarms=True, fused_adarms_reduction=True,
                             suffix_attention={"splits": 4, "block_q": 16})
    ada, suffix = policy._adarms_installation, policy._suffix_installation
    metadata = policy.metadata()["lite_optimizations"]
    assert metadata["fused_adarms_manifest"]["fuse_reduction"]
    assert metadata["suffix_attention_manifest"]["splits"] == 4
    runner = Pi05NativeGraphRunner(model)
    runner.entries["sentinel"] = object()
    policy.graph_runner = runner
    policy.close()
    assert runner._closed and not runner.entries and not ada.enabled and not suffix.enabled
    policy.close()


def test_policy_native_compile_preserved_linears_and_batched_vision_close():
    import torch
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.dtype import set_bfloat16_layout
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    with torch.device("meta"):
        model = Pi05ForActionPrediction(Pi05Config(attention_backend="torch_fa2"))
        set_bfloat16_layout(model)
    model.eval()
    policy = Pi05LitePolicy(model, _policy().processor, device="cuda", cuda_graph=True,
                             compile_sample_actions=True, compile_preserve_action_linears=True,
                             batched_vision=True)
    vision, linears = policy._vision_installation, policy._preserved_linears_installation
    metadata = policy.metadata()["lite_optimizations"]
    assert metadata["batched_vision_manifest"]["batch"] == 1
    assert metadata["preserved_action_linears_manifest"]["modules"] == ["action_in_proj", "action_out_proj"]
    assert policy.graph_runner._device is None  # No CUDA allocation in this CPU contract.
    policy.close()
    assert not vision.enabled and not linears.enabled and policy.graph_runner._closed
    assert "embed_prefix" not in model.__dict__ and "forward" not in model.action_out_proj.__dict__
    policy.close()
