"""Prepared sampler compilation contracts; CUDA cases run only by experiment owner."""

import importlib.util
from pathlib import Path

import pytest
import torch

from vvla_repro_pi05.lite.compilation import (INDUCTOR_OPTIONS, compilation_manifest,
                                      compile_prepared_run, validate_compile_model)
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner


def _probe_module():
    path = Path(__file__).parents[2] / "benchmarks/pi05-lite-bf16-20261105/audit/compile_cpu_probe.py"
    spec = importlib.util.spec_from_file_location("compile_cpu_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_off_strict_option_and_no_inductor_cuda_graph(monkeypatch):
    model = torch.nn.Linear(2, 2).eval()
    assert not Pi05NativeGraphRunner(model).compile_sample_actions
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            Pi05NativeGraphRunner(model, compile_sample_actions=value)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    run = lambda: None
    assert compile_prepared_run(model, run).compiled is run
    assert calls == [{"backend": "inductor", "fullgraph": True, "dynamic": False,
                      "options": INDUCTOR_OPTIONS}]
    assert not calls[0]["options"]["triton.cudagraphs"]
    assert calls[0]["options"]["emulate_precision_casts"]
    first = compilation_manifest(True)
    calls[0]["options"]["triton.cudagraphs"] = True
    assert compilation_manifest(True) == first  # Compiler cannot mutate identity.


@pytest.mark.parametrize("cuda_graph,extra", [(False, {})])
def test_from_pretrained_rejects_invalid_combinations_before_model_or_weight_load(monkeypatch, tmp_path,
                                                                                  cuda_graph, extra):
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid compile combination reached model allocation")
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    with pytest.raises(ValueError, match="[Cc]ompil"):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, cuda_graph=cuda_graph,
                                          config_overrides={"lite_compile_sample_actions": True, **extra})


def test_actual_policy_constructor_records_stable_compile_identity_without_cuda_work():
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    assert _split_lite_overrides({})[1]["compile_sample_actions"] is False
    original = _policy()
    # Selecting a device label and constructing a runner performs no CUDA API
    # calls or allocation; this CPU test never submits inference.
    policy = Pi05LitePolicy(original.model, original.processor, device="cuda", cuda_graph=True,
                             compile_sample_actions=True)
    expected = compilation_manifest(True)
    first = policy.metadata()
    assert first["lite_optimizations"]["compile_sample_actions"]
    assert first["lite_optimizations"]["compilation_manifest"] == expected
    assert policy.metadata() == first
    assert policy.graph_runner._device is None and not policy.graph_runner.entries
    policy.close()


@pytest.mark.parametrize("attribute", ["_lite_expert_qkv_installation", "_lite_expert_mlp_installation",
                                       ])
def test_incompatible_checked_helpers_fail_before_tracing(attribute):
    model = torch.nn.Linear(2, 2).eval()
    setattr(model, attribute, object())
    with pytest.raises(ValueError, match="not validated"):
        Pi05NativeGraphRunner(model, compile_sample_actions=True)
    # Original noncompiled path remains available.
    assert not Pi05NativeGraphRunner(model).compile_sample_actions


def test_train_and_graph_break_fail_closed_without_eager_fallback():
    with pytest.raises(ValueError, match="eval"):
        validate_compile_model(torch.nn.Linear(2, 2))
    calls = []

    def backend(gm, inputs):
        calls.append(gm)
        return gm.forward

    value = torch.tensor([1.0])
    def invalid():
        return value + value.data_ptr()
    compiled = compile_prepared_run(torch.nn.Linear(1, 1).eval(), invalid, backend=backend)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled()
    assert not calls


@pytest.mark.parametrize("adarms,gelu_residual", [(False, False), (True, False), (True, True)])
def test_real_small_core_one_graph_prepared_scope_and_fresh_tensor_values(monkeypatch, adarms, gelu_residual):
    from vvla_repro_pi05.lite import compilation
    from vvla_repro_pi05.lite.fused_adarms import install_fused_adarms
    probe = _probe_module()
    model = probe.small_core()
    if adarms:
        install_fused_adarms(model)
    if gelu_residual:
        from vvla_repro_pi05.lite.fused_expert_mlp import install_expert_mlp
        from vvla_repro_pi05.lite.fused_gated_residual import install_gated_residual
        install_expert_mlp(model, fuse_gelu_mul=True)
        install_gated_residual(model)
    graphs = []
    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward
    original_compile = compilation.compile_prepared_run
    monkeypatch.setattr(compilation, "compile_prepared_run",
                        lambda model, run, **kw: original_compile(model, run, backend=backend, **kw))
    runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
    with torch.inference_mode():
        inputs = probe.prepared(model)
        context, metadata = inputs["inference_context"], inputs["attention_metadata"]
        def forbidden(*args, **kwargs):
            raise AssertionError("Lifecycle/mask preparation entered the compiled region")
        monkeypatch.setattr(model, "prepare_inference_context", forbidden)
        monkeypatch.setattr(model, "prepare_attention_metadata", forbidden)
        run = runner._make_run(inputs["images"], inputs["image_masks"], inputs["lang_tokens"],
                               inputs["lang_masks"], inputs["noise"], 2, metadata, context)
        reference = model.sample_actions(**inputs)
        actual = run()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        saved = actual.clone()
        inputs["noise"].add_(0.25)
        inputs["images"][0].mul_(0.5)
        inputs["lang_tokens"][0, 1] = 5
        changed = run()
        torch.testing.assert_close(changed, model.sample_actions(**inputs), rtol=0, atol=0)
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        assert not torch.equal(changed, actual)
    assert len(graphs) == 1 and len(list(graphs[0].graph.nodes)) > 500
    runner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; run by experiment owner")
def test_cuda_compiled_graph_masks_stream_weight_invalidation_and_owned_outputs():
    from test_lite_host_layout import _LayoutModel, _inputs
    model = _LayoutModel("cuda")
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True, elide_identity_gathers=True,
                                   compile_sample_actions=True)
    with torch.inference_mode():
        retained = None
        for valid, value in ((2, 1.0), (3, 2.0), (2, 3.0)):
            inputs = _inputs(valid, value=value, device="cuda")
            expected = model.sample_actions(**{k: v for k, v in inputs.items() if k != "host_layout"})
            output = runner(**inputs)
            torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
            if retained is None:
                retained = output, output.clone()
            else:
                torch.testing.assert_close(*retained, rtol=0, atol=0)
        assert runner.stats["compiled_entries"] == 1
        assert runner.stats["metadata_updates"] == 2
        assert all(entry.compiled_callable is not None for entry in runner.entries.values())
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            model.weight.add_(0.125)
            inputs = _inputs(device="cuda")
            output = runner(**inputs)
        other.synchronize()
        expected = model.sample_actions(**{k: v for k, v in inputs.items() if k != "host_layout"})
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
        assert runner.stats["invalidations"] == 1 and runner.stats["compiled_entries"] == 2
        torch.testing.assert_close(*retained, rtol=0, atol=0)
    runner.close()
    assert not runner.entries


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; run by experiment owner")
@pytest.mark.parametrize("gelu_residual", [False, True])
def test_cuda_small_core_fa2_adarms_fullgraph_and_native_capture(gelu_residual):
    from vvla_repro_pi05.lite.fused_adarms import install_fused_adarms
    from vvla_repro_pi05.lite.layout import HostMaskLayout
    probe = _probe_module()
    model = probe.small_core(attention_backend="torch_fa2").cuda()
    handle = install_fused_adarms(model)
    other_handles = []
    if gelu_residual:
        from vvla_repro_pi05.lite.fused_expert_mlp import install_expert_mlp
        from vvla_repro_pi05.lite.fused_gated_residual import install_gated_residual
        other_handles = [install_expert_mlp(model, fuse_gelu_mul=True), install_gated_residual(model)]
    model.paligemma_with_expert.prefix_attention.elide_identity_gathers = True
    model.paligemma_with_expert.suffix_attention.elide_identity_gathers = True
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True, elide_identity_gathers=True,
                                   compile_sample_actions=True)
    with torch.inference_mode():
        # Construct inputs on CPU without asking the CUDA model to process them.
        source = dict(images=[torch.rand(1, 3, 4, 4).bfloat16().cuda(), -torch.ones(1, 3, 4, 4).bfloat16().cuda()],
                      image_masks=[torch.tensor([True], device="cuda"), torch.tensor([False], device="cuda")],
                      lang_tokens=torch.tensor([[2, 3, 4, 0, 0, 0]], device="cuda"),
                      lang_masks=torch.tensor([[True, True, True, False, False, False]], device="cuda"),
                      noise=torch.randn(1, 4, 8, device="cuda"), num_steps=2)
        host = HostMaskLayout.from_processor((True, False), (True, True, True, False, False, False),
                                            source["images"], source["image_masks"], source["lang_masks"])
        expected = model.sample_actions(**source)
        first = runner(**source, host_layout=host)
        second = runner(**source, host_layout=host)
        # Inductor can change reduction order; whole-checkpoint gates remain required.
        torch.testing.assert_close(first, expected, rtol=0.02, atol=0.01)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        assert runner.stats["compiled_entries"] == 1 and not runner.last_call["cold"]
    runner.close()
    for installed in reversed(other_handles):
        installed.uninstall()
    handle.uninstall()
