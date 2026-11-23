"""Scoped packed weights and opaque FP32 action boundaries, CPU contracts."""
from types import SimpleNamespace

import pytest
import torch

from test_lite_compilation import _probe_module
from vvla_repro_pi05.lite.compilation import compile_prepared_run, install_preserved_action_linears
from vvla_repro_pi05.lite.fused_qkv import install_expert_qkv
from vvla_repro_pi05.lite.fused_expert_mlp import install_expert_mlp
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner


def _packed_model():
    probe = _probe_module()
    model = probe.small_core()
    handles = install_expert_qkv(model), install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True)
    return probe, model, handles


def _scoped(model, inputs, runner, backend):
    runner._token = inputs["inference_context"]["token"]
    return compile_prepared_run(model, lambda: model.sample_actions(**inputs), backend=backend,
                                owner=runner, context=inputs["inference_context"])


def test_packed_real_core_fullgraph_matches_and_scopes_cannot_escape():
    probe, model, handles = _packed_model()
    runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
    graphs = []
    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward
    with torch.inference_mode():
        inputs = probe.prepared(model)
        compiled = _scoped(model, inputs, runner, backend)
        with pytest.raises(RuntimeError, match="locked owner"):
            compiled()
        with runner._call_lock:
            expected = model.sample_actions(**inputs)
            first = compiled()
            torch.testing.assert_close(first, expected, rtol=0, atol=0)
            inputs["noise"].mul_(0.5)
            second = compiled()
            torch.testing.assert_close(second, model.sample_actions(**inputs), rtol=0, atol=0)
        assert len(graphs) == 1
        members = [*handles[0]._packers, *handles[1]._candidates]
        assert all(member._prepared_compile_token is None for member in members)
        with compiled.scope_lock, runner._call_lock, pytest.raises(RuntimeError, match="Another packed"):
            compiled()
        # Normal eager calls retain the original stale-source guard.
        q = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj
        q.weight.add_(0.125)
        with pytest.raises(RuntimeError, match="prepare_inference_context"):
            model.sample_actions(**inputs)
        with runner._call_lock, pytest.raises(RuntimeError, match="inputs changed"):
            compiled()
        updated = probe.prepared(model)
        assert updated["inference_context"]["token"] != inputs["inference_context"]["token"]
        with runner._call_lock, pytest.raises(RuntimeError, match="prepared generation"):
            compiled()
        fresh = _scoped(model, updated, runner, backend)
        with runner._call_lock:
            torch.testing.assert_close(fresh(), model.sample_actions(**updated), rtol=0, atol=0)
        # Explicit epochs also retire compiled assumptions after unsupported
        # loader-style .data writes; packed addresses remain fixed.
        address = handles[0]._packers[0]._packed.data_ptr()
        q.weight.data.add_(0.125)
        model.invalidate_inference_caches(weights_changed=False)
        reloaded = probe.prepared(model)
        with runner._call_lock, pytest.raises(RuntimeError, match="prepared generation"):
            fresh()
        assert handles[0]._packers[0]._packed.data_ptr() == address
        fresh = _scoped(model, reloaded, runner, backend)
        with runner._call_lock:
            torch.testing.assert_close(fresh(), model.sample_actions(**reloaded), rtol=0, atol=0)
        runner.entries["cpu-entry"] = SimpleNamespace(compiled_callable=fresh)
        runner.clear()
        with pytest.raises(RuntimeError, match="retired"):
            fresh()
    runner.close()
    for handle in reversed(handles):
        handle.uninstall()


def test_packed_scope_failure_restores_tokens_and_direct_compile_is_rejected():
    probe, model, handles = _packed_model()
    runner = Pi05NativeGraphRunner(model, compile_sample_actions=True)
    with torch.inference_mode():
        inputs = probe.prepared(model)
        with pytest.raises(ValueError, match="native owner"):
            compile_prepared_run(model, lambda: model.sample_actions(**inputs), backend="eager")
        def failing_backend(*args):
            raise RuntimeError("deliberate compile failure")
        compiled = _scoped(model, inputs, runner, failing_backend)
        with runner._call_lock, pytest.raises(Exception, match="deliberate compile failure"):
            compiled()
        assert all(member._prepared_compile_token is None for member in (*handles[0]._packers, *handles[1]._candidates))
        for member, fn in ((handles[0]._packers[0], "project"), (handles[1]._candidates[0], "forward")):
            x = torch.randn(1, 4, 16).bfloat16()
            direct = torch.compile(getattr(member, fn), backend="eager", fullgraph=True, dynamic=False)
            with pytest.raises(Exception, match="prepared scope|Observed exception"):
                direct(x)
        # Deliberate private packed corruption is rejected outside Dynamo too.
        handles[0]._packers[0]._packed.add_(1)
        with runner._call_lock, pytest.raises(RuntimeError, match="modified"):
            compiled()
    runner.close()
    for handle in reversed(handles):
        handle.uninstall()


def test_opaque_action_linears_keep_original_parameters_and_two_native_ops_per_step():
    probe = _probe_module()
    model = probe.small_core()
    parameters = tuple((name, id(value)) for name, value in model.named_parameters())
    keys = tuple(model.state_dict())
    handle = install_preserved_action_linears(model)
    assert tuple((name, id(value)) for name, value in model.named_parameters()) == parameters
    assert tuple(model.state_dict()) == keys
    graphs = []
    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward
    with torch.inference_mode():
        inputs = probe.prepared(model)
        compiled = compile_prepared_run(model, lambda: model.sample_actions(**inputs), backend=backend)
        torch.testing.assert_close(compiled(), model.sample_actions(**inputs), rtol=0, atol=0)
        targets = [str(node.target) for node in graphs[0].graph.nodes]
        assert targets.count("vvla_repro_pi05_lite.preserved_fp32_linear.default") == 4  # in+out, two steps
        # Native op always consumes current original FP32 weights; no packed copy.
        model.action_out_proj.bias.add_(0.125)
        torch.testing.assert_close(compiled(), model.sample_actions(**inputs), rtol=0, atol=0)
    handle.uninstall()
    assert "forward" not in model.action_in_proj.__dict__ and "forward" not in model.action_out_proj.__dict__


def test_graph_metadata_failure_retires_compiled_entry_without_cuda_work():
    from test_lite_host_layout import _LayoutModel, _inputs, _get, _context
    from vvla_repro_pi05.lite.graph import _Entry
    from vvla_repro_pi05.lite.layout import copy_layout_tree
    model = _LayoutModel()
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True)
    a, b = _inputs(2), _inputs(3)
    template = _get(runner._host_layouts, model, a)
    context = _context(model, a, template)
    template.materialize(model, a["image_masks"], a["lang_masks"])
    context["positions"] = copy_layout_tree(template.positions)
    runner._token = context["token"]
    tensors = [*a["images"], *a["image_masks"], a["lang_tokens"], a["lang_masks"], a["noise"]]
    key = (a["num_steps"], tuple((tuple(x.shape), x.dtype, x.device) for x in tensors),
           context["empty_pattern"], context["packing"]["pads"].shape[1])
    compiled = compile_prepared_run(model, lambda: a["noise"], backend="eager")
    metadata = copy_layout_tree(template.metadata)
    metadata[0].extra["invalid-field"] = True
    runner.entries[key] = _Entry(tensors, None, a["noise"], metadata, a["host_layout"].mask_bytes, context, compiled)
    with torch.inference_mode(), pytest.raises(ValueError, match="metadata fields"):
        runner._call(b["images"], b["image_masks"], b["lang_tokens"], b["lang_masks"],
                     b["noise"], b["num_steps"], b["host_layout"])
    assert key not in runner.entries and not compiled.active
    runner.close()


def test_policy_combines_scoped_packing_and_action_boundary_with_stable_metadata_and_close():
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    assert not _split_lite_overrides({})[1]["compile_preserve_action_linears"]
    model = _probe_module().small_core()
    processor = _policy().processor
    with pytest.raises(ValueError, match="requires compiled"):
        Pi05LitePolicy(model, processor, device="cpu", cuda_graph=False,
                       compile_preserve_action_linears=True)
    # A CUDA device label does not allocate anything during construction.
    policy = Pi05LitePolicy(model, processor, device="cuda", cuda_graph=True,
                           compile_sample_actions=True, compile_preserve_action_linears=True,
                           fused_expert_qkv=True, fused_expert_gate_up=True)
    handles = (policy._preserved_linears_installation, policy._mlp_installation, policy._qkv_installation)
    first = policy.metadata()
    assert first["lite_optimizations"]["preserved_action_linears_manifest"] == handles[0].manifest()
    assert policy.metadata() == first and policy.graph_runner._device is None
    policy.close()
    assert all(not handle.enabled for handle in handles)
    assert "forward" not in model.action_in_proj.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; run by experiment owner")
def test_cuda_native_compiled_packing_action_boundary_and_inplace_invalidation():
    from vvla_repro_pi05.lite.layout import HostMaskLayout
    probe = _probe_module()
    model = probe.small_core(attention_backend="torch_fa2").cuda()
    qkv, mlp = install_expert_qkv(model), install_expert_mlp(model, concatenate=True, fuse_gelu_mul=True)
    linear = install_preserved_action_linears(model)
    runner = Pi05NativeGraphRunner(model, host_layout_cache=True, compile_sample_actions=True)
    with torch.inference_mode():
        inputs = dict(images=[torch.rand(1, 3, 4, 4, device="cuda").bfloat16(),
                              -torch.ones(1, 3, 4, 4, device="cuda").bfloat16()],
                      image_masks=[torch.tensor([True], device="cuda"), torch.tensor([False], device="cuda")],
                      lang_tokens=torch.tensor([[2, 3, 4, 0, 0, 0]], device="cuda"),
                      lang_masks=torch.tensor([[True, True, True, False, False, False]], device="cuda"),
                      noise=torch.randn(1, 4, 8, device="cuda"), num_steps=2)
        host = HostMaskLayout.from_processor((True, False), (True, True, True, False, False, False),
                                            inputs["images"], inputs["image_masks"], inputs["lang_masks"])
        expected = model.sample_actions(**inputs)
        first = runner(**inputs, host_layout=host)
        saved = first.clone()
        torch.testing.assert_close(first, expected, rtol=0.02, atol=0.02)
        for valid in (2, 3):
            masks = tuple(i < valid for i in range(6))
            inputs["lang_masks"] = torch.tensor([masks], device="cuda")
            host = HostMaskLayout.from_processor((True, False), masks, inputs["images"],
                                                inputs["image_masks"], inputs["lang_masks"])
            actual = runner(**inputs, host_layout=host)
            torch.testing.assert_close(actual, model.sample_actions(**inputs), rtol=0.02, atol=0.02)
        assert runner.stats["compiled_entries"] == 1 and runner.stats["metadata_updates"] == 2
        old_entry = next(iter(runner.entries.values()))
        model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.add_(0.125)
        changed = runner(**inputs, host_layout=host)
        torch.testing.assert_close(changed, model.sample_actions(**inputs), rtol=0.02, atol=0.02)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        assert not old_entry.compiled_callable.active and runner.stats["invalidations"] == 1
    runner.close()
    linear.uninstall(); mlp.uninstall(); qkv.uninstall()
