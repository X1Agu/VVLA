"""Focused CUDA contracts; root owns device scheduling, never run locally.

Use live CUDA storage and actual shared-core/native graph/expert operators.
v2 is the before/mutate/restore control for v3. Request identity lookup stays off.
No checkpoint, full rollout, compiler optimization or tolerance relaxation.
"""
import json

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite import request_identity as lookup
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import install_native_lifecycle_signature, load_native_lifecycle
from vvla_runtime.pointwise import rotary_pos_emb

from test_lite_expert_cublaslt import cuda_profile, make_case, close_case
from test_lite_host_layout import _LayoutModel, _inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contracts require root's existing remote queue")


@pytest.fixture(scope="module")
def collectors():
    initialized = torch.cuda.is_initialized()
    result = {variant: load_native_lifecycle(variant=variant, verbose=True) for variant in ("v2", "v3")}
    assert torch.cuda.is_initialized() == initialized
    return result


def assert_lookup_off(model):
    assert getattr(model, "_lite_request_identity_installation", None) is None
    assert getattr(lookup._LOCAL, "scope", None) is None


def exact(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().contiguous().reshape(-1).view(torch.uint8)), "Raw output bytes differ"


def assert_reuse(fresh, signature, variant):
    for current, previous in zip(fresh, signature[-3:], strict=True):
        if variant == "v3":
            assert current is previous
        elif previous:
            # CPython shares the empty tuple even when v2 freshly materializes
            # a signature; identity only distinguishes nonempty child tuples.
            assert current is not previous


def test_cuda_real_storage_pointer_version_shape_offset_device_and_inference(collectors, record_property):
    model = nn.Module().eval()
    backing = torch.arange(64, device="cuda", dtype=torch.bfloat16).reshape(8, 8)
    model.weight = nn.Parameter(backing[1:5, ::2])
    model.register_buffer("view", backing[2:6, 1::2])
    with torch.inference_mode():
        model.register_buffer("inference", torch.arange(3, device="cuda"))
    assert not model.weight.is_contiguous() and model.weight.storage_offset() == 8
    assert model.weight.data_ptr() == backing.data_ptr() + 8 * backing.element_size()
    assert model.view.storage_offset() == 17
    snapshots = []

    def check(label, prior=None, changed=False):
        expected = collect_lifecycle_signature(model)
        v2 = collectors["v2"].signature(model, previous=prior)
        v3 = collectors["v3"].signature(model, previous=prior)
        assert v2 == v3 == expected, label
        if prior is not None:
            assert (v3 != prior) is changed, label
            if not changed:
                assert all(v3[index] is prior[index] for index in range(3)), label
        # Reuse is proved on a freshly checked value for every changed state too.
        warm = collectors["v3"].signature(model, previous=tuple(v3))
        assert all(warm[index] is v3[index] for index in range(3)), label
        snapshots.append(label)
        return tuple(v3)

    current = check("initial CUDA views and inference buffer")
    assert current[0][0][1] == model.weight.data_ptr()
    assert current[0][0][4] == torch.device("cuda", torch.cuda.current_device())
    assert current[1][-1][2] is None
    current = check("unchanged reuse", current)
    with torch.no_grad():
        saved = model.weight.detach().clone()
        model.weight.add_(1)
        current = check("in-place version change", current, True)
        model.weight.copy_(saved)
        current = check("restore values increments version", current, True)
        exact(model.weight, saved)
        data = model.weight.data
        object_id, version = id(model.weight), model.weight._version
        model.weight.data = data.clone()
        assert id(model.weight) == object_id and model.weight._version == version
        current = check("new storage same Parameter and version", current, True)
        model.weight.data = data
        current = check("original offset storage restored", current, True)
        pointer = model.weight.data_ptr()
        model.weight.data = data.view(2, 8)
        assert model.weight.data_ptr() == pointer and model.weight._version == version
        current = check("shape-only change with fixed pointer/version", current, True)
        model.weight.data = data
        current = check("shape restored", current, True)
        model.weight.data = backing[2:6, ::2]
        assert model.weight.shape == data.shape and model.weight.storage_offset() == 16
        assert model.weight._version == version
        current = check("different offset in the same allocation", current, True)
        model.weight.data = data
        current = check("offset restored", current, True)
        model.weight.data = data.float()
        current = check("dtype change", current, True)
        model.weight.data = data
        current = check("dtype restored", current, True)
        view = model.view
        model.view = view.cpu()
        current = check("registered buffer device changed to CPU", current, True)
        model.view = view
        current = check("registered CUDA buffer restored", current, True)
        inference = model.inference
        with torch.inference_mode():
            model.inference = torch.ones_like(inference)
        current = check("replace unversioned inference buffer", current, True)
        assert current[1][-1][2] is None
        model.inference = inference
        check("restore unversioned inference buffer", current, True)
    record_property("native_v3_cuda_tensor_contract", json.dumps(dict(
        phases=snapshots, v2_v3_python_exact=True, lookup=False,
        note="Unversioned inference-buffer content writes still require explicit invalidation; no stronger guarantee claimed")))


@pytest.mark.parametrize("elide_identity_gathers", [False, True])
def test_cuda_native_tiny_graph_before_mutate_restore_owned_output_and_generation(
        collectors, elide_identity_gathers, record_property):
    model = _LayoutModel("cuda")
    inputs = _inputs(device="cuda")
    saved = model.weight.detach().clone()
    results, evidence = {}, {}
    for variant in ("v2", "v3"):
        handle = install_native_lifecycle_signature(model, collector=collectors[variant])
        runner = Pi05NativeGraphRunner(model, host_layout_cache=True,
                                       elide_identity_gathers=elide_identity_gathers)
        retained, values, generations, cold = [], [], [], []
        try:
            for phase in ("before", "mutate", "restore", "new_empty_module", "restore_topology"):
                if phase == "mutate":
                    with torch.no_grad(): model.weight.add_(.75)
                elif phase == "restore":
                    with torch.no_grad(): model.weight.copy_(saved)
                elif phase == "new_empty_module":
                    model.add_module("temporary_empty", nn.Identity())
                elif phase == "restore_topology":
                    del model.temporary_empty
                assert_lookup_off(model)
                output = runner(**inputs)
                generations.append(model._inference_cache.generation)
                cold.append(runner.last_call["cold"])
                entry = next(iter(runner.entries.values()))
                assert output.data_ptr() != entry.output.data_ptr()
                expected = model.sample_actions(**{key: value for key, value in inputs.items() if key != "host_layout"})
                exact(output, expected)
                signature = model._inference_cache.signature
                fresh = handle.signature(previous_full_signature=signature)
                assert_reuse(fresh, signature, variant)
                replay = runner(**inputs)
                exact(replay, output)
                assert not runner.last_call["cold"] and replay.data_ptr() != output.data_ptr()
                assert model._inference_cache.generation == generations[-1]
                # Real new request data must reach the graph without changing
                # the retained output, cache generation, or captured graph.
                changed_inputs = _inputs(device="cuda", value=2.0)
                newer = runner(**changed_inputs)
                expected_new = model.sample_actions(**{key: value for key, value in changed_inputs.items() if key != "host_layout"})
                exact(newer, expected_new)
                assert not torch.equal(newer, output) and not runner.last_call["cold"]
                retained.append((output, output.clone()))
                for old, snapshot in retained:
                    exact(old, snapshot)
                values.append(output.clone())
            assert cold == [True] * 5
            assert [value - generations[0] for value in generations] == list(range(5))
            assert runner.stats["captures"] == 5 and runner.stats["invalidations"] == 4
            assert not torch.equal(values[0], values[1])
            for restored in values[2:]: exact(restored, values[0])
            results[variant] = values
            evidence[variant] = dict(generation_deltas=[value - generations[0] for value in generations],
                                     captures=runner.stats["captures"], invalidations=runner.stats["invalidations"],
                                     replays=runner.stats["replays"], owned_outputs_preserved=True)
        finally:
            runner.close()
            handle.uninstall()
            with torch.no_grad(): model.weight.copy_(saved)
            if hasattr(model, "temporary_empty"): del model.temporary_empty
    for actual, expected in zip(results["v3"], results["v2"], strict=True): exact(actual, expected)
    assert evidence["v3"] == evidence["v2"]
    record_property("native_v3_cuda_graph_contract", json.dumps(dict(
        lookup=False, elide_identity_gathers=elide_identity_gathers, v2_v3_bitwise=True, evidence=evidence)))


def test_cuda_v3_real_expert_lt_qkv_mlp_rope_capture_before_mutate_restore(collectors, record_property):
    profile = cuda_profile()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model = qkv = mlp = lt = runner = native = None
    labels = ("q", "k", "v", "attention_out", "rope_q", "rope_k", "mlp")
    results, evidence = {}, {}
    try:
        model, qkv, mlp, lt, runner = make_case(profile, False, device="cuda", fake=False)
        sources = (model.layer.self_attn.q_proj.weight, model.layer.self_attn.o_proj.weight,
                   model.layer.mlp.gate_proj.weight)
        saved = [value.detach().clone() for value in sources]
        generator = torch.Generator(device="cuda").manual_seed(197)
        x = torch.randn(1, 50, 1024, device="cuda", dtype=torch.bfloat16, generator=generator) / 8
        y = torch.randn(1, 50, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) / 8
        angle = torch.randn(1, 50, 256, device="cuda", generator=generator)
        cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
        images = [torch.zeros(1, 3, 2, 2, device="cuda")]
        image_masks = [torch.ones(1, dtype=torch.bool, device="cuda")]
        lang_masks = torch.ones(1, 4, dtype=torch.bool, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def run():
            assert_lookup_off(model)
            q, k, v = qkv._packers[0].project(x)
            qr, kr = rotary_pos_emb(q.view(1, 50, 8, 256).transpose(1, 2),
                                   k.view(1, 50, 1, 256).transpose(1, 2), cos, sin)
            return q, k, v, model.layer.self_attn.o_proj(y), qr, kr, model.layer.mlp(x)

        def exact_outputs(actual, expected):
            for label, left, right in zip(labels, actual, expected, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda message, name=label: name + ": " + message)
                assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                                   right.detach().contiguous().reshape(-1).view(torch.uint8)), label + ": raw bytes differ"

        def capture():
            graph = None
            try:
                run()
                eager = tuple(value.clone() for value in run())
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream): captured = run()
                graph.replay()
                first = tuple(value.clone() for value in captured)
                graph.replay()
                exact_outputs(tuple(value.clone() for value in captured), first)
                exact_outputs(first, eager)
                stream.synchronize()
                return first
            finally:
                stream.synchronize()
                if graph is not None: graph.reset()

        for variant in ("v2", "v3"):
            native = install_native_lifecycle_signature(model, collector=collectors[variant])
            values, generations = [], []
            try:
                with torch.no_grad(), torch.cuda.stream(stream), runner._call_lock:
                    for phase in ("before", "mutate", "restore"):
                        if phase == "mutate":
                            for value, increment in zip(sources, (.125, .0625, .03125), strict=True): value.add_(increment)
                        elif phase == "restore":
                            for value, original in zip(sources, saved, strict=True): value.copy_(original)
                        context = model.prepare_inference_context(images, image_masks, lang_masks, 10, positions=False)
                        runner._token = context["token"]
                        generations.append(context["token"][1])
                        assert_lookup_off(model)
                        signature = model._inference_cache.signature
                        fresh = native.signature(previous_full_signature=signature)
                        assert_reuse(fresh, signature, variant)
                        # Private packed/KN buffers are refreshed by the actual
                        # shared-core lifecycle; no duplicate manual prepare.
                        values.append(capture())
                    assert [value - generations[0] for value in generations] == [0, 1, 2]
                    for index in (0, 3, 6): assert not torch.equal(values[0][index], values[1][index])
                    exact_outputs(values[2], values[0])
                    stream.synchronize()
                results[variant] = values
                evidence[variant] = dict(generation_deltas=[value - generations[0] for value in generations],
                                         captured_and_repeated=3, restored_bitwise=True)
            finally:
                stream.synchronize()
                native.uninstall()
                native = None
        for actual, expected in zip(results["v3"], results["v2"], strict=True): exact_outputs(actual, expected)
        record_property("native_v3_real_expert_contract", json.dumps(dict(
            profile=profile, lookup=False, expert_layers=1, compiled=False, labels=labels,
            real_cublaslt=True, v2_v3_bitwise=True, evidence=evidence,
            native_v3=collectors["v3"].manifest()), sort_keys=True))
    finally:
        if runner is not None:
            torch.cuda.synchronize()
            runner.close()
        if native is not None and native.enabled: native.uninstall()
        if lt is not None: close_case(qkv, mlp, lt, runner)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
