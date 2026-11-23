"""One focused real-Lt CUDA integration per audited card; scheduled remotely.

No compiler is added here: Q/K/V, RoPE and MLP must all match the lookup-off
execution bitwise. The separate Lt compiled tests retain their independent
Lt-free compiled RoPE control unchanged.
"""
from contextlib import contextmanager
import json

import pytest
import torch

from vvla_repro_pi05.lite import request_identity as lookup
from vvla_repro_pi05.lite.native_lifecycle import install_native_lifecycle_signature, load_native_lifecycle
from vvla_runtime.pointwise import rotary_pos_emb

from test_lite_expert_cublaslt import cuda_profile, make_case, close_case


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA remote queue only')
def test_cuda_request_identity_native_collect_capture_mutation_and_private_corruption(monkeypatch, record_property):
    profile = cuda_profile()
    # Load the host-only collector before making a CUDA model. No checkpoint or
    # full model is needed: this is one real 1024/4096 expert layer and real Lt.
    collector = load_native_lifecycle(variant='v2')
    before_tf32 = torch.backends.cuda.matmul.allow_tf32
    before_bf16_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model, qkv, mlp, lt, runner = make_case(profile, False, device='cuda', fake=False)
    native = identities_handle = None
    scopes, collected, run_calls = [], [], []
    labels = ('q', 'k', 'v', 'attention_out', 'rope_q', 'rope_k', 'mlp')
    stream = torch.cuda.Stream()
    try:
        native = install_native_lifecycle_signature(model, collector=collector)
        original_signature = native.signature
        def observe_collect(**kwargs):
            result = original_signature(**kwargs)
            collected.append((len(result.parameters), len(result.buffers), len(result.topology)))
            assert all(len(value) == 6 for group in result[:2] for value in group)
            return result
        monkeypatch.setattr(native, 'signature', observe_collect)
        x = torch.randn(1, 50, 1024, device='cuda', dtype=torch.bfloat16) / 8
        y = torch.randn(1, 50, 2048, device='cuda', dtype=torch.bfloat16) / 8
        angle = torch.randn(1, 50, 256, device='cuda')
        cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
        images = [torch.zeros(1, 3, 2, 2, device='cuda')]
        image_masks = [torch.ones(1, dtype=torch.bool, device='cuda')]
        lang_masks = torch.ones(1, 4, dtype=torch.bool, device='cuda')
        stream.wait_stream(torch.cuda.current_stream())

        def assert_scope_expired():
            assert getattr(lookup._LOCAL, 'scope', None) is None
            for scope in scopes:
                assert not scope._active and scope._registered == scope._private == {}

        def prepare_current():
            # Actual shared-core hook performs full collect -> refresh -> scoped
            # QKV/MLP/Lt prepare. There is no extra standalone prepare afterward.
            before = len(collected)
            context = model.prepare_inference_context(images, image_masks, lang_masks, 10, positions=False)
            assert len(collected) == before + 1
            assert_scope_expired()
            runner._token = context['token']
            return context

        def run():
            # Capture and replay begin only after the identity scope has expired.
            assert_scope_expired()
            run_calls.append(True)
            q, k, v = qkv._packers[0].project(x)
            qr, kr = rotary_pos_emb(q.view(1, 50, 8, 256).transpose(1, 2),
                                   k.view(1, 50, 1, 256).transpose(1, 2), cos, sin)
            return q, k, v, model.layer.self_attn.o_proj(y), qr, kr, model.layer.mlp(x)

        def exact(actual, expected):
            for name, a, b in zip(labels, actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0,
                                           msg=lambda message, label=name: label + ': ' + message)

        def capture_outputs():
            # Real graph, same static tensors, owned clones. Retire each graph
            # before weight/installation changes, as the native owner requires.
            graph = None
            try:
                run()
                eager = tuple(value.clone() for value in run())
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                graph.replay()
                first = tuple(value.clone() for value in captured)
                graph.replay()
                second = tuple(value.clone() for value in captured)
                exact(first, second)
                exact(first, eager)
                stream.synchronize()
                return first
            finally:
                stream.synchronize()
                if graph is not None: graph.reset()

        with torch.no_grad(), torch.cuda.stream(stream):
            with runner._call_lock:
                prepare_current()
                baseline = capture_outputs()
            identities_handle = lookup.install_request_identity_lookup(model)
            original_scope = identities_handle.prepare_scope
            @contextmanager
            def observe_scope(state, generation):
                with original_scope(state, generation) as scope:
                    scopes.append(scope)
                    yield scope
            monkeypatch.setattr(identities_handle, 'prepare_scope', observe_scope)
            with runner._call_lock:
                context = prepare_current()
                generation = context['token'][1]
                enabled = capture_outputs()
                exact(enabled, baseline)
                assert len(scopes) == 1
                cache_tensors = [qkv._packers[0]._packed]
                if mlp._candidates[0]._packed is not None:
                    cache_tensors.append(mlp._candidates[0]._packed)
                cache_tensors.extend(site._weight for site in lt.layers[0].sites.values())
                addresses = tuple(tensor.data_ptr() for tensor in cache_tensors)
                model.layer.self_attn.q_proj.weight.add_(.125)
                context = prepare_current()
                assert context['token'][1] != generation
                assert tuple(tensor.data_ptr() for tensor in cache_tensors) == addresses
                updated = capture_outputs()
                assert not torch.equal(updated[0], baseline[0])
                # Check independent private caches. Native full signatures must
                # remain unchanged; the new scope must still detect corruption.
                corruptions = [('qkv_packed', qkv._packers[0]._packed)]
                if mlp._candidates[0]._packed is not None:
                    corruptions.append(('mlp_packed', mlp._candidates[0]._packed))
                kn_site = next(site for site in lt.layers[0].sites.values()
                               if site.spec['layout'] == 'contiguous_kn')
                corruptions.append(('lt_kn', kn_site._weight))
                rejected = []
                for name, tensor in corruptions:
                    generation, executions = model._inference_cache.generation, len(run_calls)
                    collections = len(collected)
                    tensor.add_(1)
                    with pytest.raises(RuntimeError, match='cache'):
                        prepare_current()
                    assert_scope_expired()
                    assert len(collected) == collections + 1
                    assert model._inference_cache.generation == generation
                    assert len(run_calls) == executions  # No capture/replay on failed prepare.
                    rejected.append(name)
                    model.invalidate_inference_caches(weights_changed=False)
                    prepare_current()
                    assert tuple(tensor.data_ptr() for tensor in cache_tensors) == addresses
                    exact(tuple(value.clone() for value in run()), updated)
                stream.synchronize()
            # Updated-weight lookup-off control, same allocations and native
            # collector, after all enabled scopes and graphs have been retired.
            identities_handle.uninstall()
            identities_handle = None
            with runner._call_lock:
                prepare_current()
                updated_off = capture_outputs()
                exact(updated, updated_off)
            evidence = dict(profile=profile, expert_layers=1, compiled=False,
                            original_fields=6, native_collections=len(collected), scopes=len(scopes),
                            scope_expired_before_capture=True, private_corruptions_rejected=rejected,
                            all_output_boundaries=list(labels), lookup_on_off_bitwise=True,
                            updated_on_off_bitwise=True, replay_bitwise=True, packed_addresses_preserved=True,
                            native=collector.manifest())
            record_property('request_identity_cuda_evidence', json.dumps(evidence, sort_keys=True))
    finally:
        torch.cuda.synchronize()
        runner.close()
        if identities_handle is not None and identities_handle.enabled: identities_handle.uninstall()
        if native is not None and native.enabled: native.uninstall()
        close_case(qkv, mlp, lt, runner)
        torch.backends.cuda.matmul.allow_tf32 = before_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before_bf16_reduction
