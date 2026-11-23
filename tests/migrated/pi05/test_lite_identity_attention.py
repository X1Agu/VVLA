"""Identity gather candidate: actual row proof, fallback, and graph transitions."""

import pytest
import torch

from vvla_repro_pi05.lite.attention import (
    Pi05FlashAttention, annotate_identity_gathers, identity_gather_signature,
    prepare_pi05_static_metadata, select_packed_qkv,
)
from vvla_repro_pi05.lite.pi05 import _split_lite_overrides


def _metadata(q, k):
    result = prepare_pi05_static_metadata(q, k)
    annotate_identity_gathers(result)
    return result


def test_candidate_is_explicit_boolean_and_leaves_core_overrides_untouched():
    original = {"pack_prefix": True, "lite_elide_identity_gathers": True}
    core, lite = _split_lite_overrides(original)
    assert core == {"pack_prefix": True} and lite == {
        "elide_identity_gathers": True, "host_layout_cache": False, "fused_adarms": False,
        "skip_weight_init": False, "fused_expert_qkv": False,
        "fused_expert_gate_up": False, "fused_expert_gelu": False,
        "fused_gated_residual": False, "compile_sample_actions": False,
        "fused_adarms_reduction": False, "suffix_attention": None,
        "compile_preserve_action_linears": False, "batched_vision": False, "fused_vision_qkv": False, "action_norm_cache": False, "suffix_fa2": None, "single_pass_lifecycle": False, "native_lifecycle": False, "native_lifecycle_variant": "v1", "expert_cublaslt": None, "gpu_uint8_prepare": False, "fused_residual_adarms": False, "request_identity_lookup": False, "packed_input_metadata": False, "vision_fc2_cublaslt": False,
    }
    assert original["lite_elide_identity_gathers"] is True
    assert _split_lite_overrides(None) == ({}, {
        "elide_identity_gathers": False, "host_layout_cache": False, "fused_adarms": False,
        "skip_weight_init": False, "fused_expert_qkv": False,
        "fused_expert_gate_up": False, "fused_expert_gelu": False,
        "fused_gated_residual": False, "compile_sample_actions": False,
        "fused_adarms_reduction": False, "suffix_attention": None,
        "compile_preserve_action_linears": False, "batched_vision": False, "fused_vision_qkv": False, "action_norm_cache": False, "suffix_fa2": None, "single_pass_lifecycle": False, "native_lifecycle": False, "native_lifecycle_variant": "v1", "expert_cublaslt": None, "gpu_uint8_prepare": False, "fused_residual_adarms": False, "request_identity_lookup": False, "packed_input_metadata": False, "vision_fc2_cublaslt": False,
    })
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="must be a boolean"):
            _split_lite_overrides({"lite_elide_identity_gathers": invalid})


def test_b1_prefix_and_suffix_proofs_keep_padding_holes():
    prefix = torch.tensor([[True, True, True, False]])
    suffix = torch.ones((1, 2), dtype=torch.bool)
    p = _metadata(prefix, prefix)
    s = _metadata(suffix, torch.cat((prefix, suffix), 1))
    assert identity_gather_signature((p, s)) == ((True, True), (True, False))
    assert s.extra["k_indices"].tolist() == [0, 1, 2, 4, 5, 3]
    # Equal allocated lengths are insufficient proof, including across batches.
    nonidentity = _metadata(torch.tensor([[True, False], [True, False]]),
                            torch.tensor([[False, True], [True, True]]))
    assert identity_gather_signature((nonidentity,)) == ((False, False),)


@pytest.mark.parametrize("strided_columns", [False, True])
def test_selector_preserves_values_and_only_aliases_proven_compatible_rows(strided_columns):
    qmask = torch.tensor([[True, True, False, False]])
    kmask = torch.tensor([[True, False, True, False, True]])
    metadata = _metadata(qmask, kmask)
    step = 2 if strided_columns else 1
    q = torch.arange(4 * 2 * 8 * step).reshape(1, 4, 2, 8 * step)[..., ::step]
    k = torch.arange(5 * 8 * step).reshape(1, 5, 1, 8 * step)[..., ::step]
    v = k + 5
    base = select_packed_qkv(q, k, v, metadata)
    fast = select_packed_qkv(q, k, v, metadata, elide_identity_gathers=True)
    for expected, actual in zip(base, fast, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (fast[0].data_ptr() == q.data_ptr()) is (not strided_columns)
    assert fast[1].data_ptr() != k.data_ptr() and fast[2].data_ptr() != v.data_ptr()


def test_signature_changes_only_when_captured_gather_branch_must_change():
    prefix2 = torch.tensor([[True, True, False, False]])
    prefix3 = torch.tensor([[True, True, True, False]])
    prefix4 = torch.ones_like(prefix2)
    q = torch.ones((1, 2), dtype=torch.bool)
    signatures = [identity_gather_signature((_metadata(q, torch.cat((p, q), 1)),))
                  for p in (prefix2, prefix3, prefix4)]
    assert signatures[0] == signatures[1] == ((True, False),)
    assert signatures[2] == ((True, True),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["torch_fa2", "flash_attn"])
@pytest.mark.parametrize("layout", ["prefix", "suffix", "arbitrary_b2"])
def test_identity_candidate_bitwise_matches_fa2_before_and_after_capture(backend, layout):
    if backend == "flash_attn":
        pytest.importorskip("flash_attn")
    if layout == "prefix":
        qm = km = torch.arange(64, device="cuda")[None] < 57
    elif layout == "suffix":
        qm = torch.ones((1, 50), dtype=torch.bool, device="cuda")
        km = torch.cat((torch.arange(64, device="cuda")[None] < 57, qm), 1)
    else:
        qm = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
        km = torch.tensor([[True, False, True, True], [False, True, False, True]], device="cuda")
    b, sq = qm.shape
    sk = km.shape[1]
    rng = torch.Generator(device="cuda").manual_seed(21)
    # Match production GQA head dimension; transpose layout mimics RoPE output.
    q = torch.randn((b, 8, sq, 256), device="cuda", dtype=torch.bfloat16, generator=rng).transpose(1, 2)
    k = torch.randn((b, 1, sk, 256), device="cuda", dtype=torch.bfloat16, generator=rng).transpose(1, 2)
    v = torch.randn((b, 1, sk, 256), device="cuda", dtype=torch.bfloat16, generator=rng).transpose(1, 2)
    metadata = _metadata(qm, km)
    base = Pi05FlashAttention(256**-0.5, backend=backend)
    fast = Pi05FlashAttention(256**-0.5, backend=backend, elide_identity_gathers=True)
    expected = base(q, k, v, metadata)
    actual = fast(q, k, v, metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fast(q, k, v, metadata)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fast(q, k, v, metadata)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_recaptures_if_identity_proof_changes_and_refreshes_if_it_does_not():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner

    class SmallAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Pi05FlashAttention(32**-0.5, elide_identity_gathers=True)

        def prepare_inference_context(self, *args, **kwargs):
            return {"token": 0, "empty_pattern": (False,), "packing": None, "positions": None}

        def prepare_position_context(self, *args, **kwargs):
            return None

        def prepare_attention_metadata(self, image_masks, lang_masks, **kwargs):
            return prepare_pi05_static_metadata(lang_masks, lang_masks), None

        def sample_actions(self, images, image_masks, lang_tokens, lang_masks, noise, num_steps,
                           attention_metadata=None, inference_context=None):
            if attention_metadata is None:
                attention_metadata = self.prepare_attention_metadata(image_masks, lang_masks)
            q, k, v = images
            return self.attention(q, k, v, attention_metadata[0])

    model = SmallAttention()
    runner = Pi05NativeGraphRunner(model, elide_identity_gathers=True)
    rng = torch.Generator(device="cuda").manual_seed(9)
    images = [torch.randn((1, 4, h, 32), device="cuda", dtype=torch.bfloat16, generator=rng)
              for h in (4, 2, 2)]
    inputs = dict(images=images, image_masks=[torch.ones(1, device="cuda", dtype=torch.bool)] * 3,
                  lang_tokens=torch.ones((1, 4), device="cuda", dtype=torch.long),
                  noise=torch.zeros((1, 2, 4), device="cuda"), num_steps=2)
    retained = None
    for row, cold in (([1, 1, 0, 0], True), ([1, 1, 1, 0], False), ([1, 0, 1, 0], True)):
        args = dict(inputs, lang_masks=torch.tensor([row], device="cuda", dtype=torch.bool))
        actual = runner(**args)
        expected = model.sample_actions(**args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runner.last_call["cold"] is cold
        if retained is None:
            retained = actual, actual.clone()
        else:
            torch.testing.assert_close(*retained, rtol=0, atol=0)
    assert runner.stats["captures"] == 2 and runner.stats["identity_recaptures"] == 1
    assert runner.stats["metadata_updates"] == 1
    runner.close()
