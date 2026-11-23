"""CPU contracts for the default-off suffix backend; never launch a GPU kernel."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_repro_pi05.lite.attention import Pi05FlashAttention, prepare_pi05_static_metadata, prepare_pi05_attention_metadata
from vvla_repro_pi05.lite.suffix_attention import (
    INSTALLATION, Pi05SplitSuffixAttention, _allocate_workspace, _validate_call, install_suffix_attention,
)


def _inputs(prefix=7, *, eager=False):
    q = torch.empty((1, 8, 50, 256), dtype=torch.bfloat16)
    pk, pv = [torch.empty((1, 1, prefix, 256), dtype=torch.bfloat16) for _ in range(2)]
    sk, sv = [torch.empty((1, 1, 50, 256), dtype=torch.bfloat16) for _ in range(2)]
    queries = torch.ones((1, 50), dtype=torch.bool)
    prefix_keep = (torch.arange(prefix) % 3 != 1)[None]
    keys = torch.cat((prefix_keep, queries), 1)
    if eager:
        visible = keys[:, None, None, :].expand(-1, 1, 50, -1)
        metadata = prepare_pi05_attention_metadata(torch.where(visible, 0.0, -2.3819763e38), prefix + 50)
    else:
        metadata = prepare_pi05_static_metadata(queries, keys, all_queries_valid=True)
    return (q, pk, pv, sk, sv, metadata), keys


@pytest.mark.parametrize("eager", [False, True])
def test_existing_metadata_maps_live_logical_keys_to_exact_original_physical_order(eager):
    args, keys = _inputs(eager=eager)
    m = _validate_call(*args, require_cuda=False)
    count = int(m['cu_k'][1])
    physical = m['k_indices'][:count]
    torch.testing.assert_close(physical, keys.flatten().nonzero().flatten(), rtol=0, atol=0)
    # Suffix begins at physical capacity7, not at live-prefix count5.
    assert physical[-50:].tolist() == list(range(7, 57))
    assert physical.tolist() == sorted(physical.tolist())


def test_metadata_refresh_is_consumed_without_caching_values_or_host_tensor_reads(monkeypatch):
    args, keys = _inputs()
    old = args[-1]
    keys[0, :7] = torch.tensor([False, True, False, False, True, False, False])
    new = prepare_pi05_static_metadata(torch.ones((1, 50), dtype=torch.bool), keys, all_queries_valid=True)
    addresses = {k: v.data_ptr() for k, v in old.extra.items() if isinstance(v, torch.Tensor)}
    for key, value in new.extra.items():
        if isinstance(value, torch.Tensor):
            old.extra[key].copy_(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('Device data must not be copied or inspected by validation')
    with monkeypatch.context() as patch:
        for name in ('cpu', 'numpy', 'item', '__bool__'):
            patch.setattr(torch.Tensor, name, forbidden)
        result = _validate_call(*args, require_cuda=False)
    assert result is old.extra
    assert addresses == {k: v.data_ptr() for k, v in result.items() if isinstance(v, torch.Tensor)}
    assert int(result['cu_k'][1]) == 52
    assert result['k_indices'][:52].tolist() == [1, 4, *range(7, 57)]


def test_bf16_geometry_real_token_strides_and_explicit_unsupported_rejection():
    args, _ = _inputs()
    q, pk, pv, sk, sv, metadata = args
    packed = torch.empty((1, 50, 2560), dtype=torch.bfloat16)
    q2, sk2, sv2 = (t.view(1, 50, -1, 256).transpose(1, 2) for t in packed.split((2048, 256, 256), -1))
    assert sv2.stride(2) == 2560
    _validate_call(q2, pk, pv, sk2, sv2, metadata, require_cuda=False)
    for bad in (
        (q[:, :, :49], pk, pv, sk, sv, metadata),
        (q.expand(2, -1, -1, -1), pk, pv, sk, sv, metadata),
        (q, pk.float(), pv, sk, sv, metadata),
        (q, pk.expand(-1, 2, -1, -1), pv, sk, sv, metadata),
        (q, pk, pv, sk, torch.empty((1, 1, 50, 512), dtype=torch.bfloat16)[..., ::2], metadata),
    ):
        with pytest.raises(ValueError):
            _validate_call(*bad, require_cuda=False)
    metadata.extra['all_queries_valid'] = False
    with pytest.raises(ValueError, match='all_queries_valid'):
        _validate_call(*args, require_cuda=False)


@pytest.mark.parametrize('splits', [0, 3, 16, True, 4.0])
def test_split_options_do_not_silently_select_another_algorithm(splits):
    with pytest.raises(ValueError, match='split count'):
        Pi05SplitSuffixAttention(splits=splits)


def test_workspace_and_outputs_are_owned_per_call_and_no_silent_cpu_fallback():
    first = _allocate_workspace('cpu', 4)
    second = _allocate_workspace('cpu', 4)
    assert not ({t.data_ptr() for t in first} & {t.data_ptr() for t in second})
    assert first[0].shape == (4, 8, 50, 256) and first[0].dtype == torch.float32
    assert first[-1].shape == (1, 50, 8, 256) and first[-1].dtype == torch.bfloat16
    assert first[-1].transpose(1, 2).transpose(1, 2).reshape(1, 50, 2048).data_ptr() == first[-1].data_ptr()
    backend = Pi05SplitSuffixAttention()
    assert not list(backend.parameters()) and not list(backend.buffers())
    with pytest.raises(AttributeError):
        backend.splits = 8
    args, _ = _inputs()
    with torch.no_grad(), pytest.raises(ValueError, match='requires CUDA'):
        backend.forward_split(*args)
    with pytest.raises(ValueError, match='without gradients'):
        backend.forward_split(*args)
    with pytest.raises(RuntimeError, match='before core KV concatenation'):
        backend(*args)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_horizon = 50
        self.sentinel = nn.Parameter(torch.ones(1))
        self.paligemma_with_expert = nn.Module()
        inner = self.paligemma_with_expert
        inner.prefix_attention = Pi05FlashAttention(1 / 16)
        inner.suffix_attention = Pi05FlashAttention(1 / 16)
        inner.gemma_expert = nn.Module()
        inner.gemma_expert.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=1, head_dim=256)
        self.generation = 0
        self.eval()

    def invalidate_inference_caches(self, **kwargs):
        self.generation += 1


def test_install_preserves_prefix_parameters_and_requires_idle_clear_graph_lifecycle():
    from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
    model = _Model()
    inner = model.paligemma_with_expert
    prefix, suffix = inner.prefix_attention, inner.suffix_attention
    parameters = {k: id(v) for k, v in model.named_parameters()}
    state = tuple(model.state_dict())
    assert not hasattr(suffix, 'forward_split') and not hasattr(model, INSTALLATION)
    runner = Pi05NativeGraphRunner(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):
        install_suffix_attention(model)
    runner.clear()
    runner._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='active'):
            install_suffix_attention(model)
    finally:
        runner._call_lock.release()
    handle = install_suffix_attention(model, splits=2, block_q=32)
    assert model.generation == 1 and inner.prefix_attention is prefix
    assert inner.suffix_attention.splits == 2 and inner.suffix_attention.block_q == 32
    assert {k: id(v) for k, v in model.named_parameters()} == parameters
    assert tuple(model.state_dict()) == state
    manifest = handle.manifest()
    assert manifest['splits'] == 2 and manifest['causal'] is False
    with pytest.raises(ValueError, match='already installed'):
        install_suffix_attention(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):
        handle.uninstall()
    assert handle.manifest() == manifest
    runner.close()
    handle.uninstall()
    assert inner.prefix_attention is prefix and inner.suffix_attention is suffix
    assert model.generation == 2 and not handle.enabled and not hasattr(model, INSTALLATION)


def test_install_rejects_other_geometry_and_detects_external_dispatch_change():
    model = _Model()
    model.action_horizon = 51
    with pytest.raises(ValueError, match='horizon50'):
        install_suffix_attention(model)
    model.action_horizon = 50
    handle = install_suffix_attention(model)
    model.paligemma_with_expert.suffix_attention = nn.Identity()
    with pytest.raises(RuntimeError, match='modified externally'):
        handle.uninstall()


@pytest.mark.parametrize('use_split', [False, True])
def test_actual_core_dispatch_is_after_rope_and_default_cat_semantics_are_unchanged(monkeypatch, use_split):
    import vvla_repro_pi05.models.pi05.modeling_pi05 as core
    class Norm(nn.Module):
        def forward(self, x, cond, modulation=None):
            return x, None
    layer = nn.Module()
    layer.input_layernorm = Norm()
    layer.post_attention_layernorm = Norm()
    layer.self_attn = nn.Module()
    layer.self_attn.head_dim = 4
    layer.self_attn.num_key_value_groups = 1
    for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        setattr(layer.self_attn, name, nn.Linear(4, 4, bias=False, dtype=torch.bfloat16))
    layer.mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    object.__setattr__(layer.mlp, 'up_proj', layer.mlp)
    expert = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    prefix = (torch.ones((1, 1, 2, 4)), torch.full((1, 1, 2, 4), 2.0))
    rotated, called = [], []
    def rope(q, k, *args, **kwargs):
        rotated[:] = [q + 1, k + 2]
        return tuple(rotated)
    monkeypatch.setattr(core, 'apply_rotary_pos_emb', rope)
    def split(q, pk, pv, sk, sv, metadata):
        assert q is rotated[0] and sk is rotated[1]
        assert pk is prefix[0] and pv is prefix[1]
        called.append('split')
        return q
    def joined(q, k, v, *args, **kwargs):
        assert q is rotated[0]
        torch.testing.assert_close(k[:, :, :2], prefix[0].bfloat16())
        torch.testing.assert_close(v[:, :, :2], prefix[1].bfloat16())
        torch.testing.assert_close(k[:, :, 2:], rotated[1])
        called.append('default')
        return q
    monkeypatch.setattr(core, '_attend', joined)
    if use_split:
        def forbidden(*args, **kwargs):
            raise AssertionError('Split branch must bypass concatenation and default attention')
        monkeypatch.setattr(core.torch, 'cat', forbidden)
        monkeypatch.setattr(core, '_attend', forbidden)
    with torch.no_grad():
        result = core._compute_layer_suffix_only(
            0, torch.randn((1, 3, 4)), prefix, None, None, expert, None,
            attention=SimpleNamespace(forward_split=split) if use_split else None,
            rotary=(torch.ones((1, 3, 4)), torch.zeros((1, 3, 4))),
        )
    assert result.shape == (1, 3, 4) and torch.isfinite(result).all()
    assert called == ['split' if use_split else 'default']
