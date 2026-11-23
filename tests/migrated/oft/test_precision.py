"""Precision coverage and native-kernel contracts; GPU cases use random weights."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vvla_runtime.precision import QuantizedLinear, selected_linears
from vvla_oft.recipe import OFTExecutionRecipe, recipe_from_config


def fixture_components():
    layer = SimpleNamespace(
        mlp=nn.ModuleDict({n: nn.Linear(32, 32, bias=False) for n in ('gate_proj', 'up_proj', 'down_proj')}),
        self_attn=nn.ModuleDict({n: nn.Linear(32, 32, bias=False) for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj')}),
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer]), config=SimpleNamespace(hidden_act='silu')),
        vision_backbone=nn.Sequential(nn.Conv2d(3, 8, 2), nn.Sequential(nn.Linear(32, 32))),
        projector=nn.Sequential(nn.Linear(32, 32)),
    )
    return model, nn.Sequential(nn.Linear(32, 7)), nn.Sequential(nn.Linear(8, 32))


def test_explicit_scope_selects_every_requested_component_and_leaves_convolution():
    model, head, proprio = fixture_components()
    default = selected_linears(model, OFTExecutionRecipe(mlp_precision='fp8'))
    expanded = selected_linears(model, OFTExecutionRecipe(mlp_precision='fp8', fp8_scope='all_linear'), head, proprio)
    assert len(default) == 3
    assert len(expanded) == 11
    assert {name for name, _, _, _ in expanded} == {
        'llama.0.gate_proj', 'llama.0.up_proj', 'llama.0.down_proj',
        'llama.0.attention.q_proj', 'llama.0.attention.k_proj',
        'llama.0.attention.v_proj', 'llama.0.attention.o_proj',
        'vision.1.0', 'multimodal_projector.0', 'action_head.0', 'proprio_projector.0',
    }
    assert isinstance(model.vision_backbone[0], nn.Conv2d)


def test_expanded_scope_cannot_silently_omit_action_head():
    model, _, proprio = fixture_components()
    with pytest.raises(ValueError, match='action_head'):
        selected_linears(model, OFTExecutionRecipe(mlp_precision='fp8', fp8_scope='all_linear'), None, proprio)


@pytest.mark.parametrize("scope,count", [("llama", 7), ("llama_vision", 8)])
def test_attention_scopes_leave_output_head_at_its_original_precision(scope, count):
    model, head, proprio = fixture_components()
    recipe = OFTExecutionRecipe(mlp_precision="fp8", fp8_scope=scope)
    selected = selected_linears(model, recipe, head, proprio)
    assert len(selected) == count
    assert all(name.startswith(("llama.", "vision.")) for name, *_ in selected)


def test_batched_camera_features_preserve_camera_and_prompt_batch_order():
    from vvla_oft.native_adapter import batched_camera_features
    camera = SimpleNamespace(get_num_images_in_input=lambda: 2, use_fused_vision_backbone=True,
                             featurizer=lambda x: x.flatten(2).transpose(1, 2),
                             fused_featurizer=lambda x: 3 * x.flatten(2).transpose(1, 2))
    pixels = torch.arange(3 * 12 * 4 * 4).reshape(3, 12, 4, 4)
    expected = []
    for view in pixels.split(6, dim=1):
        expected.append(torch.cat((camera.featurizer(view[:, :3]), camera.fused_featurizer(view[:, 3:])), dim=-1))
    torch.testing.assert_close(batched_camera_features(camera, pixels), torch.cat(expected, dim=1))


@pytest.mark.parametrize('change', [dict(mlp_precision='bf16', fp8_scope='all_linear'),
    dict(mlp_precision='fp8', fp8_scope='all_linear', fuse_qkv=True),
    dict(reference_compatible=True, fuse_silu_mul=True)])
def test_incompatible_execution_recipes_are_rejected(change):
    with pytest.raises(ValueError):
        OFTExecutionRecipe(**change)


def test_actual_worker_recipe_preserves_all_fusion_flags():
    flags = dict.fromkeys(('fuse_rope', 'fuse_silu_mul', 'fuse_rmsnorm', 'fuse_residual_square'), True)
    recipe = recipe_from_config(dict(backend='vllm_fa2', mlp_precision='fp8', fp8_scope='all_linear', **flags))
    assert all(getattr(recipe, key) for key in flags)
    assert recipe.pack_sequence and recipe.precompute_positions and recipe.fp8_scope == 'all_linear'


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA debug device required')
@pytest.mark.parametrize('k,n', [(32, 32), (8, 32), (32, 7), (17, 19)])
@pytest.mark.parametrize('cpu_reference', [False, True])
def test_fp8_bias_alignment_graph_and_stale_parameters(k, n, cpu_reference):
    torch.manual_seed(17)
    original = nn.Linear(k, n, bias=True, device='cuda', dtype=torch.bfloat16).eval()
    module = QuantizedLinear(original, 'fp8', cpu_reference=cpu_reference).eval()
    assert module.weight.device.type == ('cpu' if cpu_reference else 'cuda')
    assert module.packed_weight.is_cuda and module.bias.is_cuda
    x = torch.randn(2, 8, k, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        actual = module(x)
        # A kernel arithmetic check, not a model-quality acceptance threshold.
        torch.testing.assert_close(actual, original(x), rtol=0.1, atol=0.08)
        assert actual.shape == (2, 8, n) and torch.isfinite(actual).all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): module(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        x.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(captured, module(x), rtol=0, atol=0)
        module.assert_fresh()
        original.bias.add_(1)
        with pytest.raises(RuntimeError, match='bias changed'):
            module.assert_fresh()
        with pytest.raises(ValueError, match='geometry'):
            module(torch.zeros(2, k * 2, device='cuda', dtype=torch.bfloat16))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA debug device required')
def test_cpu_reference_weight_change_invalidates_quantized_recipe():
    original = nn.Linear(32, 32, device='cuda', dtype=torch.bfloat16).eval()
    module = QuantizedLinear(original, 'fp8', cpu_reference=True)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match='weight or bias changed'):
        module.assert_fresh()


def test_shared_fp8_gate_up_quantizes_once_and_preserves_results(monkeypatch):
    from vvla_runtime import precision
    from vvla_runtime.precision import FusedLlamaMLP
    calls = []
    def quantize(x):
        calls.append(x.clone())
        scale = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
        return (x.float()/scale).to(torch.float8_e4m3fn), scale
    monkeypatch.setattr(precision.native, 'quantize_fp8', quantize)
    monkeypatch.setattr(precision.native, 'fp8_mm', lambda a,b,sa,sb: ((a.float()*sa) @ (b.float()*sb)).to(torch.bfloat16))
    def projection():
        m = object.__new__(QuantizedLinear)
        nn.Module.__init__(m)
        m.precision='fp8'; m.in_features=m.padded_in_features=32
        m.out_features=m.padded_out_features=32
        m.register_buffer('packed_weight',torch.randn(32,32).to(torch.float8_e4m3fn))
        m.register_buffer('weight_scale',torch.ones(32,1))
        m.bias=None
        return m.eval()
    gate,up=projection(),projection()
    x=torch.randn(2,3,32,dtype=torch.bfloat16)
    with torch.no_grad():
        expected=(gate(x),up(x)); calls.clear()
        q,scale=gate.quantize_input(x)
        actual=(gate(quantized=q,quant_scale=scale,original_shape=x.shape),
                up(quantized=q,quant_scale=scale,original_shape=x.shape))
        assert len(calls)==1
        for left,right in zip(actual,expected):torch.testing.assert_close(left,right,rtol=0,atol=0)
        with pytest.raises(ValueError,match='FP8 activation'):
            up(quantized=q,quant_scale=scale[:,0],original_shape=x.shape)
    recipe=recipe_from_config(dict(backend='vllm_fa2',mlp_precision='fp8',fuse_silu_mul=True,share_mlp_quant=True,fuse_qkv=True))
    assert recipe.share_mlp_quant and recipe.fuse_qkv and recipe.fp8_scope=='mlp'


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='L4 validation only')
def test_generic_qkv_fusion_executes_on_cuda_and_detects_source_changes():
    from vvla_runtime.gemm import PackedQKV
    attention=nn.Module()
    for name in ('q_proj','k_proj','v_proj'):
        setattr(attention,name,nn.Linear(32,32,bias=False,device='cuda',dtype=torch.bfloat16).eval())
    packed=PackedQKV(attention).eval()
    assert packed.use_packed
    x=torch.randn(1,17,32,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        for actual,name in zip(packed(x),('q_proj','k_proj','v_proj')):
            torch.testing.assert_close(actual,getattr(attention,name)(x))
        attention.q_proj.weight.add_(1)
        with pytest.raises(RuntimeError,match='sources or layout changed'):packed.assert_fresh()
