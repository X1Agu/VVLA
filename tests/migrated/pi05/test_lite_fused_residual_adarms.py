"""Deferred Colab tests for the single attention-residual/postnorm fusion."""
import pytest
import torch

from vvla_repro_pi05.lite.fused_adarms import fused_adarms, install_fused_adarms
from vvla_repro_pi05.lite.fused_gated_residual import fused_gated_residual, install_gated_residual
from vvla_repro_pi05.lite.fused_residual_adarms import (
    INSTALLATION, DISPATCH, fused_residual_adarms, install_residual_adarms,
)
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.models.pi05.modeling_pi05 import _gated_residual
from test_lite_fused_adarms import SmallExpert, norm


def install(model):
    adarms = install_fused_adarms(model, fuse_reduction=True)
    residual = install_gated_residual(model)
    combined = install_residual_adarms(model)
    return adarms, residual, combined


def close(handles):
    for handle in reversed(handles):
        handle.uninstall()


def bitwise(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    torch.testing.assert_close(actual.contiguous().view(dtype), expected.contiguous().view(dtype), rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(2, 8), (2, 5, 8)])
@pytest.mark.parametrize('precomputed', [False, True])
def test_cpu_fallback_keeps_rounding_broadcast_gate_and_gradients(shape, precomputed):
    model = SmallExpert()
    handles = install(model)
    layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
    r = torch.randn(shape, requires_grad=True)
    out = torch.randn(shape).bfloat16().requires_grad_()
    incoming_mod = torch.randn(2, 24, requires_grad=True)
    gate = incoming_mod.chunk(3, -1)[2]
    if len(shape) == 3:
        gate = gate.unsqueeze(1)
    cond = torch.randn(2, 8, requires_grad=True)
    modulation = layer.post_attention_layernorm.dense(cond) if precomputed else None
    expected_r = _gated_residual(r, out, gate)
    expected_norm, expected_gate = layer.post_attention_layernorm(expected_r, cond, modulation=modulation)
    actual = getattr(layer, DISPATCH)(r, out, gate, cond, modulation=modulation)
    for left, right in zip(actual, (expected_r, expected_norm, expected_gate), strict=True):
        bitwise(left, right)
    variables = (r, out, incoming_mod, cond, layer.post_attention_layernorm.dense.weight)
    expected_grads = torch.autograd.grad(sum(v.float().square().sum() for v in (expected_r, expected_norm, expected_gate)),
                                         variables, retain_graph=True)
    actual_grads = torch.autograd.grad(sum(v.float().square().sum() for v in actual), variables)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        bitwise(left, right)
    assert actual[0].dtype == actual[2].dtype == torch.float32
    assert actual[1].dtype == torch.bfloat16
    assert actual[0].data_ptr() != r.data_ptr()
    close(handles)


def test_installation_preserves_modules_parameters_state_and_requires_prerequisites():
    model = SmallExpert()
    with pytest.raises(ValueError, match='requires existing'):
        install_residual_adarms(model)
    a = install_fused_adarms(model, fuse_reduction=False)
    r = install_gated_residual(model)
    with pytest.raises(ValueError, match='full-reduction'):
        install_residual_adarms(model)
    r.uninstall()
    a.uninstall()
    parameters = {name: id(value) for name, value in model.named_parameters()}
    modules = {name: id(value) for name, value in model.named_modules()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    handles = install(model)
    combined = handles[-1]
    assert combined.manifest()['pairs_per_layer'] == 1
    assert combined.manifest()['enable_fp_fusion'] is False
    combined.prepare(0)
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert {name: id(value) for name, value in model.named_modules()} == modules
    for name, value in model.state_dict().items():
        bitwise(value, state[name])
    close(handles)
    assert not hasattr(model, INSTALLATION)
    assert all(not hasattr(layer, DISPATCH) for layer in model.paligemma_with_expert.gemma_expert.model.layers)


def test_full_lifecycle_guard_inplace_change_topology_and_dispatch_rejection():
    model = SmallExpert()
    handles = install(model)
    combined = handles[-1]
    layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
    combined.prepare(0)
    with torch.no_grad():
        layer.post_attention_layernorm.dense.bias.add_(.125)
    combined.prepare(1)  # No stale cache exists; original live parameters are consumed.
    old_eps = layer.post_attention_layernorm.eps
    layer.post_attention_layernorm.eps *= 2
    with pytest.raises(ValueError, match='epsilon'):
        combined.prepare(2)
    layer.post_attention_layernorm.eps = old_eps
    original = layer.mlp.up_proj
    layer.mlp.up_proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='consumer dispatch'):
        combined.prepare(3)
    layer.mlp.up_proj = original
    original_forward = layer.post_attention_layernorm.forward
    layer.post_attention_layernorm.forward = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match='dispatch'):
        combined.prepare(4)
    layer.post_attention_layernorm.forward = original_forward
    combined.prepare(5)
    close(handles)


def test_idle_empty_runner_required_for_install_and_uninstall():
    model = SmallExpert()
    a, r = install_fused_adarms(model, fuse_reduction=True), install_gated_residual(model)
    runner = Pi05NativeGraphRunner(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):
        install_residual_adarms(model)
    runner.clear()
    with runner._call_lock, pytest.raises(RuntimeError, match='active'):
        install_residual_adarms(model)
    combined = install_residual_adarms(model)
    runner.entries['live'] = object()
    with pytest.raises(ValueError, match='Clear all live'):
        combined.uninstall()
    runner.close()
    close((a, r, combined))


def test_actual_core_opt_in_uses_one_pair_per_layer_and_default_restored(monkeypatch):
    from test_lite_compilation import _probe_module
    from vvla_repro_pi05.lite import fused_residual_adarms as module
    probe = _probe_module()
    model = probe.small_core()
    a, r = install_fused_adarms(model, fuse_reduction=True), install_gated_residual(model)
    with torch.inference_mode():
        prepared = probe.prepared(model)
        source = {key: value for key, value in prepared.items() if key not in ('inference_context', 'attention_metadata')}
        baseline = model.sample_actions(**source)
    combined = install_residual_adarms(model)
    calls = []
    original = module.fused_residual_adarms
    def recording(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'fused_residual_adarms', recording)
    with torch.inference_mode():
        actual = model.sample_actions(**source)
        bitwise(actual, baseline)
    expert = model.paligemma_with_expert.gemma_expert.model
    assert len(calls) == len(expert.layers) * source['num_steps']
    with torch.no_grad():
        expert.layers[0].post_attention_layernorm.dense.bias.add_(.125)
    with torch.inference_mode():
        changed = model.sample_actions(**source)
    combined.uninstall()
    with torch.inference_mode():
        bitwise(changed, model.sample_actions(**source))
    r.uninstall()
    a.uninstall()


def test_cpu_fullgraph_trace_keeps_tensor_work_and_live_modulation():
    model = SmallExpert()
    handles = install(model)
    layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
    handles[-1].prepare(0)
    fn = getattr(layer, DISPATCH)
    r, out, gate = torch.randn(2, 5, 8), torch.randn(2, 5, 8).bfloat16(), torch.randn(2, 1, 24).chunk(3, -1)[2]
    modulation = torch.randn(2, 24)
    with torch.no_grad():
        compiled = torch.compile(fn, backend='eager', fullgraph=True, dynamic=False)
        first = compiled(r, out, gate, modulation=modulation)
        for left, right in zip(first, fn(r, out, gate, modulation=modulation), strict=True):
            bitwise(left, right)
        modulation.add_(.125)
        for left, right in zip(compiled(r, out, gate, modulation=modulation),
                               fn(r, out, gate, modulation=modulation), strict=True):
            bitwise(left, right)
    close(handles)


def test_policy_flag_is_default_off_strict_and_records_enabled_manifest():
    from test_lite_pi05 import _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    original = _policy()
    assert _split_lite_overrides({})[1]['fused_residual_adarms'] is False
    assert original._lite_optimizations['fused_residual_adarms'] is False
    assert original._residual_adarms_installation is None
    for invalid in (1, 'true', None):
        with pytest.raises(ValueError, match='lite_fused_residual_adarms must be a boolean'):
            _split_lite_overrides({'lite_fused_residual_adarms': invalid})
    model = SmallExpert()
    model.action_horizon, model.action_dim = 2, 4
    policy = Pi05LitePolicy(model, original.processor, device='cpu', cuda_graph=False,
                            fused_adarms=True, fused_adarms_reduction=True,
                            fused_gated_residual=True, fused_residual_adarms=True)
    try:
        metadata = policy.metadata()
        options = metadata['lite_optimizations']
        assert options['fused_residual_adarms'] is True
        assert options['fused_residual_adarms_manifest'] == policy._residual_adarms_installation.manifest()
        assert options['fused_residual_adarms_manifest']['pairs_per_layer'] == 1
        options['fused_residual_adarms_manifest']['pairs_per_layer'] = 99
        assert policy.metadata()['lite_optimizations']['fused_residual_adarms_manifest']['pairs_per_layer'] == 1
    finally:
        policy.close()
        original.close()


@pytest.mark.parametrize('missing', ['fused_adarms', 'fused_adarms_reduction', 'fused_gated_residual'])
def test_policy_rejects_each_missing_prerequisite_before_model_or_weight_loading(monkeypatch, tmp_path, missing):
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    monkeypatch.setattr(pi05, '_resolve_checkpoint', lambda *args: tmp_path)
    monkeypatch.setattr(pi05, 'load_lite_config', lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid residual-AdaRMS dependencies reached model allocation')
    monkeypatch.setattr(pi05, '_construct_pi05_model', forbidden)
    overrides = {f'lite_{name}': True for name in
                 ('fused_adarms', 'fused_adarms_reduction', 'fused_gated_residual', 'fused_residual_adarms')}
    overrides[f'lite_{missing}'] = False
    with pytest.raises(ValueError, match='Residual-AdaRMS fusion requires'):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, config_overrides=overrides)


def test_policy_close_retires_graph_then_pair_then_prerequisites(monkeypatch):
    from types import SimpleNamespace
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy
    model = SmallExpert()
    original_modules = {name: id(module) for name, module in model.named_modules()}
    original_parameters = {name: id(value) for name, value in model.named_parameters()}
    # A CUDA device label only constructs the runner. No CUDA API or allocation
    # is used because this lifecycle test never submits inference.
    policy = Pi05LitePolicy(model, SimpleNamespace(config=SimpleNamespace()),
                            device='cuda', cuda_graph=True, fused_adarms=True,
                            fused_adarms_reduction=True, fused_gated_residual=True,
                            fused_residual_adarms=True)
    assert policy.graph_runner._device is None
    policy.graph_runner.entries['retirement_contract'] = object()
    calls = []
    targets = [('graph', policy.graph_runner, 'close'),
               ('pair', policy._residual_adarms_installation, 'uninstall'),
               ('residual', policy._residual_installation, 'uninstall'),
               ('adarms', policy._adarms_installation, 'uninstall')]
    for name, target, method in targets:
        original = getattr(target, method)
        def record_then_call(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(target, method, record_then_call)
    policy.close()
    policy.close()  # Idempotent policy cleanup does not double-uninstall.
    assert calls == ['graph', 'pair', 'residual', 'adarms']
    assert not policy.graph_runner.entries and policy.graph_runner._closed
    assert not hasattr(model, INSTALLATION)
    assert not hasattr(model, '_lite_gated_residual_installation')
    assert not hasattr(model, '_lite_fused_adarms_installation')
    assert {name: id(module) for name, module in model.named_modules()} == original_modules
    assert {name: id(value) for name, value in model.named_parameters()} == original_parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only remote experiment owner executes CUDA')
@pytest.mark.parametrize('batch,width,full_gate,out_dtype,mod_dtype', [
    (1, 1024, False, torch.bfloat16, torch.float32),
    (2, 1024, True, torch.bfloat16, torch.float32),
    (2, 1000, False, torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize('compiled', [False, True])
def test_cuda_bitwise_existing_pair_capture_refresh_and_no_fma(batch, width, full_gate, out_dtype, mod_dtype, compiled):
    from vvla_repro_pi05.lite.compilation import INDUCTOR_OPTIONS
    module = norm(width, device='cuda')
    shape = (batch, 50, width)
    r = torch.randn(shape, device='cuda')
    out = torch.randn(shape, device='cuda', dtype=out_dtype)
    incoming_mod = torch.randn(batch, 50 if full_gate else 1, 3 * width, device='cuda')
    gate = incoming_mod.chunk(3, -1)[2]
    modulation = torch.randn(batch, 3 * width, device='cuda', dtype=mod_dtype) / 8
    def reference():
        updated = fused_gated_residual(r, out, gate)
        normalized, next_gate = fused_adarms(module, updated, modulation=modulation,
                                              output_dtype=torch.bfloat16, fuse_reduction=True)
        return updated, normalized, next_gate
    def run():
        return fused_residual_adarms(module, r, out, gate, modulation=modulation)
    graph = None
    try:
        with torch.inference_mode():
            # Cancellation distinguishes rounded multiply then add from illicit FMA.
            gate[0, 0, 0] = 1 + 2**-23
            out[0, 0, 0] = 1 + 2**-7
            r[0, 0, 0] = -(gate[0, 0, 0] * out[0, 0, 0])
            r[0, 0, 1], gate[0, 0, 1], out[0, 0, 1] = -0.0, -0.0, 1.0
            r[0, 0, 2], gate[0, 0, 2], out[0, 0, 2] = 2**-149, 0.0, 1.0
            expected = tuple(value.clone() for value in reference())
            assert expected[0][0, 0, 0].item() == 0.0
            for left, right in zip(run(), expected, strict=True):
                bitwise(left, right)
            execute = torch.compile(run, backend='inductor', fullgraph=True, dynamic=False,
                                    options=dict(INDUCTOR_OPTIONS)) if compiled else run
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute()
                warm = execute()
                for left, right in zip(warm, expected, strict=True):
                    bitwise(left, right)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = execute()
            with torch.cuda.stream(stream):
                graph.replay()
                retained = tuple(value.clone() for value in captured)
                graph.replay()
                for left, right in zip(captured, retained, strict=True):
                    bitwise(left, right)
                r.mul_(.25)
                gate.add_(.125)
                modulation.add_(.0625)
                refreshed = reference()
                graph.replay()
                for left, right in zip(captured, refreshed, strict=True):
                    bitwise(left, right)
                # A caller's previous output remains owned and unchanged.
                for left, right in zip(retained, expected, strict=True):
                    bitwise(left, right)
                stream.synchronize()
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only remote experiment owner executes CUDA')
def test_cuda_cond_dense_parameter_update_is_live_and_original_gate_returned():
    module = norm(1024, device='cuda')
    r, out, gate = (torch.randn(1, 50, 1024, device='cuda'),
                    torch.randn(1, 50, 1024, device='cuda').bfloat16(),
                    torch.randn(1, 1, 3072, device='cuda').chunk(3, -1)[2])
    cond = torch.randn(1, 1024, device='cuda') / 8
    with torch.no_grad():
        for update in (0.0, .125):
            module.dense.bias.add_(update)
            expected_r = fused_gated_residual(r, out, gate)
            expected_norm, expected_gate = fused_adarms(module, expected_r, cond,
                                                        output_dtype=torch.bfloat16, fuse_reduction=True)
            actual = fused_residual_adarms(module, r, out, gate, cond)
            for left, right in zip(actual, (expected_r, expected_norm, expected_gate), strict=True):
                bitwise(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Only remote experiment owner executes CUDA')
@pytest.mark.parametrize('compiled', [False, True])
def test_cuda_real_core_native_graph_pair_parity_inplace_invalidation_and_owned_output(compiled):
    from test_lite_compilation import _probe_module
    from vvla_repro_pi05.lite.layout import HostMaskLayout
    probe = _probe_module()
    model = probe.small_core(attention_backend='torch_fa2').cuda()
    a, r = install_fused_adarms(model, fuse_reduction=True), install_gated_residual(model)
    runners = []
    combined = None
    def runner():
        value = Pi05NativeGraphRunner(model, host_layout_cache=True, compile_sample_actions=compiled)
        runners.append(value)
        return value
    source = dict(images=[torch.rand(1, 3, 4, 4, device='cuda').bfloat16(),
                          -torch.ones(1, 3, 4, 4, device='cuda').bfloat16()],
                  image_masks=[torch.tensor([True], device='cuda'), torch.tensor([False], device='cuda')],
                  lang_tokens=torch.tensor([[2, 3, 4, 0, 0, 0]], device='cuda'),
                  lang_masks=torch.tensor([[True, True, True, False, False, False]], device='cuda'),
                  noise=torch.randn(1, 4, 8, device='cuda'), num_steps=2)
    host = HostMaskLayout.from_processor((True, False), (True, True, True, False, False, False),
                                         source['images'], source['image_masks'], source['lang_masks'])
    try:
        baseline = runner()
        with torch.inference_mode():
            expected = baseline(**source, host_layout=host)
        baseline.close()
        combined = install_residual_adarms(model)
        fused = runner()
        with torch.inference_mode():
            first = fused(**source, host_layout=host)
            retained = first.clone()
            second = fused(**source, host_layout=host)
            bitwise(first, expected)
            bitwise(second, first)
        with torch.no_grad():
            model.paligemma_with_expert.gemma_expert.model.layers[0].post_attention_layernorm.dense.bias.add_(.125)
        with torch.inference_mode():
            changed = fused(**source, host_layout=host)
            assert fused.stats['invalidations'] == 1
            bitwise(first, retained)
        fused.close()
        combined.uninstall()
        combined = None
        updated_baseline = runner()
        with torch.inference_mode():
            bitwise(changed, updated_baseline(**source, host_layout=host))
        updated_baseline.close()
    finally:
        for owner in runners:
            owner.close()
        if combined is not None and combined.enabled:
            combined.uninstall()
        r.uninstall()
        a.uninstall()
