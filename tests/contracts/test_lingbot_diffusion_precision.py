import pytest
import torch

from vvla_lingbot.diffusion import euler_update, sampling_clock
from vvla_lingbot.presets import resolve_recipe


@pytest.mark.parametrize('dtype,midpoint', [
    (torch.float32, 0.49999991059303284), (torch.bfloat16, 0.494140625),
])
def test_ten_repeated_additions(dtype, midpoint):
    dt, times = sampling_clock(10, dtype)
    assert len(times) == 10
    assert all(t.dtype == dtype for t in times)
    assert times[5].item() == midpoint
    assert times[0].item() == 1.0
    assert times[-1].item() + dt.item() < -dt.item() / 2


def test_euler_retains_float32_carry_between_network_calls():
    noise = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    v = torch.tensor([0.01, -0.01], dtype=torch.bfloat16)
    full, rounded = noise.float(), noise.clone()
    dt, _ = sampling_clock(10, torch.float32)
    expected = noise.float()
    for _ in range(10):
        euler_update(full, v, dt)
        euler_update(rounded, v, dt.bfloat16())
        expected = expected + v.float() * dt
    torch.testing.assert_close(expected, full, rtol=0, atol=0)
    assert torch.equal(noise, rounded)
    assert not torch.equal(full.bfloat16(), rounded)


def test_mixed_recipes_reach_both_pool_types_and_preserve_bf16_defaults():
    base = resolve_recipe('bf16')
    assert base['resolved'] == 'posttopk-bf16'
    assert 'denoise_precision' not in base['options']
    for precision in ('fp32-clock', 'fp32-clock-accum', 'fp32-heads', 'fp16-expert', 'fp16-prefix-expert'):
        for prefix, batch, capacity in [('cache32', False, 32), ('true-batch', True, 1)]:
            name = prefix + '-posttopk-' + precision
            recipe = resolve_recipe(name)
            assert recipe['resolved'] == name and recipe['posttopk']
            assert recipe['options']['denoise_precision'] == precision
            assert recipe['true_batch'] == batch and not recipe['customized']
            assert recipe['options']['graph_cache_capacity'] == capacity
