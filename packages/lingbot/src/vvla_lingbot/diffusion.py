"""Numerical contracts for the LingBot clock and action integrator."""
import torch


def sampling_clock(steps, dtype):
    dt = torch.tensor(-1.0 / steps, dtype=dtype)
    time = torch.tensor(1.0, dtype=dtype)
    values = []
    while bool(time >= -dt / 2):
        values.append(time.clone())
        time += dt
    return dt, values


def euler_update(latent, velocity, dt):
    # The multiplication and addition both use the accumulator's precision.
    # In the BF16 recipe each operation rounds separately, as in upstream.
    latent.add_(dt * velocity.to(latent.dtype))
    return latent
