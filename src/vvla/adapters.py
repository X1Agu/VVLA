"""CPU validation shared by installed backends; model semantics stay explicit."""
import numpy as np
from .errors import InputError


def validate_batch(observations, seeds, *, sizes, seed_bits=63):
    if type(observations) is not list or len(observations) not in sizes or any(type(o) is not dict for o in observations):
        raise InputError(f"Expected raw observation dictionaries with batch size in {list(sizes)}")
    if type(seeds) is not list or len(seeds) != len(observations) or any(type(s) is not int or not 0 <= s < 2**seed_bits for s in seeds):
        raise InputError(f"Expected one uint{seed_bits} seed for each real request")


def image(value, *, name, allow_float=False):
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise InputError(f"{name} must be a CPU HWC RGB array")
    if value.dtype == np.uint8:
        return
    if allow_float and value.dtype.kind == "f" and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1:
        return
    raise InputError(f"{name} must use uint8" + (" or finite floats in [0,1]" if allow_float else ""))


def state(value, width):
    result = np.asarray(value)
    if result.shape != (width,) or result.dtype.kind not in "fi" or not np.isfinite(result).all():
        raise InputError(f"State must contain {width} finite numeric values")

