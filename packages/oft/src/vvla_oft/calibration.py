"""Validate immutable calibration evidence before packing NVFP4 weights."""
import hashlib
import json
import math
from pathlib import Path

def load_calibration(config, checkpoint_revision):
    enabled = config.get('mlp_precision', 'bf16').startswith('nvfp4')
    path, expected = config.get('calibration_path'), config.get('calibration_sha256')
    if not enabled:
        if path is not None or expected is not None:
            raise ValueError('Unexpected calibration on non-NVFP4 recipe')
        return None
    if not path or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('NVFP4 requires immutable calibration evidence')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Calibration SHA256 mismatch')
    data = json.loads(raw)
    if data.get('status') != 'complete' or data.get('checkpoint_revision') != checkpoint_revision:
        raise ValueError('Calibration is incomplete or belongs to another checkpoint')
    maxima = data.get('maxima')
    if not isinstance(maxima, dict) or not maxima:
        raise ValueError('Missing activation calibration')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in maxima.values()):
        raise ValueError('Activation maxima must be positive and finite')
    return maxima
