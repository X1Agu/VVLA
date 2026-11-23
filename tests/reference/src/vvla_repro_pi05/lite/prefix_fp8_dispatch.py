"""Explicit report-selected installation, never a hardware-based fallback.

Only the original report's SHA-verified variant selects an owner. Each owner
independently re-reads and fully checks that report against the actual runtime,
source, libraries and device before installing anything. Dispatch itself makes
no backend, model-precision or quality claim.
"""
import hashlib
import json
from pathlib import Path


def install_prefix_fp8(model, evidence):
    from .prefix_fp8_evidence import validate_evidence_config
    validate_evidence_config(evidence)
    raw = Path(evidence['backend_report']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence['backend_report_sha256']:
        raise ValueError('Prefix FP8 dispatch requires the original report byte SHA')
    payload = json.loads(raw)
    if type(payload) is not dict:
        raise ValueError('Prefix FP8 dispatch requires an explicit backend report object')
    variant = payload.get('quantizer_variant')
    if variant == 'triton-eager-rn-native-sm120-v1':
        from .prefix_fp8 import install_prefix_fp8 as selected
    elif variant == 'sm89-down-scalar-fp32-native-v1':
        from .prefix_fp8_down import install_prefix_fp8_down as selected
    else:
        raise ValueError('Unsupported explicit prefix FP8 backend report variant')
    return selected(model, evidence)
