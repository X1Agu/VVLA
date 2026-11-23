"""Explicit report dispatch contracts; fake installers never authorize a model."""
import hashlib
import json
import sys
from types import ModuleType

import pytest

from vvla_repro_pi05.lite.prefix_fp8_dispatch import install_prefix_fp8


def evidence(tmp_path, payload):
    raw = json.dumps(payload, sort_keys=True).encode()
    path = tmp_path / 'backend.json'
    path.write_bytes(raw)
    return dict(backend_report=str(path), backend_report_sha256=hashlib.sha256(raw).hexdigest(),
        bf16_control={key: 'b' * 64 for key in ('recipe_sha256', 'config_sha256', 'identity_sha256', 'content_sha256')})


def installers(monkeypatch):
    calls = []
    for name, function in (('prefix_fp8', 'install_prefix_fp8'), ('prefix_fp8_down', 'install_prefix_fp8_down')):
        module = ModuleType('vvla_repro_pi05.lite.' + name)
        def selected(model, value, name=name):
            calls.append((name, model, value))
            return name
        setattr(module, function, selected)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize(('variant', 'owner'), [
    ('triton-eager-rn-native-sm120-v1', 'prefix_fp8'),
    ('sm89-down-scalar-fp32-native-v1', 'prefix_fp8_down')])
def test_only_sha_verified_explicit_variant_selects_owner_not_hardware(tmp_path, monkeypatch, variant, owner):
    calls = installers(monkeypatch)
    value = evidence(tmp_path, {'quantizer_variant': variant, 'capability': ['deliberately ignored at dispatch']})
    model = object()
    assert install_prefix_fp8(model, value) == owner
    assert calls == [(owner, model, value)]


@pytest.mark.parametrize('payload', [[], {}, {'quantizer_variant': 'triton-eager-rn'},
    {'quantizer_variant': 'unknown', 'capability': [8, 9]}, {'quantizer_variant': None}])
def test_unknown_variant_cannot_fall_back_to_either_owner(tmp_path, monkeypatch, payload):
    calls = installers(monkeypatch)
    with pytest.raises(ValueError):
        install_prefix_fp8(object(), evidence(tmp_path, payload))
    assert calls == []


def test_byte_sha_is_checked_before_json_and_owner_selection(tmp_path, monkeypatch):
    calls = installers(monkeypatch)
    value = evidence(tmp_path, {'quantizer_variant': 'sm89-down-scalar-fp32-native-v1'})
    (tmp_path / 'backend.json').write_bytes(b'not even JSON')
    with pytest.raises(ValueError, match='original report byte SHA'):
        install_prefix_fp8(object(), value)
    assert calls == []


def test_selected_owner_failure_is_not_swallowed_or_retried_on_another_backend(tmp_path, monkeypatch):
    calls = installers(monkeypatch)
    def unsupported(model, value):
        raise RuntimeError('new owner independent full evidence/runtime check failed')
    sys.modules['vvla_repro_pi05.lite.prefix_fp8_down'].install_prefix_fp8_down = unsupported
    with pytest.raises(RuntimeError, match='independent full evidence'):
        install_prefix_fp8(object(), evidence(tmp_path, {'quantizer_variant': 'sm89-down-scalar-fp32-native-v1'}))
    assert calls == []
