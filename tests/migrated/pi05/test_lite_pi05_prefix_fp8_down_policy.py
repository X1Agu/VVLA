"""Actual policy/explicit dispatch, with fictional installers and CPU stubs."""
import hashlib
import json
import sys

import pytest

from vvla_repro_pi05.lite import pi05
from vvla_repro_pi05.lite.prefix_fp8_dispatch import install_prefix_fp8 as real_dispatch
from test_lite_pi05_prefix_fp8_policy import evidence, enabled, model_processor, module, rig


def selected_evidence(tmp_path, variant):
    value = evidence()
    raw = json.dumps({'quantizer_variant': variant}).encode()
    path = tmp_path / 'backend.json'
    path.write_bytes(raw)
    value.update(backend_report=str(path), backend_report_sha256=hashlib.sha256(raw).hexdigest())
    return value


@pytest.mark.parametrize('variant', ['triton-eager-rn-native-sm120-v1', 'sm89-down-scalar-fp32-native-v1'])
def test_real_policy_uses_verified_variant_preserves_same_option_metadata_and_owner_close(tmp_path, monkeypatch, variant):
    events, handles = rig(monkeypatch)
    fake_install = sys.modules['vvla_repro_pi05.lite.prefix_fp8'].install_prefix_fp8
    selected = []
    def down_install(model, value):
        selected.append('down17')
        result = fake_install(model, value)
        result.document = dict(precision='mixed_bf16_fp8_prefix', sites=['fictional_down17'],
                              scope='prefix_mlp_down_layers_0_16')
        return result
    module(monkeypatch, 'prefix_fp8_down', install_prefix_fp8_down=down_install)
    module(monkeypatch, 'prefix_fp8_dispatch', install_prefix_fp8=real_dispatch)
    options = enabled()
    options['prefix_fp8_evidence'] = selected_evidence(tmp_path, variant)
    policy = pi05.Pi05LitePolicy(*model_processor(), **options)
    try:
        assert selected == (['down17'] if variant.startswith('sm89-') else [])
        assert events == ['prefix.install', 'native.install', 'runner.init', 'prefix.bind']
        metadata = policy.metadata()
        assert metadata['precision'] == 'mixed_bf16_fp8_prefix'
        assert metadata['quantization_manifest'] == handles[0].manifest()
        assert metadata['recipe_components']['bf16_control'] == options['prefix_fp8_evidence']['bf16_control']
        assert metadata['recipe_sha256'] == pi05._json_hash(metadata['recipe_components'])
        assert policy.validate_precision_manifest() == handles[0].manifest()
        assert policy.precision_backend_report() is handles[0].raw_report
    finally:
        policy.close()
    assert events[-3:] == ['runner.close', 'prefix.uninstall', 'native.uninstall']


@pytest.mark.parametrize('failure', ['retained_install', 'runner', 'bind'])
def test_down_dispatch_keeps_policy_constructor_rollback_contract(tmp_path, monkeypatch, failure):
    events, handles = rig(monkeypatch, failure=failure)
    fake_install = sys.modules['vvla_repro_pi05.lite.prefix_fp8'].install_prefix_fp8
    module(monkeypatch, 'prefix_fp8_down', install_prefix_fp8_down=fake_install)
    module(monkeypatch, 'prefix_fp8_dispatch', install_prefix_fp8=real_dispatch)
    options = enabled()
    options['prefix_fp8_evidence'] = selected_evidence(tmp_path, 'sm89-down-scalar-fp32-native-v1')
    model, processor = model_processor()
    failed = pi05.Pi05LitePolicy.__new__(pi05.Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        failed.__init__(model, processor, **options)
    assert failed._closed and failed._prefix_fp8_installation is None
    assert not hasattr(model, '_lite_prefix_fp8_installation')
    assert 'prefix.uninstall' in events
    if failure == 'bind':
        assert events.index('runner.close') < events.index('prefix.uninstall') < events.index('native.uninstall')
