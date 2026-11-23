"""CPU wiring and failure retirement; fictional owners are not GPU evidence."""
from copy import deepcopy

import pytest

from vvla_repro_pi05.lite import pi05
from test_lite_pi05_prefix_fp8_policy import enabled, model_processor, module, rig


def fusion_evidence():
    return dict(contracts_root='/fictional/executed-contracts', result_sha256='a' * 64,
                source_manifest_sha256='b' * 64)


def fusion_rig(monkeypatch, *, failure=None):
    events, handles = rig(monkeypatch)

    class Fusion:
        def __init__(self, model):
            self.model = model
            self.base = model._lite_prefix_fp8_installation

        def uninstall(self):
            if self.base.runner is not None:
                assert self.base.runner._closed
            assert 'prefix.uninstall' not in events
            events.append('gelu.uninstall')
            self.base.document.pop('gelu_fusion')
            del self.model._lite_prefix_fp8_gelu_installation

    def install(model, evidence):
        assert evidence == fusion_evidence()
        assert 'prefix.install' in events and 'runner.init' not in events
        events.append('gelu.install')
        owner = Fusion(model)
        model._lite_prefix_fp8_gelu_installation = owner
        owner.base.document['gelu_fusion'] = {'fictional': True}
        if failure == 'retained_install':
            raise RuntimeError('injected gelu retained install')
        return owner

    module(monkeypatch, 'prefix_fp8_gelu_owner', install_prefix_fp8_gelu=install)
    return events, handles


def test_disabled_fusion_preserves_default_option_shape_and_owner_absence(monkeypatch):
    assert pi05._split_lite_overrides({}) == pi05._split_lite_overrides({'lite_prefix_fp8_gelu': False})
    events, _ = fusion_rig(monkeypatch)
    policy = pi05.Pi05LitePolicy(*model_processor(), **enabled())
    try:
        assert policy._prefix_fp8_gelu_installation is None
        assert 'prefix_fp8_gelu' not in policy.metadata()['lite_optimizations']
        assert 'gelu_fusion' not in policy.metadata()['quantization_manifest']
        assert 'gelu.install' not in events
    finally:
        policy.close()


@pytest.mark.parametrize('value', [0, 1, None, {}, 'true'])
def test_fusion_option_requires_exact_bool(value):
    with pytest.raises(ValueError, match='must be a boolean'):
        pi05._split_lite_overrides({'lite_prefix_fp8': True, 'lite_prefix_fp8_gelu': value})


def test_fusion_requires_base_owner_and_its_own_explicit_evidence(monkeypatch):
    events, _ = fusion_rig(monkeypatch)
    with pytest.raises(ValueError, match='requires'):
        pi05._split_lite_overrides({'lite_prefix_fp8_gelu': True})
    with pytest.raises(ValueError, match='only allowed'):
        pi05.Pi05LitePolicy(*model_processor(), **enabled(), prefix_fp8_gelu_evidence=fusion_evidence())
    with pytest.raises(ValueError):
        pi05.Pi05LitePolicy(*model_processor(), **enabled(), prefix_fp8_gelu=True)
    assert events == []


def test_fusion_metadata_and_close_preserve_base_and_retire_graph_first(monkeypatch):
    events, handles = fusion_rig(monkeypatch)
    policy = pi05.Pi05LitePolicy(*model_processor(), **enabled(), prefix_fp8_gelu=True,
                               prefix_fp8_gelu_evidence=fusion_evidence())
    assert events == ['prefix.install', 'gelu.install', 'native.install', 'runner.init', 'prefix.bind']
    data = policy.metadata()
    assert data['lite_optimizations']['prefix_fp8_gelu'] is True
    assert data['quantization_manifest']['gelu_fusion'] == {'fictional': True}
    assert policy.validate_precision_manifest() == data['quantization_manifest']
    altered = deepcopy(data)
    altered['quantization_manifest']['gelu_fusion']['fictional'] = False
    assert policy.metadata()['quantization_manifest']['gelu_fusion']['fictional'] is True
    assert policy.precision_backend_report() is handles[0].raw_report
    policy.close()
    assert events[-4:] == ['runner.close', 'gelu.uninstall', 'prefix.uninstall', 'native.uninstall']


def test_retained_partial_fusion_is_removed_before_base_owner(monkeypatch):
    events, _ = fusion_rig(monkeypatch, failure='retained_install')
    model, processor = model_processor()
    failed = pi05.Pi05LitePolicy.__new__(pi05.Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected gelu'):
        failed.__init__(model, processor, **enabled(), prefix_fp8_gelu=True,
                        prefix_fp8_gelu_evidence=fusion_evidence())
    assert failed._closed
    assert failed._prefix_fp8_gelu_installation is None
    assert not hasattr(model, '_lite_prefix_fp8_gelu_installation')
    assert events[-2:] == ['gelu.uninstall', 'prefix.uninstall']
