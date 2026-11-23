"""Remote CPU wiring with fictional owners; no GPU evidence is fabricated."""
from copy import deepcopy
import sys

import pytest

from vvla_repro_pi05.lite import pi05
from vvla_repro_pi05.lite.prefix_fp8 import PrefixFP8Installation as ActualPrefixOwner
from prefix_fp8_inventory_fixtures import make_inventory_bundle
from test_lite_pi05_prefix_fp8_policy import enabled, model_processor, module, rig


def options(**changes):
    result = enabled()
    result.update(native_lifecycle_variant='v3', prefix_fp8_inventory=True)
    result.update(changes)
    return result


def inventory_rig(monkeypatch, *, failure=None, gelu=False):
    events, bases = rig(monkeypatch, failure='runner' if failure == 'runner' else None)
    bundle = make_inventory_bundle()
    dispatch = sys.modules['vvla_repro_pi05.lite.prefix_fp8_dispatch']
    original_install = dispatch.install_prefix_fp8
    owners = []

    def install_base(model, value):
        base = original_install(model, value)
        base._manifest = deepcopy(bundle['quantization_manifest'])
        base._manifest.pop('inventory_reuse')
        base._model = lambda: model
        # Exercise the actual base manifest extension code, rather than
        # manufacturing inventory_reuse in the policy fixture.
        base.manifest = lambda: ActualPrefixOwner._manifest_with_extension(base)
        return base

    monkeypatch.setattr(dispatch, 'install_prefix_fp8', install_base)

    class Native:
        def __init__(self, model): self.model = model
        def manifest(self): return deepcopy(bundle['native_manifest'])
        def uninstall(self):
            assert not hasattr(self.model, '_lite_prefix_fp8_inventory_installation')
            assert 'prefix.uninstall' in events
            events.append('native.uninstall')
            del self.model._lite_native_lifecycle_signature_installation

    def native_install(model, *, collector):
        assert collector == 'v3'
        events.append('native.install')
        native = Native(model)
        model._lite_native_lifecycle_signature_installation = native
        return native

    module(monkeypatch, 'native_lifecycle', install_native_lifecycle_signature=native_install,
           load_native_lifecycle=lambda *, variant: variant)

    class Inventory:
        def __init__(self, model):
            self.model, self.enabled = model, True
            self._native = model._lite_native_lifecycle_signature_installation
            self._prefix = lambda: model._lite_prefix_fp8_installation
            self.document = deepcopy(bundle['extension'])
            self.fail_close = failure == 'close'
        def manifest(self):
            if not self.enabled: raise RuntimeError('fictional retired inventory')
            return deepcopy(self.document)
        def uninstall(self):
            assert all(runner._closed for runner in self.model._graph_runners)
            assert 'prefix.uninstall' not in events and 'native.uninstall' not in events
            if gelu: assert 'gelu.uninstall' not in events
            if self.fail_close:
                self.fail_close = False
                raise RuntimeError('injected inventory retirement')
            events.append('inventory.uninstall')
            self.enabled = False
            del self.model._lite_prefix_fp8_inventory_installation

    def install(model):
        assert events[-1] == 'native.install'
        assert not model._graph_runners
        events.append('inventory.install')
        owner = Inventory(model)
        owners.append(owner)
        model._lite_prefix_fp8_inventory_installation = owner
        if failure == 'retained_install': raise RuntimeError('injected inventory retained install')
        return owner

    module(monkeypatch, 'prefix_fp8_inventory', install_prefix_fp8_inventory=install,
           PrefixFP8InventoryInstallation=Inventory)
    if gelu:
        class Gelu:
            def __init__(self, model):
                self.model, self.base, self.enabled = model, model._lite_prefix_fp8_installation, True
            def manifest(self): return {'fictional_gelu_only': True}
            def uninstall(self):
                assert 'inventory.uninstall' in events
                events.append('gelu.uninstall')
                del self.model._lite_prefix_fp8_gelu_installation
        def install_gelu(model, evidence):
            events.append('gelu.install')
            handle = Gelu(model)
            model._lite_prefix_fp8_gelu_installation = handle
            return handle
        module(monkeypatch, 'prefix_fp8_gelu_owner', install_prefix_fp8_gelu=install_gelu,
               PrefixFP8GeluInstallation=Gelu)
    return events, bases, owners, bundle


def test_absent_and_explicit_false_preserve_option_metadata_and_mixed_recipe(monkeypatch):
    assert pi05._split_lite_overrides({}) == pi05._split_lite_overrides({'lite_prefix_fp8_inventory': False})
    events, _, owners, _ = inventory_rig(monkeypatch)
    snapshots = []
    for kwargs in ({}, {'prefix_fp8_inventory': False}):
        policy = pi05.Pi05LitePolicy(*model_processor(), **options(**{'prefix_fp8_inventory': False, **kwargs}))
        try:
            metadata = policy.metadata()
            assert 'prefix_fp8_inventory' not in metadata['lite_optimizations']
            assert 'prefix_fp8_inventory_manifest' not in metadata['lite_optimizations']
            assert 'inventory_reuse' not in metadata['quantization_manifest']
            snapshots.append(metadata)
        finally: policy.close()
    assert snapshots[0] == snapshots[1]
    assert not owners and 'inventory.install' not in events


@pytest.mark.parametrize('value', [0, 1, None, {}, 'true'])
def test_option_and_constructor_require_exact_boolean_before_install(monkeypatch, value):
    events, _, _, _ = inventory_rig(monkeypatch)
    with pytest.raises(ValueError, match='must be a boolean'):
        pi05._split_lite_overrides({'lite_prefix_fp8_inventory': value})
    with pytest.raises(ValueError, match='must be a boolean'):
        pi05.Pi05LitePolicy(*model_processor(), **options(prefix_fp8_inventory=value))
    assert events == []


@pytest.mark.parametrize('change', [dict(prefix_fp8=False, prefix_fp8_evidence=None),
    dict(native_lifecycle=False), dict(native_lifecycle_variant='v1'), dict(native_lifecycle_variant='v2'),
    dict(compile_sample_actions=False), dict(cuda_graph=False, device='cpu')])
def test_all_dependencies_reject_before_installation(monkeypatch, change):
    events, _, _, _ = inventory_rig(monkeypatch)
    with pytest.raises(ValueError, match='Prefix FP8 inventory requires'):
        pi05.Pi05LitePolicy(*model_processor(), **options(**change))
    assert events == []


def test_actual_manifest_hook_recipe_live_hook_and_detached_output(monkeypatch):
    events, _, owners, bundle = inventory_rig(monkeypatch)
    policy = pi05.Pi05LitePolicy(*model_processor(), **options())
    try:
        assert events == ['prefix.install', 'native.install', 'inventory.install', 'runner.init', 'prefix.bind']
        data = policy.metadata()
        ext = data['quantization_manifest']['inventory_reuse']
        assert ext == owners[0].manifest() == bundle['extension']
        assert data['lite_optimizations']['prefix_fp8_inventory_manifest'] == ext
        assert data['lite_optimizations']['prefix_fp8_inventory'] is True
        assert policy.validate_precision_manifest() == data['quantization_manifest']
        components = data['recipe_components']
        assert components['quantization_manifest_sha256'] == pi05._json_hash(data['quantization_manifest'])
        assert data['recipe_sha256'] == pi05._json_hash(components)
        without = deepcopy(data['quantization_manifest']); without.pop('inventory_reuse')
        assert pi05._json_hash(without) != components['quantization_manifest_sha256']
        ext['module_count'] += 1
        assert policy.metadata()['quantization_manifest']['inventory_reuse'] == bundle['extension']
    finally: policy.close()
    assert events[-4:] == ['runner.close', 'inventory.uninstall', 'prefix.uninstall', 'native.uninstall']


def test_inventory_and_gelu_are_independent_and_close_in_dependency_order(monkeypatch):
    from test_lite_pi05_prefix_fp8_gelu_policy import fusion_evidence
    events, _, _, _ = inventory_rig(monkeypatch, gelu=True)
    policy = pi05.Pi05LitePolicy(*model_processor(), **options(prefix_fp8_gelu=True,
                               prefix_fp8_gelu_evidence=fusion_evidence()))
    data = policy.metadata()
    assert 'inventory_reuse' in data['quantization_manifest'] and 'gelu_fusion' in data['quantization_manifest']
    policy.close()
    assert events[-5:] == ['runner.close', 'inventory.uninstall', 'gelu.uninstall', 'prefix.uninstall', 'native.uninstall']


@pytest.mark.parametrize('failure', ['retained_install', 'runner'])
def test_constructor_rollback_recovers_inventory_before_base_and_native(monkeypatch, failure):
    events, _, owners, _ = inventory_rig(monkeypatch, failure=failure)
    model, processor = model_processor()
    policy = pi05.Pi05LitePolicy.__new__(pi05.Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        policy.__init__(model, processor, **options())
    assert policy._closed and policy._prefix_fp8_inventory_installation is None
    assert not owners[0].enabled and not hasattr(model, '_lite_prefix_fp8_inventory_installation')
    assert events[-3:] == ['inventory.uninstall', 'prefix.uninstall', 'native.uninstall']


@pytest.mark.parametrize('failure', ['graph', 'close'])
def test_failed_drain_or_inventory_retirement_keeps_handles_until_retry(monkeypatch, failure):
    events, bases, owners, _ = inventory_rig(monkeypatch, failure=failure)
    policy = pi05.Pi05LitePolicy(*model_processor(), **options())
    if failure == 'graph':
        original = policy.graph_runner.close
        def fail(): raise RuntimeError('injected graph retirement')
        monkeypatch.setattr(policy.graph_runner, 'close', fail)
    with pytest.raises(RuntimeError, match='injected'): policy.close()
    assert not policy._closed
    assert policy._prefix_fp8_inventory_installation is owners[0]
    assert policy._prefix_fp8_installation is bases[0]
    assert owners[0].enabled and 'prefix.uninstall' not in events and 'native.uninstall' not in events
    if failure == 'graph': monkeypatch.setattr(policy.graph_runner, 'close', original)
    policy.close()
    assert policy._closed and policy._prefix_fp8_inventory_installation is None


@pytest.mark.parametrize('change', ['native_metadata', 'inventory_metadata', 'owner_detached'])
def test_live_evidence_rejects_stale_policy_bindings(monkeypatch, change):
    _, _, owners, _ = inventory_rig(monkeypatch)
    policy = pi05.Pi05LitePolicy(*model_processor(), **options())
    try:
        if change == 'native_metadata': policy._lite_optimizations['native_lifecycle_manifest']['enabled'] = False
        elif change == 'inventory_metadata': policy._lite_optimizations['prefix_fp8_inventory_manifest']['module_count'] += 1
        else: del policy.model._lite_prefix_fp8_inventory_installation
        with pytest.raises(RuntimeError, match='inventory metadata'):
            policy.validate_precision_manifest()
    finally:
        if change == 'owner_detached': policy.model._lite_prefix_fp8_inventory_installation = owners[0]
        policy.close()
