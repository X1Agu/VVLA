"""Policy option/ownership tests; scheduled Colab execution only.

These tests use the real native installation and request-lookup handles with a
test-only Python collector backend. Native C++ identity equivalence and the
locked request scope itself have separate helper contracts.
"""
import pytest

from test_lite_fused_adarms import SmallExpert
from test_lite_pi05 import _policy
from vvla_repro_pi05.lite import native_lifecycle, request_identity
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides


def model_and_processor():
    model = SmallExpert()
    model.action_horizon, model.action_dim = 2, 4
    return model, _policy().processor


def native_loader(monkeypatch, calls):
    class Extension:
        @staticmethod
        def collect(model):
            return collect_lifecycle_signature(model)
    def load(*, variant):
        calls.append(('load', variant))
        return native_lifecycle.NativeLifecycleCollector(Extension(), {
            'collector_variant': variant, 'version': 'test-only-python-collector',
            'source_sha256': {}, 'inventory_cache': False, 'signature_cache': False,
        })
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', load)


def test_request_lookup_default_off_and_strict_boolean_never_loads_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled/invalid lookup attempted optional installation')
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', forbidden)
    monkeypatch.setattr(request_identity, 'install_request_identity_lookup', forbidden)
    assert _split_lite_overrides({})[1]['request_identity_lookup'] is False
    model, processor = model_and_processor()
    policy = Pi05LitePolicy(model, processor, device='cpu', cuda_graph=False)
    assert policy._request_identity_installation is None
    assert policy.metadata()['lite_optimizations']['request_identity_lookup'] is False
    assert 'request_identity_lookup_manifest' not in policy.metadata()['lite_optimizations']
    policy.close()
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='request_identity_lookup must be a boolean'):
            _split_lite_overrides({'lite_request_identity_lookup': value})
        failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
        with pytest.raises(ValueError, match='request_identity_lookup must be a boolean'):
            failed.__init__(model, processor, device='cpu', cuda_graph=False, request_identity_lookup=value)
        assert failed._closed and failed._request_identity_installation is None


@pytest.mark.parametrize('native,graphs', [(False, True), (True, False), (False, False)])
def test_request_lookup_dependencies_rejected_before_model_or_extension(monkeypatch, tmp_path, native, graphs):
    from vvla_repro_pi05.lite import pi05
    from vvla_repro_pi05.models.pi05.config import Pi05Config
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid request lookup dependencies reached allocation or installation')
    monkeypatch.setattr(native_lifecycle, 'load_native_lifecycle', forbidden)
    monkeypatch.setattr(request_identity, 'install_request_identity_lookup', forbidden)
    monkeypatch.setattr(pi05, '_construct_pi05_model', forbidden)
    monkeypatch.setattr(pi05, '_resolve_checkpoint', lambda *args: tmp_path)
    monkeypatch.setattr(pi05, 'load_lite_config', lambda path: (Pi05Config(), {}))
    with pytest.raises(ValueError, match='requires lite_native_lifecycle and native CUDA graphs'):
        Pi05LitePolicy.from_pretrained(tmp_path, cuda_graph=graphs, config_overrides={
            'lite_request_identity_lookup': True, 'lite_native_lifecycle': native})
    model, processor = model_and_processor()
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(ValueError, match='requires lite_native_lifecycle and native CUDA graphs'):
        failed.__init__(model, processor, device='cuda' if graphs else 'cpu', cuda_graph=graphs,
                        native_lifecycle=native, request_identity_lookup=True)
    assert failed._closed and failed._native_lifecycle_installation is None


@pytest.mark.parametrize('variant', ['v1', 'v2'])
def test_policy_installs_after_native_records_manifest_and_closes_graph_lookup_native(monkeypatch, variant):
    calls = []
    native_loader(monkeypatch, calls)
    model, processor = model_and_processor()
    parameters = {name: id(value) for name, value in model.named_parameters()}
    modules = {name: id(value) for name, value in model.named_modules()}
    actual_install = request_identity.install_request_identity_lookup
    def install(model):
        assert getattr(model, native_lifecycle.INSTALLATION).enabled
        assert not tuple(model._graph_runners)
        calls.append(('install_lookup', variant))
        return actual_install(model)
    monkeypatch.setattr(request_identity, 'install_request_identity_lookup', install)
    policy = Pi05LitePolicy(model, processor, device='cuda', cuda_graph=True,
                            native_lifecycle=True, native_lifecycle_variant=variant,
                            request_identity_lookup=True)
    assert calls == [('load', variant), ('install_lookup', variant)]
    lookup, native, runner = policy._request_identity_installation, policy._native_lifecycle_installation, policy.graph_runner
    assert runner._device is None  # Constructor only; no CUDA work in this test.
    options = policy.metadata()['lite_optimizations']
    assert options['request_identity_lookup'] is True
    assert options['request_identity_lookup_manifest'] == lookup.manifest()
    requirements = options['request_identity_lookup_manifest']['requirements']
    assert 'native lifecycle collector v1/v2' in requirements and 'locked native runner' in requirements
    assert options['native_lifecycle_manifest']['collector_variant'] == variant
    options['request_identity_lookup_manifest']['fields'].clear()
    assert policy.metadata()['lite_optimizations']['request_identity_lookup_manifest']['fields']
    runner.entries['retirement_sentinel'] = object()
    original_close, original_lookup, original_native = runner.close, lookup.uninstall, native.uninstall
    def close_graph():
        calls.append('graph')
        return original_close()
    def uninstall_lookup():
        assert runner._closed and not runner.entries and native.enabled
        calls.append('lookup')
        return original_lookup()
    def uninstall_native():
        assert not lookup.enabled and not hasattr(model, request_identity.INSTALLATION)
        calls.append('native')
        return original_native()
    monkeypatch.setattr(runner, 'close', close_graph)
    monkeypatch.setattr(lookup, 'uninstall', uninstall_lookup)
    monkeypatch.setattr(native, 'uninstall', uninstall_native)
    policy.close()
    policy.close()
    assert calls[-3:] == ['graph', 'lookup', 'native']
    assert policy._request_identity_installation is policy._native_lifecycle_installation is None
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert {name: id(value) for name, value in model.named_modules()} == modules


@pytest.mark.parametrize('stage', ['lookup_install', 'lookup_manifest', 'runner'])
def test_late_lookup_failure_rolls_back_before_native_dependency(monkeypatch, stage):
    from vvla_repro_pi05.lite import graph
    calls, handles = [], []
    native_loader(monkeypatch, calls)
    model, processor = model_and_processor()
    original_parameters = tuple(id(value) for value in model.parameters())
    actual_install = request_identity.install_request_identity_lookup
    def install(model):
        assert getattr(model, native_lifecycle.INSTALLATION).enabled
        if stage == 'lookup_install':
            raise RuntimeError('injected lookup install failure')
        handle = actual_install(model)
        handles.append(handle)
        return handle
    monkeypatch.setattr(request_identity, 'install_request_identity_lookup', install)
    if stage == 'lookup_manifest':
        def fail_manifest(self):
            raise RuntimeError('injected lookup manifest failure')
        monkeypatch.setattr(request_identity.RequestIdentityInstallation, 'manifest', fail_manifest)
    if stage == 'runner':
        def fail_runner(*args, **kwargs):
            raise RuntimeError('injected runner failure after lookup installation')
        monkeypatch.setattr(graph, 'Pi05NativeGraphRunner', fail_runner)
    actual_uninstall = native_lifecycle.NativeLifecycleSignatureInstallation.uninstall
    def uninstall_native(self):
        assert not hasattr(model, request_identity.INSTALLATION)
        assert all(not handle.enabled for handle in handles)
        calls.append('native_uninstall')
        return actual_uninstall(self)
    monkeypatch.setattr(native_lifecycle.NativeLifecycleSignatureInstallation, 'uninstall', uninstall_native)
    failed = Pi05LitePolicy.__new__(Pi05LitePolicy)
    with pytest.raises(RuntimeError, match='injected'):
        failed.__init__(model, processor, device='cuda', cuda_graph=True, native_lifecycle=True,
                        native_lifecycle_variant='v2', request_identity_lookup=True)
    assert failed._closed and failed.graph_runner is None
    assert failed._request_identity_installation is failed._native_lifecycle_installation is None
    assert calls == [('load', 'v2'), 'native_uninstall']
    assert not hasattr(model, native_lifecycle.INSTALLATION)
    assert tuple(id(value) for value in model.parameters()) == original_parameters
    failed.close()
