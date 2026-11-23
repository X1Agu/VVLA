"""Remote-only CPU contracts. No local execution or arithmetic claim.

The meta rig deliberately supplies fictional evidence/packing. The final test
uses the real v3 collector on the scheduled host; it does not launch GPU work.
"""
import cProfile
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_lite_prefix_fp8 import rig, evidence
from prefix_fp8_inventory_fixtures import (evidence as inventory_evidence,
                                          make_inventory_bundle)
from vvla_repro_pi05.lite import prefix_fp8 as owner
from vvla_repro_pi05.lite import prefix_fp8_inventory as inventory
from vvla_repro_pi05.lite.graph import Pi05NativeGraphRunner
from vvla_repro_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_repro_pi05.lite.native_lifecycle import (NativeLifecycleCollector,
    install_native_lifecycle_signature, load_native_lifecycle)


HEAD = ((True, True, True, True, True), 0.004, 4.0, 64, 'highest', False)


class FreshPythonReference:
    """Fake extension with actual fresh reference reads and v3 tuple reuse."""
    def __init__(self):
        self.calls = 0

    def collect(self, model, previous=None):
        self.calls += 1
        snapshot = tuple(collect_lifecycle_signature(model))
        return previous if previous is not None and snapshot == previous else snapshot


def setup(rig, *, collector=None):
    extension = FreshPythonReference()
    if collector is None:
        # Only the fake collector needs fictional complete build/backend
        # records. Bind their source fields to the files actually read by the
        # cold manifest; the fake binary/toolchain remain explicitly fictional.
        # The supplied real-v3 collector path below is deliberately untouched.
        bundle = make_inventory_bundle()
        rig.backend.update(bundle['quantization_manifest']['backend'])
        manifest = bundle['native_manifest']
        root = Path(__file__).resolve().parents[2]
        manifest['source_sha256'] = {name: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for name, path in inventory_evidence.NATIVE_SOURCE_PATHS.items()}
        build = {key: manifest[key] for key in
            (*inventory_evidence.NATIVE_BUILD_CONSTANTS, *inventory_evidence.NATIVE_VARIABLE_BUILD)}
        manifest['module_name'] = 'vvla_native_lifecycle_' + hashlib.sha256(
            json.dumps(build, sort_keys=True).encode()).hexdigest()[:20]
        collector = NativeLifecycleCollector(extension, manifest)
    base = owner.install_prefix_fp8(rig.model, evidence())
    native = install_native_lifecycle_signature(rig.model, collector=collector)
    fast = inventory.install_prefix_fp8_inventory(rig.model)
    runner = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    base.bind_runner(runner)
    return SimpleNamespace(model=rig.model, base=base, native=native, fast=fast,
                           runner=runner, extension=extension)


def prepare(case):
    with case.runner._call_lock:
        return case.fast.prepare_context(HEAD)


def cleanup(case):
    case.runner.close()
    case.fast.uninstall()
    case.base.uninstall()
    case.native.uninstall()


def module_dfs_calls(operation):
    profile = cProfile.Profile()
    profile.runcall(operation)
    codes = {nn.Module.modules.__code__, nn.Module.named_modules.__code__}
    return sum(row.callcount for row in profile.getstats() if row.code in codes)


def test_warm_reuses_only_dfs_and_still_reads_all_153_site_tensors(rig, monkeypatch):
    case = setup(rig)
    try:
        prepare(case)  # First full validation intentionally keeps full scans.
        actual_identity = owner.tensor_identity
        reads = []
        monkeypatch.setattr(owner, 'tensor_identity', lambda value: (reads.append(value), actual_identity(value))[1])
        before = case.extension.calls
        assert module_dfs_calls(lambda: prepare(case)) == 0
        assert case.extension.calls == before + 1
        expected = [t for site in case.base.sites for t in (site.weight, site.quantized, site.scale)]
        assert len(reads) == 153 and all(a is b for a, b in zip(reads, expected, strict=True))
        # Standalone prepare and cold manifest retain the original DFS path.
        with case.runner._call_lock:
            assert module_dfs_calls(lambda: case.base.prepare(case.base._generation)) > 0
        assert module_dfs_calls(case.base.manifest) > 0
        assert case.fast.manifest()['module_count'] == len(tuple(case.model.modules()))
    finally:
        cleanup(case)


def test_actual_core_handoff_collects_once_and_keeps_complete_shared_signature(rig):
    from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    model = rig.model
    model.config = SimpleNamespace(min_period=0.004, max_period=4.0, prefix_pack_bucket_size=64)
    model.optimization_flags = lambda: (False, False, True, False, False)
    model.paligemma_with_expert.paligemma.model.vision_tower = nn.Identity().eval()
    case = setup(rig)
    try:
        def actual_prepare():
            with case.runner._call_lock:
                return Pi05ForActionPrediction.prepare_inference_context(model, [], [],
                    torch.ones((1, 2), dtype=torch.bool), 10, positions=False,
                    prepared_layout={'packing': None, 'empty_pattern': ()})
        before = case.extension.calls
        context = actual_prepare()
        assert case.extension.calls == before + 1
        assert context['token'] == (model.optimization_flags(), case.base._generation)
        assert len(model._inference_cache.signature) == 9
        assert model._inference_cache.signature[-3:] == tuple(collect_lifecycle_signature(model))
        before = case.extension.calls
        assert module_dfs_calls(actual_prepare) == 0
        assert case.extension.calls == before + 1
        assert case.native._fresh_snapshot is None
    finally:
        cleanup(case)


def test_proof_expires_on_success_exception_cross_thread_and_no_cached_signature_substitute(rig):
    case = setup(rig)
    retained = []
    try:
        with case.runner._call_lock:
            def consume(snapshot, proof):
                retained.append(proof)
                assert case.fast.modules(proof) is not None
                errors = []
                def other_thread():
                    try:
                        case.fast.modules(proof)
                    except RuntimeError as exc:
                        errors.append(str(exc))
                thread = threading.Thread(target=other_thread)
                thread.start()
                thread.join()
                assert len(errors) == 1 and 'fresh' in errors[0]
            case.native.consume_fresh_signature(consume)
            with pytest.raises(RuntimeError, match='fresh'):
                case.fast.modules(retained[-1])
            with pytest.raises(RuntimeError, match='fresh'):
                case.fast.modules(case.native.signature())
            def fail(snapshot, proof):
                retained.append(proof)
                raise ValueError('consumer failed')
            with pytest.raises(ValueError, match='consumer failed'):
                case.native.consume_fresh_signature(fail)
            assert case.native._fresh_snapshot is None
            with pytest.raises(RuntimeError, match='fresh'):
                case.fast.modules(retained[-1])
        prepare(case)  # Failure leaves no persistent active proof.
    finally:
        cleanup(case)


@pytest.mark.parametrize('mutation', ['weight', 'packed', 'scale', 'packed_replace',
    'module_replace', 'none_slot', 'alias_slot', 'order', 'forward', 'nonprefix_forward', 'invalidation'])
def test_original_mutation_rejection_and_no_implicit_repack(rig, mutation):
    case = setup(rig)
    try:
        prepare(case)
        original = tuple(id(site.quantized) for site in case.base.sites)
        site = case.base.sites[0]
        with torch.no_grad():
            if mutation == 'weight': site.weight.add_(1)
            elif mutation == 'packed': site.quantized.copy_(torch.empty_like(site.quantized))
            elif mutation == 'scale': site.scale.add_(1)
            elif mutation == 'packed_replace': site.quantized = torch.empty_like(site.quantized)
            elif mutation == 'module_replace':
                owner._layers(case.model)[0].mlp.gate_proj = nn.Linear(2048, 16384, bias=False,
                    device='meta', dtype=torch.bfloat16).eval()
            elif mutation == 'none_slot': case.model.add_module('new_none_slot', None)
            elif mutation == 'alias_slot': case.model.alias = case.model.other
            elif mutation == 'order':
                case.model._modules['paligemma_with_expert'] = case.model._modules.pop('paligemma_with_expert')
            elif mutation == 'forward': site.module.forward = lambda x: x
            elif mutation == 'nonprefix_forward': case.model.other.forward = lambda x: x
            elif mutation == 'invalidation': case.model.invalidate_inference_caches()
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            prepare(case)
        if mutation != 'packed_replace':
            assert tuple(id(site.quantized) for site in case.base.sites) == original
    finally:
        cleanup(case)


def test_standard_access_override_falls_back_to_original_iterator_once(rig):
    case = setup(rig)
    try:
        prepare(case)
        calls = []
        def modules():
            calls.append('original path')
            return nn.Module.modules(case.model)
        case.model.modules = modules
        prepare(case)
        assert calls == ['original path']
        del case.model.modules
        prepare(case)
    finally:
        cleanup(case)


def test_unknown_class_descriptor_is_not_invoked_by_eligibility(rig, monkeypatch):
    case = setup(rig)
    try:
        prepare(case)
        calls = []
        class Descriptor:
            def __get__(self, instance, cls):
                calls.append(instance)
                return lambda: nn.Module.modules(instance)
        monkeypatch.setattr(type(case.model), 'modules', Descriptor())
        # Only the original fallback invocation executes the descriptor. The
        # eligibility path does not call descriptor.__get__(None, class).
        prepare(case)
        assert calls == [case.model]
    finally:
        monkeypatch.undo()
        cleanup(case)


def test_custom_getattribute_uses_original_dfs_and_recovery_needs_fresh_call(rig, monkeypatch):
    case = setup(rig)
    try:
        prepare(case)
        def custom_getattribute(self, name):
            return object.__getattribute__(self, name)
        with monkeypatch.context() as patch:
            patch.setattr(type(case.model), '__getattribute__', custom_getattribute)
            assert module_dfs_calls(lambda: prepare(case)) > 0
        assert module_dfs_calls(lambda: prepare(case)) == 0
    finally:
        cleanup(case)


@pytest.mark.parametrize('variant', ['v1', 'v2'])
def test_no_implicit_upgrade_or_default_inventory_installation(rig, variant):
    base = owner.install_prefix_fp8(rig.model, evidence())
    native = install_native_lifecycle_signature(rig.model,
        collector=NativeLifecycleCollector(FreshPythonReference(), {'collector_variant': variant}))
    try:
        assert not hasattr(rig.model, inventory.INSTALLATION)
        with pytest.raises(ValueError, match='native lifecycle v3'):
            inventory.install_prefix_fp8_inventory(rig.model)
        assert not hasattr(rig.model, inventory.INSTALLATION)
    finally:
        base.uninstall()
        native.uninstall()


def test_intervening_registration_change_does_not_hide_added_quantized_module(rig, monkeypatch):
    from vvla_repro_pi05.models.pi05.precision_pi05 import QuantizedLinear
    case = setup(rig)
    extra = QuantizedLinear.__new__(QuantizedLinear)
    nn.Module.__init__(extra)
    extra.eval()
    try:
        prepare(case)
        original = case.model.optimization_flags
        def change_after_snapshot():
            case.model.add_module('unapproved_quantized', extra)
            return original()
        monkeypatch.setattr(case.model, 'optimization_flags', change_after_snapshot)
        # Snapshot was taken before optimization_flags. The final flat slot
        # check notices the insertion and the original DFS rejects it.
        with pytest.raises(ValueError, match='pre-existing quantized'):
            prepare(case)
    finally:
        monkeypatch.undo()
        cleanup(case)


def test_unique_runner_and_retirement_order_and_install_rollback(rig, monkeypatch):
    case = setup(rig)
    try:
        with pytest.raises(RuntimeError, match='unique locked'):
            case.fast.prepare_context(HEAD)
        prepare(case)
        with pytest.raises(RuntimeError, match='inventory'):
            case.base.uninstall()
        with pytest.raises(RuntimeError, match='inventory'):
            case.native.uninstall()
        second = Pi05NativeGraphRunner(case.model, compile_sample_actions=True)
        try:
            with pytest.raises(RuntimeError, match='unique locked'):
                prepare(case)
        finally:
            second.close()
        case.runner.entries['retained'] = object()
        with pytest.raises(ValueError, match='Clear all live'):
            case.fast.uninstall()
        case.runner.entries.clear()
        case.runner._device = torch.device('meta')
        with pytest.raises(RuntimeError, match='Drain native graph'):
            case.fast.uninstall()
        case.runner.clear()
    finally:
        cleanup(case)


def test_absent_lazy_registry_cold_manifest_and_caller_failure_cleanup(rig):
    # MetaPrefix normally creates this registry eagerly for older owner tests.
    # Remove it BEFORE either owner is installed, matching the real Pi05 model
    # constructor: Pi05NativeGraphRunner creates it only when a runner is built.
    # Packing/evidence are fictional, but all owner/manifest/cleanup methods
    # below are the actual production methods, with no dummy runner.
    del rig.model._graph_runners
    base = owner.install_prefix_fp8(rig.model, evidence())
    native = install_native_lifecycle_signature(rig.model,
        collector=NativeLifecycleCollector(FreshPythonReference(), {'collector_variant': 'v3'}))
    fast = None
    try:
        with pytest.raises(RuntimeError, match='caller failed before runner construction'):
            try:
                fast = inventory.install_prefix_fp8_inventory(rig.model)
                document = fast.manifest()
                assert document['enabled'] is True and document['cold_standard_inventory'] is True
                assert document['module_count'] == len(tuple(rig.model.modules()))
                assert not hasattr(rig.model, '_graph_runners')
                raise RuntimeError('caller failed before runner construction')
            finally:
                if fast is not None:
                    fast.uninstall()
        assert not hasattr(rig.model, inventory.INSTALLATION)
        assert not hasattr(rig.model, '_graph_runners')
        assert not fast.enabled and fast._modules is None
        base._validate(full=True)
    finally:
        if fast is not None and fast.enabled:
            fast.uninstall()
        base.uninstall()
        native.uninstall()


def test_absent_lazy_registry_install_failure_rolls_back_and_allows_cold_retry(rig, monkeypatch):
    del rig.model._graph_runners
    base = owner.install_prefix_fp8(rig.model, evidence())
    native = install_native_lifecycle_signature(rig.model,
        collector=NativeLifecycleCollector(FreshPythonReference(), {'collector_variant': 'v3'}))
    original_invalidate = rig.model.invalidate_inference_caches
    fast = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(rig.model, 'invalidate_inference_caches',
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError('injected invalidate')))
            with pytest.raises(RuntimeError, match='injected invalidate'):
                inventory.install_prefix_fp8_inventory(rig.model)
        assert not hasattr(rig.model, inventory.INSTALLATION)
        assert not hasattr(rig.model, '_graph_runners')
        assert rig.model.invalidate_inference_caches == original_invalidate
        fast = inventory.install_prefix_fp8_inventory(rig.model)
        assert fast.manifest()['enabled'] is True
    finally:
        if fast is not None:
            fast.uninstall()
        base.uninstall()
        native.uninstall()


@pytest.mark.parametrize('state', ['nonempty_registry', 'bound_owner', 'activated_epoch'])
def test_inventory_rejects_every_non_cold_owner_even_without_live_graph_entries(rig, state):
    base = owner.install_prefix_fp8(rig.model, evidence())
    native = install_native_lifecycle_signature(rig.model,
        collector=NativeLifecycleCollector(FreshPythonReference(), {'collector_variant': 'v3'}))
    runner = Pi05NativeGraphRunner(rig.model, compile_sample_actions=True)
    try:
        if state != 'nonempty_registry':
            base.bind_runner(runner)
            if state == 'activated_epoch':
                generation = rig.model._inference_cache.refresh((*HEAD, *native.signature()))
                with runner._call_lock:
                    base.prepare(generation)
                assert base._generation is not None
            runner.close()
            rig.model._graph_runners.clear()
            # Separately prove the base binding and activated epoch guards;
            # absence of registry membership cannot authorize a warm owner.
            if state == 'activated_epoch':
                base._runner = None
        with pytest.raises(ValueError, match='unbound cold owner'):
            inventory.install_prefix_fp8_inventory(rig.model)
        assert not hasattr(rig.model, inventory.INSTALLATION)
    finally:
        runner.close()
        base.uninstall()
        native.uninstall()


def test_real_native_v3_fresh_snapshot_reuse_meta_model(rig):
    # Root schedules this real compiler/extension contract remotely. Missing
    # tools are failures, never implicit Python fallback or successful skips.
    collector = load_native_lifecycle(variant='v3', verbose=True)
    case = setup(rig, collector=collector)
    try:
        before = torch.cuda.is_initialized()
        prepare(case)
        assert module_dfs_calls(lambda: prepare(case)) == 0
        assert torch.cuda.is_initialized() == before
        with torch.no_grad():
            case.base.sites[0].scale.add_(1)
        with pytest.raises(RuntimeError, match='packing epoch'):
            prepare(case)
    finally:
        cleanup(case)
