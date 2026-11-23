"""Default-off reuse of v3's *current call* topology for the FP8 module scan.

Only the duplicate Module.modules DFS is removed. The no-extra-quantized-module
predicate, every source/private tensor read, forward check and precision check
remain in prefix_fp8. Unsupported Python access semantics use the original DFS.
No signature or tensor identity is accepted as proof of a new collector call.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import weakref

from torch import nn

from .fused_expert_mlp import _mutation_guard
from .native_lifecycle import (INSTALLATION as NATIVE_INSTALLATION,
    NativeLifecycleSignatureInstallation, _FreshLifecycleSnapshot)

INSTALLATION = '_lite_prefix_fp8_inventory_installation'
VERSION = 'prefix-fp8-fresh-v3-module-inventory-v1'
_ABSENT = object()


def _raw_type_attribute(cls, name):
    # Do not execute unknown descriptors or metaclass __getattribute__ hooks.
    if type(cls) is not type:
        return _ABSENT
    for base in type.__getattribute__(cls, '__mro__'):
        namespace = type.__getattribute__(base, '__dict__')
        if name in namespace:
            return namespace[name]
    return _ABSENT


_STANDARD = {name: _raw_type_attribute(nn.Module, name) for name in
             ('modules', 'named_modules', '__getattribute__', '__getattr__',
              '__dict__', '__class__', '__eq__', '__hash__')}
_NATIVE_SIGNATURE = NativeLifecycleSignatureInstallation.signature
_NATIVE_CONSUMER = NativeLifecycleSignatureInstallation.consume_fresh_signature


def _standard_type(cls):
    return (type(cls) is type
            and all(_raw_type_attribute(cls, name) is binding for name, binding in _STANDARD.items())
            and _raw_type_attribute(cls, '_modules') is _ABSENT)


def _standard_instance(module):
    state = object.__getattribute__(module, '__dict__')
    return (type(state) is dict and 'modules' not in state and 'named_modules' not in state
            and type(state.get('_modules')) is dict)


def _cold_inventory(model):
    """Raw identity DFS only after validating every visited access contract."""
    stack, seen, modules = [model], set(), []
    while stack:
        module = stack.pop()
        if module is None or id(module) in seen:
            continue
        if not _standard_type(type(module)) or not _standard_instance(module):
            return None
        seen.add(id(module))
        modules.append(module)
        state = object.__getattribute__(module, '__dict__')
        stack.extend(reversed(state['_modules'].values()))
    return tuple(modules)


def _same_topology(current, original):
    if current is original:
        return True
    if type(current) is not tuple or len(current) != 7:
        return False
    nodes, expected = current[0], original[0]
    if type(nodes) is not tuple or len(nodes) != len(expected):
        return False
    for node, old in zip(nodes, expected, strict=True):
        if (type(node) is not tuple or len(node) != 5 or type(node[0]) is not int
                or node[0] != old[0] or node[1] is not old[1]
                or any(type(value) is not int for value in node[2:]) or node[2:] != old[2:]):
            return False
    # These fields originate from exact dict registries. Unknown key equality
    # must never run here; it belongs to the unchanged fallback traversal.
    for index in (1, 3, 5):
        if type(current[index]) is not tuple or any(type(x) is not str for x in current[index]):
            return False
    for index in (2, 4, 6):
        if type(current[index]) is not tuple or any(type(x) is not int for x in current[index]):
            return False
    return current[1:] == original[1:]


class PrefixFP8InventoryInstallation:
    def __init__(self, model, prefix, native):
        self._model, self._prefix, self._native = weakref.ref(model), weakref.ref(prefix), native
        self.enabled = True
        modules = _cold_inventory(model)
        snapshot = native.signature(previous_full_signature=model._inference_cache.signature)
        self._topology = snapshot.topology
        self._accepted_topology = self._topology
        if modules is not None and (tuple(id(m) for m in modules) != tuple(n[0] for n in self._topology[0])
                or any(type(k) is not str for i in (1, 3, 5) for k in self._topology[i])):
            modules = None
        self._modules = None if modules is None else tuple(weakref.ref(m) for m in modules)
        self._types = () if modules is None else tuple(dict.fromkeys(type(m) for m in modules))
        self._edges = (() if modules is None else tuple(tuple((key, id(value)) for key, value in
            object.__getattribute__(m, '__dict__')['_modules'].items()) for m in modules))
        self._inventory_binding = (self._modules, self._types, self._edges, self._topology)
        self._collector = native._collector
        self._extension = native._collector._extension
        self._module_count = 0 if modules is None else len(modules)

    def _active(self):
        model, prefix = self._model(), self._prefix()
        if (not self.enabled or model is None or prefix is None
                or getattr(model, INSTALLATION, None) is not self
                or getattr(model, '_lite_prefix_fp8_installation', None) is not prefix
                or getattr(model, NATIVE_INSTALLATION, None) is not self._native
                or type(self._native) is not NativeLifecycleSignatureInstallation
                or self._native._collector._manifest.get('collector_variant') != 'v3'
                or self._native._collector is not self._collector
                or self._collector._extension is not self._extension
                or self._native.__dict__.get('signature', _ABSENT) is not _ABSENT
                or self._native.__dict__.get('consume_fresh_signature', _ABSENT) is not _ABSENT
                or NativeLifecycleSignatureInstallation.signature is not _NATIVE_SIGNATURE
                or NativeLifecycleSignatureInstallation.consume_fresh_signature is not _NATIVE_CONSUMER):
            raise RuntimeError('Prefix FP8 inventory/native lifecycle binding changed')
        self._native._active_model()
        if any(a is not b for a, b in zip((self._modules, self._types, self._edges, self._topology),
                                         self._inventory_binding, strict=True)):
            raise RuntimeError('Prefix FP8 module inventory ownership changed')
        return model, prefix

    def modules(self, proof):
        """Return a current validated tuple, or None for the original DFS."""
        model, prefix = self._active()
        prefix._owner()  # The original unique native runner/lock contract.
        if type(proof) is not _FreshLifecycleSnapshot:
            raise RuntimeError('FP8 inventory requires a fresh native callback proof')
        snapshot = _FreshLifecycleSnapshot.require(proof, self._native, model)
        if self._modules is None or not all(_standard_type(cls) for cls in self._types):
            return None
        current = snapshot.topology
        if not _same_topology(current, self._accepted_topology):
            return None
        modules = tuple(reference() for reference in self._modules)
        for module, edges, node in zip(modules, self._edges, self._topology[0], strict=True):
            if (module is None or id(module) != node[0] or type(module) is not node[1]
                    or not _standard_instance(module)):
                return None
            children = object.__getattribute__(module, '__dict__')['_modules']
            if len(children) != len(edges):
                return None
            for (key, child), (old_key, old_id) in zip(children.items(), edges, strict=True):
                if type(key) is not str or key != old_key or id(child) != old_id:
                    return None
        # The flat edge check also closes the gap between collection and the
        # final scan if an intervening model method changed registrations. It
        # creates no recursive generators, names, visited set or second DFS.
        # Only memoize the immutable topology *value*, not whether it was read.
        # Every reuse above still requires a just-issued active native proof.
        self._accepted_topology = current
        return modules

    def prepare_context(self, signature_head):
        model, prefix = self._active()
        prefix._owner()
        if type(signature_head) is not tuple or len(signature_head) != 6:
            raise ValueError('FP8 inventory requires the complete unchanged signature prefix')

        def consume(snapshot, proof):
            tensor_state = tuple(snapshot)
            generation = model._inference_cache.refresh((*signature_head, *tensor_state))
            prefix.prepare(generation, inventory_proof=proof)
            return tensor_state, generation

        return self._native.consume_fresh_signature(consume,
            previous_full_signature=model._inference_cache.signature)

    def manifest(self):
        self._active()
        base = Path(__file__).resolve().parent
        sources = (Path(__file__), base / 'native_lifecycle.py', base / 'prefix_fp8.py',
                   base.parent / 'models/pi05/modeling_pi05.py')
        return dict(version=VERSION, enabled=self.enabled,
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            dependency='G4 prefix FP8 51-site owner; native lifecycle v3; unique compiled native runner',
            module_count=self._module_count, cold_standard_inventory=self._modules is not None,
            reused='module DFS inventory only, authorized by the same-call fresh native snapshot',
            intervening_mutation_guard='live flat child-slot identity/order/None/alias comparison before scan',
            preserved=['no-extra-QuantizedLinear scan', '51 original weight live identities',
                       '102 private packed/scale live identities', 'all forward/precision/owner checks'],
            fallback='original model.modules DFS for unsupported access or changed topology',
            snapshot_lifetime='current callback/thread only; retired on success and exception')

    def uninstall(self):
        model, _ = self._active()
        with _mutation_guard(model, 'prefix FP8 inventory'):
            if any(runner._event is not None or runner._device is not None
                   for runner in getattr(model, '_graph_runners', ())):
                raise RuntimeError('Drain native graph work before removing prefix FP8 inventory')
            delattr(model, INSTALLATION)
            self.enabled = False
            self._modules, self._types = None, ()
            self._edges = ()
            self._topology = self._accepted_topology = None
            self._native = self._collector = self._extension = self._inventory_binding = None
            model.invalidate_inference_caches(weights_changed=False)


def install_prefix_fp8_inventory(model):
    """Explicit cold opt-in; install after prefix owner and native v3.

    The model may not yet have its lazily created graph-runner registry.
    An existing runner or activated base epoch is never a cold installation.
    Uninstall after graph retirement, before the prefix/native installations.
    No arithmetic, packed tensor, or inference/compile option is changed.
    """
    from .prefix_fp8 import PrefixFP8Installation
    with _mutation_guard(model, 'prefix FP8 inventory'):
        prefix = getattr(model, '_lite_prefix_fp8_installation', None)
        native = getattr(model, NATIVE_INSTALLATION, None)
        if (getattr(model, INSTALLATION, None) is not None or type(prefix) is not PrefixFP8Installation
                or type(native) is not NativeLifecycleSignatureInstallation
                or native._collector._manifest.get('collector_variant') != 'v3'):
            raise ValueError('FP8 inventory requires the original 51-site owner and native lifecycle v3')
        if (tuple(getattr(model, '_graph_runners', ())) or prefix._runner is not None
                or prefix._generation is not None):
            raise ValueError('FP8 inventory requires an unbound cold owner before any graph runner or active epoch')
        prefix._validate(full=True)
        handle = PrefixFP8InventoryInstallation(model, prefix, native)
        setattr(model, INSTALLATION, handle)
        try:
            model.invalidate_inference_caches(weights_changed=False)
        except BaseException:
            delattr(model, INSTALLATION)
            handle.enabled = False
            handle._modules = None
            raise
        return handle
