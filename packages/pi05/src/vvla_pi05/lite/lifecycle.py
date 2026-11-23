"""Read every live tensor and registration edge in one module-tree traversal.

This helper is not installed by default. It replaces no model/recipe/cache check
on its own and retains no tensor or module inventory between calls. The tensor
fields use the original shared-core identity function unchanged; callers must
include all three returned fields in their invalidation signature.
"""

from typing import NamedTuple

from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity

VERSION = "single-dfs-full-tensor-and-registration-signature-v1"


class LifecycleSignature(NamedTuple):
    parameters: tuple
    buffers: tuple
    topology: tuple


def collect_lifecycle_signature(model):
    """Preserve native DFS/dedup order and inspect every registration slot.

    Topology includes node identity/type plus every child, parameter and buffer
    slot name/identity. Flattening the registries avoids several tiny allocations
    per leaf; per-node lengths make the flattened edge records unambiguous.
    Identity-based dedup never invokes tensor equality or reads tensor contents.
    """
    stack, visited = [model], set()
    nodes, parameter_names, parameters, buffer_names, buffers, child_names, children = [], [], [], [], [], [], []
    while stack:
        module = stack.pop()
        mid = id(module)
        if module is None or mid in visited:
            continue
        visited.add(mid)
        state = module.__dict__
        ps, bs, cs = state["_parameters"], state["_buffers"], state["_modules"]
        nodes.append((mid, type(module), len(ps), len(bs), len(cs)))
        parameter_names.extend(ps)
        parameters.extend(ps.values())
        buffer_names.extend(bs)
        buffers.extend(bs.values())
        child_names.extend(cs)
        children.extend(cs.values())
        stack.extend(reversed(cs.values()))
    pids, bids, cids = tuple(map(id, parameters)), tuple(map(id, buffers)), tuple(map(id, children))
    unique_parameters, unique_buffers = dict(zip(pids, parameters)), dict(zip(bids, buffers))
    unique_parameters.pop(id(None), None)
    unique_buffers.pop(id(None), None)
    return LifecycleSignature(
        tuple(map(tensor_identity, unique_parameters.values())),
        tuple(map(tensor_identity, unique_buffers.values())),
        (tuple(nodes), tuple(parameter_names), pids, tuple(buffer_names), bids, tuple(child_names), cids),
    )


INSTALLATION = '_lite_lifecycle_signature_installation'


class LifecycleSignatureInstallation:
    """An opt-in collector, never a cached inventory or invalidation result.

    Core callers must consume the complete returned parameters/buffers/topology
    tuple. Reading the signature occurs under the core/graph request owner;
    install/uninstall separately use the shared idle-graph mutation guard.
    """
    def __init__(self, model):
        import weakref
        self._model = weakref.ref(model)
        self.enabled = True

    def _active_model(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError('Lifecycle signature installation is no longer active')
        return model

    def signature(self):
        """Read all live registries and unchanged tensor-identity fields now."""
        return collect_lifecycle_signature(self._active_model())

    def __call__(self):
        return self.signature()

    def manifest(self):
        import hashlib
        from pathlib import Path
        identity_source = Path(__file__).resolve().parents[1] / 'models/pi05/inference_cache_pi05.py'
        return dict(version=VERSION, enabled=self.enabled,
                    inventory_cache=False, signature_cache=False,
                    returned_fields=['parameters', 'buffers', 'topology'],
                    tensor_identity_fields=['object_id', 'data_ptr', 'version', 'dtype', 'device', 'shape'],
                    topology='all module nodes/types and all child/parameter/buffer registration slots, including None and aliases',
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), identity_source)})

    def uninstall(self):
        from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
        model = self._active_model()
        with _mutation_guard(model, 'single-pass lifecycle signature'):
            delattr(model, INSTALLATION)
            model.invalidate_inference_caches(weights_changed=False)
            self.enabled = False


def install_lifecycle_signature(model):
    """Default-off installation; never replaces parameter registry validation."""
    from torch import nn
    from vvla_pi05.lite.fused_expert_mlp import _mutation_guard
    if not isinstance(model, nn.Module):
        raise TypeError('Lifecycle signature installation requires a Torch module')
    with _mutation_guard(model, 'single-pass lifecycle signature'):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError('Lifecycle signature collector is already installed')
        handle = LifecycleSignatureInstallation(model)
        setattr(model, INSTALLATION, handle)
        model.invalidate_inference_caches(weights_changed=False)
        return handle
