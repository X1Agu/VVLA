"""Versioned model state for executable caches; no GPU reads or weight copies.

Supported mutations use ordinary versioned tensors and mutation_scope. Tensor.data
writes, external raw-pointer writes and edits outside the lock are unsupported.
"""
from dataclasses import asdict, is_dataclass
import json

import torch


_STANDARD_ENUMERATORS = tuple(
    (name, getattr(torch.nn.Module, name))
    for name in ('named_modules', 'named_parameters', 'named_buffers')
)


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError as exc:
        raise ValueError(
            'Persistent model tensors need version counters; construct/install outside inference_mode'
        ) from exc


def tensor_signature(tensor):
    return (id(tensor), tensor.data_ptr(), _tensor_version(tensor),
            tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


class ModelStateContract:
    """Reuse a signature while registered state and tensor metadata are unchanged.

    Public registration slots are still checked on every request. Ordinary
    in-place tensor changes bump _version; rebinding a module, parameter or buffer
    changes a slot identity. Storage, dtype, device and layout are also checked
    to cover Module.to/_apply. Changes trigger the same full signature as before.
    Custom named_* enumerators retain the uncached path because their outputs
    need not correspond to the standard registration dictionaries.
    """

    def __init__(self, model, *, recipe=None, validate=None, native_comparator=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError('State contract requires an nn.Module')
        if native_comparator is not None:
            from .native_state import NativeStateComparator
            if type(native_comparator) is not NativeStateComparator:
                raise TypeError('native_comparator must be an explicitly loaded NativeStateComparator')
        self.native_comparator = native_comparator
        self.model = model
        self.recipe = recipe if recipe is not None else getattr(model, 'execution_signature', lambda: None)
        self.validate = validate if validate is not None else getattr(model, 'validate_execution_state', lambda: None)
        self._cached_token = None
        self._registered_modules = ()
        self._tensor_states = ()
        self._enumeration_types = ()

    def _registered_state_unchanged(self):
        # No lazy build or installation here. None means the original Python
        # path; a native result of None means an explicitly unsupported Python
        # tensor/registry/API override and executes that same reference path.
        # Native exceptions propagate: an unexpected failure must retire graphs,
        # not silently turn a broken candidate into an apparent baseline pass.
        if self.native_comparator is not None:
            result = self.native_comparator.compare(self)
            if result is not None:
                return result
        return self._registered_state_unchanged_python()

    def _registered_state_unchanged_python(self):
        """Unchanged reference comparator, also available for explicit A-B-A."""
        if not self._registered_modules or self._registered_modules[0][0] is not self.model:
            return False
        for module_type, methods in self._enumeration_types:
            if any(getattr(module_type, name) is not function for name, function in methods):
                return False
        for module, module_type, type_module, type_name, registries in self._registered_modules:
            if (type(module) is not module_type or module_type.__module__ != type_module
                    or module_type.__qualname__ != type_name):
                return False
            if any(name in module.__dict__ for name in ('named_modules', 'named_parameters', 'named_buffers')):
                return False
            if module.training:
                raise ValueError('All model submodules must remain in eval mode')
            for name, registry, slots in registries:
                current = getattr(module, name)
                if current is not registry or len(current) != len(slots):
                    return False
                for (key, value), current_key in zip(slots, current):
                    # Registration order can affect Sequential forward, and a
                    # missing registered None slot is still a structural edit.
                    if current_key != key or current[key] is not value:
                        return False
        for tensor, signature in self._tensor_states:
            # Module.to/_apply may preserve Parameter identity and _version but
            # replace its storage or layout through PyTorch's own .data setter.
            # Checking versions alone would replay a graph against stale storage.
            if (_tensor_version(tensor) != signature[2]
                    or tensor.data_ptr() != signature[1]
                    or tensor.dtype != signature[5] or tensor.device != signature[6]
                    or tensor.shape != signature[3] or tensor.stride() != signature[4]):
                return False
        return True

    def _full_signature(self, recipe):
        named_modules = tuple(self.model.named_modules())
        modules = tuple((name, id(module), type(module).__module__,
                         type(module).__qualname__, module.training)
                        for name, module in named_modules)
        if any(row[-1] for row in modules):
            raise ValueError('All model submodules must remain in eval mode')
        tensors = []
        tensor_states = {}
        for kind, iterator in [('parameter', self.model.named_parameters(remove_duplicate=False)),
                               ('buffer', self.model.named_buffers(remove_duplicate=False))]:
            for name, tensor in iterator:
                signature = tensor_signature(tensor)
                tensors.append((kind, name, signature))
                tensor_states[id(tensor)] = (tensor, signature)
        token = recipe, modules, tuple(tensors)

        # Only cache standard PyTorch registration traversal. A custom iterator
        # can derive state from values outside _modules/_parameters/_buffers.
        methods = tuple(name for name, _ in _STANDARD_ENUMERATORS)
        cacheable = all(
            all(getattr(type(module), method) is implementation
                and method not in module.__dict__ for method, implementation in _STANDARD_ENUMERATORS)
            for _, module in named_modules
        )
        if cacheable:
            self._registered_modules = tuple(
                (module, type(module), type(module).__module__, type(module).__qualname__,
                 tuple((name, getattr(module, name), tuple(getattr(module, name).items()))
                       for name in ('_modules', '_parameters', '_buffers')))
                for _, module in named_modules
            )
            self._enumeration_types = tuple(
                (module_type, tuple((method, getattr(module_type, method)) for method in methods))
                for module_type in {type(module) for _, module in named_modules}
            )
            self._tensor_states = tuple(tensor_states.values())
            self._cached_token = token
        else:
            self._registered_modules = ()
            self._tensor_states = ()
            self._cached_token = None
            self._enumeration_types = ()
        return token

    def signature(self):
        if self.model.training:
            raise ValueError('Graph inference model must remain in eval mode')
        # Keep adapter-specific packed-weight, rotary and recipe checks live.
        self.validate()
        recipe = self.recipe()
        if is_dataclass(recipe):
            recipe = asdict(recipe)
        recipe = json.dumps(recipe, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if (self._cached_token is not None and recipe == self._cached_token[0]
                and self._registered_state_unchanged()):
            return self._cached_token
        return self._full_signature(recipe)

