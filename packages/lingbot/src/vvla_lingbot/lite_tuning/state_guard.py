"""Cache traversal, retain tensor, topology, mode and configuration checks.

The caller must rebuild the engine after any mutation. A token never changes:
check() raises before a graph can replay stale state. No GPU synchronization.
"""
import copy


def tensor_token(t):
    return (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device, t._version)


class FrozenStateGuard:
    def __init__(self, flow):
        self.flow = flow
        self.config = flow.config
        self.config_value = copy.deepcopy(flow.config.to_dict())
        self.modules = []
        self.tensors = []
        self.fresh = []
        seen = set()
        for module in flow.modules():
            maps = tuple((name, tuple(getattr(module, name)), tuple(getattr(module, name).items()))
                         for name in ('_parameters', '_buffers', '_modules'))
            self.modules.append((module, type(module), module.training,
                                 module.__dict__.get('forward'), maps))
            if hasattr(module, 'assert_fresh'):
                self.fresh.append(module.assert_fresh)
            for name in ('_parameters', '_buffers'):
                for t in getattr(module, name).values():
                    if t is not None and id(t) not in seen:
                        seen.add(id(t))
                        self.tensors.append((t, tensor_token(t)))
        self.token = object()
        self.check()

    def check(self):
        for module, cls, training, forward, maps in self.modules:
            if type(module) is not cls or module.training != training or module.__dict__.get('forward') is not forward:
                raise RuntimeError('Model module type/mode/forward changed; rebuild engine')
            for name, keys, entries in maps:
                current = getattr(module, name)
                # Sequential execution and custom forwards can depend on registration
                # order. Equal key sets and object identities alone are insufficient.
                if tuple(current) != keys or any(current[key] is not value for key, value in entries):
                    raise RuntimeError('Model registration changed; rebuild engine')
        for t, expected in self.tensors:
            if tensor_token(t) != expected:
                raise RuntimeError('Model tensor identity/storage/version changed; rebuild engine')
        for callback in self.fresh:
            callback()
        if self.flow.config is not self.config or self.flow.config.to_dict() != self.config_value:
            raise RuntimeError('Model configuration changed; rebuild engine')
        return self.token
