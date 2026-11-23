"""In-process policy execution without a vLLM or Omni service runtime."""

from importlib import import_module

__all__ = ["Pi05LitePolicy", "create_adapter"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("vvla_repro_pi05.lite.pi05"), name)
