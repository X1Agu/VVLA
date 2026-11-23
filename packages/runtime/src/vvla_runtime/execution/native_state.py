"""Explicit cold-loaded OFT live-state comparator; never a model-state cache.

The Python ModelStateContract still owns its original immutable token and strong
registration/tensor references. This helper only compares those with live state.
It performs no tensor operations, GPU reads, imports, builds, or synchronization
on a request. Unsupported Python overrides use the unchanged Python comparator;
build failures and unexpected runtime errors propagate without fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import sysconfig


VERSION = 'oft-native-live-state-comparator-v1'
_SOURCE = Path(__file__).with_name('csrc') / 'native_state.cpp'
_FIELDS = ('data_ptr', '_version', 'shape', 'stride', 'dtype', 'device', '__getattribute__')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compiler_provenance(command):
    """Resolve the command selected by Torch, including explicit cache wrappers.

    This hashes files only; it does not run a compiler/version command. An
    ambiguous shell command is rejected instead of claiming the wrong driver.
    """
    if type(command) is not str or not command.strip():
        raise ValueError('Empty C++ compiler command')
    words = shlex.split(command)
    executables = []
    for index, word in enumerate(words):
        found = shutil.which(word)
        if found is None:
            raise RuntimeError('Cannot resolve actual C++ compiler executable: ' + word)
        path = Path(found).resolve(strict=True)
        wrapper = Path(word).name in ('ccache', 'sccache')
        if (wrapper and index == len(words) - 1) or (not wrapper and index != len(words) - 1):
            raise ValueError('Use an explicit compiler or cache-wrapper/compiler command without shell options')
        executables.append(dict(role='wrapper' if wrapper else 'compiler', requested=word,
                                resolved_path=str(path), sha256=_sha(path)))
    prefixes = {name: os.environ[name] for name in ('CCACHE_PREFIX', 'CCACHE_PREFIX_CPP')
                if os.environ.get(name)}
    if prefixes:
        raise ValueError('Unset implicit ccache prefix wrappers or express the complete compiler command explicitly')
    return dict(torch_selected_command=command, argv=words, executables=executables,
                implicit_wrapper_prefixes=prefixes,
                scope='actual compiler/cache-wrapper files; system headers/linker are not independently hashed')


class NativeStateComparator:
    """Loaded native code and fixed API descriptors only; no owner/tensor refs."""

    def __init__(self, extension, manifest, tensor_api, *, versioned=False):
        self._extension = extension
        self._manifest = json.loads(json.dumps(manifest, sort_keys=True))
        self._tensor_api = tensor_api
        self._versioned = versioned

    def compare(self, contract):
        provider = self._extension
        if self._versioned:
            provider = getattr(contract, '_versioned_registry_snapshot', None)
            if provider is None:
                provider = contract._versioned_registry_snapshot = self._extension.VersionedComparator()
        result = provider.compare(contract.model, contract._registered_modules,
                                         contract._tensor_states, contract._enumeration_types,
                                         self._tensor_api)
        if result is not None and type(result) is not bool:
            raise TypeError('Native OFT comparator returned an invalid result')
        return result

    def manifest(self):
        return json.loads(json.dumps(self._manifest, sort_keys=True))


def load_native_state_comparator(*, verbose=False, versioned=False):
    """Build once before serving; explicitly inject into ModelStateContract.

    Example: ModelStateContract(adapter,
        native_comparator=load_native_state_comparator()). No environment switch,
    global installation, default change, or compilation during signature().
    """
    if type(verbose) is not bool:
        raise TypeError('verbose must be bool')
    if type(versioned) is not bool or (versioned and sys.version_info[:2] != (3,12)):
        raise ValueError('Versioned registration checks require CPython 3.12')
    if sys.implementation.name != 'cpython' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('OFT native state requires CPython with the GIL')
    import torch
    from torch.utils.cpp_extension import get_cxx_compiler, load

    # Compare live class descriptors every call before reading TensorImpl fields.
    # A class/instance override falls back to Python rather than being bypassed.
    tensor_api = tuple((kind, tuple((name, getattr(torch._C.TensorBase, name))
                                    for name in _FIELDS) +
                       (('__torch_function__', vars(kind)['__torch_function__']),))
                       for kind in (torch.Tensor, torch.nn.Parameter))
    manifest = dict(version=VERSION, default_enabled=False,
        source_sha256={path.name: _sha(path) for path in
                       (_SOURCE, Path(__file__), Path(__file__).with_name('state.py'))},
        torch_version=torch.__version__, torch_git_version=torch.version.git_version,
        torch_cxx11_abi=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None),
        python_version=sys.version, python_soabi=sysconfig.get_config_var('SOABI'),
        platform=platform.platform(), machine=platform.machine(), with_cuda=False,
        extra_cflags=['-O3'], compiler_request=os.environ.get('CXX', 'default C++ compiler'),
        compiler_command=_compiler_provenance(get_cxx_compiler()),
        cache_owner='existing ModelStateContract only; original cached_token unchanged',
        inventory_cache=False, tensor_content_reads=False, gpu_operations=False,
        tensor_fields=['object_id via registered slots', 'data_ptr', 'version',
                       'shape', 'stride', 'dtype', 'device'],
        module_fields=['type identity', 'type module/qualname', 'training',
                       'registry identity', 'registry length/order/names/None/aliases',
                       'named_* class bindings and instance overrides'],
        fallback='unchanged Python comparator for unsupported module/metaclass/tensor/API/registry overrides',
        runtime_failure='propagate to existing runner retirement',
        live_adapter_validation='unchanged recipe, RoPE, layout and math validation',
        concurrency='caller request lock and GIL; no GIL release')
    if versioned:
        manifest.update(version='oft-live-registration-versions-v2', inventory_cache=True,
            cache_owner='ModelStateContract only; dictionary and type registration snapshot; tensor metadata checked at every call')
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    initialized = torch.cuda.is_initialized()
    try:
        extension = load(name='vvla_oft_native_state_' + identity[:20],
                         sources=[str(_SOURCE)], extra_cflags=['-O3'],
                         with_cuda=False, verbose=verbose)
    except Exception as exc:
        raise RuntimeError('OFT native state build/load failed; explicit candidate rejected') from exc
    if torch.cuda.is_initialized() != initialized:
        raise RuntimeError('Host-only OFT state loading unexpectedly initialized CUDA')
    manifest.update(module_name=extension.__name__, compiler=extension.compiler(),
                    binary_sha256=_sha(extension.__file__))
    return NativeStateComparator(extension, manifest, tensor_api, versioned=versioned)
