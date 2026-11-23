"""Fictional stdlib fixtures for host-path evidence, never execution receipts."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).parents[1] / 'src/vvla_repro_pi05/lite/prefix_fp8_inventory_evidence.py'
_SPEC = importlib.util.spec_from_file_location('fictional_inventory_evidence', _PATH)
evidence = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evidence)


def make_inventory_bundle():
    sources = {name: hashlib.sha256(('fictional source ' + name).encode()).hexdigest()
               for name in (*evidence.SOURCE_PATHS, *evidence.NATIVE_SOURCE_PATHS)}
    extension = dict(deepcopy(evidence.INVENTORY_CONSTANTS),
        source_sha256={name: sources[name] for name in evidence.SOURCE_PATHS},
        module_count=851, cold_standard_inventory=True)
    native = dict(deepcopy(evidence.NATIVE_BUILD_CONSTANTS),
        source_sha256={name: sources[name] for name in evidence.NATIVE_SOURCE_PATHS},
        torch_version=evidence.RUNTIME['torch_version'], torch_git_version=evidence.RUNTIME['torch_git_version'],
        torch_cxx11_abi=True, python_version='3.12 fictional CPU fixture',
        python_soabi='cpython-312-x86_64-linux-gnu', platform='fictional Linux', machine='x86_64',
        compiler_request='fictional compiler')
    identity = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    native.update(deepcopy(evidence.NATIVE_RESULT_CONSTANTS),
        module_name='vvla_native_lifecycle_' + identity[:20], compiler='fictional compiler 1',
        binary_sha256='b' * 64)
    # Intentionally a minimal base projection, not a full 51-tensor manifest.
    # Full harness tests must merge these fields into their complete fixture.
    base = dict(precision='mixed_bf16_fp8_prefix', implementation='pi05-prefix-rowwise-fp8-v1',
        scope='prefix_mlp_layers_0_16',
        sites=[{'name': f'prefix.{i}.{p}'} for i in range(17) for p in ('gate_proj', 'up_proj', 'down_proj')],
        lifecycle=dict(version='pi05-prefix-fp8-live-owner-v1', enabled=True,
            source_sha256=sources['prefix_fp8.py'], original_parameters_preserved=True,
            owner='Pi05NativeGraphRunner', fallback_permitted=False),
        backend=dict(deepcopy(evidence.RUNTIME), quantizer_variant='triton-eager-rn-native-sm120-v1',
            compile_options=deepcopy(evidence.COMPILE_OPTIONS),
            hardware={'name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition'}),
        inventory_reuse=deepcopy(extension))
    payload = {path: sources[name] for name, path in {**evidence.SOURCE_PATHS, **evidence.NATIVE_SOURCE_PATHS}.items()}
    return dict(extension=extension, native_manifest=native, quantization_manifest=base,
                source_files_sha256=payload)
