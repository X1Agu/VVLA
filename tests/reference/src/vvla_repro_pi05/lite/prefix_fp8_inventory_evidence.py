"""Pure validation of same-call inventory reuse; no runtime or speed claims.

The source-only shape check is deliberately separate from the required final
metadata binding. A full quantization manifest/backend is still validated by
its existing validator. This module only validates the additional host path.
"""
import hashlib
import json
import re

VERSION = 'prefix-fp8-fresh-v3-module-inventory-v1'
LITE = 'vllm-vla-plugin/src/vvla_repro_pi05/lite/'
CORE = 'vllm-vla-plugin/src/vvla_repro_pi05/models/pi05/'
SOURCE_PATHS = {name: LITE + name for name in
    ('prefix_fp8_inventory.py', 'native_lifecycle.py', 'prefix_fp8.py')}
SOURCE_PATHS['modeling_pi05.py'] = CORE + 'modeling_pi05.py'
NATIVE_SOURCE_PATHS = dict(SOURCE_PATHS)
NATIVE_SOURCE_PATHS.pop('prefix_fp8_inventory.py')
NATIVE_SOURCE_PATHS.pop('prefix_fp8.py')
NATIVE_SOURCE_PATHS.update({'native_lifecycle_v3.cpp': LITE + 'csrc/native_lifecycle_v3.cpp',
    'lifecycle.py': LITE + 'lifecycle.py', 'inference_cache_pi05.py': CORE + 'inference_cache_pi05.py'})
INVENTORY_CONSTANTS = dict(version=VERSION, enabled=True,
    dependency='G4 prefix FP8 51-site owner; native lifecycle v3; unique compiled native runner',
    reused='module DFS inventory only, authorized by the same-call fresh native snapshot',
    intervening_mutation_guard='live flat child-slot identity/order/None/alias comparison before scan',
    preserved=['no-extra-QuantizedLinear scan', '51 original weight live identities',
               '102 private packed/scale live identities', 'all forward/precision/owner checks'],
    fallback='original model.modules DFS for unsupported access or changed topology',
    snapshot_lifetime='current callback/thread only; retired on success and exception')
NATIVE_BUILD_CONSTANTS = dict(version='native-live-full-lifecycle-v3-validated-value-reuse',
    collector_variant='v3',
    allocation_strategy='fresh native snapshot; caller-owned immutable triple reuse only after complete live comparison',
    extra_cflags=['-O3'], with_cuda=False,
    handoff_schema='exact full tuple: six current outer fields plus three immutable lifecycle fields')
NATIVE_RESULT_CONSTANTS = dict(inventory_cache=False, signature_cache=True, fallback='explicit rejection',
    signature_cache_owner='existing Pi05InferenceCache only; collector retains no result',
    signature_reuse='existing immutable three subtuples after complete live validation',
    all_live_fields_read_every_call=True, new_persistent_model_or_tensor_references=False,
    comparison='exact scalar fields; no hash approximation or Python rich equality',
    comparison_fallback='materialize the same native snapshot; never walk model twice',
    registry_then_tensor_reads=True, enabled=True,
    returned_fields=['parameters', 'buffers', 'topology'],
    tensor_identity_fields=['object_id', 'data_ptr', 'version', 'dtype', 'device', 'shape'],
    topology='all module nodes/types and all child/parameter/buffer slots, including None and aliases',
    restrictions='CPython; exact dict registries; exact Tensor/Parameter; no shadowed identity fields')
NATIVE_VARIABLE_BUILD = ('source_sha256', 'torch_version', 'torch_git_version', 'torch_cxx11_abi',
    'python_version', 'python_soabi', 'platform', 'machine', 'compiler_request')
RUNTIME = dict(torch_version='2.13.0+cu130', cuda_version='13.0',
    torch_git_version='cf30153c4c131c8164ee7798e5022d810682e2cb', capability=[12, 0])
COMPILE_OPTIONS = {'triton.cudagraphs': False, 'emulate_precision_casts': True, 'use_fast_math': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _exact(value, expected, field):
    if canonical(value) != canonical(expected):
        raise ValueError('Inventory evidence mismatch: ' + field)


def _keys(value, expected, field):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('Inventory evidence fields: ' + field)


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Inventory evidence requires SHA256')


def validate_inventory_shape(extension):
    """Structural check only; final acceptance also requires the binding below."""
    _keys(extension, (*INVENTORY_CONSTANTS, 'source_sha256', 'module_count', 'cold_standard_inventory'), 'inventory')
    for key, expected in INVENTORY_CONSTANTS.items():
        _exact(extension[key], expected, key)
    _keys(extension['source_sha256'], SOURCE_PATHS, 'inventory sources')
    for value in extension['source_sha256'].values():
        _sha(value)
    count, standard = extension['module_count'], extension['cold_standard_inventory']
    if (type(count) is not int or type(standard) is not bool or count < 0
            or (standard and count == 0) or (not standard and count != 0)):
        raise ValueError('Inventory evidence has inconsistent cold module eligibility')
    return extension


def validate_inventory_manifest(extension, *, native_manifest, quantization_manifest, source_files_sha256=None):
    """Bind the actual v3 collector, original G4 FP8 owner and optional payload.

    ``source_files_sha256`` is an already verified deployed inventory mapping
    repository-relative paths to SHA256, never a guessed local working tree.
    Shape/metadata binding alone is not proof of GPU quality or a speedup.
    """
    validate_inventory_shape(extension)
    native = native_manifest
    _keys(native, (*NATIVE_BUILD_CONSTANTS, *NATIVE_VARIABLE_BUILD, *NATIVE_RESULT_CONSTANTS,
                   'module_name', 'compiler', 'binary_sha256'), 'native v3 manifest')
    for key, expected in {**NATIVE_BUILD_CONSTANTS, **NATIVE_RESULT_CONSTANTS}.items():
        _exact(native[key], expected, 'native.' + key)
    _keys(native['source_sha256'], NATIVE_SOURCE_PATHS, 'native v3 sources')
    for value in (*native['source_sha256'].values(), native['binary_sha256']):
        _sha(value)
    for key in ('python_version', 'python_soabi', 'platform', 'machine', 'compiler_request', 'compiler'):
        if type(native[key]) is not str or not native[key]:
            raise ValueError('Invalid native v3 runtime/compiler: ' + key)
    if type(native['torch_cxx11_abi']) is not bool:
        raise ValueError('Invalid native v3 CXX ABI')
    build = {key: native[key] for key in (*NATIVE_BUILD_CONSTANTS, *NATIVE_VARIABLE_BUILD)}
    identity = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    _exact(native['module_name'], 'vvla_native_lifecycle_' + identity[:20], 'native build identity')
    for name in ('native_lifecycle.py', 'modeling_pi05.py'):
        _exact(extension['source_sha256'][name], native['source_sha256'][name], 'same-call source.' + name)

    base = quantization_manifest
    if type(base) is not dict:
        raise ValueError('Inventory requires the full original 51-site quantization manifest')
    for key, expected in dict(precision='mixed_bf16_fp8_prefix', implementation='pi05-prefix-rowwise-fp8-v1',
                              scope='prefix_mlp_layers_0_16').items():
        _exact(base.get(key), expected, 'base.' + key)
    sites = base.get('sites')
    if type(sites) is not list or any(type(site) is not dict for site in sites):
        raise ValueError('Inventory requires all original 51-site records')
    _exact([site.get('name') for site in sites], [f'prefix.{i}.{p}' for i in range(17)
        for p in ('gate_proj', 'up_proj', 'down_proj')], 'base site order/scope')
    lifecycle, backend = base.get('lifecycle'), base.get('backend')
    if type(lifecycle) is not dict or type(backend) is not dict:
        raise ValueError('Inventory requires the original FP8 lifecycle/backend')
    for key, expected in dict(version='pi05-prefix-fp8-live-owner-v1', enabled=True,
            original_parameters_preserved=True, owner='Pi05NativeGraphRunner', fallback_permitted=False).items():
        _exact(lifecycle.get(key), expected, 'base lifecycle.' + key)
    _exact(lifecycle.get('source_sha256'), extension['source_sha256']['prefix_fp8.py'], 'original owner source')
    _exact(backend.get('quantizer_variant'), 'triton-eager-rn-native-sm120-v1', 'base backend variant')
    _exact(backend.get('compile_options'), COMPILE_OPTIONS, 'unchanged compilation options')
    for key, expected in RUNTIME.items():
        _exact(backend.get(key), expected, 'base runtime.' + key)
    for key in ('torch_version', 'torch_git_version'):
        _exact(native[key], backend[key], 'native/base runtime.' + key)
    if type(backend.get('hardware')) is not dict:
        raise ValueError('Inventory requires physical G4 hardware evidence')
    _exact(backend['hardware'].get('name'), 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'G4 hardware')
    if 'inventory_reuse' in base:
        _exact(base['inventory_reuse'], extension, 'base extension binding')
    if source_files_sha256 is not None:
        if type(source_files_sha256) is not dict:
            raise ValueError('Inventory requires a verified source SHA mapping')
        for manifest, paths in ((extension, SOURCE_PATHS), (native, NATIVE_SOURCE_PATHS)):
            for name, path in paths.items():
                value = source_files_sha256.get(path)
                _sha(value)
                _exact(value, manifest['source_sha256'][name], 'deployed source.' + path)
    return extension
