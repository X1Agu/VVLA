"""Pure source-binding contracts; fixtures do not claim a GPU run or speedup."""
from copy import deepcopy
import hashlib
import json

import pytest

from prefix_fp8_inventory_fixtures import evidence, make_inventory_bundle


def validate(bundle):
    return evidence.validate_inventory_manifest(**bundle)


def reseal_native(bundle):
    native = bundle['native_manifest']
    build = {key: native[key] for key in (*evidence.NATIVE_BUILD_CONSTANTS, *evidence.NATIVE_VARIABLE_BUILD)}
    native['module_name'] = 'vvla_native_lifecycle_' + hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()[:20]


def test_same_call_manifest_binds_native_build_owner_and_all_deployed_sources():
    bundle = make_inventory_bundle()
    before = deepcopy(bundle)
    assert validate(bundle) is bundle['extension']
    assert bundle == before
    assert set(bundle['source_files_sha256']) == set(evidence.SOURCE_PATHS.values()) | set(evidence.NATIVE_SOURCE_PATHS.values())


@pytest.mark.parametrize('change', ('unknown', 'false_enabled', 'integer_enabled', 'bool_count',
    'negative_count', 'missing_source', 'non_sha', 'dropped_live_read', 'cross_request_lifetime', 'false_fallback'))
def test_shape_rejects_weakened_claims_and_ambiguous_types(change):
    ext = make_inventory_bundle()['extension']
    if change == 'unknown': ext['quality_validated'] = True
    elif change == 'false_enabled': ext['enabled'] = False
    elif change == 'integer_enabled': ext['enabled'] = 1
    elif change == 'bool_count': ext['module_count'] = True
    elif change == 'negative_count': ext['module_count'] = -1
    elif change == 'missing_source': ext['source_sha256'].pop('prefix_fp8.py')
    elif change == 'non_sha': ext['source_sha256']['prefix_fp8.py'] = 'not a digest'
    elif change == 'dropped_live_read': ext['preserved'].remove('102 private packed/scale live identities')
    elif change == 'cross_request_lifetime': ext['snapshot_lifetime'] = 'cached across requests'
    else: ext['fallback'] = 'silently trust the old topology'
    with pytest.raises(ValueError): evidence.validate_inventory_shape(ext)


def test_unsupported_standard_inventory_truthfully_records_original_dfs_fallback():
    bundle = make_inventory_bundle()
    bundle['extension'].update(module_count=0, cold_standard_inventory=False)
    bundle['quantization_manifest']['inventory_reuse'] = deepcopy(bundle['extension'])
    validate(bundle)
    bundle['extension']['module_count'] = 1
    with pytest.raises(ValueError): validate(bundle)


@pytest.mark.parametrize('change', ('v2', 'disabled_native', 'lost_live_checks', 'mutable_cache',
    'stale_module_name', 'native_wrapper', 'core_source', 'cpp_payload', 'missing_payload',
    'owner_source', 'inventory_source', 'different_torch', 'partial_sites', 'wrong_order',
    'l4_down', 'nvfp4', 'wrong_compile', 'stale_extension'))
def test_final_binding_rejects_different_collector_payload_owner_or_precision(change):
    bundle = make_inventory_bundle()
    native, base = bundle['native_manifest'], bundle['quantization_manifest']
    if change == 'v2': native['collector_variant'] = 'v2'; reseal_native(bundle)
    elif change == 'disabled_native': native['enabled'] = False
    elif change == 'lost_live_checks': native['all_live_fields_read_every_call'] = False
    elif change == 'mutable_cache': native['inventory_cache'] = True
    elif change == 'stale_module_name': native['module_name'] = 'vvla_native_lifecycle_' + '0' * 20
    elif change in ('native_wrapper', 'core_source'):
        name = 'native_lifecycle.py' if change == 'native_wrapper' else 'modeling_pi05.py'
        native['source_sha256'][name] = '0' * 64
        reseal_native(bundle)
    elif change == 'cpp_payload': bundle['source_files_sha256'][evidence.NATIVE_SOURCE_PATHS['native_lifecycle_v3.cpp']] = '0' * 64
    elif change == 'missing_payload': bundle['source_files_sha256'].pop(evidence.SOURCE_PATHS['prefix_fp8_inventory.py'])
    elif change == 'owner_source': base['lifecycle']['source_sha256'] = '0' * 64
    elif change == 'inventory_source': bundle['source_files_sha256'][evidence.SOURCE_PATHS['prefix_fp8_inventory.py']] = '0' * 64
    elif change == 'different_torch': native['torch_version'] = '2.12.0'; reseal_native(bundle)
    elif change == 'partial_sites': base['sites'].pop()
    elif change == 'wrong_order': base['sites'][0], base['sites'][1] = base['sites'][1], base['sites'][0]
    elif change == 'l4_down': base['backend']['quantizer_variant'] = 'sm89-down-scalar-fp32-native-v1'
    elif change == 'nvfp4': base['precision'] = 'mixed_bf16_nvfp4_prefix'
    elif change == 'wrong_compile': base['backend']['compile_options']['use_fast_math'] = True
    else: base['inventory_reuse']['module_count'] += 1
    with pytest.raises(ValueError): validate(bundle)


def test_structural_check_cannot_substitute_for_final_metadata_binding():
    bundle = make_inventory_bundle()
    assert evidence.validate_inventory_shape(bundle['extension'])
    with pytest.raises(TypeError): evidence.validate_inventory_manifest(bundle['extension'])
    with pytest.raises(ValueError):
        evidence.validate_inventory_manifest(bundle['extension'], native_manifest=None,
            quantization_manifest=bundle['quantization_manifest'])
