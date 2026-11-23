"""Pure evidence for the opt-in private fixed-checkpoint serving domain.

This declares the supported API and audit scope. It certifies neither speed nor
policy quality, and never describes cold model checks as per-request checks.
"""
from copy import deepcopy
import hashlib
import json
import re

IMPLEMENTATION = 'pi05-fixed-checkpoint-engine-v1'
DEPLOYMENT_CONTRACT = 'private-from-pretrained-prefix51-immutable-v1'
CHECKPOINT_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
LITE = 'vllm-vla-plugin/src/vvla_repro_pi05/lite/'
SOURCE_PATHS = {name: LITE + name for name in ('fixed_checkpoint_engine.py',
    'fixed_checkpoint_engine_evidence.py', 'pi05.py', 'prefix_fp8.py', 'native_lifecycle.py')}
SOURCE_PATHS['modeling_pi05.py'] = 'vllm-vla-plugin/src/vvla_repro_pi05/models/pi05/modeling_pi05.py'
CONSTANTS = dict(schema=1, implementation=IMPLEMENTATION, deployment_contract=DEPLOYMENT_CONTRACT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    validation_scope='cold_seal_and_explicit_full_audit;hot_private_checkpoint_dynamic_request_only',
    public_api=['from_pretrained', 'infer', 'predict', 'reset', 'prepared_fixture',
                'preflight', 'metadata', 'validate_precision_manifest', 'precision_backend_report', 'close'],
    cold_audit_checks=['full_model_precision_and_topology', 'original_and_packed_weight_epochs',
        '51_prefix_site_bindings', 'native_v3_lifecycle', 'backend_source_and_binary',
        'owner_generation_initialization'],
    hot_request_checks=['request_and_runner_ownership', 'closed_and_thread_state',
        'autocast_and_process_math_settings', 'fresh_raw_observation_preprocessing',
        'current_prompt_state_masks_positions_and_graph_key', 'request_local_rng_or_explicit_noise',
        'actual_graph_execution_and_owned_cpu_physical_actions'],
    unsupported_mutations=['external_model_or_processor_injection', 'model_reference_escape',
        'training_or_hot_weight_update', 'hot_precision_or_recipe_reconfiguration'],
    checkpoint_changes='close_and_construct_new_engine',
    request_cache='no_observation_action_rng_result_or_whole_context_cache')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value, expected, field):
    if canonical(value) != canonical(expected):
        raise ValueError('Fixed checkpoint engine mismatch: ' + field)


def keys(value, expected, field):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('Fixed checkpoint engine fields: ' + field)


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Fixed checkpoint engine requires SHA256')


def make_manifest(*, source_sha256, base_quantization_manifest_sha256, native_lifecycle_manifest_sha256):
    value = dict(deepcopy(CONSTANTS), source_sha256=deepcopy(source_sha256),
        base_quantization_manifest_sha256=base_quantization_manifest_sha256,
        native_lifecycle_manifest_sha256=native_lifecycle_manifest_sha256)
    return validate_engine_manifest(value)


def validate_engine_manifest(value, *, base_quantization_manifest_sha256=None,
                             native_lifecycle_manifest_sha256=None):
    keys(value, (*CONSTANTS, 'source_sha256', 'base_quantization_manifest_sha256',
                 'native_lifecycle_manifest_sha256'), 'manifest')
    for name, expected in CONSTANTS.items():
        exact(value[name], expected, name)
    keys(value['source_sha256'], SOURCE_PATHS, 'source_sha256')
    for checksum in (*value['source_sha256'].values(), value['base_quantization_manifest_sha256'],
                     value['native_lifecycle_manifest_sha256']):
        sha(checksum)
    for name, expected in (('base_quantization_manifest_sha256', base_quantization_manifest_sha256),
                           ('native_lifecycle_manifest_sha256', native_lifecycle_manifest_sha256)):
        if expected is not None:
            sha(expected)
            exact(value[name], expected, name)
    return value


def validate_engine_binding(value, *, base_quantization_manifest, native_lifecycle_manifest,
                            source_files_sha256=None):
    """Bind separately full-validated base metadata and the real cold collector.

    ``source_files_sha256`` must come from a separately verified deployed payload.
    A pure declaration cannot prove encapsulation; implementation tests must.
    """
    base, native = base_quantization_manifest, native_lifecycle_manifest
    if type(base) is not dict or type(native) is not dict:
        raise ValueError('Fixed checkpoint engine needs base and native manifests')
    if any(name in base for name in ('fixed_checkpoint_engine', 'vision_fp8', 'inventory_reuse', 'gelu_fusion')):
        raise ValueError('Fixed checkpoint engine requires an undecorated selected prefix51 base')
    validate_engine_manifest(value, base_quantization_manifest_sha256=digest(base),
                             native_lifecycle_manifest_sha256=digest(native))
    for name, expected in dict(precision='mixed_bf16_fp8_prefix',
        implementation='pi05-prefix-rowwise-fp8-v1', scope='prefix_mlp_layers_0_16',
        checkpoint_sha256=CHECKPOINT_SHA256).items():
        exact(base.get(name), expected, 'base.' + name)
    backend, lifecycle = base.get('backend'), base.get('lifecycle')
    if type(backend) is not dict or type(lifecycle) is not dict:
        raise ValueError('Fixed checkpoint engine requires backend and lifecycle')
    exact(backend.get('quantizer_variant'), 'triton-eager-rn-native-sm120-v1', 'selected backend')
    exact(lifecycle.get('source_sha256'), value['source_sha256']['prefix_fp8.py'], 'prefix source')
    exact(native.get('collector_variant'), 'v3', 'native collector variant')
    exact(native.get('enabled'), True, 'native collector enabled')
    native_sources = native.get('source_sha256')
    if type(native_sources) is not dict:
        raise ValueError('Fixed checkpoint engine requires native source binding')
    for name in ('native_lifecycle.py', 'modeling_pi05.py'):
        exact(native_sources.get(name), value['source_sha256'][name], 'native source.' + name)
    for name in ('torch_version', 'torch_git_version'):
        exact(native.get(name), backend.get(name), 'native/backend.' + name)
    sha(native.get('binary_sha256'))
    if source_files_sha256 is not None:
        if type(source_files_sha256) is not dict:
            raise ValueError('Verified deployed source mapping required')
        for name, path in SOURCE_PATHS.items():
            sha(source_files_sha256.get(path))
            exact(source_files_sha256[path], value['source_sha256'][name], 'deployed source.' + path)
    return value


def deployment_record(value):
    validate_engine_manifest(value)
    return dict(implementation=IMPLEMENTATION, deployment_contract=DEPLOYMENT_CONTRACT,
                manifest_sha256=digest(value))
