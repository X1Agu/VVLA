"""Independent BF16 deployment identity; never reuse prefix-FP8 evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

CHECKPOINT = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
RESOLVED_CONFIG = '6199d7cf23b3a127ab75b72de10e2293229dd27bd5e336e0fbce3b9a11c028b7'
LINEAGES = {
    'g4': dict(identity_sha256='d4f4397d54a5ab96fed2c3a2529c331f2b65ea21dfae19d592a6f8624cf9e17d',
        recipe_sha256='1837f987f34b45d28930eb8f03d25b2d029e539e89375ff4c0aa3ff5086737d6',
        device_name='NVIDIA RTX PRO 6000 Blackwell Server Edition', capability=[12, 0], native_variant='v3'),
    'l4': dict(identity_sha256='aedddeb6f3683b00048740356270de164e00285c0816e2923798f3d1b866eb71',
        recipe_sha256='c4aa50f2a0e914456c4edb916afb547fc320d880685fde4700465ec74e185ca9',
        device_name='NVIDIA L4', capability=[8, 9], native_variant='v2'),
}
FORBIDDEN_OPTIONS = ('lite_prefix_fp8', 'lite_prefix_nvfp4', 'lite_vision_fp8',
    'lite_prefix_fp8_inventory', 'lite_prefix_fp8_gelu', 'lite_fp8_host_epochs',
    'lite_fixed_checkpoint_engine')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_origin(path, hardware_profile):
    if hardware_profile not in LINEAGES:
        raise ValueError('BF16 owner supports the separately recorded g4/l4 lineages')
    path = Path(path)
    expected = LINEAGES[hardware_profile]
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['identity_sha256']:
        raise ValueError('BF16 owner requires the exact original BF16 identity bytes')
    value = json.loads(path.read_bytes())
    metadata, config = value['metadata'], value['config']
    if (metadata['precision'] != 'bf16' or metadata['checkpoint_sha256'] != CHECKPOINT
            or metadata['recipe_sha256'] != expected['recipe_sha256']
            or digest(metadata['resolved_config']) != RESOLVED_CONFIG):
        raise ValueError('BF16 origin checkpoint, resolved config or recipe differs')
    options = config['config_overrides']
    if (config['profile'] != 'full-bf16' or config['attention_backend'] != 'torch_fa2'
            or config['cuda_graph'] is not True
            or options.get('lite_native_lifecycle') is not True
            or options.get('lite_native_lifecycle_variant') != expected['native_variant']
            or any(options.get(name, False) is not False for name in FORBIDDEN_OPTIONS)):
        raise ValueError('BF16 origin has an unsupported graph, precision or lifecycle recipe')
    return deepcopy(dict(profile=hardware_profile, identity_path=str(path.resolve()),
        identity_sha256=expected['identity_sha256'], recipe_sha256=expected['recipe_sha256'],
        config=config, metadata=metadata))


def validate_base(metadata, origin):
    if (metadata['precision'] != 'bf16' or metadata['checkpoint_sha256'] != CHECKPOINT
            or digest(metadata['resolved_config']) != RESOLVED_CONFIG
            or metadata['action_shape'] != [50, 7] or metadata['denoise_steps'] != 10
            or metadata['batch'] != 1 or metadata['sampler_dtype'] != 'float32'
            or metadata['attention_backend'] != 'torch_fa2' or metadata['cuda_graph'] is not True):
        raise ValueError('Actual BF16 model does not implement the bound B1/H50/10-step recipe')
    if any(name in metadata for name in ('quantization_manifest', 'bf16_deployment_manifest')):
        raise ValueError('BF16 seal requires an undecorated actual BF16 loader')
    for field in ('float32_matmul_precision', 'cuda_blas_preference', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32'):
        if metadata[field] != origin['metadata'][field]:
            raise ValueError('BF16 process math differs from original recipe: ' + field)
    actual = metadata['lite_optimizations']
    for name, value in origin['config']['config_overrides'].items():
        if actual.get(name.removeprefix('lite_')) != value:
            raise ValueError('BF16 optimization changed: ' + name)
    if any(actual.get(name.removeprefix('lite_'), False) is not False for name in FORBIDDEN_OPTIONS):
        raise ValueError('A quantized or foreign private owner entered the BF16 domain')
    native = actual['native_lifecycle_manifest']
    if native['enabled'] is not True or native['collector_variant'] != LINEAGES[origin['profile']]['native_variant']:
        raise ValueError('BF16 requires the original complete native lifecycle collector')


def make_manifest(*, origin, hardware, source_sha256, base_metadata, precision, process_math):
    validate_base(base_metadata, origin)
    if precision['precision'] != 'bf16' or precision['quantized_owners']:
        raise ValueError('BF16 deployment cannot borrow mixed-precision qualification')
    return dict(schema=1, implementation='pi05-private-bf16-checkpoint-v1',
        deployment_contract='private-B1-original-BF16-recipe-v1', precision='bf16', sampler_dtype='float32',
        origin={k: origin[k] for k in ('profile', 'identity_sha256', 'recipe_sha256')},
        checkpoint_sha256=CHECKPOINT, resolved_config_sha256=RESOLVED_CONFIG,
        hardware=deepcopy(hardware), source_sha256=deepcopy(source_sha256),
        actual_base_recipe_sha256=base_metadata['recipe_sha256'],
        native_lifecycle_manifest_sha256=digest(base_metadata['lite_optimizations']['native_lifecycle_manifest']),
        precision_manifest_sha256=digest(precision),
        process_math=deepcopy(process_math),
        cold_audit_checks=['full_model_collector_and_installed_owners', 'registered_parameter_dtypes',
                           'no_quantized_owner', 'actual_source_native_and_original_recipe'],
        hot_request_checks=['private_model_generation_and_signature', 'serialized_policy_and_runner',
            'thread_owner_handoff', 'autocast_and_process_math', 'fresh_raw_inputs_masks_positions_graph_key',
            'current_independent_rng_or_explicit_noise', 'actual_graph_execution_and_owned_CPU_actions'],
        unsupported_mutations=['external_model_processor_or_weight_injection', 'private_reference_access_or_monkeypatching',
                               'hot_training_precision_recipe_or_weight_changes'],
        reconstruction='close_and_construct_from_checkpoint', policy_quality_passed=False,
        qualification_scope='cold seal and explicit audit; separate same-card model/performance/quality campaign required')
