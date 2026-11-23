"""Serving presets, including the complete historical L4 and Blackwell controls.

These are execution choices, not historical qualification requirements. A
checkpoint or GPU UUID never selects admission. Explicit hardware presets keep
their numerical recipe; unsupported geometry is rejected instead of relabelled.
"""
from copy import deepcopy

from vvla.errors import InputError, UnsupportedConfiguration


_COMMON = dict.fromkeys((
    'lite_action_norm_cache', 'lite_batched_vision',
    'lite_compile_preserve_action_linears', 'lite_compile_sample_actions',
    'lite_elide_identity_gathers', 'lite_fused_adarms', 'lite_fused_adarms_reduction',
    'lite_fused_expert_gelu', 'lite_fused_expert_qkv', 'lite_fused_gated_residual',
    'lite_gpu_uint8_prepare', 'lite_host_layout_cache', 'lite_native_lifecycle',
    'lite_skip_weight_init',
), True)

_SM89 = dict(_COMMON,
    lite_expert_cublaslt={'include_down': True, 'profile': 'l4_exact_three'},
    lite_fused_residual_adarms=False, lite_native_lifecycle_variant='v2',
    lite_single_pass_lifecycle=False, lite_suffix_fa2={'splits': 8})

_SM120 = dict(_COMMON,
    lite_expert_cublaslt={'include_down': False, 'profile': 'g4_best_three'},
    lite_fused_expert_gate_up=True, lite_fused_residual_adarms=True,
    lite_native_lifecycle_variant='v3', lite_packed_input_metadata=True,
    lite_request_identity_lookup=False, lite_single_pass_lifecycle=False,
    lite_suffix_fa2={'splits': 8}, lite_vision_fc2_cublaslt=True)

_ALIASES = {'bf16-l4': 'bf16-sm89', 'bf16-g4': 'bf16-sm120',
            'bf16-blackwell': 'bf16-sm120', 'prefix-fp8-g4': 'prefix-fp8-sm120'}
RECIPE_NAMES = ('bf16', 'fp8', 'bf16-reference', 'bf16-optimized', 'bf16-portable',
                'bf16-sm89', 'bf16-sm120', 'bf16-batch-sm89', 'bf16-batch-sm120',
                'prefix-fp8-batch-sm120',
                'prefix-fp8-sm120', 'prefix-fp8-down-sm89', *_ALIASES)


def resolve_recipe(name, *, capability, horizon, max_batch=1, input_contract='lerobot',lt_tactics_available=False):
    """Return fresh loader options and a serializable selection receipt."""
    requested = name
    name = _ALIASES.get(name, name)
    if name not in RECIPE_NAMES:
        raise UnsupportedConfiguration(f'Unknown Pi0.5 BF16 recipe: {requested}')
    capability = tuple(capability)
    fallback=None
    if name == 'fp8':
        if horizon != 50 or input_contract == 'openpi-libero-pi05':
            raise UnsupportedConfiguration('fp8 selects the measured H50 prefix recipe; H10 precision requires an explicit experimental recipe')
        if capability == (12,0):
            name = 'prefix-fp8-sm120' if max_batch == 1 else 'prefix-fp8-batch-sm120'
        elif capability == (8,9) and max_batch == 1:
            name = 'prefix-fp8-down-sm89'
        else:
            raise UnsupportedConfiguration('fp8 supports SM120 H50 and SM89 H50 B1; no BF16 fallback is implicit')
    if name in ('bf16', 'bf16-optimized'):
        name = ('bf16-sm89' if capability == (8, 9) else
                'bf16-sm120' if capability == (12, 0) else 'bf16-portable')
        if horizon != 50 or max_batch != 1:
            name = 'bf16-portable'
            if horizon == 50 and capability == (8,9):
                name = 'bf16-batch-sm89'
            elif horizon == 50 and capability == (12,0):
                if lt_tactics_available:name = 'bf16-batch-sm120'
                else:fallback=dict(from_recipe='bf16-batch-sm120',reason='missing_lt_tactics')
    if name in ('bf16-sm89', 'bf16-sm120', 'prefix-fp8-sm120', 'prefix-fp8-down-sm89'):
        is_sm89 = name in ('bf16-sm89', 'prefix-fp8-down-sm89')
        required = (8, 9) if is_sm89 else (12, 0)
        if capability != required or horizon != 50 or max_batch != 1:
            raise UnsupportedConfiguration(
                f'{name} preserves the B1/H50 recipe on SM{required[0]}{required[1]}; '
                'select an explicit batched or portable recipe for other geometry')
        overrides = deepcopy(_SM89 if is_sm89 else _SM120)
        if name.startswith('prefix-fp8'):
            overrides['lite_prefix_fp8'] = True
    elif name == 'bf16-batch-sm89':
        if capability != (8,9) or horizon != 50:
            raise UnsupportedConfiguration('bf16-batch-sm89 preserves the L4 H50 batch recipe')
        overrides=deepcopy(_COMMON)
        overrides.update(lite_batched_vision=False, lite_batch_geometry='sm89',
                         lite_native_lifecycle_variant='v2', lite_fused_expert_gate_up=False,
                         lite_fused_residual_adarms=False, lite_packed_input_metadata=False,
                         lite_expert_cublaslt=None, lite_suffix_fa2=None)
    elif name in ('bf16-batch-sm120','prefix-fp8-batch-sm120'):
        if capability!=(12,0) or horizon!=50:
            raise UnsupportedConfiguration('The historical Blackwell batch recipe requires SM120/H50')
        overrides=deepcopy(_SM120)
        overrides.update(lite_batched_vision=False,lite_batch_geometry='sm120',
            lite_expert_cublaslt=None,lite_suffix_fa2=None,lite_vision_fc2_cublaslt=False,
            lite_prefix_fp8=name.startswith('prefix-fp8'))
    elif name == 'bf16-portable':
        overrides = deepcopy(_COMMON)
        overrides.update(lite_native_lifecycle_variant='v3',
                         lite_fused_expert_gate_up=True, lite_fused_residual_adarms=True)
        # These existing owners are specifically B1. Batched owners are selected
        # separately, never installed and then silently bypassed at runtime.
        if max_batch != 1:
            overrides.update(lite_batched_vision=False, lite_batch_geometry='portable')
    else:
        overrides = {}
    adapter_adjustments={}
    if input_contract=='openpi-libero-pi05':
        if name not in ('bf16-portable','bf16-reference'):
            raise UnsupportedConfiguration('OpenPI preprocessing requires the portable or reference recipe')
        for key in ('lite_action_norm_cache','lite_gpu_uint8_prepare','lite_packed_input_metadata'):
            if overrides.get(key):
                overrides[key]=False
                adapter_adjustments[key]='LeRobot-only processor hook; preserve OpenPI preprocessing'
    return dict(requested=requested, resolved=name, capability=list(capability),fallback=fallback,
                horizon=horizon, max_batch=max_batch, customized=False,
                input_contract=input_contract,adapter_adjustments=adapter_adjustments,
                historical_configuration=(name in ('bf16-sm89', 'bf16-sm120', 'bf16-batch-sm89',
                    'bf16-batch-sm120','prefix-fp8-batch-sm120','prefix-fp8-sm120', 'prefix-fp8-down-sm89')),
                required_assets=['lt_tactics'] if name in ('bf16-batch-sm120','prefix-fp8-batch-sm120') else [],
                full_weight_release_qualified=False,
                loader=dict(profile='bf16' if name == 'bf16-reference' else 'full-bf16',
                            cuda_graph=True, attention_backend='torch_fa2',
                            config_overrides=overrides))


def merge_options(selection, options):
    """Keep named historical recipes exact; make intentional ablations explicit."""
    options = deepcopy(options)
    allow = options.pop('allow_recipe_overrides', False)
    if type(allow) is not bool:
        raise InputError('allow_recipe_overrides must be boolean')
    loader = deepcopy(selection['loader'])
    overrides = options.pop('config_overrides', {})
    if type(overrides) is not dict:
        raise InputError('config_overrides must be a dictionary')
    changed = {key for key, value in overrides.items()
               if key in loader['config_overrides'] and loader['config_overrides'][key] != value}
    changed.update(key for key, value in options.items() if key in loader and loader[key] != value)
    if changed and selection['historical_configuration'] and not allow:
        raise InputError('Historical recipe overrides change ' + ', '.join(sorted(changed)) +
                         '; set allow_recipe_overrides=True for a labelled ablation')
    loader['config_overrides'].update(overrides)
    loader.update(options)
    selection['customized'] = bool(changed or overrides.keys() - selection['loader']['config_overrides'].keys())
    selection['loader'] = deepcopy(loader)
    return loader
