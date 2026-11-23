"""Complete OFT serving recipes, independent of author paths or audit files."""
from copy import deepcopy

from vvla.errors import InputError, UnsupportedConfiguration

_L4 = dict(backend='vllm_fa2', fuse_qkv=False, fuse_residual_square=False,
           fuse_rmsnorm=False, fuse_rope=True, fuse_silu_mul=False,
           qkvo_layout='original', reference_compatible=False)
_G4 = dict(_L4, fuse_residual_square=True, fuse_rmsnorm=True, fuse_silu_mul=True)
_G4_FP8 = dict(backend='vllm_fa2', fuse_residual_square=True, fuse_rmsnorm=True,
               fuse_rope=True, fuse_silu_mul=True, mlp_precision='fp8')
_ALIASES = {'bf16-l4': 'bf16-sm89', 'bf16-g4': 'bf16-sm120',
            'mlp-fp8-g4': 'mlp-fp8-sm120'}
RECIPE_NAMES = ('bf16', 'bf16-optimized', 'bf16-reference', 'bf16-portable',
                'official-bf16', 'bf16-sm89', 'bf16-sm120', 'mlp-fp8',
                'mlp-fp8-sm120', *_ALIASES)


def resolve_recipe(name, *, capability, batch_size=1):
    requested = name
    name = _ALIASES.get(name, name)
    capability = tuple(capability)
    if name not in RECIPE_NAMES:
        raise UnsupportedConfiguration('Unknown OFT execution recipe: ' + requested)
    if name in ('bf16', 'bf16-optimized'):
        name = ('bf16-sm89' if capability == (8, 9) else
                'bf16-sm120' if capability == (12, 0) else 'bf16-portable')
    elif name == 'mlp-fp8':
        name = 'mlp-fp8-sm120' if capability == (12, 0) else 'mlp-fp8'
    required = {'bf16-sm89': (8, 9), 'bf16-sm120': (12, 0), 'mlp-fp8-sm120': (12, 0)}.get(name)
    if required and capability != required:
        raise UnsupportedConfiguration(f'{name} requires compute capability {required}')
    if name == 'mlp-fp8' and capability < (8, 9):
        raise UnsupportedConfiguration('OFT native FP8 requires FP8-capable hardware')
    options = (_L4 if name == 'bf16-sm89' else _G4 if name == 'bf16-sm120' else
               _G4_FP8 if name.startswith('mlp-fp8') else
               dict(backend='sdpa', reference_compatible=True) if name == 'bf16-reference' else
               dict(backend='sdpa', fuse_rope=True, fuse_silu_mul=True,
                    fuse_rmsnorm=True, fuse_residual_square=True))
    loader = dict(batch_size=batch_size, execution='native',
                  cuda_graph=name != 'bf16-reference', native_options=deepcopy(options))
    if name == 'official-bf16':
        loader = dict(batch_size=batch_size, execution='official', cuda_graph=False)
    return dict(requested=requested, resolved=name, capability=list(capability),
                blas_library='cublaslt' if name.startswith('mlp-fp8') else 'cublas',
                historical_configuration=bool(required and batch_size == 1),
                full_weight_release_qualified=False, customized=False, loader=loader)


def merge_options(selection, options):
    options = deepcopy(options)
    allow = options.pop('allow_recipe_overrides', False)
    if type(allow) is not bool:
        raise InputError('allow_recipe_overrides must be boolean')
    loader = deepcopy(selection['loader'])
    native = options.pop('native_options', {})
    if type(native) is not dict:
        raise InputError('native_options must be a dictionary')
    defaults = loader.get('native_options', {})
    changed = {key for key, value in native.items() if key in defaults and defaults[key] != value}
    changed.update(key for key, value in options.items() if key in loader and loader[key] != value)
    if changed and selection['historical_configuration'] and not allow:
        raise InputError('Historical OFT recipe changed: ' + ', '.join(sorted(changed)) +
                         '; use allow_recipe_overrides=True for a labelled ablation')
    if native or 'native_options' in loader:
        loader['native_options'] = dict(defaults, **native)
    loader.update(options)
    selection['customized'] = bool(changed or native.keys() - defaults.keys()
                                   or options.get('paired_image_preprocessing'))
    selection['loader'] = deepcopy(loader)
    return loader
