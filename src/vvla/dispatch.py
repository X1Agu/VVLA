"""Select model and serving settings in the isolated worker before loading weights."""
from copy import deepcopy
from functools import lru_cache
from importlib import resources, util
import json
import os
from pathlib import Path

from .assets import resolve_assets
from .errors import InputError, UnsupportedConfiguration

GIB = 1024 ** 3


def card_name(name, sm, total):
    name = name.upper()
    if sm == (12, 0) and 'PRO 6000' in name and 'SERVER' in name and total >= 90 * GIB:
        return 'G4'
    if sm == (8, 0) and 'A100' in name and total >= 75 * GIB:
        return 'A100 80 GB'
    if 22 * GIB <= total <= 26 * GIB:
        for model, cap in (('RTX 4090', (8, 9)), ('RTX 3090', (8, 6))):
            if cap == sm and name.endswith(model):
                return model
    return 'unmeasured'


def hardware(device):
    import torch
    if not torch.cuda.is_available():
        raise UnsupportedConfiguration('CUDA is unavailable in the selected worker environment')
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    sm = (props.major, props.minor)
    return dict(name=props.name, capability=list(sm), total_bytes=total, free_bytes=free,
                profile=card_name(props.name, sm, total), torch=str(torch.__version__),
                cuda=torch.version.cuda)


@lru_cache(maxsize=1)
def profiles():
    rows = json.loads(resources.files('vvla').joinpath('data/online-profiles.json').read_bytes())['profiles']
    return {(r['model'], r['gpu'], r['recipe']): r for r in rows}


def serving_defaults(model, dispatch):
    defaults = {key: dispatch[key] for key in ('max_batch', 'wait_ms')}
    key = (model, dispatch['hardware_profile']['profile'], dispatch['resolved_recipe'])
    network = profiles().get(key, {}).get('network_fifo')
    if network and network['max_batch'] == defaults['max_batch']:
        defaults['network_wait_ms'] = network['wait_ms']
    return defaults


def select(config, gpu, *, horizon=50, input_contract='lerobot', native_fa2=False):
    """Pure resolution step. Returned options are independent of the frozen presets."""
    model, requested = config['model'], config.get('recipe') or 'auto'
    options = deepcopy(config.get('options', {}))
    size_key = 'batch_size' if model == 'oft' else 'max_batch'
    width = config.get('max_batch')
    if model != 'lingbot' and size_key in options:
        if width is not None and options[size_key] != width:
            raise InputError(f'max_batch conflicts with options.{size_key}')
        width = options[size_key]
    if width is not None and (type(width) is not int or width < 1):
        raise InputError("max_batch must be a positive integer")
    mode = config.get('mode') or ('fixed' if width and width > 1 else 'direct')
    if mode not in ('direct', 'fixed', 'serve'):
        raise InputError('mode must be direct, fixed or serve')
    if mode == 'direct' and width not in (None, 1):
        raise InputError('dedicated B1 requires max_batch=1')
    card, sm = gpu['profile'], tuple(gpu['capability'])
    free = gpu['free_bytes'] / GIB
    reasons = []
    scope = 'measured hardware' if card != 'unmeasured' else 'unmeasured hardware'
    if card == 'unmeasured':
        reasons.append('device is outside the four measured hardware profiles')
    batch = width or (1 if mode == 'direct' else 4)
    wait = 0
    settings = {}
    recipe = requested
    reference = None
    if model == 'pi05':
        if not width and mode != 'direct' and card == 'G4':
            batch = 16
        if requested in ('bf16-reference', 'bf16-sm89', 'bf16-sm120', 'prefix-fp8-down-sm89', 'prefix-fp8-sm120') and width is None:
            batch = 1
        if batch not in (1, 2, 4, 8, 16, 32):
            raise UnsupportedConfiguration('Pi0.5 supports exact widths 1, 2, 4, 8, 16 and 32')
        if requested in ('auto', 'prefix-fp8'):
            if requested == 'prefix-fp8':
                recipe = 'fp8'
            elif horizon != 50 or input_contract == 'openpi-libero-pi05':
                recipe = 'bf16-portable'
                reasons.append('checkpoint uses the portable input geometry')
            elif card == 'G4':
                recipe = 'prefix-fp8-sm120' if batch == 1 else 'prefix-fp8-batch-sm120'
            elif card == 'RTX 4090':
                recipe = 'bf16-sm89' if batch == 1 else 'bf16-batch-sm89'
            else:
                recipe = 'bf16-portable'
        settings = {'max_batch': batch}
        wait = 2 if card == 'A100 80 GB' and batch > 1 else 0
    elif model == 'oft':
        if requested in ('bf16-reference', 'official-bf16') and width is None:
            batch = 1
        if not 1 <= batch <= 32:
            raise UnsupportedConfiguration('OFT batch width must be between 1 and 32')
        tuned = requested in ('auto', 'bf16', 'mixed-fp8')
        if tuned:
            mixed = requested == 'mixed-fp8'
            if mixed and (sm not in ((8, 9), (12, 0)) or not native_fa2):
                raise UnsupportedConfiguration('mixed-fp8 requires SM89/SM120 and the compatible vLLM FA2 native backend')
            template_card = card
            if card == 'unmeasured':
                template_card = 'RTX 3090'
                if mixed:
                    template_card = 'G4' if sm == (12, 0) else 'RTX 4090'
            recipes = {'G4': 'bf16-sm120', 'RTX 4090': 'bf16-sm89'}
            recipe = ('mlp-fp8-sm120' if sm == (12, 0) else 'mlp-fp8') if mixed else recipes.get(template_card, 'bf16-portable')
            reference = profiles()[model, template_card, recipe]
            settings = deepcopy(reference['options'])
            native = settings['native_options']
            if not mixed:
                if card == 'unmeasured':
                    native.update(backend='vllm_fa2' if native_fa2 else 'sdpa',
                                  packed_single_kv=native_fa2, pack_gate_up=free >= 28,
                                  fuse_qkv=free >= 20)
                if not native_fa2:
                    recipe = 'bf16-portable'
                    native.update(backend='sdpa', packed_single_kv=False,
                                  fuse_rmsnorm=True, fuse_residual_square=True, fuse_silu_mul=True)
                    if template_card in ('G4', 'RTX 4090', 'A100 80 GB') and not native_fa2:
                        reasons.append('vLLM FA2 unavailable, using SDPA')
                if native.get('pack_gate_up') and free < 28:
                    native['pack_gate_up'] = False
                    reasons.append('available memory below 28 GiB, disabled gate/up packing')
                if native.get('fuse_qkv') and free < 20:
                    native['fuse_qkv'] = False
                    reasons.append('available memory below 20 GiB, disabled QKV packing')
            wait = reference['fifo']['wait_ms'] if batch > 1 else 0
        settings['batch_size'] = batch
    elif model == 'lingbot':
        if requested in ('auto', 'recommended'):
            recipe = 'true-batch-posttopk-fp16-prefix-expert' if mode == 'fixed' else 'cache32-posttopk-fp16-prefix-expert'
        elif requested == 'split-stage':
            recipe = 'split-stage-posttopk-fp16-prefix-expert'
        elif requested == 'true-batch':
            recipe = 'true-batch-posttopk-fp16-prefix-expert'
        batched = recipe.startswith(('true-batch-', 'split-stage-'))
        if not batched:
            if width not in (None, 1):
                raise UnsupportedConfiguration('Dedicated LingBot serves real B1. Choose true-batch or split-stage for a wider engine.')
            batch = 1
        if batched and batch not in ((1, 2, 4) if recipe.startswith('split-stage') else (1, 2, 4, 8)):
            raise UnsupportedConfiguration('Unsupported LingBot batch width for the selected engine')
        if free < 18:
            reasons.append('less than 18 GiB available for the full LingBot checkpoint')
    else:
        raise InputError('Unknown model: ' + model)
    if reference is None:
        reference = profiles().get((model, card, recipe))
    if model == 'oft' and card in ('RTX 4090', 'RTX 3090') and requested in ('auto', 'bf16'):
        reasons.append('24 GiB profile disables gate/up packing' + (' and QKV fusion' if card == 'RTX 4090' else ''))
    if 'native_options' in options and 'native_options' in settings:
        if not isinstance(options['native_options'], dict):
            raise InputError('native_options must be an object')
        options['native_options'] = {**settings['native_options'], **options['native_options']}
    settings.update(options)
    return dict(requested_recipe=requested, selected_recipe=recipe, mode=mode, options=settings,
                hardware_profile=gpu, evidence_scope=scope, fallback_reasons=reasons,
                max_batch=batch, wait_ms=wait,
                source_profile_sha256=reference['receipt']['sha256'] if reference and reference['gpu']==card else None)


def prepare_config(config):
    """Called by the actual model worker, after CUDA_VISIBLE_DEVICES is mapped."""
    config = dict(config)
    gpu = hardware(config['device'])
    automatic = config.get('recipe') in (None, 'auto')
    root = Path(config['checkpoint'])
    installed = Path(os.sys.prefix) / 'share/vvla/assets.json'
    defaults = json.loads(installed.read_text()) if installed.is_file() else {}
    if config['model'] == 'lingbot':
        for key, relative in [('qwen', 'qwen'), ('norm_stats','robotwin_norm.json'),
                              ('training_config', 'lingbotvla_cli.yaml')]:
            if (root / relative).exists():
                defaults[key] = str(root / relative)
    nested = None
    if config['model'] == 'lingbot' and not (root / 'config.json').exists():
        candidates = list(root.glob('checkpoints/*/hf_ckpt/config.json'))
        if len(candidates) > 1:
            raise InputError('Multiple LingBot checkpoints found. Select one hf_ckpt directory.')
        if candidates:
            nested = candidates[0].parent
            config['checkpoint'] = str(nested)
    assets = resolve_assets(root, config.get('assets', {}), defaults=defaults, offline=config.get('offline', False),
                            optional_missing=('lt_tactics',) if automatic else ())
    if nested is not None:
        assets = resolve_assets(nested, assets, offline=config.get('offline', False))
        if 'qwen' not in assets:
            from .hub import fetch, PROCESSOR_FILES, QWEN_PROCESSOR
            assets['qwen'] = fetch(QWEN_PROCESSOR, offline=config.get('offline', False), patterns=PROCESSOR_FILES)
    native_path = native_directory()
    if 'native' not in assets and native_path:
        assets['native'] = native_path
    kwargs = {}
    if config['model'] == 'pi05':
        from vvla_pi05.lite.pi05 import load_lite_config
        model_config, _ = load_lite_config(config['checkpoint'])
        overrides = config.get('options', {}).get('config_overrides', {})
        kwargs.update(horizon=overrides.get('chunk_size', model_config.chunk_size),
                      input_contract=overrides.get('input_contract', model_config.input_contract))
    if config['model'] == 'oft' and config.get('recipe') in (None, 'auto', 'bf16', 'mixed-fp8'):
        kwargs['native_fa2'], native_error = oft_fa2(assets.get('native'), gpu)
    else:
        native_error = None
    dispatch = select(config, gpu, **kwargs)
    if dispatch['selected_recipe'].startswith('split-stage-'):
        dispatch['experimental_high_throughput'] = True
    if native_error:
        dispatch['native_probe'] = native_error
    config.update(max_batch=dispatch['max_batch'], recipe=dispatch['selected_recipe'], options=dispatch['options'], assets=assets,
                  auto_tactics_fallback=dispatch['requested_recipe'] == 'auto')
    if config['model'] == 'pi05' and gpu['profile'] == 'G4' and dispatch['max_batch'] > 1:
        table = resources.files('vvla_pi05').joinpath('lite/lt-tactics-pro6000.json')
        if table.is_file():
            assets.setdefault('lt_tactics', str(table))
    return config, dispatch


def native_directory():
    configured = os.environ.get('VLA_VLLM_NATIVE_PATH')
    if configured:
        return configured
    target = Path(os.sys.prefix) / 'native'
    if (target / 'vllm').is_dir():
        return str(target)
    spec = util.find_spec('vllm')
    if spec and spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).parent)
    return None


def oft_fa2(path, gpu):
    if not path or tuple(gpu['capability']) == (8, 6):
        return False, 'no configured compatible OFT native target'
    import torch
    from vvla_runtime.execution import native
    os.environ['VLA_VLLM_NATIVE_PATH'] = str(path)
    try:
        native.load_native()
        q = torch.zeros((1, 1, 64), dtype=torch.bfloat16, device='cuda')
        cu = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
        native.flash_attention(q, q, q, cu_q=cu, cu_k=cu, max_q=1, max_k=1)
        torch.cuda.synchronize()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return False, str(error)
    return True, None
