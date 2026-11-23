import json
import sys

import pytest

from vvla.assets import resolve_assets
from vvla.dispatch import GIB, card_name, profiles, select, serving_defaults
from vvla.environments import worker_python
from vvla.errors import InputError
from vvla.hub import checkpoint_path, fetch

DEVICES = {
    'G4': ('NVIDIA RTX PRO 6000 Blackwell Server Edition', (12, 0), 96),
    'RTX 4090': ('NVIDIA GeForce RTX 4090', (8, 9), 24),
    'RTX 3090': ('NVIDIA GeForce RTX 3090', (8, 6), 24),
    'A100 80 GB': ('NVIDIA A100-SXM4-80GB', (8, 0), 80),
}


def gpu(card, free=None):
    name, sm, total = DEVICES[card]
    return dict(name=name, capability=sm, total_bytes=total*GIB, free_bytes=(free or total-1)*GIB,
                profile=card_name(name, sm, total*GIB))


@pytest.mark.parametrize('card', DEVICES)
@pytest.mark.parametrize('model', ['pi05', 'oft', 'lingbot'])
def test_public_automatic_online_profile(card, model):
    device = gpu(card)
    assert device['profile'] == card
    plan = select(dict(model=model, mode='serve'), device, native_fa2=card!='RTX 3090')
    ref = profiles()[model, card, plan['selected_recipe']]
    assert plan['source_profile_sha256'] == ref['receipt']['sha256']
    for key in ('max_batch', 'wait_ms'):
        assert plan[key] == ref['fifo'][key]
    if model == 'oft':
        assert plan['options'] == ref['options']
    if model == 'lingbot':
        assert plan['max_batch'] == 1 and plan['selected_recipe'].startswith('cache32-')


@pytest.mark.parametrize('card', DEVICES)
@pytest.mark.parametrize('model', ['pi05','oft','lingbot'])
def test_dedicated_defaults_are_b1(card, model):
    plan = select(dict(model=model), gpu(card), native_fa2=card!='RTX 3090')
    assert (plan['max_batch'], plan['wait_ms']) == (1, 0)


@pytest.mark.parametrize('card', ['G4','RTX 4090'])
def test_explicit_mixed_uses_full_language_scope(card):
    plan = select(dict(model='oft', recipe='mixed-fp8', mode='serve'), gpu(card), native_fa2=True)
    assert plan['options']['native_options']['fp8_scope'] == 'llama'
    with pytest.raises(InputError, match='FP8|fp8'):
        select(dict(model='oft', recipe='mixed-fp8'), gpu(card), native_fa2=False)


@pytest.mark.parametrize('card', ['RTX 3090','A100 80 GB'])
def test_explicit_mixed_on_ampere_is_rejected(card):
    with pytest.raises(InputError):
        select(dict(model='oft', recipe='mixed-fp8'), gpu(card), native_fa2=True)


def test_busy_gpu_and_unknown_card_fallbacks():
    plan = select(dict(model='oft'), gpu('A100 80 GB', 14), native_fa2=False)
    opts = plan['options']['native_options']
    assert opts['backend'] == 'sdpa' and not opts['pack_gate_up'] and not opts['fuse_qkv']
    assert len(plan['fallback_reasons']) >= 3
    assert card_name('NVIDIA GeForce RTX 4070 Laptop GPU', (8,9), 8*GIB) == 'unmeasured'
    assert card_name('NVIDIA A100-SXM4-40GB', (8,0), 40*GIB) == 'unmeasured'
    for mode, prefix in [('serve','cache32-'), ('fixed','true-batch-')]:
        selected = select(dict(model='lingbot', mode=mode), gpu('G4'))
        assert selected['selected_recipe'].startswith(prefix)
    assert select(dict(model='pi05', mode='serve'), gpu('G4'), horizon=10)['selected_recipe']=='bf16-portable'


def test_unsupported_width_and_explicit_override():
    with pytest.raises(InputError):
        select(dict(model='lingbot',mode='serve',max_batch=4), gpu('G4'))
    with pytest.raises(InputError):
        select(dict(model='oft',options=dict(batch_size='4')), gpu('RTX 4090'))
    plan=select(dict(model='lingbot',mode='serve',recipe='split-stage',max_batch=4), gpu('G4'))
    assert plan['selected_recipe'].startswith('split-stage-') and plan['max_batch']==4


def test_latest_formal_serving_routes():
    mixed = select(dict(model='oft', recipe='mixed-fp8', mode='serve'), gpu('RTX 4090'), native_fa2=True)
    assert mixed['wait_ms'] == 0
    native = mixed['options']['native_options']
    assert native['fp8_scope'] == 'llama' and not native['fuse_qkv'] and not native['pack_gate_up']
    split = select(dict(model='lingbot', recipe='split-stage', mode='serve'), gpu('A100 80 GB'))
    assert split['source_profile_sha256'] and split['max_batch'] == 4

    plan = select(dict(model='pi05', mode='serve'), gpu('G4'))
    plan['resolved_recipe'] = plan['selected_recipe']
    assert serving_defaults('pi05', plan) == dict(max_batch=16, wait_ms=0, network_wait_ms=2)
    assert 'network_wait_ms' not in serving_defaults('pi05', dict(plan, max_batch=4))
    assert 'network_wait_ms' not in serving_defaults('pi05', dict(plan, resolved_recipe='bf16-portable'))


def test_worker_lookup_priority(tmp_path,monkeypatch):
    target=tmp_path/'oft/bin/python'
    target.parent.mkdir(parents=True)
    target.touch()
    monkeypatch.setenv('VVLA_ENV_ROOT', str(tmp_path))
    assert worker_python('oft')==str(target)
    monkeypatch.setenv('VVLA_OFT_PYTHON','/custom/python')
    assert worker_python('oft')=='/custom/python'
    assert worker_python('oft','/explicit')=='/explicit'


def test_hub_checkpoint_and_nested_asset_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def download(**args):
        calls.append(args)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules,'huggingface_hub', SimpleNamespace(snapshot_download=download))
    (tmp_path/'norm.json').write_text('{}')
    assert checkpoint_path('org/repo',revision='sha',offline=True)==str(tmp_path)
    (tmp_path/'vvla.json').write_text(json.dumps(dict(schema=1, assets=dict(norm_stats='hf://org/assets@v1/norm.json'))))
    assert resolve_assets(tmp_path, {}, offline=True)['norm_stats']==str(tmp_path/'norm.json')
    assert [(x['repo_id'],x['revision'],x['local_files_only']) for x in calls] == [('org/repo','sha',True),('org/assets','v1',True)]
    with pytest.raises(InputError):
        fetch('hf://org/repo/../x')


@pytest.mark.parametrize('width', [1,2,3,4])
def test_oft_collate_uses_real_tensor_rows_at_capacity_four(width):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from vvla_oft.service import OFTService
    service = OFTService.__new__(OFTService)
    service.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    service.batch_size, service.capacity, service.device = 4, 128, 'cpu'
    service.prepare_inputs = None
    rows=[([1, 2, 3+i], torch.full((2,3,2,2), i), np.zeros(8)) for i in range(width)]
    values, count = service.collate(rows)
    assert count == width
    assert all(t.shape[0] == width for t in values.values())
    assert values['input_ids'].shape[1] == 128


def test_published_lingbot_repo_layout_and_processor_subset(tmp_path, monkeypatch):
    from vvla import dispatch, hub
    nested = tmp_path/'checkpoints/global_step_50000/hf_ckpt'
    nested.mkdir(parents=True)
    (nested/'config.json').write_text('{}')
    (tmp_path/'lingbotvla_cli.yaml').touch()
    monkeypatch.setattr(dispatch,'hardware', lambda _:gpu('G4'))
    monkeypatch.setattr(dispatch,'native_directory',lambda:None)
    calls=[]
    def download(spec, **kwargs):
        calls.append((spec, kwargs))
        return str(tmp_path)
    monkeypatch.setattr(hub,'fetch',download)
    cfg, plan=dispatch.prepare_config(dict(model='lingbot',device='cuda:0',checkpoint=str(tmp_path),offline=True))
    assert cfg['checkpoint']==str(nested) and cfg['assets']['training_config']==str(tmp_path/'lingbotvla_cli.yaml')
    assert plan['max_batch']==1 and cfg['assets']['qwen']==str(tmp_path)
    assert calls == [(hub.QWEN_PROCESSOR,dict(offline=True,patterns=hub.PROCESSOR_FILES))]
    assert all('safetensors' not in pattern for pattern in hub.PROCESSOR_FILES)
    calls.clear()
    assert hub.pull(hub.LINGBOT_ROBOTWIN,revision='sha',offline=True)==str(tmp_path)
    assert len(calls)==2 and calls[0][1]['revision']=='sha'
