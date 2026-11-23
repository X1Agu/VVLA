import json
import os
from pathlib import Path
import shutil

import pytest
from vvla_lingbot.config import LiteRuntimeConfig
from vvla_lingbot.prepare_source import prepare
from vvla_lingbot import runtime


def source_asset(tmp_path):
    root=tmp_path/'original'
    files={
        'lingbotvla/data/__init__.py':'from .vla_data import TrainingDataset\n',
        'lingbotvla/data/vla_data/__init__.py':'from .dataset import TrainingDataset\n',
        'lingbotvla/data/vla_data/dataset.py':"raise RuntimeError('Training import must stay lazy')\n",
        'lingbotvla/data/vla_data/utils.py':'class FeatureTransform: pass\n',
        'configs/robot_configs/robotwin.yaml':'robot: robotwin\n',
        'deploy/lingbot_vla_v2_policy.py':'''from pathlib import Path
class LingbotVLAv2Server:
    def __init__(self,
        robot_norm_path=None,
    ):
        self.robot_norm_path = robot_norm_path
    def initialize(self,path_to_pi_model):
        training_config_path = Path(path_to_pi_model).parent.parent.parent/'lingbotvla_cli.yaml'
    def reset(self,robo_name):
        robot_config = f'configs/robot_configs/{robo_name}.yaml'
'''}
    for name,value in files.items():
        path=root/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text(value)
    return root,files


def test_preparation_preserves_original_and_math_with_reviewable_diff(tmp_path):
    source,files=source_asset(tmp_path)
    target=prepare(source,tmp_path/'prepared')
    assert all((source/name).read_text()==value for name,value in files.items())
    assert (source/'lingbotvla/data/vla_data/utils.py').read_bytes()==(target/'lingbotvla/data/vla_data/utils.py').read_bytes()
    manifest=json.loads((target/'VVLA_INFERENCE_SOURCE.json').read_text())
    assert len(manifest['changed_files'])==3
    assert 'training_config_path' in (target/'VVLA_INFERENCE_SOURCE.patch').read_text()
    with pytest.raises(FileExistsError):prepare(source,target)


def test_flat_assets_default_skips_weight_content_hash_but_strict_mode_reads_it(tmp_path,monkeypatch):
    source,_=source_asset(tmp_path);prepared=prepare(source,tmp_path/'prepared')
    checkpoint=tmp_path/'flat-model';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'weight payload')
    (checkpoint/'lingbotvla_cli.yaml').write_text('model: {}\n')
    qwen=tmp_path/'qwen';qwen.mkdir();(qwen/'model.safetensors').write_bytes(b'qwen weight payload')
    (qwen/'config.json').write_text('{}')
    stats=tmp_path/'stats.json';stats.write_text('{}')
    args=dict(checkpoint=str(checkpoint),official_root=str(prepared),qwen=str(qwen),norm_stats=str(stats))
    config=LiteRuntimeConfig.from_mapping(args)
    assert config.training_config==checkpoint/'lingbotvla_cli.yaml'
    observed=[];original=runtime.file_sha256
    def track(path):observed.append(Path(path));return original(path)
    monkeypatch.setattr(runtime,'file_sha256',track)
    metadata=runtime.validate_assets(config)
    assert not metadata['weights_content_hashed'] and metadata['checkpoint_sha256'] is None
    assert not any(p.suffix=='.safetensors' for p in observed)
    observed.clear()
    strict=runtime.validate_assets(LiteRuntimeConfig.from_mapping(dict(args,asset_integrity='sha256')))
    assert strict['weights_content_hashed'] and strict['checkpoint_sha256']
    assert len([p for p in observed if p.suffix=='.safetensors'])==2
    with pytest.raises(ValueError,match='Prepare the upstream inference asset'):
        runtime.validate_assets(LiteRuntimeConfig.from_mapping(dict(args,official_root=str(source))))
    with pytest.raises(ValueError,match='calibration asset'):
        LiteRuntimeConfig.from_mapping(dict(args,mode='framework-nvfp4'))
    calibration=tmp_path/'calibration.json';calibration.write_text('{"maxima":{}}')
    nvfp4=LiteRuntimeConfig.from_mapping(dict(args,mode='framework-nvfp4',calibration=str(calibration)))
    assert nvfp4.to_mapping()['calibration']==str(calibration)
    with pytest.raises(ValueError,match='original B1 engine'):
        LiteRuntimeConfig.from_mapping(dict(args,mode='framework-fp8',posttopk=True))


def test_preprocessing_identity_survives_copy_but_detects_same_size_template_edit(tmp_path):
    source,_=source_asset(tmp_path);prepared=prepare(source,tmp_path/'prepared')
    checkpoint=tmp_path/'model';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'weights')
    (checkpoint/'lingbotvla_cli.yaml').write_text('model: {}\n')
    qwen=tmp_path/'qwen';qwen.mkdir()
    (qwen/'config.json').write_text('{}')
    (qwen/'chat_template.jinja').write_text('{{ a }}')
    stats=tmp_path/'stats.json';stats.write_text('{}')
    args=dict(checkpoint=str(checkpoint),official_root=str(prepared),qwen=str(qwen),norm_stats=str(stats))
    first=runtime.validate_assets(LiteRuntimeConfig.from_mapping(args))
    copied=tmp_path/'copied';shutil.copytree(qwen,copied)
    for path in copied.iterdir():
        stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+10_000_000_000))
    cache=copied/'.cache/huggingface/download';cache.mkdir(parents=True)
    (cache/'download.json').write_text('{"time": 123}')
    selected=LiteRuntimeConfig.from_mapping(dict(args,qwen=str(copied)))
    second=runtime.validate_assets(selected)
    assert first['qwen_metadata_sha256']==second['qwen_metadata_sha256']
    assert first['qwen_inventory']!=second['qwen_inventory']
    template=copied/'chat_template.jinja';stat=template.stat()
    template.write_text('{{ b }}');os.utime(template,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    changed=runtime.validate_assets(selected)
    assert changed['qwen_metadata_sha256']!=second['qwen_metadata_sha256']
