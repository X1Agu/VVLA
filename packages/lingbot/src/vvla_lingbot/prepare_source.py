"""Prepare an explicit LingBot source asset before deployment.

Keeps model/preprocessing mathematics unchanged. Data package exports become
lazy, and the policy receives explicit training/robot config paths. All changed
files and a reviewable diff are recorded. Nothing patches loaded Python code.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def lazy_exports(source):
    tree=ast.parse(source)
    exports={}
    for node in tree.body:
        if not isinstance(node,ast.ImportFrom) or any(n.name=='*' for n in node.names):
            raise ValueError('Expected only explicit imports in the upstream data package initializer')
        for name in node.names:
            exports[name.asname or name.name]=('.'*node.level+(node.module or ''),name.name)
    license_text=source[:source.index('from ')]
    return license_text+'''# Inference preparation: retain training exports without eager imports.
from importlib import import_module
_EXPORTS = '''+repr(exports)+'''
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
'''


def prepare(source,target):
    source,target=Path(source).resolve(strict=True),Path(target).resolve()
    if target.exists():raise FileExistsError('Choose a new output directory; existing source assets are never overwritten')
    if target.is_relative_to(source):raise ValueError('Prepared source must be outside the original source asset')
    selected=[p for directory in ('deploy','lingbotvla','configs') for p in (source/directory).rglob('*')
              if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml')]
    selected += [p for p in source.iterdir() if p.is_file() and p.name.startswith(('LICENSE','NOTICE'))]
    originals={str(p.relative_to(source)):p.read_bytes() for p in selected}
    changed={}
    for name in ('lingbotvla/data/__init__.py','lingbotvla/data/vla_data/__init__.py'):
        changed[name]=lazy_exports(originals[name].decode())
    name='deploy/lingbot_vla_v2_policy.py'
    policy=originals[name].decode()
    replacements={
        '        robot_norm_path=None,':'        robot_norm_path=None,\n        training_config_path=None,\n        robot_config_path=None,',
        '        self.robot_norm_path = robot_norm_path':'        self.robot_norm_path = robot_norm_path\n        self.training_config_path = training_config_path\n        self.robot_config_path = robot_config_path',
        "        training_config_path = Path(path_to_pi_model).parent.parent.parent/'lingbotvla_cli.yaml'":
            "        training_config_path = Path(self.training_config_path) if self.training_config_path is not None else Path(path_to_pi_model)/'lingbotvla_cli.yaml'",
        "        robot_config = f'configs/robot_configs/{robo_name}.yaml'":
            "        robot_config = self.robot_config_path or f'configs/robot_configs/{robo_name}.yaml'",
    }
    for old,new in replacements.items():
        if policy.count(old)!=1:raise ValueError('Unsupported upstream policy source; inspect the explicit config-path patch')
        policy=policy.replace(old,new)
    websocket='try:\n    from .websocket_policy_server import WebsocketPolicyServer\nexcept ImportError:\n    from deploy.websocket_policy_server import WebsocketPolicyServer\n'
    if websocket in policy:
        if policy.count(websocket)!=1 or policy.count('def main():')!=1:
            raise ValueError('Unsupported upstream CLI import layout')
        policy=policy.replace(websocket,'').replace('def main():',
            'def main():\n'+''.join('    '+line+'\n' for line in websocket.splitlines()))
    changed[name]=policy
    for name,text in changed.items():ast.parse(text)
    target.mkdir(parents=True)
    for name,data in originals.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(changed[name].encode() if name in changed else data)
    manifest=dict(schema=1,source=str(source),scope='inference import isolation and explicit asset paths only',
        original_files={name:hashlib.sha256(data).hexdigest() for name,data in originals.items()},
        changed_files={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in changed},
        mathematical_source_files_unchanged=True)
    (target/'VVLA_INFERENCE_SOURCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    diff=''.join(''.join(difflib.unified_diff(originals[name].decode().splitlines(True),text.splitlines(True),
                fromfile='original/'+name,tofile='inference/'+name)) for name,text in changed.items())
    (target/'VVLA_INFERENCE_SOURCE.patch').write_text(diff)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(prepare(args.source,args.output))


if __name__=='__main__':
    main()
