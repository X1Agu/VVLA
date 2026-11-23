"""Run real GPU contracts against installed wheels, outside the source tree."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from check_release import run

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--python',type=Path,required=True)
p.add_argument('--tools-bin',type=Path,help='Installed compiler/ninja executables, not model source')
args=p.parse_args()
work=Path(tempfile.mkdtemp(prefix='installed-gpu-',dir=root/'artifacts'))
(work/'artifacts').mkdir()
(work/'artifacts/performance-alignment').mkdir()
env=dict(os.environ);env.pop('PYTHONPATH',None)
env['PATH']=str(args.tools_bin or args.python.parent)+os.pathsep+env['PATH']
probe=work/'origins.py'
probe.write_text('''import importlib,json,pathlib,sysconfig,torch
expected=pathlib.Path(sysconfig.get_paths()['purelib']).resolve()
records={}
for name in ('vvla','vvla_runtime','vvla_pi05','vvla_lingbot'):
    path=pathlib.Path(importlib.import_module(name).__file__).resolve()
    assert path.is_relative_to(expected),(name,str(path))
    records[name]=str(path)
records['torch']=str(torch.__version__)
records['gpu']=torch.cuda.get_device_name(0)
print(json.dumps(records,indent=2))
''')
run([args.python,'-I',probe],cwd=work,env=env,log=work/'origins.json')
run([args.python,'-I','-m','pytest','--import-mode=importlib','-c',root/'pyproject.toml',
     '-q',root/'tests/gpu','--show-capture=no','--junitxml='+str(work/'installed-gpu.xml')],
    cwd=work,env=env,log=work/'pytest.log')
(work/'result.json').write_text(json.dumps(dict(status='passed',installed_backend=True,
    real_gpu=True,synthetic_weights=True,full_weight_qualified=False),indent=2)+'\n')
print(json.dumps(dict(status='passed',artifacts=str(work))))
