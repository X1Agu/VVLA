from pathlib import Path
import json
import os
import subprocess
import sys
import torch

root=Path(__file__).resolve().parents[1]
if not torch.cuda.is_available():raise SystemExit('Local GPU check requires an actual CUDA device')
properties=torch.cuda.get_device_properties(0)
if properties.total_memory<4*1024**3:raise SystemExit('Insufficient local memory for production-width tiny checks')
artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)
(artifacts/'performance-alignment').mkdir(exist_ok=True)
paths=[root/'src',*(p/'src' for p in (root/'packages').iterdir())]
env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,paths)),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
record=dict(gpu=properties.name,capability=list(torch.cuda.get_device_capability(0)),total_memory=properties.total_memory,
            torch=str(torch.__version__),cuda=str(torch.version.cuda),scope='local mechanism tests; no full-model quality qualification')
(artifacts/'local-gpu.json').write_text(json.dumps(record,indent=2))
raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q','tests/gpu','--junitxml=artifacts/local-gpu.xml',*sys.argv[1:]],cwd=root,env=env))
