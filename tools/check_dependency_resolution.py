"""Resolve declared locks plus built wheels without installing ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('backend',choices=['pi05','oft','lingbot']);p.add_argument('--uv',type=Path,required=True)
p.add_argument('--wheels',type=Path,required=True)
p.add_argument('--native-wheels',type=Path,help='Local ABI-matched FlashAttention build artifacts for LingBot resolution')
args=p.parse_args()
directory=root/'packages'/args.backend/'environments'
lock=directory/'linux-py312-cu130.lock'
selected=[p.resolve() for p in args.wheels.glob('*.whl') if p.name.startswith(('vvla-','vvla_runtime-',f'vvla_{args.backend}-'))]
if len(selected)!=3:raise ValueError('Expected exactly three selected distribution wheels')
inputs=root/'artifacts'/f'{args.backend}-resolution.in'
inputs.write_text('-r '+str(lock)+'\n'+'\n'.join(map(str,selected))+'\n')
output=root/'artifacts'/f'{args.backend}-resolved-requirements.txt'
command=[str(args.uv),'pip','compile',str(inputs),'--python',sys.executable,
         '--no-header','--no-annotate','--output-file',str(output)]
if args.native_wheels:command.extend(['--find-links',str(args.native_wheels)])
result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
log=root/'artifacts'/f'{args.backend}-uv-resolution.log';log.write_text(result.stdout)
def requirements(path):
    records={}
    for text in path.read_text().splitlines():
        line=text.strip()
        if not line or line.startswith('#') or line in set(map(str,selected)):continue
        req=Requirement(line);name=canonicalize_name(req.name)
        if not name.startswith('vvla'):records[name]=req
    return records
if result.returncode==0:
    pinned,resolved=requirements(lock),requirements(output)
    if pinned.keys()!=resolved.keys() or any(str(pinned[n].specifier)!=str(resolved[n].specifier)
          or pinned[n].url!=resolved[n].url for n in pinned):
        raise RuntimeError('Resolved dependencies differ from the complete pinned closure; inspect '+str(output))
record=json.loads((directory/'lock-status.json').read_text())
lock_sha=hashlib.sha256(lock.read_bytes()).hexdigest()
installation=(record.get('fresh_dependency_installation','not executed')
              if record.get('input_lock_sha256')==lock_sha else 'not executed for this lock')
record.update(fresh_dependency_resolution='passed (resolver only)' if result.returncode==0 else 'failed',
    resolver=subprocess.check_output([str(args.uv),'--version'],text=True).strip(),
    resolver_command=command,resolver_log=str(log.relative_to(root)),
    fresh_dependency_installation=installation,
    input_lock_sha256=lock_sha)
(directory/'lock-status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(backend=args.backend,exit_code=result.returncode,log=str(log),resolved=str(output))))
raise SystemExit(result.returncode)
