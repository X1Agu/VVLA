"""Build the historical inference-only FlashAttention wheel before deployment.

Use SM80 for the historical L4/Ada build or SM120 for Blackwell. This changes
only the original disable-backward build flag; all forward kernels are intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time

SOURCE_SHA='1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d'
SETUP_BEFORE='d089d876c34366979708a87abe051bf3e30f81ee4ee685e4bb446c11f0940a73'
SETUP_AFTER='ade1404de098756025748ecb32f76e8a897732e6a0bc6508485e62b27b4a88c9'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdist',type=Path,required=True)
    p.add_argument('--cuda-home',type=Path,required=True)
    p.add_argument('--arch',choices=['80','120'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--resume',type=Path,help='Resume an existing verified work directory under --output')
    args=p.parse_args()
    if sha(args.sdist)!=SOURCE_SHA:raise ValueError('Unexpected FlashAttention 2.8.3 source archive')
    if args.jobs<1:raise ValueError('jobs must be positive')
    import torch
    toolkit=args.cuda_home.resolve()
    for relative in ('bin/nvcc','include/crt/host_config.h','include/cuda_runtime.h','nvvm/libdevice/libdevice.10.bc'):
        if not (toolkit/relative).is_file():
            raise ValueError(f'Incomplete CUDA toolkit: missing {toolkit/relative}; include matching nvcc, CRT, runtime, NVVM and CCCL packages')
    compiler=subprocess.check_output([str(toolkit/'bin/nvcc'),'--version'],text=True)
    version=re.search(r'release (\d+\.\d+)',compiler)
    if version is None or version.group(1)!=torch.version.cuda:
        raise ValueError(f'Build toolkit must match Torch CUDA {torch.version.cuda}: {compiler}')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    work=args.resume.resolve(strict=True) if args.resume else Path(tempfile.mkdtemp(prefix='flash-attn-build-',dir=output))
    if not work.is_relative_to(output) or work==output:
        raise ValueError('Build work directory must be under the output directory')
    if not args.resume:
        with tarfile.open(args.sdist) as archive:archive.extractall(work,filter='data')
    source=work/'flash_attn-2.8.3';setup=source/'setup.py'
    if not args.resume:
        if sha(setup)!=SETUP_BEFORE:raise ValueError('Unexpected upstream setup.py')
        setup.write_text(setup.read_text().replace('# "-DFLASHATTENTION_DISABLE_BACKWARD",','"-DFLASHATTENTION_DISABLE_BACKWARD",'))
    if sha(setup)!=SETUP_AFTER:raise ValueError('Historical inference build patch differs')
    if args.resume:
        with tarfile.open(args.sdist) as archive:
            for member in archive:
                if member.isfile() and member.name!='flash_attn-2.8.3/setup.py':
                    target=work/member.name
                    if not target.resolve().is_relative_to(work) or target.read_bytes()!=archive.extractfile(member).read():
                        raise ValueError('Cannot resume modified upstream source: '+member.name)
    env=dict(os.environ,CUDA_HOME=str(args.cuda_home.resolve()),MAX_JOBS=str(args.jobs),
        FLASH_ATTN_CUDA_ARCHS=args.arch,NVCC_THREADS='1',FLASH_ATTENTION_FORCE_BUILD='TRUE')
    # NVIDIA's pip runtime carries only the versioned SONAME. Provide the
    # development link in this build directory, never modify a system toolkit.
    runtime_lib=next((toolkit/directory/'libcudart.so.13' for directory in ('lib','lib64')
                      if (toolkit/directory/'libcudart.so.13').is_file()),None)
    if runtime_lib is None:raise ValueError('CUDA 13 runtime shared library is missing')
    links=work/'link-libs';links.mkdir(exist_ok=True)
    link=links/'libcudart.so'
    if link.is_symlink():
        if link.resolve()!=runtime_lib.resolve():raise ValueError('Unexpected existing CUDA runtime linker target')
    elif link.exists():raise ValueError('Unexpected existing CUDA runtime linker file')
    else:link.symlink_to(runtime_lib)
    env['LIBRARY_PATH']=str(links)+os.pathsep+env.get('LIBRARY_PATH','')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(args.cuda_home/'bin')+os.pathsep+env['PATH']
    command=[sys.executable,'-m','build','--wheel','--no-isolation','--outdir',str(output),str(source)]
    log_path=work/(f'build-resume-{args.jobs}-{time.time_ns()}.log' if args.resume else 'build.log')
    with log_path.open('w') as log:
        status=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    record=dict(status='passed' if status==0 else 'failed',source_sha256=SOURCE_SHA,
        setup_before=SETUP_BEFORE,setup_after=SETUP_AFTER,arch=args.arch,torch=torch.__version__,
        torch_cuda=torch.version.cuda,jobs=args.jobs,nvcc_threads=1,disable_backward=True,
        compiler=compiler,
        log=str(log_path),command=command,resumed=bool(args.resume))
    if status==0:
        record['wheels']={p.name:sha(p) for p in output.glob('flash_attn-2.8.3*.whl')}
    (work/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    raise SystemExit(status)


if __name__=='__main__':main()
