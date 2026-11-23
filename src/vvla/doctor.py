"""Inspect each worker in its own interpreter and CUDA visibility scope."""
from importlib import metadata, resources, util
import json
import os
import subprocess
import sys

from .config import MODELS
from .dispatch import hardware, native_directory, oft_fa2, select
from .environments import worker_python
from .workers.device import resolve_device


def inspect_worker(model):
    from . import __version__
    result = dict(model=model, python=sys.executable, python_version=sys.version.split()[0],
                  sdk_version=__version__, issues=[])
    for dist in ('vvla-runtime', 'vvla-' + model, 'torch'):
        try:
            result[dist] = metadata.version(dist)
            if dist.startswith('vvla-') and result[dist] != __version__:
                result['issues'].append(f'{dist} version differs from SDK')
        except metadata.PackageNotFoundError:
            result['issues'].append('missing ' + dist)
    if not result['issues']:
        try:
            import platform
            if sys.version_info[:2] != (3,12) or (platform.system(), platform.machine()) != ('Linux','x86_64'):
                result['issues'].append('worker differs from the supported Linux x86_64 / Python 3.12 platform')
            __import__('vvla_'+model+'.backend')
            gpu = hardware('cuda:0')
            result['gpu'] = gpu
            if gpu['torch'].split('+')[0] != '2.13.0' or gpu['cuda'] != '13.0':
                result['issues'].append('worker differs from the supported Torch 2.13.0 / CUDA 13.0 build')
            path = native_directory()
            native_ok = False
            if model == 'oft':
                native_ok, result['native_probe'] = oft_fa2(path, gpu)
                result['vllm_fa2_available'] = native_ok
                result['packaged_cpp'] = [name for name in ('native_state.cpp','paired_jpeg.cpp')
                    if resources.files('vvla_runtime').joinpath('execution/csrc', name).is_file()]
            if model == 'lingbot':
                if util.find_spec('flash_attn') is None:
                    result['issues'].append('missing compatible flash-attn')
                if path:
                    os.environ['VLA_VLLM_NATIVE_PATH'] = path
                from vvla_runtime.execution.native import load_native
                load_native()
            if model == 'pi05':
                result['packaged_lt_table'] = resources.files('vvla_pi05').joinpath('lite/lt-tactics-pro6000.json').is_file()
            result['suggested'] = {}
            for mode in ('direct', 'serve'):
                plan = select(dict(model=model, mode=mode), gpu, native_fa2=native_ok)
                result['suggested'][mode] = {k:plan[k] for k in ('selected_recipe','max_batch','wait_ms','fallback_reasons')}
        except Exception as error:
            result['issues'].append(str(error))
    result['ready'] = not result['issues']
    return result


def diagnose(model=None, *, device='cuda:0', interpreter=None):
    mapping = resolve_device(device, os.environ)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=mapping['visible_token'])
    env.pop('PYTHONPATH', None)
    result = {'device_mapping': mapping, 'workers': {}}
    try:
        probe = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10, check=True)
        result['driver_versions'] = sorted(set(probe.stdout.split()))
    except (OSError, subprocess.SubprocessError):
        result['driver_versions'] = []
    for name in [model] if model else sorted(MODELS):
        binary = worker_python(name, interpreter)
        try:
            proc = subprocess.run([binary, '-I', '-m', 'vvla.doctor', name], env=env,
                                  capture_output=True, text=True, timeout=90)
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip()[-1500:] or f'worker exited {proc.returncode}')
            result['workers'][name] = json.loads(proc.stdout)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
            result['workers'][name] = dict(ready=False, python=binary, issues=[str(error)])
    return result


if __name__ == '__main__':
    print(json.dumps(inspect_worker(sys.argv[1])))
