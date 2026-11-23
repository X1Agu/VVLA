"""Install one isolated CUDA 13.0 model worker from a VVLA release directory."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'oft': [
        ('source', 'https://github.com/moojink/openvla-oft.git', 'e4287e94541f459edc4feabc4e181f537cd569a8'),
        ('transformers_source', 'https://github.com/moojink/transformers-openvla-oft.git', 'bc339d9ad707454c0c115970db43c260067c61ab'),
    ],
    'lingbot': [('upstream', 'https://github.com/Robbyant/lingbot-vla-v2.git', 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6')],
}


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def install_sources(backend, directory, binary):
    assets = {}
    for key, url, commit in SOURCES.get(backend, []):
        path = directory/key
        run('git', 'init', '-q', path)
        run('git', '-C', path, 'fetch', '--quiet', '--depth=1', url, commit)
        run('git', '-C', path, 'checkout', '--quiet', '--detach', 'FETCH_HEAD')
        if key == 'upstream':
            prepared = directory/'source'
            if not (prepared/'VVLA_INFERENCE_SOURCE.json').is_file():
                import tempfile
                with tempfile.TemporaryDirectory(dir=directory) as staging:
                    candidate = Path(staging)/'source'
                    run(binary, '-m', 'vvla_lingbot.prepare_source', '--source', path, '--output', candidate)
                    candidate.rename(prepared)
            assets['source'] = str(prepared)
            assets['norm_stats'] = str((path/'assets/norm_stats/robotwin.json').resolve(strict=True))
        else:
            assets[key] = str(path)
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['pi05','oft','lingbot'])
    parser.add_argument('--directory', type=Path, help='Default $VVLA_ENV_ROOT/<model> or ~/.local/share/vvla/envs/<model>')
    parser.add_argument('--wheels', type=Path, default=ROOT/'dist')
    parser.add_argument('--native-wheel', type=Path, help='Override the pinned vLLM 0.32.0 native wheel')
    parser.add_argument('--flash-attn-wheel', type=Path, help='FlashAttention 2.8.3 built for the target architecture and Torch ABI')
    parser.add_argument('--skip-sources', action='store_true', help='Supply prepared upstream sources via checkpoint/vvla.json')
    parser.add_argument('--sdpa', action='store_true', help='OFT only. Install without the vLLM native FA2 target.')
    args = parser.parse_args()
    if (platform.system(), platform.machine(), sys.version_info[:2]) != ('Linux','x86_64',(3,12)):
        parser.error('Use Linux x86_64 with Python 3.12')
    if args.sdpa and (args.backend != 'oft' or args.native_wheel):
        parser.error('--sdpa applies to OFT without an explicit native wheel')
    if args.backend == 'lingbot' and args.flash_attn_wheel is None:
        parser.error('Supply --flash-attn-wheel with a compatible 2.8.3 wheel built by tools/build_lingbot_flash_attn.py')
    for file, prefix in ((args.native_wheel,'vllm-0.32.0-'),(args.flash_attn_wheel,'flash_attn-2.8.3-')):
        if file is not None and (not file.is_file() or not file.name.startswith(prefix)):
            parser.error('Expected an existing '+prefix+'*.whl')
    env_root = Path(os.environ.get('VVLA_ENV_ROOT', Path.home()/'.local/share/vvla/envs'))
    directory = (args.directory or env_root/args.backend).expanduser().absolute()
    version = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    names = ['vvla','vvla_runtime','vvla_'+args.backend]
    wheels = [args.wheels/f'{name}-{version}-py3-none-any.whl' for name in names]
    if not all(p.is_file() for p in wheels):
        parser.error('Build the beta wheels with python tools/build_release.py or provide --wheels')
    lock = ROOT/'packages'/args.backend/'environments/linux-py312-cu130.lock'
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=True)
    record = directory/'installation.json'
    if record.exists():
        parser.error(f'Worker is already installed in {directory}')
    marker = directory/'.vvla-installing'
    # A retry only resumes a directory created by this installer.
    if any(directory.iterdir()) and not marker.exists():
        parser.error(f'Choose an empty installation directory ({directory})')
    import fcntl
    with marker.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Installation is already running in {directory}')
        run(sys.executable, '-m', 'venv', directory)
        binary = directory/'bin/python'
        run(binary, '-m', 'pip', 'install', '-r', lock, *wheels, *([args.flash_attn_wheel] if args.flash_attn_wheel else []))
        run(binary, '-c', "import torch; assert torch.__version__.split('+')[0]=='2.13.0' and torch.version.cuda=='13.0', torch.__version__")
        run(binary, '-m', 'pip', 'check')
        native = args.native_wheel or 'vllm==0.32.0'
        if args.native_wheel or (args.backend in ('oft','lingbot') and not args.sdpa):
            run(binary, '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps','--upgrade','--target', directory/'native', native)
        assets = {} if args.skip_sources else install_sources(args.backend, directory, binary)
        share = directory/'share/vvla'
        share.mkdir(parents=True, exist_ok=True)
        (share/'assets.json').write_text(json.dumps(assets,indent=2)+'\n')
        details = dict(backend=args.backend, version=version, worker_python=str(binary), assets=assets)
        record.write_text(json.dumps(details,indent=2)+'\n')
        marker.unlink()
        print(json.dumps(details,indent=2))


if __name__ == '__main__':
    main()
