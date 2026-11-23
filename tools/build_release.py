"""Build five sdists and their wheels into dist/ for a local beta distribution."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('runtime', 'pi05', 'oft', 'lingbot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for source in [ROOT, *(ROOT/'packages'/name for name in PACKAGES)]:
        subprocess.run([sys.executable, '-m', 'build', '--outdir', str(output), str(source)], check=True)
    import tomllib
    version = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    archives = sorted([*output.glob(f'vvla*-{version}.tar.gz'), *output.glob(f'vvla*-{version}-*.whl')])
    if len(archives) != 10:
        raise RuntimeError('Expected a wheel and source distribution for each of the five packages')
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in archives))


if __name__ == '__main__':
    main()
