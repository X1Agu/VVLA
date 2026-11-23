"""Build sdists, rebuild wheels, then exercise installed resources out of tree.

The artifact environment reuses only installed ML dependencies from --ml-python.
No source PYTHONPATH or editable install is used. This is a packaging check, not
qualification of all three incompatible model environments.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env, log):
    result = subprocess.run(list(map(str, command)), cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-python", default=sys.executable)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="release-check-", dir=ROOT / "artifacts"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MAX_JOBS"] = "1"
    env["PATH"] = str(Path(args.ml_python).parent) + os.pathsep + env["PATH"]
    wheels = work / "wheels"
    wheels.mkdir()
    records = []
    for name, source in [("vvla", ROOT)] + [(p.name, p) for p in sorted((ROOT / "packages").iterdir()) if (p / "pyproject.toml").exists()]:
        stage = work / name
        stage.mkdir()
        run([args.ml_python, "-m", "build", "--sdist", "--no-isolation", "--outdir", stage, source],
            cwd=work, env=env, log=stage / "sdist.log")
        sdist = next(stage.glob("*.tar.gz"))
        with tarfile.open(sdist) as archive:
            archive.extractall(stage / "unpacked", filter="data")
        unpacked = next((stage / "unpacked").iterdir())
        run([args.ml_python, "-m", "build", "--wheel", "--no-isolation", "--outdir", wheels, unpacked],
            cwd=work, env=env, log=stage / "wheel.log")
        records.append({"package": name, "sdist": str(sdist), "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest()})
    run([args.ml_python, "-m", "venv", work / "installed"], cwd=work, env=env, log=work / "venv.log")
    python = work / "installed/bin/python"
    site = next((work / "installed/lib").glob("python*/site-packages"))
    ml_site = Path(subprocess.check_output([args.ml_python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip())
    # A path entry does not execute .pth/editable hooks located in the reused
    # dependency directory. Model and SDK imports must resolve to this new venv.
    (site / "ml-dependencies.pth").write_text(str(ml_site) + "\n")
    run([python, "-m", "pip", "install", "--ignore-installed", "--no-deps", "--no-index", *sorted(wheels.glob("*.whl"))],
        cwd=work, env=env, log=work / "install.log")
    # Install the console executable in the worker venv, not only its import
    # package via the reused dependency path. Runtime declares this dependency.
    run([python,'-m','pip','install','--ignore-installed','--no-deps','ninja==1.13.2'],
        cwd=work,env=env,log=work/'ninja-install.log')
    run([work / "installed/bin/vvla", "--version"], cwd=work, env=env, log=work / "cli.log")
    probe = work / "probe.py"
    probe.write_text('''import importlib, importlib.resources as r, importlib.metadata as md, json, pathlib, torch
from packaging.requirements import Requirement
from vvla_pi05.lite.lifecycle import collect_lifecycle_signature
from vvla_pi05.lite.native_lifecycle import load_native_lifecycle
from vvla_pi05.lite.fp8_tensor_epochs import load_native_epochs
expected=pathlib.Path(__file__).parent/'installed'
result={'imports':{},'native':[],'cuda_initialized_before':torch.cuda.is_initialized()}
result['core_build_dependencies']={}
for package,dependency in [('vvla-runtime','ninja'),('vvla-pi05','nvidia-nvvm')]:
    requirements=[Requirement(text) for text in md.requires(package)]
    matches=[req for req in requirements if req.name==dependency and req.marker is None]
    assert len(matches)==1,(package,dependency)
    version=md.version(dependency)
    assert version in matches[0].specifier
    result['core_build_dependencies'][dependency]=version
for name in ['vvla','vvla_runtime','vvla_pi05','vvla_oft','vvla_lingbot']:
    path=pathlib.Path(importlib.import_module(name).__file__).resolve()
    assert path.is_relative_to(expected), (name,str(path))
    result['imports'][name]=str(path)
from vvla.dispatch import profiles
assert ('lingbot', 'A100 80 GB', 'split-stage-posttopk-fp16-prefix-expert') in profiles()
for name in ('pi05-g4-prefix-fp8', 'oft-g4-bf16'):
    assert json.loads(r.files('vvla').joinpath('data', name + '.json').read_bytes())['service_ms']['1'] > 0
assert r.files('vvla.integrations.omni').joinpath('controller/config.json').is_file()
resources={name:r.files('vvla_pi05').joinpath('lite/csrc/'+name).read_bytes() for name in
    ['native_lifecycle.cpp','native_lifecycle_v2.cpp','native_lifecycle_v3.cpp','fp8_tensor_epochs.cpp']}
result['resources']={n:len(v) for n,v in resources.items()}
for name in ('native_state.cpp', 'paired_jpeg.cpp'):
    assert r.files('vvla_runtime').joinpath('execution/csrc', name).is_file(), name
from vvla_pi05.lite.dynamic_cublaslt import load_tactics
tactics = load_tactics(r.files('vvla_pi05').joinpath('lite/lt-tactics-pro6000.json'))
result['lt_tactics_sha256'] = tactics['_report_sha256']
assert result['lt_tactics_sha256'] == 'dbd132e9fc6fc33a2da11f2ca8be18d4f165e823f6b2f8e7e6804bc2c7aa8791'
if NATIVE:
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Linear(8,2))
    for variant in ['v1','v2','v3']:
        collector=load_native_lifecycle(variant=variant)
        assert collector.signature(model)==collect_lifecycle_signature(model)
        old=collector.signature(model)
        with torch.no_grad(): model[0].weight.add_(1)
        assert collector.signature(model)!=old
        result['native'].append(collector.manifest())
    extension,manifest=load_native_epochs()
    from vvla_pi05.models.pi05.inference_cache_pi05 import tensor_identity
    tensor=torch.ones(4)
    identity=tensor_identity(tensor)
    extension.validate_tensor_epochs((tensor,),(identity,))
    tensor.add_(1)
    try: extension.validate_tensor_epochs((tensor,),(identity,))
    except RuntimeError: pass
    else: raise AssertionError('native epochs accepted mutation')
    result['native'].append(manifest)
    from vvla_runtime.execution.native_state import load_native_state_comparator
    from vvla_runtime.execution.state import ModelStateContract
    comparator=load_native_state_comparator()
    model=torch.nn.Linear(4,8).eval()
    contract=ModelStateContract(model,native_comparator=comparator)
    contract.signature()
    assert comparator.compare(contract) is True
    with torch.no_grad():model.weight.add_(1)
    assert comparator.compare(contract) is False
    result['native'].append(comparator.manifest())
result['cuda_initialized_after']=torch.cuda.is_initialized()
assert not result['cuda_initialized_after']
pathlib.Path(__file__).with_name('installed-check.json').write_text(json.dumps(result,indent=2))
'''.replace("if NATIVE:", "if " + repr(args.native) + ":"))
    run([python, "-I", probe], cwd=work, env=env, log=work / "probe.log")
    # A separately named fixture exercises process transport and faults, without
    # modifying or shadowing any installed model backend.
    shutil.copyfile(ROOT/'tests/packaging/worker_fixture.py',site/'_vvla_test_worker.py')
    shutil.copyfile(ROOT/'tests/packaging/probe_worker.py',work/'probe_worker.py')
    worker_env=dict(env,VVLA_CHECK_NATIVE_COLD='1' if args.native else '0')
    run([python,'-I',work/'probe_worker.py'],cwd=work,env=worker_env,log=work/'worker-protocol.json')
    records.extend({"wheel": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in wheels.glob("*.whl"))
    for wheel in wheels.glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if any(part in name for part in ('benchmarks/repro','vvla_repro_pi05','tests/migrated','_evidence.py')):
                    raise RuntimeError('Reproduction-only file leaked into wheel: '+name)
    (work / "build-records.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"status": "passed", "artifacts": str(work), "native": args.native}))


if __name__ == "__main__":
    (ROOT / "artifacts").mkdir(exist_ok=True)
    main()

