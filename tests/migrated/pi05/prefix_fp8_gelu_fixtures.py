"""Fictional GELU evidence for remote CPU contracts, never runtime authorization.

Only stdlib is imported. These generated reports are not copies of an executed
job. source_root optionally supplies source bytes for owner lifecycle doubles;
it does not turn the fictional execution reports into real evidence.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
_path = REPO / 'vllm-vla-plugin/src/vvla_repro_pi05/lite/prefix_fp8_gelu_evidence.py'
_spec = importlib.util.spec_from_file_location('_fictional_gelu_evidence', _path)
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def _xml(label):
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests=str(sum(evidence.COVERAGE[label].values())),
                          failures='0', errors='0', skipped='0')
    for name, count in evidence.COVERAGE[label].items():
        for index in range(count):
            ET.SubElement(suite, 'testcase', classname='tests.test_lite_prefix_fp8_gelu',
                          name=f'{name}[fictional-{index}]', time='0.001')
    return ET.tostring(root, encoding='utf-8')


def make_evidence(tmp_path, *, source_root=None):
    """Return config/blobs/record/hardware/source_root for explicit test use."""
    tmp_path = Path(tmp_path)
    root = tmp_path / 'fictional-contracts'
    root.mkdir(parents=True)
    actual_source = None if source_root is None else Path(source_root)
    sources = tmp_path / 'fictional-source'
    inventory = {}
    for name in evidence.SOURCE_FILES:
        content = (('# fictional source: ' + name + '\n').encode() if actual_source is None
                   else (actual_source / name).read_bytes())
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        inventory[name] = dict(bytes=len(content), sha256=sha(content))
    hardware = dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
        uuid='GPU-11111111-2222-3333-4444-555555555555', driver_version='fictional-driver',
        hostname='fictional-contract-only', memory_mib=97887)
    blobs = {'SOURCE_PAYLOAD.json': raw(dict(base_commit='a' * 40, files=inventory)),
             'cpu.xml': _xml('cpu'), 'cuda.xml': _xml('cuda')}
    source_sha = sha(blobs['SOURCE_PAYLOAD.json'])
    tests = {}
    for label, pid in (('cpu', 102), ('cuda', 103)):
        tests[label] = dict(actual_cuda=label == 'cuda', coverage=evidence.COVERAGE[label],
            junit_sha256=sha(blobs[label + '.xml']), model_loaded=False, no_skip=True,
            passed=sum(evidence.COVERAGE[label].values()), performance_measured=False)
        blobs[label + '-RESULT.json'] = raw(dict(confirmed_exit=True, error=None, finished=2.0,
                                                log_sha256='b' * 64, pid=pid, returncode=0))
    blobs['RESULT.json'] = raw(dict(all_owned_processes_retired=True, contracts_only=True,
        started=1.0, finished=3.0, hardware=hardware, model_loaded=False, passed=18,
        performance_measured=False, production_changed=False,
        source=dict(files=len(inventory), source_manifest_sha256=source_sha), status='complete', tests=tests))
    blobs['exit.json'] = raw(dict(exit_code=0, finished=4.0))
    result_sha = sha(blobs['RESULT.json'])
    blobs['ROOT_RETIREMENT.json'] = raw(dict(all_retired=True, compute='',
        exit_sha256=sha(blobs['exit.json']), observed_at=5.0, pids=[101, 102, 103],
        result_sha256=result_sha, source_manifest_sha256=source_sha))
    for name, value in blobs.items(): (root / name).write_bytes(value)
    config = dict(contracts_root=str(root.resolve()), result_sha256=result_sha,
                  source_manifest_sha256=source_sha)
    record = dict(schema=1, source_manifest_sha256=source_sha, result_sha256=result_sha,
        files_sha256={name: sha(value) for name, value in blobs.items()},
        source_files_sha256={name: item['sha256'] for name, item in inventory.items()},
        hardware=deepcopy(hardware), cpu_cases=15, cuda_cases=3, scope=evidence.SCOPE)
    return dict(config=config, blobs=blobs, record=record, hardware=hardware, source_root=sources)


def make_extension(record):
    """Manifest fixture; meaningful rejection tests mutate its independent gates."""
    files = record['source_files_sha256']
    implementation = dict(version='pi05-prefix-gelu-fp8-fusion-sm120-v1',
        operator='vvla_repro_pi05_lite::prefix_gelu_fp8_quantize_rows_v1',
        source_sha256={name: files[evidence.LITE + name]
            for name in ('prefix_fp8_gelu.py', 'prefix_fp8_gelu_kernel.py')},
        reference_source_sha256={**{name: files[evidence.LITE + name]
            for name in ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')},
            'models/pi05/fused_mlp_pi05.py': files['vllm-vla-plugin/src/vvla_repro_pi05/models/pi05/fused_mlp_pi05.py']},
        required_runtime=deepcopy(evidence.RUNTIME), compile_options=deepcopy(evidence.OPTIONS),
        supported_capacities=[576, 640], width=16384, output_width=2048,
        input_dtype='bfloat16', output_dtypes=['float8_e4m3fn', 'float32'],
        intermediate_rounding=['GELU_tanh_to_BF16_RTNE', 'rounded_GELU_times_up_to_BF16_RTNE'],
        quantizer_arithmetic=deepcopy(evidence.ARITHMETIC),
        gemm_operator='vvla_repro_pi05_lite::prefix_fp8_native_scaled_mm_v1', use_fast_accum=False,
        fp_fusion=False, hidden_materialization=False, persistent_tensors=False,
        output_ownership='fresh_nonaliasing_outputs', fallback_permitted=False,
        installation='none; explicit primitive only',
        evidence_requirement='new executed exact candidate contracts and owner evidence; old backend report is insufficient',
        performance_validated=False, quality_validated=False)
    return dict(schema=1, version=evidence.VERSION, enabled=True,
        scope='prefix_mlp_gelu_down_conversion_layers_0_16', modules=[f'prefix.{i}.mlp' for i in range(17)],
        implementation=implementation, contracts=deepcopy(record), owner_source_sha256='c' * 64,
        evidence_validator_sha256='d' * 64, lifecycle=deepcopy(evidence.LIFECYCLE),
        arithmetic_scope='only GELU_tanh/BF16 multiply/down input quantization; gate/up and GEMM unchanged',
        fallback_permitted=False, quality_validated=False, performance_validated=False)


def make_backend(record):
    """Only fields consumed by extension binding; not a full base FP8 report."""
    return dict(hardware=deepcopy(record['hardware']), quantizer_variant='triton-eager-rn-native-sm120-v1',
        quantizer_files_sha256={name: record['source_files_sha256'][evidence.LITE + name]
            for name in ('prefix_fp8_native.py', 'prefix_fp8_native_kernel.py')},
        compile_options=deepcopy(evidence.OPTIONS), torch_version=evidence.RUNTIME['torch'],
        cuda_version=evidence.RUNTIME['cuda'], torch_git_version=evidence.RUNTIME['torch_git'],
        capability=deepcopy(evidence.RUNTIME['capability']))
