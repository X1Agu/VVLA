"""Bind the preparation-only G4 qualification before full-call experiments."""
import hashlib
import json
from pathlib import Path

VERSION = 'pi05-b32-coalesced-preparation-only-v1'
SOURCE = '9be109d82c27dffd5dec1f7f5eca4d717e39f92a84185cf299d163ef634b12f4'
CONFIG = 'fe43bdcbfbe1e2cde0322caeb02a4a9301e42df55a03f5071955471dc7ccbed9'
FORMAL = 'd06dd65b6f27f8a8623b5621a4973e360ab5429a064816072704b1e60ed8a7b0'
PREFLIGHT = '546d3e13d13238e2609e108e5f58925544baa8811b964b933e1be26395fe3dfe'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load_evidence(config):
    if type(config) is not dict or set(config) != {'backend_report', 'backend_report_sha256'}:
        raise ValueError('Coalesced inputs require their actual measured preparation report and SHA')
    path = Path(config['backend_report'])
    if path.is_symlink() or not path.is_file() or sha(path) != config['backend_report_sha256']:
        raise ValueError('Coalesced preparation report changed')
    report = json.loads(path.read_bytes())
    if (report.get('version') != VERSION or report.get('status') != 'complete'
            or type(report.get('batch_size')) is not int or report['batch_size'] != 32
            or report.get('source_inventory_sha256') != SOURCE or report.get('config_sha256') != CONFIG
            or report.get('formal_content_sha256') != FORMAL or report.get('preflight_content_sha256') != PREFLIGHT
            or report.get('candidate_source_sha256') != sha(Path(__file__).with_name('batch_coalesced_prepare.py'))
            or report.get('accepted_for_full_call_screen') is not True
            or report.get('decision') != 'eligible_for_full_call_screen'
            or any(report.get(key) is not False for key in
                ('policy_loaded', 'model_loaded', 'graph_capture', 'production_modified', 'quality_claim'))
            or any(report.get(key) is not True for key in
                ('retained_outputs_unchanged', 'observations_unchanged', 'initial_prepared_reference_exact'))):
        raise ValueError('Incomplete or foreign G4 coalesced preparation qualification')
    status, cleanup = path.parent / 'STATUS.json', path.parent / 'CLEANUP.json'
    if (status.is_symlink() or cleanup.is_symlink() or wire(json.loads(status.read_bytes())) != wire(report)
            or wire(json.loads(cleanup.read_bytes())) != wire({'passed': True, 'errors': []})):
        raise ValueError('Preparation did not complete and retire its original owners')
    rows = report.get('parity')
    if type(rows) is not list or len(rows) != 6:
        raise ValueError('Require initial32 and all five changing complete fixtures')
    for index, row in enumerate(rows):
        if (type(row['index']) is not int or row['index'] != index or row['exact'] is not True
                or wire(row['control']) != wire(row['candidate'])
                or set(row['artifacts']) != {'control', 'candidate'}):
            raise ValueError('Complete preparation numerical or typed-layout parity failed')
        for mode, digest in row['artifacts'].items():
            raw = path.parent / f'FIXTURE-{index}-{mode}.npz'
            if raw.is_symlink() or sha(raw) != digest:
                raise ValueError('Actual preparation qualification arrays changed')
    return dict(schema=1, backend_report=str(path.resolve()), backend_report_sha256=sha(path),
        cleanup_sha256=sha(cleanup), candidate_source_sha256=report['candidate_source_sha256'],
        source_inventory_sha256=SOURCE, config_sha256=CONFIG, formal_content_sha256=FORMAL,
        preflight_content_sha256=PREFLIGHT, hardware=report['hardware'], runtime=report['runtime'],
        tokenizer_files_sha256=report['tokenizer_files_sha256'],
        accepted_scope='preparation only; complete full-call replay and closed-loop quality remain mandatory',
        full_call_qualified=False, closed_loop_quality_qualified=False)


def validate_runtime(config, device, *, tokenizer=None):
    import torch
    from .prefix_fp8 import _hardware
    evidence = load_evidence(config)
    report = json.loads(Path(evidence['backend_report']).read_bytes())
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, torch_git=torch.version.git_version)
    binary_root = Path(torch.__file__).resolve().parent
    binary = {name: sha(binary_root / name) for name in ('lib/libtorch_cuda.so', 'lib/libtorch_cpu.so')}
    if (runtime != report['runtime'] or _hardware(device) != report['hardware']
            or binary != report['torch_binary_sha256']):
        raise ValueError('Coalesced preparation card, runtime or binary differs from its qualification')
    root = Path(report['input_paths']['tokenizer']).resolve(strict=True)
    if tokenizer is not None and Path(tokenizer).resolve(strict=True) != root:
        raise ValueError('Coalesced preparation requires its actual qualified tokenizer')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected tokenizer symlink')
        if path.is_file():files[str(path.relative_to(root))]=sha(path)
    if files != report['tokenizer_files_sha256']:
        raise ValueError('Actual qualified tokenizer files changed')
    return evidence
