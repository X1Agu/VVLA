"""Strict evidence for the measured B32 Split1 replacement of B1 Split8."""
import hashlib
import inspect
import json
from pathlib import Path

VERSION='pi05-b32-paged-suffix-split1-support-v1'
SOURCES=('vllm-vla-plugin/src/vvla_repro_pi05/lite/batch_suffix_fa2.py',
    'vllm-vla-plugin/src/vvla_repro_pi05/lite/attention.py','vllm-vla-plugin/src/vvla_repro_pi05/lite/suffix_fa2.py',
    'benchmarks/pi05-b32-20261110/profiling/probe_batch_suffix_routes.py')
CHECKS={'B1_paged8_oracle_passed','all32_written','changed_lane_effective','changed_lane_isolated','oracle_passed','repeat_exact'}


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parents[4]
    return {name:sha(root/name)for name in SOURCES}


def read_support(config):
    if type(config)is not dict or set(config)!={'backend_report','backend_report_sha256'}:
        raise ValueError('B32 suffix requires its new measured report and exact SHA')
    path=Path(config['backend_report'])
    if path.is_symlink()or not path.is_file()or sha(path)!=config['backend_report_sha256']:
        raise ValueError('B32 suffix support report changed')
    report=json.loads(path.read_text())
    if (report.get('status')!='complete'or report.get('implementation')!=VERSION or report.get('batch')!=32
            or report.get('selected_splits')!=1 or report.get('policy_loaded')is not False
            or report.get('policy_quality_passed')is not False or report.get('source_sha256')!=source_hashes()
            or report.get('cleanup')!=dict(graphs_retired=True,settings_restored=True)
            or set(report.get('cases',{}))!={'576','640'}):raise ValueError('Incomplete B32 suffix support identity')
    for capacity,row in report['cases'].items():
        if (row.get('status')!='passed'or set(row.get('checks',{}))!=CHECKS
                or any(value is not True for value in row['checks'].values())):
            raise ValueError('B32 suffix repeat/oracle/capture/lane qualification failed '+capacity)
        raw=path.parent/row['raw_path']
        if Path(row['raw_path']).name!=row['raw_path']or raw.is_symlink()or sha(raw)!=row['raw_sha256']:
            raise ValueError('B32 suffix raw arrays changed')
    return report


def validate_runtime(config,device):
    import torch
    from .suffix_fa2 import _public_api
    from .prefix_fp8 import _hardware
    report=read_support(config)
    actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,torch_git=torch.version.git_version)
    files=dict(varlen_api=sha(Path(inspect.getfile(_public_api()))),
        libtorch_cuda=sha(Path(torch.__file__).resolve().parent/'lib/libtorch_cuda.so'))
    if (report['runtime']!=actual or report['hardware']!=_hardware(device)or report['torch_files_sha256']!=files):
        raise ValueError('B32 suffix current hardware/runtime/binary differs from executed support')
    return report
