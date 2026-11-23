"""Export supplied Blackwell batch Lt choices without tuning or author locks.

Run with the installed Pi0.5 backend interpreter. This performs no GPU work.
An existing report is required; missing algorithms are never synthesized.
Fresh probe selections retain their origin instead of becoming historical data.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from vvla_pi05.lite.dynamic_cublaslt import load_tactics,VERSION,BATCH_SIZES,BASE_SHAPES
    report=load_tactics(args.report)
    result=dict(implementation=VERSION,
        batches={str(batch):{name:dict(selected_spec=report['batches'][str(batch)][name]['selected_spec'])
                            for name in BASE_SHAPES} for batch in BATCH_SIZES},
        provenance=dict(original_report_sha256=report['_report_sha256'],
                        selection_origin=report.get('selection_origin','supplied_report_unspecified'),
                        source_gpu_uuid=report.get('gpu_uuid'),
                        source_runtime=report.get('runtime'),
                        selection='Unmodified supplied semantic specifications; exporter performs no tuning',
                        qualification='No quality or target-GPU qualification transferred'))
    payload=(json.dumps(result,indent=2)+'\n').encode()
    with args.output.open('xb') as output:output.write(payload)
    load_tactics(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()),sha256=hashlib.sha256(payload).hexdigest(),
                         geometries=len(BATCH_SIZES)*len(BASE_SHAPES),gpu_execution=False)))


if __name__=='__main__':main()
