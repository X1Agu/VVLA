"""Infer real observations from a wire JSON file using an installed worker.

Config is the JSON form of vvla.LoadConfig. Input contains observations, seeds
and optional request_ids. Model assets and worker Python are explicit.
"""
import argparse
import json
from pathlib import Path

from vvla import load
from vvla.protocol.wire import dumps, loads

p=argparse.ArgumentParser(description=__doc__)
for name in ('config','input','output'):p.add_argument('--'+name,type=Path,required=True)
args=p.parse_args()
if args.output.exists():raise SystemExit('Choose a new output file')
config=json.loads(args.config.read_text())
batch=loads(args.input.read_text())
with load(**config) as engine:
    actions=engine.infer_batch(batch['observations'],batch['seeds'],request_ids=batch.get('request_ids'))
    result=dict(actions=actions,metadata=engine.metadata())
args.output.write_text(dumps(result)+'\n')
