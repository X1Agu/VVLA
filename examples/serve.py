"""Submit concurrent real observations to one bounded local model service."""
import argparse
import asyncio
import json
from pathlib import Path

from vvla.protocol.wire import dumps, loads
from vvla.serve import DynamicBatchServer


async def main(args):
    config=json.loads(args.config.read_text())
    batch=loads(args.input.read_text())
    observations,seeds=batch['observations'],batch['seeds']
    if len(observations)!=len(seeds):raise ValueError('Every observation requires a seed')
    ids=batch.get('request_ids',[str(i) for i in range(len(seeds))])
    if len(ids)!=len(seeds) or len(set(ids))!=len(ids):raise ValueError('Every request requires a unique ID')
    if args.max_batch is not None:
        config['max_batch'] = args.max_batch
    server = DynamicBatchServer.from_model(**config, wait_ms=args.wait_ms, queue_capacity=128)
    await server.start()
    try:
        actions=await asyncio.gather(*(server.infer(obs,seed=seed,request_id=identity,timeout_s=60)
            for obs,seed,identity in zip(observations,seeds,ids)))
        result=dict(actions=actions,request_ids=ids,health=server.health())
    finally:
        await server.close()
    args.output.write_text(dumps(result)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-batch',type=int,help='Maximum real batch width (LingBot defaults to its backend maximum)')
    p.add_argument('--wait-ms',type=float,help='Additional time to wait for requests')
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Choose a new output file')
    asyncio.run(main(args))
