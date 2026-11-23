"""Measure VVLA with application observations supplied by the caller."""
import asyncio
from collections import Counter
import math
from pathlib import Path
import time
import uuid

from .api import load
from .client import VvlaClient
from .protocol.binary import MAGIC, decode_frame
from .protocol.wire import loads
from .serve import DynamicBatchServer


ONLINE_CLIENTS = {f'c{count}': count for count in (1, 8, 16, 32)}


def read_inputs(path):
    data = Path(path).read_bytes()
    value = decode_frame(data) if data.startswith(MAGIC) else loads(data.decode())
    if not isinstance(value, dict) or not {'observations', 'seeds'} <= value.keys():
        raise ValueError('Input requires observations and seeds')
    observations, seeds = value['observations'], value['seeds']
    if not isinstance(observations, list) or not observations or not all(isinstance(o, dict) for o in observations):
        raise ValueError('Input needs a nonempty observations list')
    if not isinstance(seeds, list) or len(seeds) != len(observations) or any(type(s) is not int for s in seeds):
        raise ValueError('Input needs one integer seed for each observation')
    return observations, seeds


def summary(mode, latencies, rows, seconds, batches):
    ordered = sorted(latencies)
    return dict(mode=mode, actions=rows, elapsed_s=seconds, actions_per_s=rows/seconds,
                latency_samples=len(ordered), p50_ms=ordered[math.ceil(.5*len(ordered))-1] if ordered else None,
                p95_ms=ordered[math.ceil(.95*len(ordered))-1] if ordered else None,
                model_calls_by_width=dict(batches))


async def run(args, config):
    obs, seeds = read_inputs(args.input)
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError('iterations must be positive and warmup nonnegative')
    clients_count = ONLINE_CLIENTS.get(args.mode)
    window = getattr(args, 'duration', None)
    if window is not None and (not clients_count or not math.isfinite(window) or window <= 0):
        raise ValueError('--duration requires an online mode and a finite positive window in seconds')
    if clients_count is None:
        if args.url:
            raise ValueError('B1 and fixed runs use the direct SDK. Select an online mode with a network URL.')
        width = 1 if args.mode == 'b1' else args.batch_size
        if config.get('max_batch') not in (None, width):
            raise ValueError('max_batch conflicts with the selected benchmark width')
        if width < 1 or len(obs) < width:
            raise ValueError('Supply at least one observation and seed per fixed batch row')
        cfg = dict(config, mode='direct' if args.mode == 'b1' else 'fixed', max_batch=width)
        batches = Counter()
        times = []
        with load(**cfg) as engine:
            for i in range(args.warmup + args.iterations):
                indices = [(i*width+j) % len(obs) for j in range(width)]
                if i == args.warmup:
                    start = time.perf_counter()
                before = time.perf_counter()
                actions = engine.infer_batch([obs[j] for j in indices], [seeds[j] for j in indices])
                elapsed = (time.perf_counter()-before)*1000
                if len(actions) != width:
                    raise RuntimeError('Incomplete batch return')
                actual = engine.last_execution['actual_tensor_batch']
                if actual != width:
                    raise RuntimeError(f'Expected a real B{width} model call, got B{actual}')
                if i >= args.warmup:
                    times.append(elapsed)
                    batches[actual] += 1
            result = summary(args.mode, times, width*args.iterations, time.perf_counter()-start, batches)
            result['dispatch'] = engine.metadata().get('dispatch')
        return result

    service = None
    clients = []
    try:
        if args.url:
            for _ in range(clients_count):
                client = VvlaClient(args.url, transport=args.transport)
                await client.__aenter__()
                clients.append(client)
        else:
            service = DynamicBatchServer.from_model(**config)
            await service.start()
        run_id = uuid.uuid4().hex
        async def lane(index, count, phase, cutoff=None):
            target = clients[index] if clients else service
            readings = []
            i = 0
            while cutoff is not None or i < count:
                if cutoff is not None and time.perf_counter() >= cutoff:
                    break
                sample = (clients_count*i+index) % len(obs)
                identity = f'{run_id}-{phase}-{index}-{i}'
                before = time.perf_counter()
                reply = await target.infer(obs[sample], seed=seeds[sample], request_id=identity, timeout_s=args.timeout)
                finished = time.perf_counter()
                if phase == 'measure':
                    readings.append((finished, (finished-before)*1000, reply['scheduler']))
                i += 1
            return readings

        async def run_lanes(count, phase, cutoff=None):
            tasks = [asyncio.create_task(lane(i, count, phase, cutoff)) for i in range(clients_count)]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        await run_lanes(args.warmup, 'warm')
        start = time.perf_counter()
        cutoff = None if window is None else start + window
        runs = await run_lanes(args.iterations, 'measure', cutoff)
        drained = time.perf_counter()
        times, calls = [], {}
        drain_actions = 0
        for readings in runs:
            for finished, elapsed, event in readings:
                if cutoff is not None and finished >= cutoff:
                    drain_actions += 1
                    continue
                times.append(elapsed)
                if 'model_batch' in event:
                    calls[event['generation'], event['model_call']] = event['model_batch']
        result = summary(args.mode, times, len(times), window or drained-start, Counter(calls.values()))
        result['clients'] = clients_count
        if cutoff is not None:
            result.update(drain_actions=drain_actions, drain_s=max(0., drained-cutoff))
        return result
    finally:
        await asyncio.gather(*(client.close() for client in clients))
        if service is not None:
            await service.close()
