"""Command-line entry points for installing assets, serving and measuring VVLA."""
import argparse
import asyncio
from importlib import resources
import json
from pathlib import Path
import sys

from . import __version__
from .config import MODELS
from .errors import VVLAError


def model_flags(parser):
    parser.add_argument('--config', type=Path, help='SDK load configuration as JSON')
    parser.add_argument('--model', choices=sorted(MODELS))
    parser.add_argument('--checkpoint', help='Local checkpoint or Hub repository')
    parser.add_argument('--offline', action='store_true', default=None, help='Use cached Hub files only')
    parser.add_argument('--revision', help='Hub commit or revision')
    parser.add_argument('--device', help='Visible CUDA device, e.g. cuda:0')
    parser.add_argument('--recipe', help='auto or an explicit model recipe')
    parser.add_argument('--worker-python', help='Isolated backend interpreter')
    parser.add_argument('--asset', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--max-batch', type=int, help='Maximum real model width')


def configuration(args):
    data = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(data, dict):
        raise ValueError('--config must contain an object')
    for name in ('model','checkpoint','revision','device','recipe','worker_python','max_batch','offline'):
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    assets = dict(data.get('assets') or {})
    for field in args.asset:
        key, sep, path = field.partition('=')
        if not key or not sep or not path:
            raise ValueError('--asset requires NAME=PATH')
        assets[key] = path
    if assets:
        data['assets'] = assets
    if 'model' not in data or 'checkpoint' not in data:
        raise ValueError('Provide --model and --checkpoint, or --config')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vvla')
    parser.add_argument('--version', action='version', version=__version__)
    subs = parser.add_subparsers(dest='command', required=True)
    doctor = subs.add_parser('doctor', help='Inspect isolated workers and GPU routes')
    doctor.add_argument('--model', choices=sorted(MODELS))
    doctor.add_argument('--worker-python')
    doctor.add_argument('--device', default='cuda:0')
    doctor.add_argument('--json', action='store_true')
    pull = subs.add_parser('pull', help='Cache a Hub checkpoint or asset')
    pull.add_argument('repo')
    pull.add_argument('--revision')
    pull.add_argument('--offline', action='store_true')
    serve = subs.add_parser('serve', help='Run the binary HTTP and WebSocket service')
    model_flags(serve)
    serve.add_argument('--backend', choices=['vvla','omni'], default='vvla')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8000)
    serve.add_argument('--wait-ms', type=float)
    serve.add_argument('--dispatch-cap', type=int, help='Limit native scheduler batches independently of the loaded model width')
    serve.add_argument('--queue-capacity', type=int, default=4096)
    serve.add_argument('--policy', choices=['fifo','edf','slack-fifo','slack-edf','cohort-slack-edf'], default='fifo')
    serve.add_argument('--service-profile', help='Service-cost JSON file, pi05-g4-prefix-fp8 or oft-g4-bf16')
    serve.add_argument('--max-request-mib', type=int, default=16)
    from .bench import ONLINE_CLIENTS
    bench = subs.add_parser('bench', help='Measure dedicated B1, fixed batches or independent clients')
    model_flags(bench)
    bench.add_argument('--mode', choices=['b1', 'fixed', *ONLINE_CLIENTS], default='b1')
    bench.add_argument('--input', type=Path, required=True, help='Observation/seed corpus in VVLA binary or JSON wire format')
    bench.add_argument('--batch-size', type=int, default=4)
    length = bench.add_mutually_exclusive_group()
    length.add_argument('--iterations', type=int, default=20, help='Measured calls per client')
    length.add_argument('--duration', type=float, help='Online measurement window in seconds')
    bench.add_argument('--warmup', type=int, default=2, help='Unmeasured calls per client')
    bench.add_argument('--url', help='Existing server URL for an online benchmark')
    bench.add_argument('--transport', choices=['http', 'ws'], default='http')
    bench.add_argument('--timeout', type=float, default=60)
    bench.add_argument('--output', type=Path, help='Write the JSON result to this file')
    bench.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            from .doctor import diagnose
            result = diagnose(args.model, device=args.device, interpreter=args.worker_python)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for model, info in result['workers'].items():
                    print(f"{model:8} {info.get('gpu',{}).get('name','-')}  {info['python']}")
                    if info['issues']:
                        print('  ' + ', '.join(info['issues']))
                    else:
                        print('  ' + json.dumps(info['suggested']))
            return 0 if all(x['ready'] for x in result['workers'].values()) else 1
        if args.command == 'pull':
            from .hub import pull as pull_model
            print(pull_model(args.repo, revision=args.revision, offline=args.offline))
        elif args.command == 'serve':
            from aiohttp import web
            from .serve import DynamicBatchServer
            from .serve.network import create_app
            from .serve.policy import scheduling_policy
            policy = {}
            if args.service_profile:
                profile = Path(args.service_profile)
                if not profile.is_file() and args.service_profile in ('pi05-g4-prefix-fp8', 'oft-g4-bf16'):
                    profile = resources.files('vvla').joinpath('data', args.service_profile + '.json')
                policy = json.loads(profile.read_text(encoding='utf-8'))
            if not isinstance(policy, dict):
                raise ValueError('--service-profile must contain a JSON object')
            policy['name'] = args.policy
            if args.policy == 'cohort-slack-edf':
                policy.update(name='slack-edf', cohort_admission=True)
            config = configuration(args)
            dispatch_cap = config.pop('dispatch_cap', None)
            if args.dispatch_cap is not None:
                dispatch_cap = args.dispatch_cap
            if args.backend == 'omni':
                if args.policy != 'fifo' or args.service_profile or dispatch_cap is not None:
                    raise ValueError('Omni uses its upstream scheduler')
                from .integrations.omni.frontend import OmniService
                service = OmniService(config, wait_ms=args.wait_ms, queue_capacity=args.queue_capacity)
            else:
                if args.policy in ('slack-fifo', 'slack-edf', 'cohort-slack-edf') and 'service_ms' not in policy:
                    raise ValueError('Provide --service-profile with a service_ms map for the selected policy')
                service = DynamicBatchServer.from_model(**config, dispatch_cap=dispatch_cap, policy=scheduling_policy(policy),
                             wait_ms=args.wait_ms, queue_capacity=args.queue_capacity)
            app = create_app(service, max_request_bytes=args.max_request_mib*1024**2)
            web.run_app(app, host=args.host, port=args.port, handler_cancellation=True)
        else:
            from .bench import run
            record = asyncio.run(run(args, {} if args.url else configuration(args)))
            output = json.dumps(record, indent=2)
            if args.output:
                args.output.write_text(output+'\n')
            if args.json:
                print(output)
            else:
                print('mode   actions   actions/s    P50 ms    P95 ms')
                latency = ['n/a' if record[key] is None else f'{record[key]:.2f}' for key in ('p50_ms', 'p95_ms')]
                print(f"{record['mode']:5} {record['actions']:8} {record['actions_per_s']:11.2f} {latency[0]:>9} {latency[1]:>9}")
        return 0
    except ImportError as error:
        print(f'vvla: missing dependency ({error}). Install vvla[serve] for network use or vvla[hub] for Hub assets.', file=sys.stderr)
        return 2
    except (VVLAError, OSError, ValueError, RuntimeError) as error:
        print(f'vvla: {error}', file=sys.stderr)
        return 2
