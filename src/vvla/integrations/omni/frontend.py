"""Optional AsyncOmni service using the same SDK, wire format and action contract."""
import asyncio
from collections import Counter
from copy import deepcopy
from importlib import resources
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from vvla.config import LoadConfig
from vvla.hub import checkpoint_path
from vvla.workers.device import resolve_device


def _probe(config):
    """Only read the checkpoint geometry and device profile in the selected worker."""
    from vvla.dispatch import prepare_config
    prepared, plan = prepare_config(config)
    model = config['model']
    if model == 'pi05':
        from vvla_pi05.lite.pi05 import load_lite_config
        cfg, _ = load_lite_config(config['checkpoint'])
        overrides = prepared['options'].get('config_overrides', {})
        shape = [overrides.get('chunk_size', cfg.chunk_size), overrides.get('action_dim', cfg.action_dim)]
    else:
        shape = [8, 7] if model == 'oft' else [50, 14]
    return dict(shape=shape, max_batch=plan['max_batch'], wait_ms=plan['wait_ms'])


class OmniService:
    def __init__(self, config, *, wait_ms=None, queue_capacity=4096):
        self.config = dict(config, mode='serve')
        self.wait_ms = wait_ms
        self.capacity = queue_capacity
        if queue_capacity < 1 or (wait_ms is not None and (not math.isfinite(wait_ms) or wait_ms < 0)):
            raise ValueError('Invalid Omni queue or wait')
        self.pending = set()
        self.counts = Counter()
        self.omni = self.temp = None
        self.info = None

    def _configure(self):
        from .plugin import register
        import yaml
        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        register(strict=True)
        self.params = OmniDiffusionSamplingParams
        config = dict(self.config)
        config['checkpoint'] = checkpoint_path(str(config['checkpoint']), revision=config.pop('revision', None),
                                               offline=config.get('offline', False))
        cfg = LoadConfig(**config)
        mapping = resolve_device(cfg.device, os.environ)
        payload = dict(cfg.to_dict(), device='cuda:0')
        proc = subprocess.run([cfg.worker_python, '-I', '-m', __name__], input=json.dumps(payload),
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=mapping['visible_token']),
                              capture_output=True, text=True, timeout=cfg.timeout_s)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:] or 'Omni worker profile probe failed')
        profile = json.loads(proc.stdout.strip().splitlines()[-1])
        self.width = profile['max_batch']
        if self.wait_ms is None:
            self.wait_ms = profile['wait_ms']
        h, d = profile['shape']
        self.temp = tempfile.TemporaryDirectory(prefix='vvla-omni-')
        config['worker_python'] = cfg.worker_python
        # The stage is launched with exactly the selected device visible.
        config['device'] = 'cuda:0'
        deploy = dict(pipeline='vvla_dynamic', async_chunk=False, distributed_executor_backend='mp', dtype='bfloat16',
            stages=[dict(stage_id=0, devices=cfg.device.split(':')[1], max_num_seqs=self.width,
                         request_batch_max_wait_ms=self.wait_ms, enforce_eager=True, model_class_name='VvlaPipeline',
                         custom_pipeline_args=dict(engine=config),
                         model_config=dict(policy_server_config=dict(action_horizon=h, action_dim=d, needs_session_id=False)))])
        path = Path(self.temp.name) / 'deploy.yaml'
        path.write_text(yaml.safe_dump(deploy, sort_keys=False))
        controller = resources.files(__package__).joinpath('controller')
        return AsyncOmni(model=str(controller), deploy_config=str(path), async_chunk=False, log_stats=False,
                         stage_init_timeout=1200, init_timeout=1800)

    async def start(self):
        try:
            self.omni = self._configure()
            self.info = (await self._dispatch(dict(op='metadata', request_id=uuid.uuid4().hex)))['result']
        except BaseException:
            await self.close()
            raise

    async def _dispatch(self, command):
        sampling = self.params(num_inference_steps=10, seed=0, extra_args={'robot_obs': command})
        value = None
        async for output in self.omni.generate({'prompt': 'robot observation'}, request_id=command['request_id'],
                                              sampling_params_list=[sampling]):
            if getattr(output, 'finished', False):
                if value is not None:
                    raise RuntimeError('Multiple terminal outputs')
                value = output.multimodal_output['actions']
        if value is None or value['request_id'] != command['request_id']:
            raise RuntimeError('Missing or mismatched terminal output')
        return value

    async def infer(self, observation, *, seed, request_id=None, timeout_s=30., target_slo_ms=None,
                    valid_until_at=None, _owned=False):
        identity = request_id or uuid.uuid4().hex
        if self.info is None:
            raise RuntimeError('Omni is not running')
        if identity in self.pending:
            raise ValueError('Duplicate live request ID')
        if type(seed) is not int or not 0 <= seed < 2**self.info['capabilities']['seed_bits']:
            raise ValueError('Invalid per-row seed')
        if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Positive finite timeout required')
        if target_slo_ms is not None and (type(target_slo_ms) not in (int,float) or not math.isfinite(target_slo_ms) or target_slo_ms<=0):
            raise ValueError('Positive finite target SLO required')
        self.counts['offered'] += 1
        if len(self.pending) >= self.capacity:
            self.counts['queue_full'] += 1
            raise OverflowError('Omni ingress capacity exhausted')
        self.pending.add(identity)
        now = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                result = await self._dispatch(dict(op='infer', request_id=identity, seed=seed, admitted_at=now,
                    observation=observation if _owned else deepcopy(observation),
                    deadline_at=now+timeout_s, valid_until_at=valid_until_at))
            if result['status'] == 'expired':
                raise TimeoutError('Request expired in Omni')
            self.counts['completed'] += 1
            event = result['scheduler']
            delivered = time.perf_counter()
            event['slo_met'] = delivered <= now + target_slo_ms / 1000 if target_slo_ms is not None else None
            event['action_valid'] = delivered <= valid_until_at if valid_until_at is not None else None
            return dict(result=result['result'], scheduler=event)
        except (asyncio.CancelledError, TimeoutError):
            await self.omni.abort(identity)
            self.counts['cancelled_or_expired'] += 1
            raise
        finally:
            self.pending.discard(identity)

    def health(self):
        return dict(status='running' if self.info else 'closed', pending=len(self.pending), **self.counts)

    async def metrics(self):
        data = await self._dispatch(dict(op='metrics', request_id=uuid.uuid4().hex))
        return dict(self.health(), **data['result'])

    async def metadata(self):
        data = await self._dispatch(dict(op='metadata', request_id=uuid.uuid4().hex))
        return dict(data['result'], scheduling=dict(policy={'name':'omni'}, max_batch=self.width, wait_ms=self.wait_ms))

    async def close(self):
        try:
            if self.omni is not None:
                value = self.omni.close()
                if inspect.isawaitable(value):
                    await value
        finally:
            self.omni = self.info = None
            if self.temp is not None:
                self.temp.cleanup()


if __name__ == '__main__':
    import sys
    print(json.dumps(_probe(json.load(sys.stdin))))
