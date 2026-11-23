import asyncio
import math
import time
import uuid

from ..errors import AdmissionRejected
from .batching import DynamicBatcher, Request


def observation_key(observation, *, per_request_text=()):
    """Conservative grouping. Model/checkpoint/recipe are fixed by the owner."""
    def shape(value,path=()):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return (tuple(value.shape), str(value.dtype))
        if isinstance(value, dict):
            return tuple((k, shape(v,(*path,k))) for k, v in sorted(value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(shape(v,path) for v in value)
        if len(path)==1 and path[0] in per_request_text and type(value) is str:
            return ('per_request_text',)
        # Normalization/domain strings still separate incompatible requests.
        return value if isinstance(value, str) else type(value).__name__
    return shape(observation)


class DynamicBatchServer:
    """Async local service for one model owner. All clients share its queue.

    No socket or cross-model scheduler is implicit. Applications may expose
    this service through their own transport without changing the model API.
    """

    def __init__(self, engine_factory, *, max_batch=None, wait_ms=None, queue_capacity=4096,
                 event_capacity=4096, compatibility_key=None, policy=None, target_slo_ms=None):
        self.factory = engine_factory
        self.max_batch = max_batch
        self.options = dict(wait_ms=wait_ms, queue_capacity=queue_capacity, event_capacity=event_capacity,policy=policy)
        if target_slo_ms is not None and (type(target_slo_ms) not in (int,float)
                or not math.isfinite(target_slo_ms) or target_slo_ms<=0):
            raise ValueError('target_slo_ms must be finite and positive')
        self.target_slo_ms=target_slo_ms
        self.key = compatibility_key
        self._custom_key = compatibility_key
        self.batcher = None
        self.generation = 0
        self._network = False
        self._transition = asyncio.Lock()

    @classmethod
    def from_model(cls, model, *, checkpoint, max_batch=None, dispatch_cap=None, wait_ms=None, policy=None,
                   queue_capacity=4096, target_slo_ms=None, **load_options):
        from ..api import load
        if load_options.pop('mode', 'serve') != 'serve':
            raise ValueError('A server requires mode=serve')
        return cls(lambda: load(model, checkpoint=checkpoint, max_batch=max_batch, mode='serve', **load_options),
                   max_batch=max_batch if dispatch_cap is None else dispatch_cap,
                   wait_ms=wait_ms, policy=policy, queue_capacity=queue_capacity,
                   target_slo_ms=target_slo_ms)

    async def start(self, *, network=False):
        async with self._transition:
            if self.batcher is not None:
                raise RuntimeError("Server already started")
            self._network = network
            await self._start()

    async def _start(self):
        engine = await asyncio.to_thread(self.factory)
        try:
            capabilities=engine.capabilities()
            supported = tuple(capabilities["batch_sizes"])
            text_fields=tuple(capabilities.get('batch_independent_text_fields',()))
            self.key=self._custom_key or (lambda observation: observation_key(observation,per_request_text=text_fields))
            defaults = capabilities.get('serving_defaults', {})
            options = dict(self.options)
            cap = self.max_batch if self.max_batch is not None else defaults.get('max_batch', max(supported))
            if options['wait_ms'] is None:
                wait = defaults.get('wait_ms', 2.)
                if self._network and options['policy'] is None and cap == defaults.get('max_batch'):
                    wait = defaults.get('network_wait_ms', wait)
                options['wait_ms'] = wait
            batcher = DynamicBatcher(engine, supported=supported, max_batch=cap, **options)
            self.capabilities = capabilities
            await batcher.start()
        except BaseException:
            await asyncio.to_thread(engine.close)
            raise
        self.batcher = batcher
        self.generation += 1

    async def infer(self, observation, *, seed, request_id=None, timeout_s=30., target_slo_ms=None,
                    slo_deadline_at=None, valid_until_at=None, _owned=False):
        if self.batcher is None:
            raise RuntimeError("Server is not started")
        bits = self.capabilities.get('seed_bits', 63)
        if type(seed) is not int or not 0 <= seed < 2**bits:
            raise ValueError(f'Expected uint{bits} seed')
        now = time.perf_counter()
        target=self.target_slo_ms if target_slo_ms is None else target_slo_ms
        if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or timeout_s<=0:
            raise ValueError('timeout_s must be finite and positive')
        if target is not None and (type(target) not in (int,float) or not math.isfinite(target) or target<=0):
            raise ValueError('target_slo_ms must be finite and positive')
        if slo_deadline_at is not None and (type(slo_deadline_at) not in (int,float) or not math.isfinite(slo_deadline_at)):
            raise ValueError('slo_deadline_at must use the local perf_counter clock')
        slo=now+target/1000 if target is not None else None
        if slo_deadline_at is not None:slo=slo_deadline_at
        request = Request(request_id or uuid.uuid4().hex, observation, seed,
                          now, now + timeout_s, self.key(observation),slo,valid_until_at)
        # Client hard timeout never releases an in-flight GPU owner; it drains
        # on its single thread (the isolated worker has its own fatal timeout).
        generation = self.generation
        batcher = self.batcher
        try:
            response = await asyncio.wait_for(batcher.submit(request, owned=_owned), timeout=timeout_s)
        except AdmissionRejected:
            raise
        except TimeoutError:
            batcher.client_timeouts += 1
            raise
        except asyncio.CancelledError:
            batcher.client_cancellations += 1
            raise
        response['scheduler']['generation'] = generation
        return response

    def health(self):
        return {"generation": self.generation,
                **(self.batcher.health() if self.batcher else {"status": "new"})}

    def metrics(self):
        batcher = self.batcher
        if batcher is None:
            return self.health()
        recent = [e for e in batcher.events if 'executor_ms' in e]
        def percentile(key):
            if not recent:
                return None
            values = sorted(e[key] for e in recent)
            return values[math.ceil(len(values) * .95) - 1]
        return dict(self.health(), model_batches=dict(batcher.model_batches), graph_builds=batcher.graph_builds,
                    graph_rebuilds=batcher.graph_rebuilds, graph_fallbacks=batcher.graph_fallbacks, graph_replays=batcher.graph_replays,
                    outcomes=dict(batcher.outcomes), client_timeouts=batcher.client_timeouts,
                    client_cancellations=batcher.client_cancellations, recent_requests=len(recent),
                    queue_p95_ms=percentile('queue_ms'), execution_p95_ms=percentile('executor_ms'))

    async def metadata(self):
        if self.batcher is None:
            raise RuntimeError('Server is not started')
        engine = self.batcher.executor
        data = await asyncio.get_running_loop().run_in_executor(self.batcher.pool, engine.metadata)
        return dict(engine=data, capabilities=self.capabilities, scheduling=dict(
            self.scheduling_metadata(), max_batch=self.batcher.max_batch, wait_ms=self.batcher.wait_s*1000))

    def scheduling_metadata(self):
        policy=self.options['policy']
        return dict(policy=policy.metadata() if policy else {'name':'fifo'},target_slo_ms=self.target_slo_ms)

    async def restart(self):
        async with self._transition:
            if self.batcher is None or self.batcher.failure is None:
                raise RuntimeError("Only a failed, drained owner can be restarted")
            try:
                await self.batcher.close()
            except Exception:
                pass  # Failed owner is disposed by close before a fresh process starts.
            if not self.batcher.owner_closed:
                raise RuntimeError('Old owner failed retirement; restart requires explicit process disposal')
            self.batcher = None
            await self._start()

    async def close(self):
        async with self._transition:
            if self.batcher is not None:
                await self.batcher.close()

