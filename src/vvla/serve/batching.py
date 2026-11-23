"""Minimal independent-B1 FIFO batching control over one real batch executor.

This module does no model computation and cannot establish a GPU speedup.
Both serial Lite and dynamic Lite use this exact queue with different limits.
The executor receives only real requests: a partial queue is decomposed into
supported true batch sizes, never padded with inactive or synthetic lanes.
"""
import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from copy import deepcopy
import math
import time

from vvla.errors import AdmissionRejected, InputError
from .policy import EDFPolicy


@dataclass(frozen=True)
class Request:
    request_id: str
    observation: dict
    seed: int
    admitted_at: float
    deadline_at: float
    compatibility: tuple = ()
    slo_deadline_at: float | None = None
    valid_until_at: float | None = None

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError('Each independent request needs a nonempty ID')
        if type(self.observation) is not dict:
            raise TypeError('Each request contains one raw observation dictionary')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Every request has its own uint63 seed')
        if any(type(x) not in (int,float) or not math.isfinite(x)
               for x in (self.admitted_at,self.deadline_at)):
            raise ValueError('Admission/deadline timestamps must be finite')
        if type(self.compatibility) is not tuple:
            raise ValueError('compatibility must be an immutable tuple')
        hash(self.compatibility)
        if self.slo_deadline_at is not None and (type(self.slo_deadline_at) not in (int, float)
                or not math.isfinite(self.slo_deadline_at)):
            raise ValueError('SLO deadline must be a finite timestamp')
        if self.valid_until_at is not None and (type(self.valid_until_at) not in (int, float)
                or not math.isfinite(self.valid_until_at)):
            raise ValueError('Action validity deadline must be a finite timestamp')


def true_batch_sizes(count, supported=(1,2,4,8,16,32)):
    """Return exact real batch sizes covering count, with no dummy work."""
    if type(count) is not int or count < 0:
        raise ValueError('count must be a nonnegative integer')
    if not supported or supported[0]!=1 or tuple(sorted(set(supported)))!=tuple(supported):
        raise ValueError('Supported batches must be ascending, unique, and include B1')
    if any(type(b) is not int or b < 1 for b in supported):
        raise ValueError('Supported batches must be positive integers')
    output=[]
    while count:
        batch=max(b for b in supported if b<=count)
        output.append(batch);count-=batch
    return output


class DynamicBatcher:
    """One GPU-owner thread, independent B1 admissions, bounded queueing window.

    executor.infer_requests(list[Request]) must return one result per input, in
    that order. Cancellation never reassigns a seed/output to another request.
    Deadlines are checked immediately before submission; in-flight work drains
    even if its client cancels. Completion after deadline is explicitly marked.
    """
    def __init__(self, executor, *, supported=(1,2,4,8,16,32),max_batch=32,
                 wait_ms=2.,queue_capacity=4096,event_capacity=4096,clock=time.perf_counter,policy=None):
        true_batch_sizes(0,supported)
        if type(max_batch) is not int or max_batch not in supported:
            raise ValueError('max_batch must be one of the real supported sizes')
        if type(wait_ms) not in (int,float) or not math.isfinite(wait_ms) or wait_ms<0:
            raise ValueError('wait_ms must be finite and nonnegative')
        if type(queue_capacity) is not int or queue_capacity<1:
            raise ValueError('queue_capacity must be positive')
        if type(event_capacity) is not int or event_capacity<1:
            raise ValueError('event_capacity must be positive')
        self.executor,self.supported,self.max_batch=executor,tuple(b for b in supported if b<=max_batch),max_batch
        if policy is not None:
            if not isinstance(policy, EDFPolicy):raise TypeError('policy must be EDFPolicy, SlackPolicy or None')
            policy.validate(self.supported)
        self.policy=policy
        self.inflight_until=None
        self.total_offered=0
        self.total_rejected=0
        self.wait_s,self.queue_capacity,self.clock=wait_ms/1000,queue_capacity,clock
        self.queue=deque();self.pending={};self.condition=asyncio.Condition()
        self._admission_cutoffs = {}
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vvla-gpu-owner')
        self.worker=None;self.closing=False;self.failure=None
        self.events=deque(maxlen=event_capacity)
        self._close_task=None
        self.total_calls = 0
        self.model_batches = Counter()
        self.graph_builds = self.graph_rebuilds = self.graph_replays = self.graph_fallbacks = 0
        self.outcomes = Counter()
        self.client_timeouts = self.client_cancellations = 0
        self.total_completed=0
        self.owner_closed=False

    async def start(self):
        if self.worker is not None or self.closing:
            raise RuntimeError('Batcher must start exactly once')
        self.worker=asyncio.create_task(self._run())

    async def submit(self,request, *, owned=False):
        if type(request) is not Request:
            raise TypeError('Batcher admits one independently identified B1 Request')
        validate=getattr(self.executor,'validate_request',None)
        if validate is not None:validate(request)
        # Admission owns a snapshot; client mutation cannot change queued work.
        if not owned:
            request=Request(request.request_id,deepcopy(request.observation),request.seed,
                            request.admitted_at,request.deadline_at,request.compatibility,request.slo_deadline_at,
                            request.valid_until_at)
        async with self.condition:
            if self.worker is None or self.closing or self.failure is not None:
                raise RuntimeError('Batcher is not accepting requests') from self.failure
            if request.request_id in self.pending:
                raise ValueError('Duplicate live request ID')
            self.total_offered+=1
            self._discard()
            if len(self.queue)>=self.queue_capacity:
                self.outcomes['queue_full'] += 1
                raise OverflowError('Request queue is full')
            if self.policy is not None and self.policy.predictive:
                now=self.clock()
                available=max(now,self.inflight_until or now)
                if self.inflight_until is not None and now >= self.inflight_until:
                    # A service estimate is not the executor's hard timeout.
                    # Reserve one B1 cost, then recheck at actual dispatch.
                    available=now+self._service_cost(1,(request,))
                queued=[r for r,f in self.queue if not f.cancelled()]
                finish=self.policy.predict_finish(queued,request,available,self.supported,self.wait_s)
                if finish is None:
                    cutoff = None
                    if self.inflight_until is None and getattr(self.policy, 'cohort_admission', False):
                        cutoff = self.policy.collection_cutoff(queued, request, now, self.supported, self.wait_s)
                    if cutoff is None:
                        raise self._slack_rejection(request,'admission_insufficient_slack',now)
                    self._admission_cutoffs[request.request_id] = cutoff
            future=asyncio.get_running_loop().create_future()
            self.pending[request.request_id]=future
            self.queue.append((request,future))
            self.condition.notify_all()
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            async with self.condition:self.condition.notify_all()
            raise

    def _service_cost(self, batch, members):
        cost = getattr(self.policy, 'service_cost', None)
        return cost(batch, members) if cost is not None else self.policy.cost(batch)

    def _slack_rejection(self,request,reason,now):
        event=dict(request_id=request.request_id,status=reason,submitted_to_model=False,
                   policy=self.policy.name,slack_ms=(self.policy.deadline(request)-now)*1000)
        self.events.append(event)
        self.total_rejected+=1
        self.outcomes[reason] += 1
        return AdmissionRejected(event)

    def _resolve_admissions(self, now, dispatch):
        """Resolve provisional futures against the actual ordered queue under the lock."""
        rows = [r for r, _ in self.queue]
        kept = deque()
        for i, (request, future) in enumerate(self.queue):
            cutoff = self._admission_cutoffs.get(request.request_id)
            if cutoff is not None and (dispatch or now >= cutoff):
                finish = self.policy.predict_finish(rows[:i], request, now, self.supported,
                                                    self.wait_s, following=rows[i + 1:])
                del self._admission_cutoffs[request.request_id]
                if finish is None:
                    self.pending.pop(request.request_id)
                    future.set_exception(self._slack_rejection(request, 'admission_insufficient_slack', now))
                    continue
            kept.append((request, future))
        self.queue = kept

    def _discard(self, *, dispatch=False):
        kept=deque();now=self.clock()
        while self.queue:
            request,future=self.queue.popleft()
            expiry=min(request.deadline_at,request.valid_until_at) if request.valid_until_at is not None else request.deadline_at
            reason='cancelled' if future.cancelled() else 'expired' if expiry<=now else None
            if reason is None and self.policy is not None and self.policy.predictive and now+self._service_cost(1,(request,))>self.policy.deadline(request):
                reason='dispatch_insufficient_slack'
            if reason:
                self.pending.pop(request.request_id,None)
                self._admission_cutoffs.pop(request.request_id, None)
                if reason=='dispatch_insufficient_slack':
                    error=self._slack_rejection(request,reason,now)
                else:
                    self.outcomes[reason] += 1
                    error=TimeoutError('Deadline elapsed in queue')
                    self.events.append(dict(request_id=request.request_id,status=reason,submitted_to_model=False))
                if not future.done():future.set_exception(error)
            else:kept.append((request,future))
        if self.policy is not None:
            pairs={r.request_id:(r,f) for r,f in kept}
            kept=deque(pairs[r.request_id] for r in self.policy.order([r for r,_ in kept]))
        self.queue=kept
        if self._admission_cutoffs:
            self._resolve_admissions(now, dispatch)

    async def _run(self):
        loop=asyncio.get_running_loop()
        try:
            while True:
                async with self.condition:
                    self._discard()
                    while not self.queue and not self.closing:
                        await self.condition.wait();self._discard()
                    if not self.queue and self.closing:return
                    first=self.queue[0][0]
                    while sum(r.compatibility==first.compatibility for r,_ in self.queue)<self.max_batch and not self.closing:
                        due=min(first.admitted_at+self.wait_s,min(r.deadline_at for r,_ in self.queue))
                        now=self.clock()
                        delay=(self.policy.wait_delay([r for r,_ in self.queue],now,self.supported,self.wait_s)
                               if self.policy is not None else due-now)
                        if self._admission_cutoffs:
                            delay = min(delay, min(self._admission_cutoffs.values()) - now)
                        if delay<=0:break
                        try:await asyncio.wait_for(self.condition.wait(),timeout=delay)
                        except TimeoutError:break
                        self._discard()
                        if not self.queue:break
                        first=self.queue[0][0]
                    self._discard(dispatch=True)
                    if not self.queue:continue
                    # Execute one supported true B now; leave any remainder
                    # independently queued for the next scheduling decision.
                    compatibility=self.queue[0][0].compatibility
                    compatible=sum(r.compatibility==compatibility for r,_ in self.queue)
                    batch=max(b for b in self.supported if b<=min(compatible,self.max_batch))
                    if self.policy is not None:
                        candidates=[r for r,_ in self.queue if r.compatibility==compatibility]
                        batch=self.policy.select(candidates,self.clock(),self.supported)
                        if not batch:
                            self._discard()
                            continue
                    group=[];remaining=deque()
                    while self.queue:
                        pair=self.queue.popleft()
                        if pair[0].compatibility==compatibility and len(group)<batch:group.append(pair)
                        else:remaining.append(pair)
                    self.queue=remaining
                    estimate_s=self._service_cost(batch,(pair[0] for pair in group)) if self.policy is not None and self.policy.predictive else None
                    self.inflight_until=self.clock()+estimate_s if estimate_s is not None else None
                requests=[r for r,_ in group]
                started=self.clock()
                try:
                    results=await loop.run_in_executor(self.pool,self.executor.infer_requests,requests)
                    if type(results) is not list or len(results)!=batch:
                        raise RuntimeError('Executor did not return one ordered output per real request')
                    # Backends may reuse temporary CPU storage between calls.
                    # Transfer owned values before the next model call begins.
                    receipt = getattr(self.executor, 'last_execution', {})
                    actual = receipt.get('actual_tensor_batch')
                    self.total_calls += 1
                    if actual is not None:
                        if actual != batch:
                            raise RuntimeError(f'Model executed {actual} rows for {batch} requests')
                        self.model_batches[actual] += 1
                    self.graph_builds += int(bool(receipt.get('graph_build')))
                    self.graph_rebuilds += int(bool(receipt.get('graph_rebuild')))
                    self.graph_fallbacks += int(bool(receipt.get('graph_fallback')))
                    self.graph_replays += int(bool(receipt.get('graph_replay') and not receipt.get('graph_build')))
                    if not getattr(self.executor, 'owns_outputs', False):
                        results = [deepcopy(result) for result in results]
                except InputError as error:
                    self.outcomes['input_error'] += batch
                    self.inflight_until=None
                    for request,future in group:
                        self.pending.pop(request.request_id,None)
                        if not future.done():future.set_exception(error)
                        self.events.append(dict(request_id=request.request_id,status='input_error',submitted_to_model=False))
                    continue
                except BaseException as error:
                    self.outcomes['worker_error'] += len(self.pending)
                    self.inflight_until=None
                    for request,future in group:
                        self.pending.pop(request.request_id,None)
                        if not future.done():future.set_exception(error)
                    raise
                finished=self.clock()
                self.inflight_until=None
                for (request,future),result in zip(group,results,strict=True):
                    self.pending.pop(request.request_id,None)
                    event=dict(request_id=request.request_id,status='completed',actual_batch=batch,
                        submitted_to_model=True,queue_ms=(started-request.admitted_at)*1000,
                        executor_ms=(finished-started)*1000,deadline_met=finished<=request.deadline_at,
                        cancelled_in_flight=future.cancelled())
                    event.update(policy=self.policy.name if self.policy else 'fifo',
                        estimated_executor_ms=estimate_s*1000 if estimate_s is not None else None,
                        estimate_overrun=(finished-started)>estimate_s if estimate_s is not None else None,
                        dispatch_slack_ms=(self.policy.deadline(request)-started)*1000 if self.policy else None,
                        slo_met=finished<=request.slo_deadline_at if request.slo_deadline_at is not None else None)
                    event['action_valid']=finished<=request.valid_until_at if request.valid_until_at is not None else None
                    if actual is not None:
                        event.update(model_batch=actual, model_call=self.total_calls)
                    self.outcomes['cancelled_in_flight' if future.cancelled() else 'completed'] += 1
                    self.events.append(event)
                    self.total_completed+=1
                    if not future.done():future.set_result(dict(result=result,scheduler=event))
        except BaseException as error:
            self.failure=error
            async with self.condition:
                self._admission_cutoffs.clear()
                while self.queue:
                    request,future=self.queue.popleft()
                    self.pending.pop(request.request_id,None)
                    if not future.done():future.set_exception(error)
            raise

    def health(self):
        return dict(status='failed' if self.failure else 'closing' if self.closing else
                    'running' if self.worker is not None else 'new',queued=len(self.queue),
                    pending=len(self.pending),completed=self.total_completed,
                    events_retained=len(self.events),restart_required=self.failure is not None,
                    offered=self.total_offered,slack_rejections=self.total_rejected,
                    policy=self.policy.name if self.policy else 'fifo')

    async def close(self):
        if self._close_task is None or (self._close_task.done() and not self.owner_closed):
            self._close_task=asyncio.create_task(self._close())
        # Cancelling a client waiting for close must not release a live owner.
        return await asyncio.shield(self._close_task)

    async def _close(self):
        async with self.condition:
            self.closing=True;self.condition.notify_all()
        try:
            if self.worker is not None:await self.worker
        finally:
            # Wait for all in-flight work before shutting down its owner. No
            # daemon worker or cancelled Future releases GPU-owned references.
            try:
                if hasattr(self.executor,'close'):
                    await asyncio.get_running_loop().run_in_executor(self.pool,self.executor.close)
                self.owner_closed=True
            finally:
                if self.owner_closed:self.pool.shutdown(wait=True,cancel_futures=False)
