import asyncio
import threading
import time
import numpy as np
import pytest

from vvla.errors import InputError
from vvla.serve.batching import DynamicBatcher, Request


def request(i, *, group=(), value=None):
    now = time.perf_counter()
    return Request(str(i), {"x": np.array([i]) if value is None else value}, i, now, now+10, group)


def test_grouping_snapshot_output_ownership_and_bounded_events():
    class Executor:
        def __init__(self): self.calls=[]; self.output=np.zeros(1); self.closes=0
        def infer_requests(self, rows):
            assert len({r.compatibility for r in rows})==1
            self.calls.append([r.seed for r in rows])
            self.output[:]=rows[0].observation['x'][0]
            return [self.output for r in rows]
        def close(self):self.closes+=1
    async def run():
        e=Executor(); b=DynamicBatcher(e,max_batch=2,wait_ms=5,event_capacity=2)
        await b.start()
        mutable=np.array([123])
        tasks=[asyncio.create_task(b.submit(request(i,group=(i%2,),value=mutable if i==0 else None))) for i in range(4)]
        await asyncio.sleep(0)
        mutable[:]=-5
        output=await asyncio.gather(*tasks)
        await b.close();await b.close()
        assert e.calls==[[0,2],[1,3]] and e.closes==1
        assert output[0]['result'][0]==123 and output[1]['result'][0]==1
        output[0]['result'][:]=-1
        assert output[2]['result'][0]==123
        assert len(b.events)==2 and b.total_completed==4
    asyncio.run(run())


def test_rejected_input_does_not_poison_next_request():
    class Executor:
        def infer_requests(self, rows):
            if rows[0].seed==0:raise InputError('invalid image, before execution')
            return [r.seed for r in rows]
    async def run():
        b=DynamicBatcher(Executor(),max_batch=1,wait_ms=0)
        await b.start()
        with pytest.raises(InputError):await b.submit(request(0))
        assert (await b.submit(request(1)))['result']==1
        assert b.failure is None
        await b.close()
    asyncio.run(run())


def test_cancelling_close_waiter_does_not_release_inflight_owner():
    entered=threading.Event(); release=threading.Event(); closed=threading.Event()
    class Executor:
        def infer_requests(self, rows):
            entered.set();assert release.wait(5);return [1]
        def close(self):closed.set()
    async def run():
        b=DynamicBatcher(Executor(),max_batch=1)
        await b.start()
        output=asyncio.create_task(b.submit(request(0)))
        while not entered.is_set():await asyncio.sleep(.001)
        close=asyncio.create_task(b.close());await asyncio.sleep(0);close.cancel()
        with pytest.raises(asyncio.CancelledError):await close
        assert not closed.is_set()
        release.set();await output;await b.close()
        assert closed.is_set()
    asyncio.run(run())


def test_server_restart_waits_for_failed_owner_retirement():
    from vvla.serve import DynamicBatchServer
    from vvla.errors import WorkerFatalError
    events=[]
    class Executor:
        def __init__(self,generation):self.generation=generation;events.append(('create',generation))
        def capabilities(self):return {'batch_sizes':[1]}
        def infer_requests(self,rows):
            if self.generation==1:raise WorkerFatalError('injected device failure')
            return [row.seed for row in rows]
        def close(self):events.append(('close',self.generation))
    async def run():
        count=0
        def factory():
            nonlocal count
            count+=1;return Executor(count)
        server=DynamicBatchServer(factory,max_batch=1,wait_ms=0)
        await server.start()
        with pytest.raises(WorkerFatalError):await server.infer({'x':np.ones(1)},seed=1)
        await server.restart()
        assert events[:3]==[('create',1),('close',1),('create',2)]
        assert (await server.infer({'x':np.ones(1)},seed=7))['result']==7
        assert server.health()['generation']==2
        await server.close();await server.close()
        assert events[-1]==('close',2)
    asyncio.run(run())


def test_declared_prompt_batching_keeps_normalization_domains_separate():
    from vvla.serve import DynamicBatchServer
    class Executor:
        def __init__(self):self.groups=[]
        def capabilities(self):return dict(batch_sizes=[1,2],batch_independent_text_fields=['prompt'])
        def infer_requests(self,rows):
            self.groups.append([r.observation for r in rows])
            return [r.observation['prompt'] for r in rows]
        def close(self):pass
    async def run():
        owner=Executor();server=DynamicBatchServer(lambda:owner,wait_ms=10)
        await server.start()
        observations=[dict(prompt='pick bowl',norm_key='a'),dict(prompt='open drawer',norm_key='a'),
                      dict(prompt='pick bowl',norm_key='b')]
        output=await asyncio.gather(*(server.infer(obs,seed=i) for i,obs in enumerate(observations)))
        assert [row['result'] for row in output]==[o['prompt'] for o in observations]
        assert [len(group) for group in owner.groups]==[2,1]
        assert all(len({o['norm_key'] for o in group})==1 for group in owner.groups)
        await server.close()
    asyncio.run(run())


def test_failed_retirement_is_retryable_on_same_owner_thread():
    class Executor:
        def __init__(self):self.calls=[]
        def infer_requests(self,rows):return [0 for row in rows]
        def close(self):
            self.calls.append(threading.get_ident())
            if len(self.calls)==1:raise RuntimeError('injected retryable drain failure')
    async def run():
        owner=Executor();batcher=DynamicBatcher(owner,max_batch=1)
        await batcher.start()
        with pytest.raises(RuntimeError,match='retryable'):await batcher.close()
        assert not batcher.owner_closed
        await batcher.close()
        assert batcher.owner_closed and len(owner.calls)==2 and len(set(owner.calls))==1
    asyncio.run(run())

