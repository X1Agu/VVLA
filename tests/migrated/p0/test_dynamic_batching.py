import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest

from vvla.serve import batching as m


class Executor:
    def __init__(self):self.groups=[];self.threads=[];self.closed=False
    def infer_requests(self,requests):
        self.groups.append([r.request_id for r in requests]);self.threads.append(threading.get_ident())
        return [(r.request_id,r.seed,r.observation['value']) for r in requests]
    def close(self):self.closed=True;self.threads.append(threading.get_ident())


def request(i,deadline=1.):
    now=time.perf_counter()
    return m.Request(str(i),{'value':i},i+10,now,now+deadline)


@pytest.mark.parametrize('count',range(66))
def test_decomposition_executes_only_real_lanes(count):
    sizes=m.true_batch_sizes(count)
    assert sum(sizes)==count and all(x in (1,2,4,8,16,32) for x in sizes)


def test_independent_admissions_batch_and_keep_ids_seeds_order_and_single_owner():
    async def run():
        executor=Executor();batcher=m.DynamicBatcher(executor,max_batch=4,wait_ms=50)
        await batcher.start()
        tasks=[]
        # Separate arrivals with no client barrier and no preassembled batch.
        for i in range(7):
            tasks.append(asyncio.create_task(batcher.submit(request(i))))
            await asyncio.sleep(.001)
        result=await asyncio.gather(*tasks)
        await batcher.close()
        assert [r['result'] for r in result]==[(str(i),i+10,i) for i in range(7)]
        assert [len(g) for g in executor.groups]==[4,2,1]
        assert len(set(executor.threads))==1 and executor.closed
        assert not batcher.pending
    asyncio.run(run())


def test_cancelled_and_expired_requests_do_not_steal_other_seed_or_output():
    async def run():
        executor=Executor();b=m.DynamicBatcher(executor,max_batch=4,wait_ms=50)
        await b.start()
        cancelled=asyncio.create_task(b.submit(request(0)))
        await asyncio.sleep(.001);cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):await cancelled
        expired=asyncio.create_task(b.submit(request(1,deadline=-1)))
        live=asyncio.create_task(b.submit(request(2)))
        with pytest.raises(TimeoutError):await expired
        result=await live
        await b.close()
        assert executor.groups==[['2']] and result['result']==('2',12,2)
        assert {e['status'] for e in b.events}=={'completed','cancelled','expired'}
    asyncio.run(run())


def test_cancel_in_flight_drains_and_next_request_keeps_its_own_output():
    async def run():
        executor=Executor();started=threading.Event();release=threading.Event()
        original=executor.infer_requests
        def execute(requests):started.set();assert release.wait(2);return original(requests)
        executor.infer_requests=execute
        b=m.DynamicBatcher(executor,max_batch=1,wait_ms=0);await b.start()
        first=asyncio.create_task(b.submit(request(0)))
        while not started.is_set():await asyncio.sleep(.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):await first
        second=asyncio.create_task(b.submit(request(1)))
        release.set();out=await second;await b.close()
        assert out['result']==('1',11,1) and executor.groups==[['0'],['1']]
        assert b.events[0]['cancelled_in_flight'] and executor.closed
    asyncio.run(run())


def test_executor_failure_poison_covers_queued_and_current_requests_and_closes():
    async def run():
        executor=Executor()
        def fail(_):raise RuntimeError('injected model failure')
        executor.infer_requests=fail
        b=m.DynamicBatcher(executor,max_batch=2,wait_ms=0);await b.start()
        tasks=[asyncio.create_task(b.submit(request(i))) for i in range(4)]
        outcomes=await asyncio.gather(*tasks,return_exceptions=True)
        assert all(isinstance(x,RuntimeError) for x in outcomes)
        with pytest.raises(RuntimeError,match='not accepting'):await b.submit(request(5))
        with pytest.raises(RuntimeError,match='model failure'):await b.close()
        assert not b.pending and executor.closed
    asyncio.run(run())
