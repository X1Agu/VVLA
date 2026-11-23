import asyncio
import threading
import time

import pytest
from vvla.errors import AdmissionRejected
from vvla.serve import DynamicBatcher, DynamicBatchServer, Request, SlackPolicy, EDFPolicy, scheduling_policy


def req(identity, deadline, group=(), *, now=0., slo=None):
    return Request(str(identity), {'value': identity}, identity, now, deadline, group, slo)


def test_batch_limit_and_compatibility_consider_every_members_deadline():
    policy=SlackPolicy({1:10,2:15,4:30},safety_factor=1,margin_ms=0)
    rows=[req(0,.020),req(1,.020),req(2,.1),req(3,.1)]
    assert policy.select(rows,0.,(1,2,4))==2
    assert policy.select(rows,.011,(1,2,4))==0
    # Other domains consume a separate call; they cannot provide dummy lanes.
    assert policy.predict_finish([req(0,.1,('a',))],req(1,.019,('b',)),0.,(1,2),0.) is None
    assert policy.predict_finish([req(0,.1,('a',))],req(1,.019,('a',)),0.,(1,2),0.)==.015


def test_profile_validation_monotone_bound_and_slo_are_explicit():
    policy=SlackPolicy({'1':20,'2':10},safety_factor=1,margin_ms=0)
    assert policy.cost(2)==.020
    assert policy.deadline(req(0,1.,slo=.02))==.02
    assert policy.select([req(0,1.,slo=.01)],0.,(1,2))==0
    with pytest.raises(ValueError,match='every'):policy.validate((1,2,4))
    for costs in ({1:float('nan')},{True:2},{1:-1},{'01':2}):
        with pytest.raises(ValueError):SlackPolicy(costs)


def test_wait_window_cannot_consume_known_execution_slack():
    policy=SlackPolicy({1:10,2:15},safety_factor=1,margin_ms=0)
    assert policy.wait_delay([req(0,.012)],0.,(1,2),.100)==pytest.approx(.002)


def test_busy_owner_prediction_rejection_preserves_seed_and_owner():
    entered=threading.Event();release=threading.Event()
    class Owner:
        def __init__(self):self.calls=[]
        def capabilities(self):return dict(batch_sizes=[1])
        def infer_requests(self,rows):
            self.calls.append([r.seed for r in rows]);entered.set();assert release.wait(3)
            return [r.seed for r in rows]
        def close(self):pass
    async def run():
        owner=Owner();server=DynamicBatchServer(lambda:owner,max_batch=1,wait_ms=0,
            policy=SlackPolicy({1:500},safety_factor=1,margin_ms=0))
        await server.start()
        pending=asyncio.create_task(server.infer({},seed=7,timeout_s=3))
        while not entered.is_set():await asyncio.sleep(.001)
        with pytest.raises(AdmissionRejected) as error:
            await server.infer({},seed=8,timeout_s=3,target_slo_ms=100)
        assert error.value.event['submitted_to_model'] is False
        release.set();assert (await pending)['result']==7
        assert (await server.infer({},seed=9,timeout_s=3))['result']==9
        assert owner.calls==[[7],[9]] and server.health()['slack_rejections']==1
        await server.close()
    asyncio.run(run())


def test_dispatch_rechecks_slack_after_owner_overrun_without_silent_timeout_extension():
    clock=[0.];entered=threading.Event();release=threading.Event()
    class Owner:
        def infer_requests(self,rows):
            entered.set();assert release.wait(3);clock[0]=.05;return [r.seed for r in rows]
    async def run():
        b=DynamicBatcher(Owner(),supported=(1,),max_batch=1,wait_ms=0,clock=lambda:clock[0],
            policy=SlackPolicy({1:10},safety_factor=1,margin_ms=0))
        await b.start()
        first=asyncio.create_task(b.submit(req(0,1.)))
        while not entered.is_set():await asyncio.sleep(.001)
        second=asyncio.create_task(b.submit(req(1,1.,slo=.03)))
        await asyncio.sleep(0)
        release.set();assert (await first)['scheduler']['estimate_overrun']
        with pytest.raises(AdmissionRejected,match='dispatch_insufficient_slack'):await second
        assert b.total_completed==1
        await b.close()
    asyncio.run(run())


def test_fifo_keeps_late_slo_completion_for_control():
    class Owner:
        def capabilities(self):return {'batch_sizes':[1]}
        def infer_requests(self,rows):return [r.seed for r in rows]
        def close(self):pass
    async def run():
        server=DynamicBatchServer(Owner,wait_ms=0)
        await server.start()
        reply=await server.infer({},seed=2,slo_deadline_at=time.perf_counter()-1)
        assert reply['result']==2 and reply['scheduler']['slo_met'] is False
        await server.close()
    asyncio.run(run())


def test_three_slow_arms_then_fast_arm_prediction_and_dispatch_agree():
    clock=[0.];entered=threading.Event();release=threading.Event();calls=[]
    policy=SlackPolicy({1:30,2:35,4:56},safety_factor=1,margin_ms=0,ordering='edf')
    slow=[req(i,.25) for i in (1,2,3)];fast=req(4,.085,now=.005)
    assert policy.predict_finish(slow,fast,.03,(1,2,4),0)==pytest.approx(.065)
    fifo=SlackPolicy({1:30,2:35,4:56},safety_factor=1,margin_ms=0)
    assert fifo.predict_finish(slow,fast,.03,(1,2,4),0) is None
    class Owner:
        def infer_requests(self,rows):
            calls.append([r.seed for r in rows])
            if len(calls)==1:
                entered.set();assert release.wait(3);clock[0]=.03
            else:clock[0]+=policy.cost(len(rows))
            return [r.seed for r in rows]
    async def run():
        b=DynamicBatcher(Owner(),supported=(1,2,4),max_batch=4,wait_ms=0,
                         clock=lambda:clock[0],policy=policy)
        await b.start();block=asyncio.create_task(b.submit(req(0,1)))
        while not entered.is_set():await asyncio.sleep(.001)
        pending=[asyncio.create_task(b.submit(r)) for r in slow]
        await asyncio.sleep(0);clock[0]=.005
        pending.append(asyncio.create_task(b.submit(fast)))
        await asyncio.sleep(0);release.set();await block
        replies=await asyncio.gather(*pending);await b.close()
        assert calls==[[0],[4,1],[2,3]]
        assert [r['result'] for r in replies]==[1,2,3,4]
        assert replies[-1]['scheduler']['deadline_met']
    asyncio.run(run())


def test_tiny_estimate_overrun_accepts_healthy_request_but_hard_failure_propagates():
    clock=[0.];entered=threading.Event();release=threading.Event()
    class Owner:
        def infer_requests(self,rows):
            if rows[0].seed==0:
                entered.set();assert release.wait(3)
            if rows[0].seed==2:raise TimeoutError('executor hard timeout')
            return [r.seed for r in rows]
    async def run():
        b=DynamicBatcher(Owner(),supported=(1,),max_batch=1,wait_ms=0,clock=lambda:clock[0],
            policy=SlackPolicy({1:10},safety_factor=1,margin_ms=0))
        await b.start();first=asyncio.create_task(b.submit(req(0,1)))
        while not entered.is_set():await asyncio.sleep(.001)
        clock[0]=.0102
        second=asyncio.create_task(b.submit(req(1,.51)))
        await asyncio.sleep(0);assert len(b.queue)==1 and b.total_rejected==0
        with pytest.raises(AdmissionRejected):await b.submit(req(3,.011))
        release.set();await first;assert (await second)['result']==1
        with pytest.raises(TimeoutError,match='executor hard'):await b.submit(req(2,1))
        assert b.health()['restart_required']
        with pytest.raises(TimeoutError):await b.close()
    asyncio.run(run())


def test_edf_expiry_ties_compatibility_and_all_four_configuration_controls():
    p=EDFPolicy();rows=[req(0,.3,('a',)),req(1,.1,('b',)),req(2,.1,('a',))]
    assert [r.seed for r in p.order(rows)]==[1,2,0]
    assert p.wait_delay(rows,0,(1,2),.2)==.1
    assert scheduling_policy({'name':'fifo'}) is None
    assert scheduling_policy({'name':'edf'}).name=='edf'
    for name in ('slack-fifo','slack-edf'):
        assert scheduling_policy(dict(name=name,service_ms={1:1})).name==name
    class Owner:
        def infer_requests(self,rows):return [r.seed for r in rows]
    async def run():
        b=DynamicBatcher(Owner(),supported=(1,),max_batch=1,clock=lambda:.2,policy=p)
        await b.start()
        with pytest.raises(TimeoutError):await b.submit(req(1,.1))
        assert (await b.submit(req(2,.5)))['result']==2
        await b.close()
    asyncio.run(run())


def test_empty_buffer_slo_does_not_override_action_validity():
    r=Request('recovery',{},0,0.,1.,(),0.,.2)
    p=SlackPolicy({1:20},safety_factor=1,margin_ms=0)
    assert p.predict_finish([],r,0.,(1,),0.)==.02


@pytest.mark.parametrize('name', ['fifo', 'edf', 'slack-fifo', 'slack-edf'])
def test_closed_loop_priority_prediction_and_dispatch_use_buffer_slo(name):
    # Same action prefix: useful expiry follows arrival order, but starvation
    # urgency does not. The expired soft SLO must neither reject nor delay recovery.
    clock=[0.];entered=threading.Event();release=threading.Event();calls=[]
    config={'name':name}
    if name.startswith('slack'):
        config.update(service_ms={1:20},safety_factor=1,margin_ms=0)
    policy=scheduling_policy(config)
    slow=Request('slow',{},1,0.,1.,('slow',),.18,.2)
    fast=Request('starved',{},2,.005,1.,('fast',),.005,.205)
    edf=name in ('edf','slack-edf')
    if policy is not None and policy.predictive:
        assert policy.predict_finish([slow],fast,.02,(1,),0.)==pytest.approx(.04 if edf else .06)
    class Owner:
        def infer_requests(self,rows):
            calls.append([r.seed for r in rows])
            if len(calls)==1:entered.set();assert release.wait(3)
            clock[0]+=.02
            return [r.seed for r in rows]
    async def run():
        batcher=DynamicBatcher(Owner(),supported=(1,),max_batch=1,wait_ms=0,
                              clock=lambda:clock[0],policy=policy)
        await batcher.start()
        block=asyncio.create_task(batcher.submit(req(0,1.)))
        try:
            while not entered.is_set():await asyncio.sleep(.001)
            a=asyncio.create_task(batcher.submit(slow));await asyncio.sleep(0)
            b=asyncio.create_task(batcher.submit(fast));await asyncio.sleep(0)
            release.set();await block
            results=await asyncio.gather(a,b)
            assert calls==([[0],[2],[1]] if edf else [[0],[1],[2]])
            assert [r['result'] for r in results]==[1,2]
            assert results[1]['scheduler']['slo_met'] is False
        finally:release.set();await batcher.close()
    asyncio.run(run())


def test_soft_deadline_ties_recovery_wait_and_useful_hard_limits():
    p=SlackPolicy({1:20,2:30},safety_factor=1,margin_ms=0,ordering='edf')
    rows=[Request('a',{},1,0.,1.,(),.01,.2),Request('b',{},2,.005,1.,(),.01,.205)]
    assert [r.request_id for r in p.order(rows)]==['a','b']
    assert p.wait_delay(rows[:1],.02,(1,2),.1)==0.
    assert p.select(rows,.02,(1,2))==2
    expired=Request('expired',{},3,0.,1.,(),0.,.03)
    hard=Request('hard',{},4,0.,.03,(),0.,.2)
    assert p.predict_finish([],expired,.02,(1,2),0.) is None
    assert p.predict_finish([],hard,.02,(1,2),0.) is None


def test_server_client_hard_timeout_keeps_owner_until_drained():
    entered=threading.Event();release=threading.Event();closed=[]
    class Owner:
        def capabilities(self):return {'batch_sizes':[1]}
        def infer_requests(self,rows):entered.set();assert release.wait(3);return [0]
        def close(self):closed.append(True)
    async def run():
        s=DynamicBatchServer(Owner,wait_ms=0);await s.start()
        try:
            with pytest.raises(TimeoutError):await s.infer({},seed=0,timeout_s=.03)
            assert entered.is_set() and s.health()['pending']==1 and not closed
        finally:release.set();await s.close()
        assert closed==[True]
    asyncio.run(run())
