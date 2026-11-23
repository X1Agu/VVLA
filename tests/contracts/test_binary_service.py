import asyncio
from contextlib import asynccontextmanager
import threading

from aiohttp import ClientSession, web
import numpy as np
import pytest

from vvla import VvlaClient, RemoteError, WorkerFatalError
from vvla.serve import DynamicBatchServer
from vvla.serve.network import create_app, CONTENT_TYPE
from vvla.protocol.binary import PREFIX, MAGIC


class Owner:
    def __init__(self):
        self.last_execution = {}
        self.seen = set()
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def capabilities(self):
        return dict(batch_sizes=[1,2,4], seed_bits=32, serving_defaults=dict(max_batch=4,wait_ms=10))

    def infer_requests(self, rows):
        x = rows[0].observation
        if x.get('fail'):
            raise WorkerFatalError('owner fault')
        if x.get('block'):
            self.started.set()
            assert self.release.wait(5)
        width=len(rows)
        self.last_execution=dict(actual_tensor_batch=width, graph_build=width not in self.seen,
                                 graph_replay=width in self.seen)
        self.seen.add(width)
        return [np.array([r.observation['rgb'][0,0,0], r.seed], dtype=np.float32) for r in rows]

    def metadata(self):
        return dict(test_owner=True)

    def close(self):
        self.release.set()
        self.closed=True


@asynccontextmanager
async def running(*, queue_capacity=64, **settings):
    owners=[]
    def factory():
        owner=Owner()
        owners.append(owner)
        return owner
    server=DynamicBatchServer(factory, queue_capacity=queue_capacity, **settings)
    runner=web.AppRunner(create_app(server, max_request_bytes=8*1024), handler_cancellation=True)
    await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0)
    await site.start()
    try:
        yield 'http://127.0.0.1:'+str(runner.addresses[0][1]), owners
    finally:
        for owner in owners:
            owner.release.set()
        await runner.cleanup()
        assert all(owner.closed for owner in owners)


@pytest.mark.parametrize('transport', ['http', 'ws'])
def test_wire_action_validity_can_extend_past_soft_slo(transport):
    from vvla.serve import SlackPolicy

    async def check():
        policy = SlackPolicy({1: 10}, ordering='edf')
        async with running(max_batch=1, policy=policy) as (url, _), VvlaClient(url, transport=transport) as client:
            with pytest.raises(RemoteError, match='insufficient_slack'):
                await client.infer(observation(1), seed=3, target_slo_ms=.001)
            reply = await client.infer(observation(1), seed=3, target_slo_ms=.001, valid_for_ms=1000)
            assert reply['scheduler']['slo_met'] is False
            assert reply['scheduler']['action_valid'] is True
            with pytest.raises(RemoteError, match='insufficient_slack'):
                await client.infer(observation(1), seed=3, target_slo_ms=100, valid_for_ms=1)
            with pytest.raises(RemoteError, match='valid_for_ms'):
                await client.infer(observation(1), seed=3, valid_for_ms=-1)
    asyncio.run(check())


def test_binary_fifo_uses_network_wait_and_preserves_override(monkeypatch):
    monkeypatch.setattr(Owner, 'capabilities', lambda self: dict(
        batch_sizes=[1, 2, 4], seed_bits=32,
        serving_defaults=dict(max_batch=4, wait_ms=0, network_wait_ms=2)))

    async def check():
        local = DynamicBatchServer(Owner)
        await local.start()
        try:
            assert (await local.metadata())['scheduling']['wait_ms'] == 0
        finally:
            await local.close()
        for options, expected in [({}, 2), ({'wait_ms': 0}, 0), ({'max_batch': 2}, 0)]:
            async with running(**options) as (url, _), VvlaClient(url) as client:
                assert (await client.metadata())['scheduling']['wait_ms'] == expected
    asyncio.run(check())


def test_bench_c32_from_cli_with_binary_http(tmp_path):
    import json
    from vvla.cli import main
    from vvla.protocol.binary import encode_frame

    corpus = tmp_path / 'corpus.vvla'
    output = tmp_path / 'result.json'
    corpus.write_bytes(encode_frame(dict(observations=[observation(i) for i in range(32)], seeds=list(range(32)))))

    async def check():
        async with running() as (url, _):
            code = await asyncio.to_thread(main, ['bench', '--mode', 'c32', '--url', url, '--input', str(corpus),
                                                   '--warmup', '1', '--iterations', '2', '--output', str(output)])
            assert code == 0
        record = json.loads(output.read_text())
        assert record['clients'] == 32 and record['actions'] == record['latency_samples'] == 64
        assert sum(int(width) * count for width, count in record['model_calls_by_width'].items()) == 64
    asyncio.run(check())


@pytest.mark.parametrize('call_seconds,expected', [(2., 0), (.4, 2)])
def test_timed_bench_counts_only_returns_inside_window(monkeypatch, tmp_path, call_seconds, expected):
    from types import SimpleNamespace
    from vvla import bench
    from vvla.protocol.binary import encode_frame

    clock = [0.]
    class Service:
        async def start(self):
            pass
        async def close(self):
            pass
        async def infer(self, observation, **request):
            clock[0] += call_seconds
            return dict(scheduler=dict(generation=1, model_call=request['request_id'], model_batch=1))

    monkeypatch.setattr(bench, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(bench.DynamicBatchServer, 'from_model', lambda **config: Service())
    corpus = tmp_path / 'observations'
    corpus.write_bytes(encode_frame(dict(observations=[observation(1)], seeds=[1])))
    args = SimpleNamespace(input=corpus, mode='c1', url=None, duration=1., warmup=1, iterations=20, timeout=30)
    result = asyncio.run(bench.run(args, {}))
    assert result['actions'] == result['latency_samples'] == result['actions_per_s'] == expected
    assert result['elapsed_s'] == 1. and result['drain_actions'] == 1
    assert result['p95_ms'] == (None if not expected else pytest.approx(400.))


def test_cli_oft_frozen_policy_uses_b4_engine_with_b2_scheduler(monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from vvla import api
    from vvla.cli import main

    def load(model, **options):
        assert model == 'oft' and options['max_batch'] == 4
        return Owner()

    async def inspect(app):
        async with TestClient(TestServer(app)) as client:
            async with client.get('/metadata') as response:
                assert response.status == 200
                metadata = await response.json()
            assert max(metadata['capabilities']['batch_sizes']) == 4
            assert metadata['scheduling']['max_batch'] == 2
            assert metadata['scheduling']['wait_ms'] == 0
            assert metadata['scheduling']['policy']['cohort_admission'] is True

    monkeypatch.setattr(api, 'load', load)
    monkeypatch.setattr(web, 'run_app', lambda app, **options: asyncio.run(inspect(app)))
    assert main(['serve', '--model', 'oft', '--checkpoint', 'fixture', '--recipe', 'bf16',
                 '--max-batch', '4', '--dispatch-cap', '2', '--wait-ms', '0',
                 '--policy', 'cohort-slack-edf', '--service-profile', 'oft-g4-bf16']) == 0


def test_failed_online_bench_retires_clients_before_closing_service(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from vvla import bench
    from vvla.protocol.binary import encode_frame

    class Service:
        def __init__(self):
            self.active = 0
            self.ready = asyncio.Event()

        async def start(self):
            pass

        async def infer(self, observation, *, seed, **kwargs):
            self.active += 1
            if self.active == 8:
                self.ready.set()
            try:
                await self.ready.wait()
                if seed == 0:
                    raise RuntimeError('client failed')
                await asyncio.Future()
            finally:
                self.active -= 1

        async def close(self):
            assert self.active == 0

    monkeypatch.setattr(bench.DynamicBatchServer, 'from_model', lambda **config: Service())
    corpus = tmp_path / 'corpus'
    corpus.write_bytes(encode_frame(dict(observations=[observation(i) for i in range(8)], seeds=list(range(8)))))
    args = SimpleNamespace(input=corpus, mode='c8', url=None, warmup=0, iterations=1, timeout=3)
    with pytest.raises(RuntimeError, match='client failed'):
        asyncio.run(bench.run(args, {}))


def test_omni_slo_includes_grouped_return(monkeypatch):
    from types import SimpleNamespace
    from vvla.integrations.omni import frontend

    service = frontend.OmniService({})
    service.info = dict(capabilities=dict(seed_bits=32))
    clock = [1.]
    monkeypatch.setattr(frontend, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    async def dispatch(command):
        assert command['valid_until_at'] == 1.2
        clock[0] += .1
        return dict(status='completed', result=[1], scheduler=dict(queue_ms=5, executor_ms=10))
    monkeypatch.setattr(service, '_dispatch', dispatch)
    reply = asyncio.run(service.infer({}, seed=7, target_slo_ms=50, valid_until_at=1.2))
    assert reply['scheduler']['slo_met'] is False and reply['scheduler']['action_valid'] is True


def observation(i, **flags):
    return dict(rgb=np.full((3,4,3), i, np.uint8), **flags)


@pytest.mark.parametrize('transport', ['http', 'ws'])
def test_loopback_ids_batch_widths_and_owned_actions(transport):
    async def check():
        async with running() as (url, _), VvlaClient(url, transport=transport) as client:
            meta = await client.metadata()
            assert meta['engine']['test_owner'] and meta['scheduling']['wait_ms'] == 10
            first = await client.infer(observation(9), seed=11, request_id='dedicated')
            assert first['scheduler']['model_batch']==1
            for ids in ((3,0,2,1), (1,2,0,3), (1,4,0,3)):
                replies = await asyncio.gather(*(client.infer(observation(i), seed=i+21, request_id=str(i)) for i in ids))
                assert [r['request_id'] for r in replies] == list(map(str,ids))
                assert all(r['scheduler']['model_batch']==4 for r in replies)
                assert len({r['scheduler']['model_call'] for r in replies})==1
                for i, r in zip(ids, replies):
                    np.testing.assert_array_equal(r['result'], [i,i+21])
                    assert r['result'].flags.owndata and r['result'].dtype == np.float32
            np.testing.assert_array_equal(first['result'],[9,11])
            metrics=await client.metrics()
            assert metrics['model_batches']=={'1':1,'4':3}
            assert metrics['graph_builds']==2 and metrics['graph_replays']==2 and metrics['graph_rebuilds']==0
            assert metrics['outcomes']['completed']==13 and metrics['queue_p95_ms']>=0
            with pytest.raises(RemoteError) as error:
                await client.infer(observation(0), seed=2**32)
            assert error.value.code=='invalid_request'
    asyncio.run(check())


def test_invalid_and_oversize_http_and_restart():
    async def check():
        async with running() as (url,owners), ClientSession() as raw, VvlaClient(url) as client:
            async with raw.post(url+'/infer', data=b'junk') as response:
                assert response.status==415
            for blob,status in [(PREFIX.pack(MAGIC,2,64000),400),(b'0'*9000,413)]:
                async with raw.post(url+'/infer', data=blob, headers={'Content-Type':CONTENT_TYPE}) as response:
                    assert response.status==status
            with pytest.raises(RemoteError) as failure:
                await client.infer(observation(2,fail=True),seed=1)
            assert failure.value.code=='worker_unavailable'
            value=await client.infer(observation(7),seed=3)
            assert value['scheduler']['generation']==2 and len(owners)==2 and owners[0].closed
    asyncio.run(check())


def test_ws_cancel_queue_timeout_and_capacity():
    async def check():
        async with running(queue_capacity=1) as (url,owners), VvlaClient(url,transport='ws') as client:
            block=asyncio.create_task(client.infer(observation(2,block=True),seed=1))
            assert await asyncio.to_thread(owners[0].started.wait,2)
            waiting=asyncio.create_task(client.infer(observation(9),seed=7,request_id='cancel'))
            for _ in range(100):
                if (await client.health())['queued']==1:break
                await asyncio.sleep(.005)
            with pytest.raises(RemoteError) as full:
                await client.infer(observation(0),seed=3)
            assert full.value.code=='queue_full'
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            for _ in range(100):
                if (await client.health())['queued']==0:break
                await asyncio.sleep(.005)
            with pytest.raises(RemoteError) as expired:
                await client.infer(observation(3), seed=0, timeout_s=.04)
            assert expired.value.code=='timeout'
            owners[0].release.set()
            await block
            metrics=await client.metrics()
            assert metrics['outcomes']['queue_full']==1 and metrics['client_timeouts']==1
            assert metrics['client_cancellations']==1
    asyncio.run(check())
