"""Exercise the installed SDK/worker protocol over actual subprocess pipes."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import statistics
import shutil

import numpy as np
from vvla.api import Engine
from vvla.config import LoadConfig
from vvla.errors import BackendLoadError, InputError, WorkerFatalError

# Only backend resolution is replaced. Both ends use the installed protocol,
# entrypoint, client and API. No fixture code enters a release distribution.
real_popen=subprocess.Popen
def popen(command,*args,**kwargs):
    assert command[-2:]==['-m','vvla.workers.entry']
    command=command[:-2]+['-c',"from vvla.workers import entry; entry.BACKENDS['pi05']='_vvla_test_worker'; entry.main()"]
    return real_popen(command,*args,**kwargs)
subprocess.Popen=popen
os.environ['CUDA_VISIBLE_DEVICES']='3,7'
config=LoadConfig(model='pi05',checkpoint='fixture',device='cuda:1',worker_python=sys.executable,timeout_s=10.)
native_cold_probe=None
if os.environ.get('VVLA_CHECK_NATIVE_COLD')=='1':
    prior_path=os.environ.get('PATH')
    prior_cache=os.environ.get('TORCH_EXTENSIONS_DIR')
    cache=Path(__file__).parent/'cold-worker-extensions'
    assert not cache.exists()
    try:
        os.environ['PATH']='/usr/bin:/bin'
        # Exclude the harness venv. Some hosts also install system Ninja; the
        # child fixture must still prove that its own venv takes precedence.
        parent_ninja=shutil.which('ninja')
        os.environ['TORCH_EXTENSIONS_DIR']=str(cache)
        with Engine(replace(config,checkpoint='native-cold',timeout_s=180.)) as cold:
            native_cold_probe=cold.metadata()['native_cold_probe']
            native_cold_probe['parent_path_ninja']=parent_ninja
        assert list(cache.rglob('*.so')),'Expected a real new native compilation'
    finally:
        if prior_path is None:os.environ.pop('PATH',None)
        else:os.environ['PATH']=prior_path
        if prior_cache is None:os.environ.pop('TORCH_EXTENSIONS_DIR',None)
        else:os.environ['TORCH_EXTENSIONS_DIR']=prior_cache
def fails(kind,fn):
    try:fn()
    except kind:return
    raise AssertionError('Expected '+kind.__name__)

engine=Engine(config)
transport_diagnostic={}
try:
    assert engine.metadata()['cuda_visible_devices']=='7'
    assert engine.metadata()['device_mapping']['worker_device']=='cuda:0'
    result=engine.infer_batch([dict(value=12),dict(value=3)],[17,29],request_ids=['a','b'])
    np.testing.assert_array_equal(result,[[12,17],[3,29]])
    fails(InputError,lambda:engine.infer_batch([dict(mode='input')],[0]))
    later=engine.infer_batch([dict(value=8,mode='logs')],[42])
    np.testing.assert_array_equal(result,[[12,17],[3,29]])
    assert later[0][0]==8 and len(engine._worker.stderr_tail)<=80
    assert engine.reset('episode')=='episode'
    cameras=[dict(mode='echo_images',images=np.full((2,224,224,3),i,np.uint8)) for i in range(32)]
    copies=engine.infer_batch(cameras,list(range(32)))
    for actual,expected in zip(copies,cameras):np.testing.assert_array_equal(actual,expected['images'])
    cameras[0]['images'][:]=255
    assert copies[0][0,0,0,0]==0
    from vvla.protocol.binary import encode_frame,decode_frame
    from vvla.protocol.wire import dumps,loads
    for batch in (1,32):
        rows=[dict(value=i,images=np.zeros((2,224,224,3),np.uint8)) for i in range(batch)]
        payload=dict(observations=rows,seeds=list(range(batch)))
        timings={}
        for label,call in (
            ('legacy_json_base64_codec',lambda:loads(dumps(payload))),
            ('binary_codec',lambda:decode_frame(encode_frame(payload))),
            ('binary_real_worker_roundtrip',lambda:engine.infer_batch(rows,list(range(batch))))):
            call();samples=[]
            for _ in range(10):
                start=time.perf_counter_ns();call();samples.append((time.perf_counter_ns()-start)/1e6)
            timings[label]=dict(median_ms=statistics.median(samples),samples_ms=samples)
        transport_diagnostic[str(batch)]=timings
    fails(InputError,lambda:engine.infer_batch([dict(value=1)],[True]))
    fails(InputError,lambda:engine.infer_batch([dict(value=1),dict(value=2)],[1,2],request_ids=['same','same']))
finally:engine.close();engine.close()
assert not any(t.is_alive() for t in engine._worker._threads)
assert engine._worker._process.poll() is not None

for mode in ('fatal','exit','timeout'):
    engine=Engine(config)
    if mode=='timeout':engine._worker.config=replace(config,timeout_s=.15)
    started=time.monotonic()
    fails(WorkerFatalError,lambda:engine.infer_batch([dict(value=1,mode=mode)],[1]))
    assert time.monotonic()-started<7
    fails(WorkerFatalError,lambda:engine.infer_batch([dict(value=2)],[2]))
    engine.close()
    assert engine._worker._process.poll() is not None
    assert not any(t.is_alive() for t in engine._worker._threads)
fails(BackendLoadError,lambda:Engine(replace(config,checkpoint='load-fails')))

# An adapter failure can precede IPC; a subsequent failed graceful close must
# still acknowledge successful child-process retirement so restart can proceed.
import asyncio
from vvla.serve import DynamicBatchServer
async def close_failure_restart():
    owners=[]
    class Adapter:
        def __init__(self,engine,fail):self.engine,self.fail=engine,fail
        def capabilities(self):return self.engine.capabilities()
        def infer_requests(self,rows):
            if self.fail:raise RuntimeError('adapter failed before IPC')
            return self.engine.infer_requests(rows)
        def close(self):self.engine.close()
    def factory():
        first=not owners
        engine=Engine(replace(config,checkpoint='close-fails' if first else 'fixture'))
        owners.append(engine)
        return Adapter(engine,first)
    server=DynamicBatchServer(factory,max_batch=1,wait_ms=0)
    await server.start()
    try:await server.infer(dict(value=1),seed=2)
    except RuntimeError:pass
    else:raise AssertionError('Missing injected adapter failure')
    await server.restart()
    assert owners[0]._worker._closed and owners[0]._worker._process.poll() is not None
    assert not any(t.is_alive() for t in owners[0]._worker._threads)
    result=await server.infer(dict(value=3),seed=4)
    np.testing.assert_array_equal(result['result'],[3,4])
    await server.close()
asyncio.run(close_failure_restart())
print(json.dumps(dict(status='passed',backend='test fixture, not a model',installed_sdk=True,
    native_cold_probe=native_cold_probe,
    real_subprocess=True,device_mapping=True,input_recovery=True,fatal_exit_timeout=True,
    independent_outputs=True,bounded_logs=True,retired_io_threads=True,
    transport_diagnostic=transport_diagnostic,
    transport_workload=dict(cameras_per_request=2,image_shape=[224,224,3],dtype='uint8',
                            array_bytes_per_request=2*224*224*3),
    diagnostic_scope='CPU codec and fixture-worker transport only; no model timing')))
