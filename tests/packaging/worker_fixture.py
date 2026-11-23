"""Installed ONLY in the disposable protocol test environment; never shipped."""
import os
import time
import numpy as np
from vvla.errors import InputError


class Engine:
    def __init__(self,close_fails=False):self.buffer=np.zeros(2,dtype=np.float32);self.closed=False;self.close_fails=close_fails
    def capabilities(self):return dict(batch_sizes=[1,2,4,8,16,32],model='protocol-test-fixture')
    def metadata(self):return dict(fixture=True,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                                  native_cold_probe=getattr(self,'native_cold_probe',None))
    def infer_batch(self,observations,seeds):
        operation=observations[0].get('mode')
        if operation=='input':raise InputError('injected input rejection')
        if operation=='fatal':raise RuntimeError('injected worker failure')
        if operation=='timeout':time.sleep(10)
        if operation=='logs':
            for i in range(200):print('fixture log',i)
        if operation=='exit':os._exit(7)
        if operation=='echo_images':return [o['images'].copy() for o in observations]
        outputs=[]
        for observation,seed in zip(observations,seeds):
            self.buffer[:]=[observation['value'],seed]
            outputs.append(self.buffer.copy())
        return outputs
    def reset(self,episode_id=None):return episode_id
    def close(self):
        if self.close_fails:raise RuntimeError('injected close failure')
        self.closed=True


def load_backend(config):
    if config['checkpoint']=='load-fails':raise RuntimeError('injected load failure')
    engine=Engine(close_fails=config['checkpoint']=='close-fails')
    if config['checkpoint']=='native-cold':
        import shutil,sys,torch
        from pathlib import Path
        from vvla_pi05.lite.native_lifecycle import load_native_lifecycle
        expected=Path(sys.executable).absolute().parent
        print('vvla-cold-ninja:',shutil.which('ninja'),flush=True)
        collector=load_native_lifecycle(variant='v3')
        assert Path(shutil.which('ninja') or '').absolute()==expected/'ninja'
        model=torch.nn.Linear(4,8)
        before=collector.signature(model)
        with torch.no_grad():model.weight.add_(1)
        assert collector.signature(model)!=before
        assert not torch.cuda.is_initialized()
        engine.native_cold_probe=dict(ninja=shutil.which('ninja'),python=sys.executable,
            extension=collector.manifest(),cuda_initialized=False)
    return engine
