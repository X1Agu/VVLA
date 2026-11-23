"""Actual dual-backbone Pi0.5 math at production widths, reduced depth/vocab.

Two layers each, 28px vision and a synthetic tokenizer. These inputs/weights do
not establish checkpoint quality or real model latency.
"""
import gc
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import asyncio

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


class Tokenizer:
    def __call__(self,text,*,max_length,**kwargs):
        length=min(max_length,max(2,len(text)%max_length))
        return dict(input_ids=[1]*length+[0]*(max_length-length),attention_mask=[1]*length+[0]*(max_length-length))


def make_model(monkeypatch,horizon):
    from vvla_pi05.models.pi05 import modeling_pi05 as core
    from vvla_pi05.models.pi05.config import Pi05Config
    from vvla_pi05.lite.dtype import set_bfloat16_layout
    original_geometry=core.get_gemma_config
    original_vlm,original_expert=core.PaliGemmaForConditionalGeneration,core.GemmaForCausalLM
    def geometry(name):
        value=original_geometry(name);value.depth=2;return value
    def vlm(config):
        config.text_config.vocab_size=1024
        config.vision_config.num_hidden_layers=2
        config.vision_config.image_size=28
        return original_vlm(config)
    def expert(config):
        config.vocab_size=1024
        return original_expert(config)
    with monkeypatch.context() as patch:
        patch.setattr(core,'get_gemma_config',geometry)
        patch.setattr(core,'PaliGemmaForConditionalGeneration',vlm)
        patch.setattr(core,'GemmaForCausalLM',expert)
        config=Pi05Config(chunk_size=horizon,n_action_steps=horizon,num_inference_steps=2,
            image_resolution=(28,28),max_cameras=2,tokenizer_max_length=16,attention_backend='torch_fa2',
            vision_bf16=True,prefix_kv_only=True,precompute_time_and_positions=True,
            input_features={'observation.state':{'shape':[7]},'observation.images.front':{'shape':[3,28,28]},
                            'observation.images.wrist':{'shape':[3,28,28]}},
            output_features={'action':{'shape':[7]}},
            norm_stats={name:{'mode':'mean_std','mean':[0]*7,'std':[1]*7} for name in ('state','action')})
        model=core.Pi05ForActionPrediction(config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,core.Pi05AdaRMSNorm) and module.dense is not None:
                module.dense.weight.normal_(0,.01)
                module.dense.bias.zero_()
                module.dense.bias[2*module.dim:].fill_(.3)
    set_bfloat16_layout(model)
    return model.to('cuda').eval(),config


def observation(value=1,prompt='move',wrist=True):
    images={'observation.images.front':np.full((28,28,3),value,np.uint8)}
    if wrist:images['observation.images.wrist']=np.full((28,28,3),value+1,np.uint8)
    return dict(images=images,state=np.linspace(0,.2,7,dtype=np.float32),prompt=prompt)


@pytest.mark.parametrize('compiled',[False,True])
def test_pi05_capacity_eviction_recaptures_without_corrupting_owned_outputs(monkeypatch,compiled):
    from vvla_pi05.lite.pi05 import Pi05LitePolicy
    from vvla_pi05.lite.processor import Pi05LiteProcessor
    from vvla_pi05.backend import Pi05Engine
    model,config=make_model(monkeypatch,10)
    policy=Pi05LitePolicy(model,Pi05LiteProcessor(config,Tokenizer(),torch.device('cuda')),
        device='cuda',max_graphs=1,compile_sample_actions=compiled)
    engine=Pi05Engine(policy,max_batch=2)
    try:
        first=engine.infer_batch([observation()],[17])[0];owned=first.copy()
        engine.infer_batch([observation(),observation(7)],[17,29])
        last=engine.infer_batch([observation()],[17])[0]
        np.testing.assert_array_equal(first,owned)
        np.testing.assert_array_equal(last,owned)
        assert len(policy.graph_runner.entries)==1
        assert policy.graph_runner.stats['capacity_evictions']==2
    finally:engine.close()
    del engine,policy,model
    gc.collect();torch.cuda.empty_cache()


@pytest.mark.parametrize('horizon',[10,50])
def test_actual_pi05_eager_graph_true_batch_seed_lanes_and_lifecycle(monkeypatch,horizon):
    from vvla_pi05.lite.pi05 import Pi05LitePolicy
    from vvla_pi05.lite.processor import Pi05LiteProcessor
    from vvla_pi05.backend import Pi05Engine
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,config=make_model(monkeypatch,horizon)
    processor=Pi05LiteProcessor(config,Tokenizer(),torch.device('cuda'))
    first,second=observation(),observation(7,'move a block',False)
    eager=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=False)
    expected=[eager.infer(first,17)['actions'],eager.infer(second,29)['actions']]
    # close drops references but preserves the explicitly retained model.
    eager.close();del eager
    policy=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=True,max_graphs=2)
    engine=Pi05Engine(policy,max_batch=2)
    before=torch.cuda.get_rng_state().clone()
    actual=engine.infer_batch([first,second],[17,29])
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=.025,atol=.025)
    assert torch.equal(before,torch.cuda.get_rng_state())
    owned=[a.copy() for a in actual]
    reverse=engine.infer_batch([second,first],[29,17])
    for a,b in zip(reverse,reversed(actual)):np.testing.assert_allclose(a,b,rtol=.025,atol=.025)
    single=engine.infer_batch([first],[17])[0]
    np.testing.assert_allclose(single,expected[0],rtol=.025,atol=.025)
    for a,b in zip(actual,owned):np.testing.assert_array_equal(a,b)
    assert len(policy.graph_runner.entries)<=2
    record=dict(horizon=horizon,depth=2,vlm_width=2048,expert_width=1024,vision_width=1152,
                image_resolution=28,vocab=1024,denoising_steps=2,
                allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    artifact=Path('artifacts');artifact.mkdir(exist_ok=True)
    (artifact/f'tiny-pi05-h{horizon}.json').write_text(json.dumps(record,indent=2))
    engine.close();engine.close()
    assert policy._closed
    del engine,policy,processor,model
    gc.collect();torch.cuda.empty_cache()


def test_fifty_concurrent_requests_use_true_tiny_gpu_batches(monkeypatch):
    from vvla_pi05.lite.pi05 import Pi05LitePolicy
    from vvla_pi05.lite.processor import Pi05LiteProcessor
    from vvla_pi05.backend import Pi05Engine
    from vvla.serve.batching import DynamicBatcher, Request
    import time
    torch.manual_seed(24);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,config=make_model(monkeypatch,10)
    processor=Pi05LiteProcessor(config,Tokenizer(),torch.device('cuda'))
    policy=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=True,max_graphs=2)
    engine=Pi05Engine(policy,max_batch=2)
    class Executor:
        def infer_requests(self,rows):return engine.infer_batch([r.observation for r in rows],[r.seed for r in rows])
        def close(self):engine.close()
    async def run():
        batcher=DynamicBatcher(Executor(),supported=(1,2),max_batch=2,wait_ms=2,queue_capacity=64,event_capacity=32)
        await batcher.start();now=time.perf_counter()
        results=await asyncio.gather(*(batcher.submit(Request(str(i),observation(i%3),i%5,now,now+60)) for i in range(50)))
        assert len(results)==50 and batcher.total_completed==50 and len(batcher.events)<=32
        for i in range(15,50):
            np.testing.assert_allclose(results[i]['result'],results[i%15]['result'],rtol=.025,atol=.025)
        for row in results:assert row['result'].shape==(10,7) and np.isfinite(row['result']).all()
        keep=results[1]['result'].copy();results[0]['result'][:]=np.nan
        np.testing.assert_array_equal(results[1]['result'],keep)
        assert policy._calls==25
        await batcher.close();await batcher.close();assert policy._closed
        Path('artifacts/tiny-service-50.json').write_text(json.dumps(dict(requests=50,real_batch_size=2,
            model_calls=25,owned_outputs=True,deterministic_seed_per_lane=True,closed=True),indent=2))
    asyncio.run(run())
    del engine,policy,processor,model
    gc.collect();torch.cuda.empty_cache()


@pytest.mark.parametrize('horizon',[10,50])
@pytest.mark.parametrize('stages',[['action_attention'],['action_mlp'],['vision','prefix','action_attention','action_mlp']])
def test_actual_pi05_dynamic_fp8_stage_ablations(monkeypatch,horizon,stages):
    from vvla_pi05.lite.pi05 import Pi05LitePolicy
    from vvla_pi05.lite.processor import Pi05LiteProcessor
    from vvla_pi05.experimental.precision import install
    from vvla_pi05.backend import Pi05Engine
    torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,config=make_model(monkeypatch,horizon)
    processor=Pi05LiteProcessor(config,Tokenizer(),torch.device('cuda'))
    baseline=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=False)
    expected=baseline.infer(observation(),17)['actions'];baseline.close();del baseline
    receipt=install(model,stages)
    assert {site['stage'] for site in receipt['sites']}==set(stages) and not receipt['fallback']
    model.invalidate_inference_caches()
    eager=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=False)
    candidate=eager.infer(observation(),17)['actions'];eager.close();del eager
    policy=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=True,max_graphs=1,
                          provenance={'experimental_precision':receipt})
    engine=Pi05Engine(policy)
    try:
        actual=engine.infer_batch([observation()],[17])[0]
        np.testing.assert_array_equal(actual,candidate)
        np.testing.assert_array_equal(actual,engine.infer_batch([observation()],[17])[0])
        relative_rms=float(np.sqrt(np.mean((actual-expected)**2))/max(np.sqrt(np.mean(expected**2)),1e-6))
        # A gross mechanism threshold, set before these runs. It is not an
        # acceptable robot-policy quality threshold or a closed-loop gate.
        assert relative_rms<.15
        assert engine.metadata()['precision']=='dynamic-fp8-experimental'
        Path(f'artifacts/tiny-fp8-h{horizon}-'+('_'.join(stages))+'.json').write_text(json.dumps(
            dict(horizon=horizon,stages=stages,relative_rms=relative_rms,threshold=.15,
                 eager_graph_bitexact=True,receipt=receipt,quality_qualified=False),indent=2))
    finally:engine.close()
    del engine,policy,processor,model
    gc.collect();torch.cuda.empty_cache()


def test_actual_pi05_compiled_sampling_uses_initialized_libdevice(monkeypatch):
    from vvla_pi05.lite.pi05 import Pi05LitePolicy
    from vvla_pi05.lite.processor import Pi05LiteProcessor
    from vvla_pi05.backend import Pi05Engine
    torch.manual_seed(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,config=make_model(monkeypatch,10)
    processor=Pi05LiteProcessor(config,Tokenizer(),torch.device('cuda'))
    eager=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=False)
    expected=eager.infer(observation(),17)['actions'];eager.close();del eager
    policy=Pi05LitePolicy(model,processor,device='cuda',cuda_graph=True,max_graphs=1,compile_sample_actions=True)
    engine=Pi05Engine(policy)
    try:
        first=engine.infer_batch([observation()],[17])[0]
        np.testing.assert_allclose(first,expected,rtol=.025,atol=.025)
        np.testing.assert_array_equal(first,engine.infer_batch([observation()],[17])[0])
        receipt=engine.metadata()['lite_optimizations']['compilation_manifest']['libdevice_environment']
        assert receipt['precision_context_restored']
        Path('artifacts/tiny-compiled-libdevice.json').write_text(json.dumps(dict(receipt=receipt,
            mechanism_passed=True,full_model_speed_qualified=False),indent=2))
    finally:engine.close()
    del engine,policy,processor,model
    gc.collect();torch.cuda.empty_cache()
