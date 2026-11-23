"""Debug installed public LingBot loaders with real two-layer model math.

Only depth, vocabulary and auxiliary training heads are reduced. Vision/text/
action widths, MoE experts/top-k, H50 and ten sampler steps remain real. This
is not a trained policy or a timing/quality qualification. Run in its isolated
environment with a compatible FlashAttention wheel and native vLLM asset.
"""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np
import torch
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','qwen-config','training-config','norm-stats','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Reuse a checkpoint/qwen fixture made by this tool; still applies the two-layer test constructor')
    args=p.parse_args();work=args.output.resolve();work.mkdir(parents=True,exist_ok=False)
    site=Path(sysconfig.get_paths()['purelib']).resolve()
    for name in ('vvla','vvla_runtime','vvla_lingbot'):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(site)
    sys.path.insert(0,str(args.source.resolve()))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    qwen=(args.checkpoint.resolve().parent/'qwen3-vl-debug') if args.checkpoint else work/'qwen3-vl-debug'
    if args.checkpoint:
        if not qwen.is_dir() or not (args.checkpoint/'model.safetensors').is_file():
            raise ValueError('Expected this tool\'s synthetic checkpoint and adjacent qwen3-vl-debug asset')
    else:qwen.mkdir()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2TokenizerFast,Qwen2VLImageProcessor,Qwen3VLVideoProcessor,Qwen3VLProcessor,AutoConfig
    vocabulary={name:i for i,name in enumerate(('<|endoftext|>','<|im_start|>','<|im_end|>',
        '<|vision_start|>','<|vision_end|>','<|image_pad|>','<|video_pad|>',
        'move','the','block','open','drawer','user','assistant','<bos>'))}
    tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='<|endoftext|>'));tokenizer.pre_tokenizer=Whitespace()
    tokenizer=Qwen2TokenizerFast(tokenizer_object=tokenizer,additional_special_tokens=list(vocabulary)[:7])
    template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.chat_template=template
    processor=Qwen3VLProcessor(tokenizer=tokenizer,chat_template=template,
        image_processor=Qwen2VLImageProcessor(patch_size=16,temporal_patch_size=2,merge_size=2,
            min_pixels=224*224,max_pixels=224*224),video_processor=Qwen3VLVideoProcessor())
    if not args.checkpoint:processor.save_pretrained(qwen)
    config=json.loads(args.qwen_config.read_text())
    config['text_config'].update(num_hidden_layers=2,vocab_size=1024,bos_token_id=1,eos_token_id=2)
    config['vision_config'].update(depth=2,deepstack_visual_indexes=[0,1])
    config.update(image_token_id=5,video_token_id=6,vision_start_token_id=3,vision_end_token_id=4)
    if not args.checkpoint:(qwen/'config.json').write_text(json.dumps(config,indent=2))
    os.environ['QWEN3VL_PATH']=str(qwen)
    import deploy.lingbot_vla_v2_policy as upstream
    from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla_v2 as modeling
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    # Test-only constructor reduction, like the Pi0.5 depth fixture. Runtime
    # production code, mathematical functions and official assets stay intact.
    original=modeling.QwenvlWithExpertV2Config.__init__
    def reduced(self,*a,**kw):
        original(self,*a,**kw)
        self.qwen_expert_config.num_hidden_layers=2
        self.qwen_expert_config.vocab_size=1024
    modeling.QwenvlWithExpertV2Config.__init__=reduced
    training=yaml.safe_load(args.training_config.read_text())
    training['model'].update(tokenizer_path=str(qwen),vocab_size=1024)
    training['train'].update(token_moe_layers=[0,1],align_params={})
    checkpoint=args.checkpoint.resolve() if args.checkpoint else work/'flat-checkpoint'
    if not args.checkpoint:
        checkpoint.mkdir()
        (checkpoint/'lingbotvla_cli.yaml').write_text(yaml.safe_dump(training))
    selected=LingbotVLAV2Config(**dict(training['model'],**training['train']))
    temporary=upstream.LingbotVLAv2Server.__new__(upstream.LingbotVLAv2Server)
    temporary.config=selected;temporary.merge_qwen_config(AutoConfig.from_pretrained(qwen))
    selected.tokenizer_path=str(qwen);selected.vocab_size=1024
    selected.use_cache=True;selected.attention_implementation='eager'
    torch.manual_seed(51);torch.set_num_threads(4)
    if not args.checkpoint:
        model=upstream.LingBotVlaV2InferencePolicy(selected,eval=True).float().eval()
        from safetensors.torch import save_file
        save_file({k:v.detach().contiguous().clone() for k,v in model.state_dict().items()},str(checkpoint/'model.safetensors'))
        del model
    del temporary
    gc.collect();torch.cuda.empty_cache()
    from vvla_lingbot.backend import load_backend
    from vvla_lingbot.lite import CAMERAS
    observation={'task':'move the block','observation.state':np.zeros(14,np.float32),
        **{name:np.full((256,256,3),64+30*i,np.uint8) for i,name in enumerate(CAMERAS)}}
    other=dict(observation,task='open drawer')
    assets=dict(source=str(args.source.resolve()),qwen=str(qwen),norm_stats=str(args.norm_stats.resolve()),
                native=str(args.native.resolve()),training_config=str(checkpoint/'lingbotvla_cli.yaml'))
    results={}
    def replay_difference(left,right):
        # The historical sparse MoE uses FP32 atomic_add in its final reduce.
        # Explicit seeds guarantee identical noise, not associative GPU sums.
        # This is the existing 0.15 gross mechanism bound, not a policy gate.
        delta=left.astype(np.float64)-right.astype(np.float64)
        return dict(bitexact=bool(np.array_equal(left,right)),max_abs=float(np.max(np.abs(delta))),
                    relative_rms=float(np.sqrt(np.mean(delta**2))/max(1e-8,float(np.sqrt(np.mean(left.astype(np.float64)**2))))))
    reference=None
    for recipe in ('official-bf16','bf16','vlm-fp8','true-batch-bf16','auto','split-stage'):
        from vvla.dispatch import prepare_config
        config, route = prepare_config(dict(model='lingbot',recipe=recipe,mode='serve',
                                     checkpoint=str(checkpoint),assets=assets,device='cuda:0'))
        engine=load_backend(config)
        try:
            first=engine.infer_batch([observation],[17])[0];owned=first.copy()
            policy=getattr(engine,'_LingBotEngine__policy',None)
            noise=policy.initial_noise().copy() if policy is not None else None
            replay=engine.infer_batch([observation],[17])[0]
            replay_stats=replay_difference(first,replay)
            if noise is not None:np.testing.assert_array_equal(noise,policy.initial_noise())
            np.savez(work/(recipe+'-replay.npz'),first=first,replay=replay,
                     initial_noise=noise if noise is not None else np.empty(0,np.uint16))
            assert replay_stats['relative_rms']<.15,replay_stats
            third=engine.infer_batch([observation],[17])[0]
            warm_stats=replay_difference(replay,third)
            assert warm_stats['relative_rms']<.15,warm_stats
            assert first.shape==(50,14) and np.isfinite(first).all()
            changed=engine.infer_batch([other],[17])[0]
            assert not np.array_equal(first,changed),'Synthetic instruction did not exercise the model'
            if recipe in ('true-batch-bf16','split-stage'):
                batch=engine.infer_batch([observation,other],[17,29])
                assert all(v.shape==(50,14) and np.isfinite(v).all() for v in batch)
                again=engine.infer_batch([observation,other],[17,29])
                batch_stats=[replay_difference(a,b) for a,b in zip(batch,again)]
                assert all(v['relative_rms']<.15 for v in batch_stats),batch_stats
                row1=engine.infer_batch([other],[29])[0]
                row_stats=[replay_difference(a,b) for a,b in zip([first,row1],batch)]
                assert all(v['relative_rms']<.15 for v in row_stats),row_stats
            if recipe == 'auto':
                assert route['selected_recipe'] == 'cache32-posttopk-fp16-prefix-expert'
                assert engine.capabilities()['batch_sizes'] == [1]
            if recipe == 'split-stage':
                engine.infer_batch([observation, other, observation, other], [17, 29, 30, 31])
                engine.infer_batch([other, observation, other, observation], [29, 17, 31, 30])
                receipt = engine.execution_receipt()
                assert receipt['actual_tensor_batch'] == 4 and receipt['graph_replay'] and not receipt['graph_build']
            np.testing.assert_array_equal(first,owned)
            metadata=engine.metadata()
            if recipe=='vlm-fp8':
                assert metadata['public_recipe']['mode']=='framework-fp8'
                assert not metadata['public_recipe']['options']['sealed_guard']
                assert len(metadata['recipe']['quantized_linears'])==6
            if recipe=='official-bf16':reference=first.copy()
            paired=replay_difference(reference,first)
            assert paired['relative_rms']<.15,paired
            results[recipe]=dict(metadata=metadata,replay=replay_stats,
                                warm_replay=warm_stats,versus_official=paired,
                                initial_noise_bitexact=True if noise is not None else None,
                                instruction_sensitive=True,
                                batch_replay=batch_stats if recipe in ('true-batch-bf16','split-stage') else None,
                                batch_vs_b1=row_stats if recipe in ('true-batch-bf16','split-stage') else None)
        finally:engine.close();engine.close()
        del engine
        gc.collect();torch.cuda.empty_cache()
        (work/'result.json').write_text(json.dumps(dict(status='incomplete',results=results,
            depth=2,vision_width=1024,text_width=2560,expert_width=768,vocab_size=1024,
            synthetic_weights=True,training_alignment_heads_omitted=True,quality_qualified=False),indent=2))
        print(recipe,'passed',flush=True)
    record=json.loads((work/'result.json').read_text())
    record.update(status='passed',installed_backend=True,real_gpu=True,
                  checkpoint=str(checkpoint),qwen=str(qwen),device=torch.cuda.get_device_name(0),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (work/'result.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
