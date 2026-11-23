"""Exercise checkpoint -> public loader -> historical stack -> real GPU.

Only model depth and vocabulary are reduced. Widths, image resolution, action
geometry, preprocessing, strict weight loading, compilation and graph execution
are the production implementations. Synthetic weights do not qualify a policy.
"""
import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required')]


def _checkpoint(monkeypatch, path, *, openpi=False):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from vvla_pi05.models.pi05 import modeling_pi05 as core
    from vvla_pi05.models.pi05.config import Pi05Config
    from vvla_pi05.lite.dtype import set_bfloat16_layout
    original_geometry = core.get_gemma_config
    original_vlm = core.PaliGemmaForConditionalGeneration
    original_expert = core.GemmaForCausalLM

    def geometry(name):
        value = original_geometry(name)
        value.depth = 2
        return value

    def vlm(config):
        config.text_config.vocab_size = 1024
        config.vision_config.num_hidden_layers = 2
        return original_vlm(config)

    def expert(config):
        config.vocab_size = 1024
        return original_expert(config)

    monkeypatch.setattr(core, 'get_gemma_config', geometry)
    monkeypatch.setattr(core, 'PaliGemmaForConditionalGeneration', vlm)
    monkeypatch.setattr(core, 'GemmaForCausalLM', expert)
    horizon=10 if openpi else 50
    config = Pi05Config(chunk_size=horizon, n_action_steps=horizon, num_inference_steps=10,
        image_resolution=(224, 224), max_cameras=3, tokenizer_max_length=200,
        input_features={'observation.state': {'shape': [7]},
                        'observation.images.front': {'shape': [3, 224, 224]},
                        'observation.images.wrist': {'shape': [3, 224, 224]}},
        output_features={'action': {'shape': [7]}},
        norm_stats={key: {'mode': 'mean_std', 'mean': [0]*7, 'std': [1]*7}
                    for key in ('state', 'action')})
    if openpi:
        from vvla_pi05.openpi import CAMERAS
        config=Pi05Config(chunk_size=10,n_action_steps=10,num_inference_steps=10,
            input_contract='openpi-libero-pi05',discrete_state_input=False,use_relative_actions=False,
            image_resolution=(224,224),max_cameras=3,tokenizer_max_length=200,
            image_feature_keys=list(CAMERAS),
            input_features={'observation.state':{'shape':[8]}},output_features={'action':{'shape':[7]}},
            norm_stats={key:dict(q01=[-1]*width,q99=[1]*width) for key,width in [('state',8),('action',7)]})
    torch.manual_seed(7)
    model = core.Pi05ForActionPrediction(config)
    # Default AdaRMS gates are zero: leaving them closed would make expert
    # FP8 errors invisible at the action output. Exercise actual projections.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,core.Pi05AdaRMSNorm) and module.dense is not None:
                module.dense.weight.normal_(0,.01)
                module.dense.bias.zero_()
                module.dense.bias[2*module.dim:].fill_(.3)
    set_bfloat16_layout(model)
    save_file({name: value.detach().contiguous().clone()
               for name, value in model.state_dict().items()}, str(path/'model.safetensors'))
    (path/'config.json').write_text(json.dumps(asdict(config)))
    del model
    gc.collect()
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'move': 2, 'stop': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
                           unk_token='[UNK]').save_pretrained(path)
    if openpi:
        import sentencepiece as sp
        corpus=path/'corpus.txt';corpus.write_text('move the block\nstop and open drawer\n')
        sp.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(path/'sentencepiece'),
            vocab_size=32,hard_vocab_limit=False,minloglevel=2)


@pytest.mark.parametrize('recipe', ['bf16-sm89', 'prefix-fp8-down-sm89', 'bf16-batch-sm89', 'bf16-portable', 'dynamic-fp8'])
def test_public_checkpoint_loader_runs_preserved_and_portable_stacks(monkeypatch, tmp_path, recipe):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip('Historical SM89 recipe')
    from vvla_pi05.backend import load_backend
    _checkpoint(monkeypatch, tmp_path)
    torch.cuda.reset_peak_memory_stats()
    options={'max_batch':2} if recipe in ('bf16-batch-sm89','bf16-portable') else {}
    if recipe=='dynamic-fp8':
        options['precision_stages']=['vision','prefix','action_attention','action_mlp']
    engine = load_backend(dict(recipe=recipe, checkpoint=str(tmp_path), device='cuda:0',options=options))
    observation = dict(images={
        'observation.images.front': np.full((224, 224, 3), 20, np.uint8),
        'observation.images.wrist': np.full((224, 224, 3), 80, np.uint8)},
        state=np.linspace(0, .2, 7, dtype=np.float32), prompt='move')
    try:
        before = torch.cuda.get_rng_state().clone()
        first = engine.infer_batch([observation], [17])[0]
        replay = engine.infer_batch([observation], [17])[0]
        np.testing.assert_array_equal(first, replay)
        assert np.isfinite(first).all() and first.shape == (50, 7)
        assert torch.equal(before, torch.cuda.get_rng_state())
        changed = dict(observation, prompt='stop')
        changed['images'] = {key: np.full_like(value, 190) for key, value in observation['images'].items()}
        other = engine.infer_batch([changed], [29])[0]
        input_only=engine.infer_batch([changed],[17])[0]
        assert not np.array_equal(first,input_only), 'Closed synthetic gates hid observation-dependent math'
        assert not np.array_equal(first, other)
        np.testing.assert_array_equal(first, replay)
        metadata = engine.metadata()
        assert metadata['public_recipe']['resolved'] == ('bf16-sm89' if recipe=='dynamic-fp8' else recipe)
        assert metadata['resolved_config']['vision_bf16']
        assert metadata['lite_optimizations']['fused_expert_qkv']
        assert metadata['lite_optimizations']['compile_sample_actions']
        if recipe in ('bf16-batch-sm89','bf16-portable'):
            assert metadata['lite_optimizations']['expert_cublaslt'] is None
            assert metadata['capabilities']['batch_sizes']==[1,2]
            mixed=engine.infer_batch([observation,changed],[17,29])
            np.testing.assert_allclose(mixed[0],first,rtol=.04,atol=.04)
            np.testing.assert_allclose(mixed[1],other,rtol=.04,atol=.04)
            reversed_rows=engine.infer_batch([changed,observation],[29,17])
            for actual,expected in zip(reversed_rows,reversed(mixed)):
                np.testing.assert_allclose(actual,expected,rtol=.04,atol=.04)
        else:
            assert metadata['lite_optimizations']['expert_cublaslt']['include_down']
        if recipe.startswith('prefix'):
            assert metadata['precision'] == 'mixed_bf16_fp8_prefix'
            assert len(metadata['quantization']['sites']) == 1  # Last layer is KV-only.
        if recipe=='dynamic-fp8':
            receipt=metadata['experimental_precision']
            assert {site['stage'] for site in receipt['sites']}==set(options['precision_stages'])
            assert sum(s['name'].endswith('packed_qkv') for s in receipt['sites'])==2
            assert sum(s['name'].endswith('packed_gate_up') for s in receipt['sites'])==2
            assert not receipt['fallback']
            # Compare the same synthetic checkpoint/input/seed against the
            # complete optimized BF16 recipe, not an unfused reference path.
            engine.close()
            baseline=load_backend(dict(recipe='bf16-sm89',checkpoint=str(tmp_path),device='cuda:0'))
            try:expected=baseline.infer_batch([observation],[17])[0]
            finally:baseline.close()
            relative_rms=float(np.sqrt(np.mean((first-expected)**2))/max(np.sqrt(np.mean(expected**2)),1e-6))
            assert 0<relative_rms<.15  # Gross mechanism check; not robot-policy qualification.
            metadata['debug_comparison']=dict(optimized_bf16_relative_rms=relative_rms,threshold=.15)
        Path(f'artifacts/performance-alignment/public-{recipe}.json').write_text(json.dumps(dict(
            depth=2, vocabulary=1024, production_widths=True, image_resolution=224,nonzero_expert_gates=True,
            strict_checkpoint_load=True, replay_bitexact=True, output_shape=list(first.shape),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), metadata=metadata), indent=2))
    finally:
        engine.close()
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def test_openpi_h10_public_optimized_checkpoint_and_real_batch(monkeypatch,tmp_path):
    from vvla_pi05.backend import load_backend
    from vvla_pi05.openpi import CAMERAS
    _checkpoint(monkeypatch,tmp_path,openpi=True)
    engine=load_backend(dict(recipe='bf16',checkpoint=str(tmp_path),device='cuda:0',
        assets={'tokenizer':str(tmp_path/'sentencepiece.model')},options={'max_batch':2,'max_graphs':2}))
    observation=dict(state=np.zeros(8,np.float32),prompt='move the block',images={
        CAMERAS[0]:np.full((224,224,3),64,np.uint8),CAMERAS[1]:np.full((224,224,3),127,np.uint8)})
    other=dict(observation,prompt='open drawer')
    try:
        single=engine.infer_batch([observation],[17])[0]
        batch=engine.infer_batch([observation,other],[17,29])
        assert single.shape==(10,7) and all(np.isfinite(x).all() for x in batch)
        np.testing.assert_allclose(batch[0],single,rtol=.04,atol=.04)
        again=engine.infer_batch([observation,other],[17,29])
        for a,b in zip(batch,again):np.testing.assert_array_equal(a,b)
        metadata=engine.metadata()
        assert metadata['public_recipe']['resolved']=='bf16-portable'
        assert not metadata['lite_optimizations']['action_norm_cache']
        Path('artifacts/performance-alignment/public-openpi-h10.json').write_text(json.dumps(dict(
            depth=2,production_widths=True,strict_checkpoint_load=True,batch_sizes=[1,2],
            replay_bitexact=True,metadata=metadata),indent=2))
    finally:engine.close()
    del engine
    gc.collect();torch.cuda.empty_cache()
