"""Load pinned upstream inference definitions without its training package imports.

Namespace packages bypass only __init__ training imports, never model functions.
Selected preprocessing functions are compiled verbatim from the official AST;
source digests and extracted function names are recorded for reproducibility.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import typing

REPOSITORY_REVISION='e4287e94541f459edc4feabc4e181f537cd569a8'
TRANSFORMERS_REVISION='bc339d9ad707454c0c115970db43c260067c61ab'
CHECKPOINT_REVISION='638918f3d1c2e43a39a8a20772bdb8b91835e4b7'

def configure_upstream(root):
    root=Path(root)
    if not (root/'prismatic/extern/hf/modeling_prismatic.py').is_file():
        raise ValueError('OFT source asset lacks prismatic model definitions')
    for package in ['prismatic','prismatic.extern','prismatic.extern.hf','prismatic.models','prismatic.training','prismatic.vla']:
        if package not in sys.modules:
            module=types.ModuleType(package);module.__path__=[str(root.joinpath(*package.split('.')))];sys.modules[package]=module
    sys.path.insert(0,str(root))

def upstream_preprocessing(root,device='cuda:0'):
    import numpy as np
    import torch
    import tensorflow as tf
    from PIL import Image
    from prismatic.vla.constants import NormalizationType,ACTION_PROPRIO_NORMALIZATION_TYPE
    tf.config.set_visible_devices([], 'GPU')
    source=Path(root)/'experiments/robot/openvla_utils.py'
    wanted={'resize_image_for_policy','crop_and_resize','center_crop_image','check_image_format',
            'normalize_proprio','prepare_images_for_vla','get_vla_action'}
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(nodes)==len(wanted)
    namespace=dict(vars(typing),np=np,torch=torch,tf=tf,Image=Image,OPENVLA_IMAGE_SIZE=224,
                   DEVICE=torch.device(device),NormalizationType=NormalizationType,
                   ACTION_PROPRIO_NORMALIZATION_TYPE=ACTION_PROPRIO_NORMALIZATION_TYPE)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return types.SimpleNamespace(**{k:namespace[k] for k in wanted}),dict(source=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),functions=sorted(wanted))

def load_reference(root,checkpoint,device='cuda',*,resource_publisher=None):
    # Optional cold ownership handoff. Publish each returned allocation before
    # a subsequent transfer/loading step can fail; legacy callers are unchanged.
    if resource_publisher is not None and not callable(resource_publisher):
        raise TypeError('resource_publisher must be callable or None')
    def publish(name,value):
        if resource_publisher is not None:resource_publisher(name,value)
    configure_upstream(root)
    import torch
    import transformers
    fork=Path(os.environ.get('VVLA_OFT_TRANSFORMERS_SOURCE',Path(root).parent/'transformers-openvla-oft'))
    from transformers.models.llama import modeling_llama
    if Path(modeling_llama.__file__).read_bytes()!=(fork/'src/transformers/models/llama/modeling_llama.py').read_bytes():
        raise ValueError('Installed attention code differs from the official bidirectional fork')
    from transformers import AutoTokenizer
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor,PrismaticImageProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM
    assert (NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM)==(8,7,8)
    checkpoint=Path(checkpoint)
    cfg=OpenVLAConfig.from_pretrained(checkpoint)
    cfg._attn_implementation='sdpa'
    model,loading_info=OpenVLAForActionPrediction.from_pretrained(checkpoint,config=cfg,torch_dtype=torch.bfloat16,
                                                   attn_implementation='sdpa',low_cpu_mem_usage=True,output_loading_info=True)
    publish('model',model)
    loading_info['unmaterialized_keys']=[name for name,tensor in model.state_dict().items() if tensor.is_meta]
    if any(loading_info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','unmaterialized_keys')):
        raise ValueError('OFT checkpoint does not exactly match the supported model: '+repr(loading_info))
    model=model.to(device).eval()
    assert model.language_model.config._attn_implementation=='sdpa'
    assert all(type(layer.self_attn).__name__=='LlamaSdpaAttention' for layer in model.language_model.model.layers)
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats=json.loads((checkpoint/'dataset_statistics.json').read_text())
    dim=model.language_model.config.hidden_size
    head=L1RegressionActionHead(input_dim=dim,hidden_dim=dim,action_dim=7)
    publish('head',head)
    head=head.to(device=device,dtype=torch.bfloat16).eval()
    proprio=ProprioProjector(llm_dim=dim,proprio_dim=8)
    publish('proprio',proprio)
    proprio=proprio.to(device=device,dtype=torch.bfloat16).eval()
    for module,name in [(head,'action_head'),(proprio,'proprio_projector')]:
        files=list(checkpoint.glob(name+'--*_checkpoint.pt'));assert len(files)==1
        state=torch.load(files[0],weights_only=True,map_location='cpu')
        module.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    processor=PrismaticProcessor(PrismaticImageProcessor.from_pretrained(checkpoint),
                                AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=False))
    publish('processor',processor)
    utils,provenance=upstream_preprocessing(root,device=device)
    publish('utils',utils)
    return model,head,proprio,processor,utils,provenance
