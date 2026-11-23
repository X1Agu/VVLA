"""Prepare an explicit VVLA config for an official OpenPI Torch H10 export.

This does not convert JAX weights. It keeps the original safetensors bytes and
links them into a new asset directory. Strict runtime tensor loading still
decides model compatibility; this preparation step is not a parity certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','norm-stats','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--openpi-commit',required=True)
    args=p.parse_args()
    if len(args.openpi_commit)!=40 or any(c not in '0123456789abcdef' for c in args.openpi_commit):
        raise ValueError('Record the exact reviewed OpenPI commit')
    weights,norm,tokenizer=[path.resolve(strict=True) for path in (args.weights,args.norm_stats,args.tokenizer)]
    if weights.suffix!='.safetensors':raise ValueError('Use the official Torch safetensors export')
    raw=json.loads(norm.read_text());raw=raw.get('norm_stats',raw)
    stats={}
    for target,key,width in [('state','state',8),('action','actions',7)]:
        record=raw[key]
        stats[target]={name:record[name][:width] for name in ('q01','q99')}
        if any(len(stats[target][name])!=width or any(not math.isfinite(v) for v in stats[target][name]) for name in ('q01','q99')):
            raise ValueError('Normalization arrays must contain the complete finite physical dimensions')
        if any(a>b for a,b in zip(stats[target]['q01'],stats[target]['q99'])):
            raise ValueError('Normalization q01 cannot exceed q99')
        stats[target]['mode']='quantiles'
    config=dict(input_contract='openpi-libero-pi05',discrete_state_input=False,
        chunk_size=10,n_action_steps=10,num_inference_steps=10,max_action_dim=32,max_state_dim=32,
        image_resolution=[224,224],max_cameras=3,image_feature_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'],
        tokenizer_max_length=200,use_relative_actions=False,input_features={'observation.state':{'shape':[8]}},
        output_features={'action':{'shape':[7]}},norm_stats=stats)
    from vvla_pi05.models.pi05.config import Pi05Config
    Pi05Config(**config)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'model.safetensors').symlink_to(weights)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'vvla.json').write_text(json.dumps(dict(schema=1,assets={'tokenizer':str(tokenizer)}),indent=2)+'\n')
    (args.output/'OPENPI_ORIGIN.json').write_text(json.dumps(dict(openpi_commit=args.openpi_commit,
        source_weights=str(weights),weights_sha256=sha(weights),norm_stats_sha256=sha(norm),tokenizer_sha256=sha(tokenizer),
        weight_transformation='none; linked original Torch export',qualification=False),indent=2)+'\n')

if __name__=='__main__':main()
