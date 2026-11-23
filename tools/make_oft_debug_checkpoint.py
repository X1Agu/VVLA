"""Random-weight OFT fixture: real DINO+SigLIP, a small official Llama, and L1 head.

This is a framework/GPU debug fixture, not a trained policy or benchmark model.
Run with the isolated OFT Python after installing its worker package.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vvla_oft.reference import configure_upstream, REPOSITORY_REVISION, TRANSFORMERS_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_upstream(args.source)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    torch.manual_seed(20261030)
    torch.set_num_threads(4)
    stats = {
        "debug": {
            name: {
                "q01": [-1.0] * dim,
                "q99": [1.0] * dim,
                "min": [-1.0] * dim,
                "max": [1.0] * dim,
                "mask": [True] * dim,
            }
            for name, dim in [("action", 7), ("proprio", 8)]
        }
    }
    config = OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px",
        llm_backbone_id="llama2-7b-pure",
        image_resize_strategy="resize-naive",
        norm_stats=stats,
        text_config=dict(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=32064,
            max_position_embeddings=2048,
            pad_token_id=32000,
        ),
    )
    config._attn_implementation = "sdpa"
    model = OpenVLAForActionPrediction(config).to(dtype=torch.bfloat16).eval()
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    head = L1RegressionActionHead(input_dim=256, hidden_dim=256, action_dim=7).to(torch.bfloat16)
    proprio = ProprioProjector(llm_dim=256, proprio_dim=8).to(torch.bfloat16)
    torch.save(head.state_dict(), args.output / "action_head--debug_checkpoint.pt")
    torch.save(proprio.state_dict(), args.output / "proprio_projector--debug_checkpoint.pt")
    (args.output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    vocab = {f"token_{i}": i for i in range(32064)}
    for token, idx in [("<unk>", 0), ("<s>", 1), ("</s>", 2), ("<stop>", 29871), ("<pad>", 32000)]:
        del vocab[f"token_{idx}"]
        vocab[token] = idx
    for idx, token in enumerate(
        [
            "In",
            "What",
            "action",
            "should",
            "the",
            "robot",
            "take",
            "to",
            "pick",
            "red",
            "blue",
            "block",
            "Out",
            ":",
            "?",
        ],
        start=10,
    ):
        del vocab[f"token_{idx}"]
        vocab[token] = idx
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=128,
    ).save_pretrained(args.output)
    PrismaticImageProcessor(
        use_fused_vision_backbone=True,
        image_resize_strategy="resize-naive",
        input_sizes=[(3, 224, 224)] * 2,
        interpolations=["bicubic"] * 2,
        means=[(0.485, 0.456, 0.406), (0.5, 0.5, 0.5)],
        stds=[(0.229, 0.224, 0.225), (0.5, 0.5, 0.5)],
    ).save_pretrained(args.output)
    checksums = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                checksums[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(
        purpose="synthetic_debug_only",
        seed=20261030,
        source_revision=REPOSITORY_REVISION,
        transformers_revision=TRANSFORMERS_REVISION,
        vision="full DINOv2-L + SigLIP-SO400M",
        text="official bidirectional Llama, 2 layers, width 256",
        action_geometry=[8, 7],
        parameters=sum(p.numel() for p in model.parameters()),
        files=checksums,
    )
    (args.output / "DEBUG_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
