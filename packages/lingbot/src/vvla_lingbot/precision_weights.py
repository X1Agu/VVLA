"""Restore original FP32 tensors at explicitly selected action-network boundaries."""
from collections import defaultdict
import hashlib
import json

import torch
from safetensors import safe_open


HEADS = ("state_proj", "action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj")


def restore_action_precision(flow, checkpoint, *, half_expert=False, half_text=False):
    layers = flow.qwenvl_with_expert.qwen_expert.model.layers
    targets = {}

    def take(path, module, names):
        for name in names:
            targets["model." + path + "." + name] = (module, name)

    for name in HEADS:
        take(name, getattr(flow, name), ("weight", "bias"))
    for i, layer in enumerate(layers):
        path = "qwenvl_with_expert.qwen_expert.model.layers." + str(i) + "."
        take(path + "mlp.gate", layer.mlp.gate, ("weight",))
        take(path + "mlp", layer.mlp, ("e_score_correction_bias",))
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name)
            take(path + name, norm, ("weight",))
            for projection in ("gamma", "beta"):
                take(path + name + "." + projection, getattr(norm, projection), ("weight", "bias"))

    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        single = checkpoint / "model.safetensors"
        with safe_open(str(single), framework="pt", device="cpu") as reader:
            index = dict.fromkeys(reader.keys(), single.name)
    half_keys = set()
    if half_expert:
        prefix = 'model.qwenvl_with_expert.qwen_expert.model.'
        for key in index:
            if not key.startswith(prefix) or key in targets:
                continue
            path, name = key.removeprefix('model.').rsplit('.', 1)
            targets[key] = (flow.get_submodule(path), name)
            if '.self_attn.' in key or '.mlp.' in key:
                half_keys.add(key)
            elif key != prefix + 'norm.weight':
                raise ValueError('Unclassified original action-expert tensor ' + key)
    text_targets = set()
    if half_text:
        if not half_expert:
            raise ValueError('FP16 text is supported only with the FP16 action expert')
        text = 'model.qwenvl_with_expert.qwenvl.model.language_model.'
        for key in index:
            if not key.startswith(text+'layers.') and key != text+'norm.weight':
                continue
            path, name = key.removeprefix('model.').rsplit('.', 1)
            targets[key] = (flow.get_submodule(path), name)
            text_targets.add(key)
            if key.endswith(('_proj.weight', '_proj.bias')):
                half_keys.add(key)
            elif not key.endswith('norm.weight'):
                raise ValueError('Unclassified original text tensor ' + key)
        count = len(flow.qwenvl_with_expert.qwenvl.model.language_model.layers)
        if not text_targets or len(text_targets & half_keys) < 7*count:
            raise ValueError('Incomplete original text projections')
    shards = defaultdict(list)
    for key in targets:
        shards[index[key]].append(key)
    manifest = {}
    for shard, keys in sorted(shards.items()):
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as reader:
            for key in sorted(keys):
                module, name = targets[key]
                old = getattr(module, name)
                source = reader.get_tensor(key)
                if source.dtype != torch.float32 or source.shape != old.shape:
                    raise ValueError("Unexpected original precision or shape of " + key)
                dtype = torch.float16 if key in half_keys else torch.float32
                value = source.to(device=old.device, dtype=dtype)
                if dtype == torch.float16 and not torch.isfinite(value).all().item():
                    raise ValueError('FP16 overflow in trained weights: ' + key)
                if name in module._parameters:
                    value = torch.nn.Parameter(value, requires_grad=False)
                elif name not in module._buffers:
                    raise ValueError("Unregistered checkpoint tensor " + key)
                setattr(module, name, value)
                manifest[key] = {"numel": source.numel(), "sha256": hashlib.sha256(memoryview(source.numpy()).cast('B')).hexdigest()}
    receipt = {"tensors": len(manifest), "fp16_tensors": len(half_keys),
            "fp16_elements": sum(manifest[k]["numel"] for k in half_keys), "elements": sum(r["numel"] for r in manifest.values()),
            "tensor_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "head_modules": list(HEADS), "conditioned_norms": 2 * len(layers), "routers": len(layers),
            "weight_source": "original F32 safetensors before graph construction"}
    if half_text:
        receipt.update(text_tensors=len(text_targets), text_fp16_tensors=len(text_targets & half_keys),
                       text_fp16_elements=sum(manifest[k]['numel'] for k in text_targets & half_keys))
    return receipt
