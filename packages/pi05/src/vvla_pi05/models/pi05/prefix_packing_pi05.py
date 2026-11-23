# SPDX-License-Identifier: Apache-2.0
"""Per-row compaction of masked tokens for pi05's noncausal prefix.

Valid tokens keep their original order and position IDs. Masked query outputs
are unobserved: masked positions are excluded as keys at every prefix layer
and by the action expert. No KV is reused across observations.
Build/update outside CUDA capture; gather inside the captured computation.
"""

import torch


def prepare_prefix_packing(image_masks, lang_masks, image_tokens, bucket_size=64):
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size < 1:
        raise ValueError("prefix_pack_bucket_size must be a positive integer")
    pads = torch.cat([m[:, None].expand(-1, image_tokens) for m in image_masks] + [lang_masks], dim=1)
    if pads.dtype != torch.bool:
        raise ValueError("Prefix packing requires boolean masks")
    counts = pads.sum(1)
    low, high = counts.aminmax()
    if int(low) < 1:
        raise ValueError("Each request needs at least one visible prefix token")
    capacity = min(pads.shape[1], ((int(high) + bucket_size - 1) // bucket_size) * bucket_size)
    indices = torch.argsort((~pads).to(torch.int32), dim=1, stable=True)[:, :capacity].contiguous()
    compact_pads = torch.arange(capacity, device=pads.device)[None, :] < counts[:, None]
    positions = (pads.cumsum(1) - 1).gather(1, indices)
    return {"indices": indices, "pads": compact_pads, "positions": positions}


def pack_prefix(embeddings, packing):
    indices = packing["indices"][:, :, None].expand(-1, -1, embeddings.shape[-1])
    packed = embeddings.gather(1, indices)
    # The tail is still masked as keys; avoid carrying irrelevant values/NaNs.
    return torch.where(packing["pads"][:, :, None], packed, 0)


def copy_prefix_packing(target, source):
    if target is None or source is None:
        if target is not source:
            raise ValueError("Prefix packing changed without a new graph")
        return
    if target.keys() != source.keys():
        raise ValueError("Prefix packing fields changed")
    for name, dst in target.items():
        src = source[name]
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise ValueError("Prefix packing geometry changed without a new graph")
        dst.copy_(src)
