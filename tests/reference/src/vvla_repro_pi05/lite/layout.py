"""Processor-owned CPU masks and bounded, immutable-by-contract layout templates.

Only deterministic mask/position metadata is cached. Captured graphs receive
independent tensor copies; observations, token values, noise and prefix KV are
never cached here. Parameter lifecycle checks remain in the shared model.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch

from vvla_repro_pi05.lite.attention import AttentionMetadata, annotate_identity_gathers
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
from vvla_repro_pi05.models.pi05.prefix_packing_pi05 import prepare_prefix_packing


def copy_layout_tree(value, *, device=None):
    """Fresh storage even for same-device copies; no graph/template aliasing."""
    if isinstance(value, torch.Tensor):
        return value.clone() if device is None else value.to(device=device, copy=True)
    if isinstance(value, AttentionMetadata):
        return AttentionMetadata(copy_layout_tree(value.extra, device=device))
    if isinstance(value, dict):
        return {key: copy_layout_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(copy_layout_tree(item, device=device) for item in value)
    if value is None or type(value) in (int, bool, str, float):
        return value
    raise TypeError(f"Unsupported cached layout field: {type(value)}")


@dataclass(frozen=True)
class HostMaskLayout:
    """CPU values emitted alongside the exact tensors by Pi05LiteProcessor.

    This is an internal trusted-processor contract, not user-supplied metadata.
    Bindings reject accidental reuse with another prepared request. Inference
    tensors may lack version counters; callers must not mutate the paired inputs
    between processor construction and the serialized graph invocation.
    """

    cameras: tuple
    tokens: tuple
    bindings: tuple

    @classmethod
    def from_processor(cls, cameras, tokens, images, image_masks, lang_masks):
        result = cls(tuple(bool(x) for x in cameras), tuple(bool(x) for x in tokens),
                     tuple(tensor_identity(x) for x in (*images, *image_masks, lang_masks)))
        result.validate(images, image_masks, lang_masks)
        return result

    @property
    def mask_bytes(self):
        # Exactly the old B1 torch.bool cat(...).cpu().numpy().tobytes() bytes.
        return bytes((*self.cameras, *self.tokens))

    def validate(self, images, image_masks, lang_masks):
        if (len(images) != len(self.cameras) or len(image_masks) != len(self.cameras)
                or lang_masks.shape != (1, len(self.tokens)) or lang_masks.dtype != torch.bool
                or any(mask.shape != (1,) or mask.dtype != torch.bool for mask in image_masks)):
            raise ValueError("Host mask layout requires its original B1 mask geometry")
        if self.bindings != tuple(tensor_identity(x) for x in (*images, *image_masks, lang_masks)):
            raise ValueError("Host mask layout does not belong to these prepared tensors")
        if not any(self.cameras):
            raise ValueError("Host mask layout requires a visible camera")


@dataclass
class LayoutTemplate:
    cpu_image_masks: list
    cpu_lang_masks: torch.Tensor
    cpu_packing: dict | None
    packing: dict | None
    empty_pattern: tuple
    metadata: tuple | None = None
    positions: dict | None = None
    materialized: bool = False

    @property
    def prepared_layout(self):
        return {"packing": self.packing, "empty_pattern": self.empty_pattern}

    def materialize(self, model, image_masks, lang_masks, *, elide_identity_gathers=False):
        if self.materialized:
            return
        # Integer mask arithmetic is identical on CPU. RoPE remains on the model
        # device through the original method, preserving its FP32 rounding.
        metadata = model.prepare_attention_metadata(
            self.cpu_image_masks, self.cpu_lang_masks,
            fixed_capacity=True, prefix_packing=self.cpu_packing,
        )
        if elide_identity_gathers:
            for item in metadata:
                annotate_identity_gathers(item)
        self.metadata = copy_layout_tree(metadata, device=lang_masks.device)
        self.positions = model.prepare_position_context(image_masks, lang_masks, self.packing)
        self.materialized = True


class HostLayoutCache:
    def __init__(self, capacity=32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Host layout cache capacity must be a positive integer")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()

    def get(self, model, host, images, image_masks, lang_masks):
        host.validate(images, image_masks, lang_masks)
        flags = model.optimization_flags()
        vision = model.paligemma_with_expert.paligemma.config.vision_config
        image_tokens = (vision.image_size // vision.patch_size) ** 2
        bucket = getattr(model.config, "prefix_pack_bucket_size", 64)
        key = (host.cameras, host.tokens, flags, image_tokens, bucket, model.action_horizon,
               tuple((tuple(x.shape), x.dtype, x.device) for x in (*images, *image_masks, lang_masks)))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        cpu_images = [torch.tensor([present], dtype=torch.bool) for present in host.cameras]
        cpu_language = torch.tensor([host.tokens], dtype=torch.bool)
        packing = prepare_prefix_packing(cpu_images, cpu_language, image_tokens, bucket) if flags[3] else None
        result = LayoutTemplate(
            cpu_images, cpu_language, packing, copy_layout_tree(packing, device=lang_masks.device),
            tuple(flags[0] and not present for present in host.cameras),
        )
        if len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
        self.entries[key] = result
        return result
