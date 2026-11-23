"""Fixed-batch host mask layouts (default B32); no input values or model KV cache."""
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .layout import HostMaskLayout, LayoutTemplate, copy_layout_tree
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
from vvla_repro_pi05.models.pi05.prefix_packing_pi05 import prepare_prefix_packing


def validate_fixed_batch_size(batch_size):
    if type(batch_size) is not int or batch_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Fixed batch size must be one of 1,2,4,8,16,32")
    return batch_size


@dataclass(frozen=True)
class BatchHostMaskLayout(HostMaskLayout):
    """The inherited tuple fields hold batch-major CPU camera/token mask rows."""
    batch_size: int = 32

    @classmethod
    def from_rows(cls, rows, images, image_masks, lang_masks, *, batch_size=32):
        validate_fixed_batch_size(batch_size)
        cameras = tuple(row.cameras for row in rows)
        tokens = tuple(row.tokens for row in rows)
        result = cls(cameras, tokens, tuple(tensor_identity(x) for x in (*images, *image_masks, lang_masks)), batch_size)
        result.validate(images, image_masks, lang_masks)
        return result

    @property
    def mask_bytes(self):
        return bytes(value for cameras, tokens in zip(self.cameras, self.tokens, strict=True)
                     for value in (*cameras, *tokens))

    def validate(self, images, image_masks, lang_masks):
        expected = validate_fixed_batch_size(self.batch_size)
        batch = len(self.cameras)
        if (batch != expected or len(self.tokens) != batch or not self.cameras or not self.tokens
                or any(len(row) != len(images) or any(type(x) is not bool for x in row) for row in self.cameras)
                or any(len(row) != len(self.tokens[0]) or any(type(x) is not bool for x in row) for row in self.tokens)
                or lang_masks.shape != (batch, len(self.tokens[0])) or lang_masks.dtype != torch.bool
                or len(image_masks) != len(images)
                or any(image.ndim != 4 or image.shape[0] != batch for image in images)
                or any(mask.shape != (batch,) or mask.dtype != torch.bool for mask in image_masks)
                or any(not any(row) for row in self.cameras)):
            raise ValueError(f"B{expected} host mask layout requires all current camera/token rows")
        if self.bindings != tuple(tensor_identity(x) for x in (*images, *image_masks, lang_masks)):
            raise ValueError(f"B{expected} host mask layout does not belong to these prepared tensors")


class BatchHostLayoutCache:
    """Bounded deterministic mask/position templates for the native graph owner."""
    def __init__(self, capacity=32, *, batch_size=32):
        self._batch_size = self._installed_batch_size = validate_fixed_batch_size(batch_size)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("B32 host layout capacity must be a positive integer")
        self.capacity, self.entries = capacity, OrderedDict()
        self.hits = self.misses = 0

    @property
    def batch_size(self):
        if self._batch_size != self._installed_batch_size:
            raise RuntimeError("Fixed host layout batch changed; reconstruct its owner")
        return self._batch_size

    def clear(self):
        self.entries.clear()

    def get(self, model, host, images, image_masks, lang_masks):
        if type(host) is not BatchHostMaskLayout or host.batch_size != self.batch_size:
            raise ValueError("B32 cache requires its own processor-owned layout")
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
        cpu_images = [torch.tensor([row[slot] for row in host.cameras], dtype=torch.bool)
                      for slot in range(len(images))]
        cpu_language = torch.tensor(host.tokens, dtype=torch.bool)
        packing = prepare_prefix_packing(cpu_images, cpu_language, image_tokens, bucket) if flags[3] else None
        pattern = tuple(flags[0] and not any(row[slot] for row in host.cameras) for slot in range(len(images)))
        result = LayoutTemplate(cpu_images, cpu_language, packing,
                                copy_layout_tree(packing, device=lang_masks.device), pattern)
        if len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
        self.entries[key] = result
        return result
