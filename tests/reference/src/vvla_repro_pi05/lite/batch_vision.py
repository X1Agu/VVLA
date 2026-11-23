"""Fixed camera batching (default B32) derived from the B1 uncached path."""
import hashlib
from pathlib import Path
from types import MethodType

import torch

from .batch_layout import validate_fixed_batch_size
from .batched_vision import (INSTALLATION, _geometry, _mutation_guard, BatchedVisionInstallation)
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

BATCH = 32


def _validate_inputs(model, owner, images, image_masks, tokens, masks, empty_features):
    batch = owner.batch_size
    if _geometry(model) != owner.geometry:
        raise ValueError(f"B{batch} vision geometry changed; retire and reinstall")
    cameras, shape, image_tokens, width = owner.geometry
    if (len(images) != cameras or len(image_masks) != cameras or tokens.ndim != 2
            or tokens.shape[0] != batch or tokens.dtype != torch.int64
            or masks.shape != tokens.shape or masks.dtype != torch.bool or masks.device != tokens.device):
        raise ValueError(f"B{batch} vision requires its ordered camera slots and language rows")
    dtype = images[0].dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"B{batch} camera batching preserves native FP32/BF16 pixels")
    for image, mask in zip(images, image_masks, strict=True):
        if (tuple(image.shape) != (batch, *shape[1:]) or image.dtype != dtype or image.device != tokens.device
                or mask.shape != (batch,) or mask.dtype != torch.bool or mask.device != tokens.device):
            raise ValueError(f"B{batch} camera/image/mask geometry differs from deployment")
    if empty_features is not None:
        if len(empty_features) != cameras:
            raise ValueError(f"B{batch} requires one feature entry per camera slot")
        if any(feature is not None and (feature.shape != (batch, image_tokens, width)
               or feature.device != tokens.device or feature.dtype not in (torch.float32, torch.bfloat16))
               for feature in empty_features):
            raise ValueError(f"B{batch} empty-camera features must retain all independent sample rows")


def _embed_prefix(self, images, image_masks, lang_tokens, lang_masks, empty_features=None):
    owner = getattr(self, INSTALLATION)
    batch = owner.batch_size
    if not owner.enabled:
        raise RuntimeError(f"B{batch} camera batching is inactive")
    _validate_inputs(self, owner, images, image_masks, lang_tokens, lang_masks, empty_features)
    inner = self.paligemma_with_expert
    vision = inner.paligemma.model.vision_tower
    if (self.training or inner.training or vision.training or torch.is_grad_enabled()
            or torch.is_autocast_enabled(lang_tokens.device.type)):
        return Pi05ForActionPrediction.embed_prefix(self, images, image_masks, lang_tokens, lang_masks,
                                                     empty_features=empty_features)
    cameras, _, image_tokens, width = owner.geometry
    features = [None] * cameras if empty_features is None else list(empty_features)
    uncached = [slot for slot, value in enumerate(features) if value is None]
    if uncached:
        pixels = images[uncached[0]] if len(uncached) == 1 else torch.cat([images[slot] for slot in uncached], dim=0)
        encoded = inner.embed_image(pixels)
        if encoded.shape != (len(uncached)*batch, image_tokens, width):
            raise ValueError(f"B{batch} native camera encoder returned incorrect batch/token/width")
        for group, slot in enumerate(uncached):
            features[slot] = encoded[group*batch:(group+1)*batch]
    language = inner.embed_language_tokens(lang_tokens)
    embeddings = torch.cat([*features, language], dim=1)
    padding = torch.cat([*(mask[:, None].expand(batch, image_tokens) for mask in image_masks), lang_masks], dim=1)
    attention = torch.zeros((batch, embeddings.shape[1]), dtype=torch.bool, device=embeddings.device)
    return embeddings, padding, attention


class BatchVisionInstallation(BatchedVisionInstallation):
    def __init__(self, model, geometry, *, batch_size=32):
        super().__init__(model, geometry)
        self._batch_size = self._installed_batch_size = validate_fixed_batch_size(batch_size)

    @property
    def batch_size(self):
        if self._batch_size != self._installed_batch_size:
            raise RuntimeError("Fixed camera batch changed; retire and reinstall")
        return self._batch_size

    def manifest(self):
        batch = self.batch_size
        value = super().manifest()
        value.update(version=f"b{batch}-uncached-camera-batch-v1", batch=batch,
                     image_shape=[batch, *self.geometry[1][1:]],
                     grouping=f"camera-major groups of{batch} independent samples, split back into exact original per-sample slot order",
                     fallback=f"original per-camera B{batch} tensor path for training/grad/autocast; no B1 execution loop",
                     source_sha256={**value["source_sha256"], Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        return value

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("B32 camera batching is no longer active")
        with _mutation_guard(model):
            if getattr(model.embed_prefix, "__func__", None) is not _embed_prefix:
                raise RuntimeError("B32 embed_prefix binding changed")
            model.invalidate_inference_caches(weights_changed=False)
            delattr(model, "embed_prefix")
            delattr(model, INSTALLATION)
            self.enabled = False


def install_batch_vision(model, *, batch_size=32):
    validate_fixed_batch_size(batch_size)
    with _mutation_guard(model):
        if (getattr(model, INSTALLATION, None) is not None or "embed_prefix" in model.__dict__
                or getattr(model.embed_prefix, "__func__", None) is not Pi05ForActionPrediction.embed_prefix):
            raise ValueError("B32 camera batching requires the unchanged original prefix method")
        owner = BatchVisionInstallation(model, _geometry(model), batch_size=batch_size)
        model.invalidate_inference_caches(weights_changed=False)
        setattr(model, INSTALLATION, owner)
        model.embed_prefix = MethodType(_embed_prefix, model)
        return owner
