"""Opt-in B1 camera batching through the unchanged SigLIP/projector modules.

The cache owner still decides which slots have reusable empty-image features.
This helper only batches slots whose request-local ``empty_features`` is None;
it never drops a slot based on its mask, changes pixels or caches an observation.
"""

from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
from types import MethodType
import weakref

import torch

from vvla_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

INSTALLATION = "_lite_batched_vision_installation"
VERSION = "b1-uncached-camera-batch-v1"


def _geometry(model):
    vision = model.paligemma_with_expert.paligemma.config.vision_config
    cameras = model.config.max_cameras
    size, patch, channels = vision.image_size, vision.patch_size, vision.num_channels
    width = model.vlm_width
    if (any(type(value) is not int or value < 1 for value in (cameras, size, patch, channels, width))
            or channels != 3 or size % patch):
        raise ValueError("Batched vision requires fixed RGB camera slots and divisible square image geometry")
    return cameras, (1, channels, size, size), (size // patch) ** 2, width


def _validate_inputs(model, installation, images, image_masks, lang_tokens, lang_masks, empty_features):
    if _geometry(model) != installation.geometry:
        raise ValueError("Vision/camera geometry changed; clear graphs and reinstall batched vision")
    cameras, image_shape, image_tokens, width = installation.geometry
    if len(images) != cameras or len(image_masks) != cameras:
        raise ValueError("Batched vision requires exactly max_cameras images and masks in slot order")
    if (lang_tokens.ndim != 2 or lang_tokens.shape[0] != 1 or lang_tokens.dtype != torch.int64
            or lang_masks.shape != lang_tokens.shape or lang_masks.dtype != torch.bool
            or lang_masks.device != lang_tokens.device):
        raise ValueError("Batched vision supports B1 int64 language tokens and matching boolean masks only")
    first_dtype = images[0].dtype
    if first_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Batched vision preserves FP32/BF16 pixels; other input dtypes are unsupported")
    for image, mask in zip(images, image_masks, strict=True):
        if (tuple(image.shape) != image_shape or image.dtype != first_dtype or image.device != lang_tokens.device
                or tuple(mask.shape) != (1,) or mask.dtype != torch.bool or mask.device != lang_tokens.device):
            raise ValueError("Batched vision supports B1 same-geometry/dtype images and boolean camera masks only")
    if empty_features is not None:
        if len(empty_features) != cameras:
            raise ValueError("Batched vision requires one empty_features entry per camera slot")
        for feature in empty_features:
            if feature is not None and (tuple(feature.shape) != (1, image_tokens, width)
                                        or feature.device != lang_tokens.device
                                        or feature.dtype not in (torch.float32, torch.bfloat16)):
                raise ValueError("Cached camera features must preserve B1 token/width geometry and device")


def _batched_embed_prefix(self, images, image_masks, lang_tokens, lang_masks, empty_features=None):
    installation = getattr(self, INSTALLATION)
    if not installation.enabled:
        raise RuntimeError("Batched vision installation is inactive")
    _validate_inputs(self, installation, images, image_masks, lang_tokens, lang_masks, empty_features)
    inner = self.paligemma_with_expert
    vision = inner.paligemma.model.vision_tower
    if (self.training or inner.training or vision.training or torch.is_grad_enabled()
            or torch.is_autocast_enabled(lang_tokens.device.type)):
        return Pi05ForActionPrediction.embed_prefix(
            self, images, image_masks, lang_tokens, lang_masks, empty_features=empty_features,
        )

    cameras, _, image_tokens, width = installation.geometry
    features = [None] * cameras if empty_features is None else list(empty_features)
    uncached = [i for i, feature in enumerate(features) if feature is None]
    if uncached:
        # Keep cameras as independent batch entries, never as one long image or
        # attention sequence. No dtype conversion precedes native embed_image.
        pixels = images[uncached[0]] if len(uncached) == 1 else torch.cat([images[i] for i in uncached], dim=0)
        encoded = inner.embed_image(pixels)
        if tuple(encoded.shape) != (len(uncached), image_tokens, width):
            raise ValueError("Native batched embed_image returned unexpected camera/token/width geometry")
        for row, slot in enumerate(uncached):
            features[slot] = encoded[row:row + 1]

    # This is the original slot-major prefix order and masks. A masked but
    # uncached slot is still encoded, just as in the native implementation.
    pad_masks = [mask[:, None].expand(1, image_tokens) for mask in image_masks]
    language = inner.embed_language_tokens(lang_tokens)
    embeddings = torch.cat([*features, language], dim=1)
    padding = torch.cat([*pad_masks, lang_masks], dim=1)
    attention = torch.zeros((1, embeddings.shape[1]), dtype=torch.bool, device=embeddings.device)
    return embeddings, padding, attention


@contextmanager
def _mutation_guard(model):
    if model.training or torch.is_inference_mode_enabled():
        raise ValueError("Install/uninstall batched vision in eval mode outside inference_mode")
    with ExitStack() as locks:
        for runner in tuple(getattr(model, "_graph_runners", ())):
            lock = getattr(runner, "_call_lock", None)
            if lock is None or not lock.acquire(blocking=False):
                raise RuntimeError("Cannot mutate batched vision during an active/unknown graph request")
            locks.callback(lock.release)
            if runner.entries:
                raise ValueError("Clear all live graph runners before changing batched vision")
        if not callable(getattr(model, "invalidate_inference_caches", None)):
            raise TypeError("Batched vision installation requires the shared model lifecycle")
        yield


class BatchedVisionInstallation:
    def __init__(self, model, geometry):
        self._model = weakref.ref(model)
        self.geometry = geometry
        self.enabled = True

    def manifest(self):
        core = Path(__file__).resolve().parents[1] / "models/pi05/modeling_pi05.py"
        cameras, shape, tokens, width = self.geometry
        return {
            "version": VERSION, "enabled": self.enabled, "batch": 1,
            "camera_slots": cameras, "image_shape": list(shape), "image_tokens": tokens,
            "feature_width": width, "grouping": "all slots with empty_features=None, in original slot order",
            "cached_slots": "request-local features reused unchanged; never selected by mask here",
            "precision": "native embed_image/projector/norm dtype boundaries unchanged",
            "attention": "cameras are separate batch rows; original per-slot prefix masks",
            "weight_cache_bytes": 0, "activation_cache_bytes": 0,
            "fallback": "original per-camera path for training, gradients or autocast; B1 geometry still required",
            "numerics": "batch geometry can change GEMM/conv algorithm and rounding; parity must be measured",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), core)},
        }

    def uninstall(self):
        model = self._model()
        if model is None or not self.enabled or getattr(model, INSTALLATION, None) is not self:
            raise RuntimeError("Batched vision installation is no longer active")
        with _mutation_guard(model):
            if getattr(model.embed_prefix, "__func__", None) is not _batched_embed_prefix:
                raise RuntimeError("Batched embed_prefix was replaced externally")
            # Invalidation can fail without removing the active installation.
            model.invalidate_inference_caches(weights_changed=False)
            delattr(model, "embed_prefix")
            delattr(model, INSTALLATION)
            self.enabled = False


def install_batched_vision(model):
    """Install after checkpoint/recipe loading, before graphs; default is absent."""
    with _mutation_guard(model):
        if getattr(model, INSTALLATION, None) is not None:
            raise ValueError("Batched vision is already installed")
        if ("embed_prefix" in model.__dict__
                or getattr(model.embed_prefix, "__func__", None) is not Pi05ForActionPrediction.embed_prefix):
            raise ValueError("Batched vision requires the original unpatched Pi05 embed_prefix")
        geometry = _geometry(model)
        installation = BatchedVisionInstallation(model, geometry)
        model.invalidate_inference_caches(weights_changed=False)
        setattr(model, INSTALLATION, installation)
        model.embed_prefix = MethodType(_batched_embed_prefix, model)
        return installation
