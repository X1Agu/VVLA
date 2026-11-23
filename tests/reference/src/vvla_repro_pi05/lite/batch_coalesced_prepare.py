"""Unselected B32 preparation candidate: fresh bulk images and metadata.

The initial three-arm campaign does not use this module. State normalization,
prompt construction and tokenizer calls remain per row and use the original
functions. Homogeneous NumPy uint8 cameras use one fresh H2D image allocation;
all typed metadata uses a second allocation. Other image domains/layouts use
the existing row processor. CUDA bit parity and end-to-end benefit must be
measured before this candidate can enter a sealed deployment.
"""
from contextlib import ExitStack

import numpy as np
import torch
from torch.nn import functional as F

from .batch_processor import Pi05BatchProcessor, validate_observations
from .packed_input_metadata import build_cpu_payload
from .processor import Pi05LiteProcessor
from vvla_repro_pi05.models.pi05.processor_pi05 import (
    _extract_images, apply_norm, as_state_vector, build_pi05_prompt, tokenize_prompt,
)

VERSION = 'unselected-fresh-b32-coalesced-inputs-v1'


def _source_rows(processor, observations):
    """Resolve each camera slot using the original processor's exact rules."""
    config = processor.config
    rows = []
    shape = None
    for observation in observations:
        images = _extract_images(observation, config)
        keys = config.image_feature_keys or list(images)[:config.max_cameras]
        keys = keys[:config.max_cameras]
        if all(images.get(key) is None for key in keys):
            raise ValueError('π0.5 observation must provide at least one configured camera image')
        if not config.image_feature_keys and len(images) > config.max_cameras:
            raise ValueError('Observation exceeds configured camera capacity')
        row = [images.get(key) for key in keys]
        row += [None] * (config.max_cameras - len(row))
        for image in row:
            if image is None:
                continue
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[-1] != 3
                    or min(image.shape[:2]) <= 0):
                return None
            if shape is None:
                shape = image.shape
            elif image.shape != shape:
                return None
        rows.append(row)
    return rows, shape


def _images(rows, shape, height, width, device, gpu_uint8):
    # Camera-major order permits disjoint B-row output views without gathers.
    batch, cameras = len(rows), len(rows[0])
    active_slots = [slot for slot in range(cameras) if any(row[slot] is not None for row in rows)]
    host = np.zeros((len(active_slots) * batch, *shape), dtype=np.uint8)
    for packed, slot in enumerate(active_slots):
        for row in range(batch):
            if rows[row][slot] is not None:
                host[packed * batch + row] = rows[row][slot]
    raw = torch.from_numpy(host).permute(0, 3, 1, 2)
    if torch.device(device).type == 'cuda' and gpu_uint8:
        # Same div.rn kernel and dense physical order as the selected per-image
        # path. Its original one-image validator is intentionally not bypassed:
        # this separate helper validates its own B-row channels-last geometry.
        from .gpu_image_prepare_kernel import _uint8_unit
        import triton
        raw = raw.to(device=device, non_blocking=False)
        if (raw.stride(1) != 1 or raw.stride(3) != 3
                or raw.stride(2) != shape[1] * 3
                or raw.stride(0) != shape[0] * shape[1] * 3):
            raise ValueError('Bulk uint8 images require dense camera-major HWC storage')
        unit = torch.empty_like(raw, dtype=torch.float32, memory_format=torch.preserve_format)
        _uint8_unit[(triton.cdiv(raw.numel(), 1024),)](
            raw, unit, raw.numel(), 1024, enable_fp_fusion=False)
    else:
        unit = (raw.float() / 255.0).to(device=device, non_blocking=False)
    current_h, current_w = shape[:2]
    if (current_h, current_w) != (height, width):
        ratio = max(current_w / width, current_h / height)
        resized_h, resized_w = int(current_h / ratio), int(current_w / ratio)
        unit = F.interpolate(unit, size=(resized_h, resized_w), mode='bilinear',
                             align_corners=False).clamp(0, 1)
        top, extra_h = divmod(height - resized_h, 2)
        left, extra_w = divmod(width - resized_w, 2)
        unit = F.pad(unit, (left, left + extra_w, top, top + extra_h), value=0.0)
    result = unit * 2.0 - 1.0
    prepared = {slot: result[packed * batch:(packed + 1) * batch].contiguous()
                for packed, slot in enumerate(active_slots)}
    return [prepared[slot] if slot in prepared else torch.full((batch, 3, height, width),
                -1.0, device=device, dtype=torch.float32) for slot in range(cameras)]


def _metadata(camera_rows, token_rows, mask_rows, device):
    batch, cameras, tokens = len(camera_rows), len(camera_rows[0]), len(token_rows[0])
    # Camera masks are camera-major, tokens and text masks are batch-major.
    payload, layout = build_cpu_payload(np.asarray(camera_rows, dtype=np.bool_).T.reshape(-1),
        np.asarray(token_rows, dtype=np.int64).reshape(-1),
        np.asarray(mask_rows, dtype=np.bool_).reshape(-1))
    payload = payload.to(device=device, non_blocking=False)
    images = payload[:batch * cameras].view(torch.bool).reshape(cameras, batch)
    ids = payload[layout['token_offset']:layout['mask_offset']].view(torch.int64).reshape(batch, tokens)
    mask = payload[layout['mask_offset']:].view(torch.bool).reshape(batch, tokens)
    return [images[slot] for slot in range(cameras)], ids, mask


def build_coalesced_inputs(processor, observations, *, return_host_layout=False):
    """Prototype only; returns fresh current inputs and performs no inference.

    Installed original preparation owners are locked and validated once for
    the complete call. Unknown hooks/builders are rejected, never ignored.
    The private batch engine would still serialize the surrounding request.
    """
    validate_observations(observations, 32)
    if (type(processor) is not Pi05LiteProcessor
            or getattr(processor.build_model_inputs, '__func__', None) is not Pi05LiteProcessor.build_model_inputs):
        raise ValueError('Coalesced preparation requires the original Lite processor')
    if torch.is_autocast_enabled(torch.device(processor.device).type):
        raise ValueError('Coalesced preparation requires the selected no-autocast execution boundary')
    source = _source_rows(processor, observations)
    if source is None:
        return Pi05BatchProcessor(processor, 32).build_model_inputs(
            observations, return_host_layout=return_host_layout)
    from .gpu_image_prepare import GPUUint8Preparation
    from .packed_input_metadata import PackedInputMetadata
    hooks = [('_lite_image_prepare', '_lite_gpu_uint8_prepare_installation', GPUUint8Preparation),
             ('_lite_metadata_prepare', '_lite_packed_input_metadata_installation', PackedInputMetadata)]
    gpu_uint8 = False
    with ExitStack() as locks:
        for hook, installation, expected in hooks:
            owner = getattr(processor, hook, None)
            installed = getattr(processor, installation, None)
            if owner is None and installed is None:
                continue
            if type(owner) is not expected or installed is not owner:
                raise ValueError('Coalesced preparation cannot bypass a foreign processor hook')
            if not owner._lock.acquire(blocking=False):
                raise RuntimeError('Original preparation owner is serving another request')
            locks.callback(owner._lock.release)
            if expected is GPUUint8Preparation:
                owner._assert_installed()
                gpu_uint8 = True
            else:
                owner._active_processor()
        rows, shape = source
        config = processor.config
        cameras = [[image is not None for image in row] for row in rows]
        ids, masks = [], []
        for observation in observations:
            state = as_state_vector(observation.get('state'), config.state_dim)
            normalized = apply_norm(torch.from_numpy(state), processor._state_norm).numpy()
            prompt = build_pi05_prompt(task=observation.get('prompt', '') or '',
                normalized_state=normalized, state_num_bins=config.state_num_bins)
            row_ids, row_mask = tokenize_prompt(processor.tokenizer, prompt, config.tokenizer_max_length)
            if len(row_ids) != config.tokenizer_max_length or len(row_mask) != len(row_ids):
                raise ValueError('Tokenizer returned unexpected fixed token geometry')
            ids.append(row_ids)
            masks.append([bool(value) for value in row_mask])
        height, width = config.image_resolution
        images = _images(rows, shape, height, width, processor.device, gpu_uint8)
        image_masks, tokens, language_masks = _metadata(cameras, ids, masks, processor.device)
        if return_host_layout:
            from .batch_layout import BatchHostMaskLayout
            from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
            host = BatchHostMaskLayout(tuple(tuple(row) for row in cameras),
                tuple(tuple(row) for row in masks),
                tuple(tensor_identity(x) for x in (*images, *image_masks, language_masks)))
            host.validate(images, image_masks, language_masks)
            return images, image_masks, tokens, language_masks, host
        return images, image_masks, tokens, language_masks
