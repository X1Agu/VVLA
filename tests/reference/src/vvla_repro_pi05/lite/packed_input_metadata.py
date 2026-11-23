"""One blocking CUDA transfer for fresh camera masks, token IDs and text mask.

No persistent payload, pinned allocation, asynchronous transfer, tensor cache,
or tokenization cache is used. Returned regions share one request-owned storage
but never overlap. Existing image processing and RNG are outside this helper.
"""
import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np
import torch

from .processor import Pi05LiteProcessor

HOOK = '_lite_metadata_prepare'
INSTALLATION = '_lite_packed_input_metadata_installation'
HOOK_VERSION = 'per_request_v1'
VERSION = 'fresh-aligned-metadata-blocking-h2d-v1'


def payload_layout(cameras, tokens):
    if type(cameras) is not int or cameras < 1 or type(tokens) is not int or tokens < 1:
        raise ValueError('Metadata requires positive camera and token counts')
    token_offset = ((cameras + 7) // 8) * 8
    mask_offset = token_offset + 8 * tokens
    return dict(camera_offset=0, token_offset=token_offset, mask_offset=mask_offset,
                bytes=mask_offset + tokens, cameras=cameras, tokens=tokens)


def build_cpu_payload(camera_present, ids, mask):
    """Private fresh bytes; CPU conversions match bool/int64 model boundaries."""
    camera_values = np.asarray(camera_present, dtype=np.bool_)
    token_values = np.asarray(ids, dtype=np.int64)
    mask_values = np.asarray(mask, dtype=np.bool_)
    if (camera_values.ndim != 1 or token_values.ndim != 1 or mask_values.ndim != 1
            or token_values.shape != mask_values.shape):
        raise ValueError('Metadata expects camera vector and equally sized token/mask vectors')
    layout = payload_layout(len(camera_values), len(token_values))
    payload = np.zeros(layout['bytes'], dtype=np.uint8)
    payload[:layout['cameras']] = camera_values.view(np.uint8)
    payload[layout['token_offset']:layout['mask_offset']] = token_values.view(np.uint8)
    payload[layout['mask_offset']:] = mask_values.view(np.uint8)
    return torch.from_numpy(payload), layout


def metadata_views(payload, layout):
    """Reinterpret disjoint aligned slices; never launch a dtype cast kernel."""
    expected = payload_layout(layout['cameras'], layout['tokens'])
    if (layout != expected or payload.dtype != torch.uint8 or payload.ndim != 1
            or payload.numel() != layout['bytes'] or not payload.is_contiguous()
            or payload.storage_offset() % 8):
        raise ValueError('Metadata payload must have its original aligned uint8 geometry')
    image_masks = [payload[index:index + 1].view(torch.bool) for index in range(layout['cameras'])]
    tokens = payload[layout['token_offset']:layout['mask_offset']].view(torch.int64).reshape(1, layout['tokens'])
    masks = payload[layout['mask_offset']:].view(torch.bool).reshape(1, layout['tokens'])
    return image_masks, tokens, masks


def prepare_metadata(camera_present, ids, mask, device):
    if torch.device(device).type != 'cuda':
        # Same original typed constructors; CPU never uses packed storage.
        return ([torch.tensor([present], dtype=torch.bool, device=device) for present in camera_present],
                torch.tensor([ids], dtype=torch.int64, device=device),
                torch.tensor([mask], dtype=torch.bool, device=device))
    payload, layout = build_cpu_payload(camera_present, ids, mask)
    # Ordinary pageable blocking copy: input can be released immediately after
    # this call; no new lifetime, pinned-buffer or cross-stream owner contract.
    device_payload = payload.to(device=device, non_blocking=False)
    return metadata_views(device_payload, layout)


class PackedInputMetadata:
    def __init__(self, processor):
        self._processor = weakref.ref(processor)
        self._builder = processor.build_model_inputs.__func__
        self._lock = threading.Lock()
        self._closed = False

    def _active_processor(self):
        processor = self._processor()
        if self._closed or processor is None:
            raise RuntimeError('Packed metadata preparation is closed or processor was released')
        if (getattr(processor, HOOK, None) is not self or getattr(processor, INSTALLATION, None) is not self
                or getattr(processor.build_model_inputs, '__func__', None) is not self._builder
                or getattr(processor, 'metadata_prepare_hook_version', None) != HOOK_VERSION):
            raise RuntimeError('Packed metadata processor or installed dispatch changed')
        return processor

    def __call__(self, camera_present, ids, mask, device):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Packed metadata preparation is serving another request')
        try:
            processor = self._active_processor()
            if torch.device(device) != torch.device(processor.device):
                raise ValueError('Packed metadata target differs from processor device')
            return prepare_metadata(camera_present, ids, mask, device)
        finally:
            self._lock.release()

    def manifest(self):
        processor = self._active_processor()
        paths = (Path(__file__), Path(__file__).with_name('processor.py'))
        return dict(version=VERSION, default_enabled=False,
                    actual_selected='packed_blocking_h2d' if torch.device(processor.device).type == 'cuda' else 'native_cpu',
                    layout='camera bool bytes; zero padding to 8-byte alignment; int64 tokens; bool language mask',
                    cuda_transfers=1, transfer='fresh pageable CPU uint8 -> blocking CUDA copy',
                    output='nonoverlapping typed views of one fresh request-owned allocation',
                    inputs='same camera_present, IDs and masks returned by the original tokenizer',
                    unchanged='image operations, state normalization, prompt construction, tokenizer, RNG and graph live checks',
                    persistent_input_cache=False, pinned=False, asynchronous=False,
                    cpu_fallback='original torch.tensor constructors', hook_version=HOOK_VERSION,
                    source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close packed metadata during an active request')
        try:
            if self._closed:
                return
            processor = self._active_processor()
            delattr(processor, HOOK)
            delattr(processor, INSTALLATION)
            self._closed = True
        finally:
            self._lock.release()

    def uninstall(self):
        self.close()


def install_packed_input_metadata(processor):
    if (not isinstance(processor, Pi05LiteProcessor)
            or getattr(processor.build_model_inputs, '__func__', None) is not Pi05LiteProcessor.build_model_inputs
            or getattr(processor, 'metadata_prepare_hook_version', None) != HOOK_VERSION):
        raise ValueError('Packed metadata requires the original Lite processor with per_request_v1 hook')
    if hasattr(processor, HOOK) or hasattr(processor, INSTALLATION):
        raise ValueError('Packed metadata is already installed or the hook is occupied')
    handle = PackedInputMetadata(processor)
    setattr(processor, HOOK, handle)
    setattr(processor, INSTALLATION, handle)
    return handle
