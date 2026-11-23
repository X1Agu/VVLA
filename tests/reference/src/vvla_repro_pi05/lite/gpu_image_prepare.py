"""Opt-in per-processor GPU preparation for raw NumPy uint8 camera images.

The processor's explicit per-image hook owns selection. Tokens, state, camera
order/masks, empty slots and relative/action processing stay in that processor.
No image buffers or module-global functions are patched/cached by this helper.
"""

import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np
import torch
from torch.nn import functional as F

from vvla_repro_pi05.lite.processor import Pi05LiteProcessor, resize_unit_then_siglip as _native_resize

HOOK = "_lite_image_prepare"
INSTALLATION = "_lite_gpu_uint8_prepare_installation"
HOOK_VERSION = "per_image_v1"
VERSION = "numpy-uint8-gpu-div-rn-v1"


def _eligible(image, device):
    return (torch.device(device).type == "cuda" and isinstance(image, np.ndarray)
            and image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3
            and image.shape[0] > 0 and image.shape[1] > 0
            and not torch.is_autocast_enabled("cuda"))


def prepare_uint8_image(image, height, width, device):
    """Eligible-image path; original resize/clamp/pad/affine operation order."""
    if not _eligible(image, device):
        return _native_resize(image, height, width, device)
    from vvla_repro_pi05.lite.gpu_image_prepare_kernel import uint8_to_unit
    raw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = uint8_to_unit(raw)
    current_h, current_w = tensor.shape[-2:]
    if (current_h, current_w) != (height, width):
        ratio = max(current_w / width, current_h / height)
        resized_h, resized_w = int(current_h / ratio), int(current_w / ratio)
        tensor = F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False).clamp(0, 1)
        top, extra_h = divmod(height - resized_h, 2)
        left, extra_w = divmod(width - resized_w, 2)
        tensor = F.pad(tensor, (left, left + extra_w, top, top + extra_h), value=0.0)
    return tensor * 2.0 - 1.0


class GPUUint8Preparation:
    """Reversible stateless image hook; caller serializes whole policy requests."""

    def __init__(self, processor):
        self._processor = weakref.ref(processor)
        self._builder = processor.build_model_inputs.__func__
        self._lock = threading.Lock()
        self._closed = False

    def _assert_installed(self):
        if self._closed:
            raise RuntimeError("GPU uint8 preparation is closed")
        processor = self._processor()
        if processor is None:
            raise RuntimeError("GPU uint8 processor was released")
        if (getattr(processor, HOOK, None) is not self
                or getattr(processor, INSTALLATION, None) is not self):
            raise RuntimeError("GPU uint8 preparation hook changed externally")
        if (getattr(processor.build_model_inputs, "__func__", None) is not self._builder
                or getattr(processor, "image_prepare_hook_version", None) != HOOK_VERSION):
            raise RuntimeError("GPU uint8 preparation processor changed externally")
        return processor

    def __call__(self, image, height, width, device):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("GPU uint8 preparation is serving another image request")
        try:
            processor = self._assert_installed()
            if torch.device(device) != torch.device(processor.device):
                raise ValueError("Image hook target differs from processor device")
            return prepare_uint8_image(image, height, width, device)
        finally:
            self._lock.release()

    def manifest(self):
        processor = self._assert_installed()
        here = Path(__file__)
        paths = (here, here.with_name("gpu_image_prepare_kernel.py"), here.with_name("processor.py"))
        return dict(version=VERSION,
                    actual_selected="triton_div_rn" if torch.device(processor.device).type == "cuda" else "native",
                    eligible_input="NumPy HWC RGB uint8, CUDA target, no autocast",
                    fallback="original resize_unit_then_siglip for all other inputs/devices/autocast",
                    arithmetic="FP32 div.rn(uint8,255), then original Torch interpolate/clamp/pad/*2/-1",
                    enable_fp_fusion=False, output_dtype="float32",
                    source_layout="native dense HWC storage with [1,3,H,W] strides",
                    h2d="ordinary blocking pageable uint8; no pinned or batched staging",
                    image_cache_bytes=0, weight_cache_bytes=0,
                    input_parity="exact unit/final FP32 bits and strides required by remote contracts/preflight",
                    untouched="camera order/masks/empty slots, state/prompt/tokenization, noise and action processing",
                    hook_version=HOOK_VERSION,
                    diagnostic_origin="profiling/gpu_image_prepare.py triton_div_rn; diagnostic source retained unchanged",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close GPU uint8 preparation during an image request")
        try:
            if self._closed:
                return
            processor = self._assert_installed()
            delattr(processor, HOOK)
            delattr(processor, INSTALLATION)
            self._closed = True
        finally:
            self._lock.release()

    def uninstall(self):
        self.close()


def install_gpu_uint8_prepare(processor):
    """Install before inference, after explicit default-off policy selection.

    Construction initializes no CUDA state and imports no Triton kernel. The
    root policy owns whole-request serialization and closes after graph/request
    retirement; all actual device allocations remain request-owned.
    """
    if (not isinstance(processor, Pi05LiteProcessor)
            or getattr(processor.build_model_inputs, "__func__", None) is not Pi05LiteProcessor.build_model_inputs
            or getattr(processor, "image_prepare_hook_version", None) != HOOK_VERSION):
        raise ValueError("GPU uint8 preparation requires the original Lite processor with per_image_v1 hook")
    if hasattr(processor, HOOK) or hasattr(processor, INSTALLATION):
        raise ValueError("GPU uint8 preparation is already installed or the image hook is occupied")
    handle = GPUUint8Preparation(processor)
    setattr(processor, HOOK, handle)
    setattr(processor, INSTALLATION, handle)
    return handle
