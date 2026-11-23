"""LeRobot image operation order for the Lite BF16 comparison boundary.

State, text, camera identities and physical actions use the shared processor.
Images enter [0,1] on CPU, move to the model device, resize/pad, and only then
receive the SigLIP affine transform. The historical core processor remains
unchanged; its normalize-before-resize order has different BF16 rounding.
"""

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from vvla_repro_pi05.models.pi05.processor_pi05 import (
    Pi05Processor, _extract_images, apply_norm, as_state_vector, build_pi05_prompt, tokenize_prompt,
)

IMAGE_PREPROCESS_VERSION = "lerobot_resize_unit_then_siglip_v1"


def unit_image(image):
    """Return CPU float32 [1,C,H,W] in [0,1], preserving native strides."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Images must be HWC RGB ndarrays")
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image.unsqueeze(0) if image.ndim == 3 else image
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError("Images must be CHW tensors, optionally batched once")
        tensor = tensor.cpu()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    if not tensor.is_floating_point():
        raise TypeError("Images must use uint8 or floating dtype")
    if not torch.isfinite(tensor).all() or float(tensor.amin()) < 0 or float(tensor.amax()) > 1:
        raise ValueError("Floating images must be finite and in [0,1]")
    return tensor.float()


def resize_unit_then_siglip(image, height, width, device):
    tensor = unit_image(image).to(device)
    current_h, current_w = tensor.shape[-2:]
    if (current_h, current_w) != (height, width):
        ratio = max(current_w / width, current_h / height)
        resized_h, resized_w = int(current_h / ratio), int(current_w / ratio)
        tensor = F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False).clamp(0, 1)
        top, extra_h = divmod(height - resized_h, 2)
        left, extra_w = divmod(width - resized_w, 2)
        tensor = F.pad(tensor, (left, left + extra_w, top, top + extra_h), value=0.0)
    return tensor * 2.0 - 1.0


class Pi05LiteProcessor(Pi05Processor):
    image_preprocess_version = IMAGE_PREPROCESS_VERSION
    image_prepare_hook_version = "per_image_v1"
    metadata_prepare_hook_version = "per_request_v1"

    def build_model_inputs(self, robot_obs, *, return_host_layout=False):
        config = self.config
        image_prepare = getattr(self, "_lite_image_prepare", resize_unit_then_siglip)
        metadata_prepare = getattr(self, "_lite_metadata_prepare", None)
        observations = _extract_images(robot_obs, config)
        feature_keys = config.image_feature_keys or list(observations)[:config.max_cameras]
        camera_keys = feature_keys[:config.max_cameras]
        if all(observations.get(key) is None for key in camera_keys):
            raise ValueError("π0.5 observation must provide at least one configured camera image")
        if not config.image_feature_keys and len(observations) > config.max_cameras:
            raise ValueError("Observation exceeds configured camera capacity")
        height, width = config.image_resolution
        images, image_masks, camera_present = [], [], []
        for key in camera_keys:
            image = observations.get(key)
            images.append(
                torch.full((1, 3, height, width), -1.0, device=self.device) if image is None
                else image_prepare(image, height, width, self.device)
            )
            if metadata_prepare is None:
                image_masks.append(torch.tensor([image is not None], dtype=torch.bool, device=self.device))
            camera_present.append(image is not None)
        while len(images) < config.max_cameras:
            images.append(torch.full((1, 3, height, width), -1.0, device=self.device))
            if metadata_prepare is None:
                image_masks.append(torch.zeros(1, dtype=torch.bool, device=self.device))
            camera_present.append(False)
        state = as_state_vector(robot_obs.get("state"), config.state_dim)
        normalized = apply_norm(torch.from_numpy(state), self._state_norm).numpy()
        prompt = build_pi05_prompt(
            task=robot_obs.get("prompt", "") or "", normalized_state=normalized, state_num_bins=config.state_num_bins,
        )
        ids, mask = tokenize_prompt(self.tokenizer, prompt, config.tokenizer_max_length)
        if metadata_prepare is None:
            tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
            masks = torch.tensor([mask], dtype=torch.bool, device=self.device)
        else:
            image_masks, tokens, masks = metadata_prepare(camera_present, ids, mask, self.device)
        if return_host_layout:
            from vvla_repro_pi05.lite.layout import HostMaskLayout
            host = HostMaskLayout.from_processor(camera_present, mask, images, image_masks, masks)
            return images, image_masks, tokens, masks, host
        return images, image_masks, tokens, masks
