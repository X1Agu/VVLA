"""True sample batching around the unchanged checkpoint input/output transforms.

The first implementation prepares each raw observation with the original
processor, then concatenates the current tensors on the batch axis. Model
execution is one B32 call, never 32 B1 calls. Output normalization/state shifts
operate on the complete batch and cross the CPU boundary once.
"""
import numpy as np
import torch

from vvla_repro_pi05.models.pi05.processor_pi05 import as_state_vector


def validate_observations(observations, batch_size):
    if type(observations) is not list or len(observations) != batch_size:
        raise ValueError(f"Expected exactly {batch_size} independent observations")
    if any(type(value) is not dict for value in observations):
        raise TypeError("Each batch slot requires its own raw observation dictionary")


class Pi05BatchProcessor:
    """Internal adapter; its single-row processor stays private to the engine."""
    def __init__(self, processor, batch_size, *, output_processor=None):
        self._single, self.batch_size = processor, batch_size
        self._output = processor if output_processor is None else output_processor

    def build_model_inputs(self, observations, *, return_host_layout=False):
        validate_observations(observations, self.batch_size)
        prepared = [self._single.build_model_inputs(value, return_host_layout=True)
                    if return_host_layout else self._single.build_model_inputs(value) for value in observations]
        rows = [row[:4] for row in prepared]
        views = len(rows[0][0])
        if views < 1 or any(len(row) != 4 or len(row[0]) != views or len(row[1]) != views for row in rows):
            raise ValueError("Every sample must preserve the same configured camera slots")
        for images, image_masks, tokens, masks in rows:
            if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.int64
                    or masks.shape != tokens.shape or masks.dtype != torch.bool
                    or any(image.ndim != 4 or image.shape[0] != 1 for image in images)
                    or any(mask.shape != (1,) or mask.dtype != torch.bool for mask in image_masks)):
                raise ValueError("Invalid single-sample prepared image/token/mask geometry")
        images = [torch.cat([row[0][view] for row in rows], dim=0).contiguous() for view in range(views)]
        image_masks = [torch.cat([row[1][view] for row in rows], dim=0).contiguous() for view in range(views)]
        tokens = torch.cat([row[2] for row in rows], dim=0).contiguous()
        masks = torch.cat([row[3] for row in rows], dim=0).contiguous()
        if return_host_layout:
            from .batch_layout import BatchHostMaskLayout
            host = BatchHostMaskLayout.from_rows([row[4] for row in prepared], images, image_masks, masks,
                                                batch_size=self.batch_size)
            return images, image_masks, tokens, masks, host
        return images, image_masks, tokens, masks

    def build_model_outputs(self, actions, observations):
        validate_observations(observations, self.batch_size)
        config = self._single.config
        if (not isinstance(actions, torch.Tensor) or actions.ndim != 3
                or actions.shape[:2] != (self.batch_size, config.chunk_size)
                or actions.shape[-1] != config.max_action_dim):
            raise ValueError("Model must return one complete action chunk for every batch slot")
        states = np.stack([as_state_vector(value.get("state"), config.state_dim) for value in observations])
        # The shared RelativeActionsStep already accepts (B,D) states. The
        # original processor's squeeze(0) preserves B when B>1, so its exact
        # operation order is retained without per-slot blocking CPU copies.
        result = self._output.build_model_outputs(actions, {"state": states})
        if self.batch_size == 1 and np.shape(result) == (config.chunk_size, config.action_dim):
            # Preserve the original processor's B1 squeeze arithmetic, then
            # restore only the API's explicit sample axis before the same copy.
            result = result[None, ...]
        owned = np.array(result, dtype=np.float32, order="C", copy=True)
        expected = (self.batch_size, config.chunk_size, config.action_dim)
        if owned.shape != expected or not np.isfinite(owned).all():
            raise ValueError(f"Batched physical actions must be finite float32{expected}")
        return owned
