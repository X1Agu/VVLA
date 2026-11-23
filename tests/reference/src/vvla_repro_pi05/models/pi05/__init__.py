# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserved pi05 public exports, loaded only when requested.

Configuration tools must not import the model, Omni, or initialize CUDA.
Numerical implementations remain in the separately checksummed modules.
"""

from importlib import import_module

_EXPORTS = {
    "Pi05Config": "vvla_repro_pi05.models.pi05.config",
    "UnsupportedCheckpointCapabilityError": "vvla_repro_pi05.models.pi05.config",
    "resolve_excluded_action_indices": "vvla_repro_pi05.models.pi05.config",
    "Pi05ForActionPrediction": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "PaliGemmaWithActionExpertPi05": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "Pi05AdaRMSNorm": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "GemmaVariantConfig": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "get_gemma_config": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "create_sinusoidal_pos_embedding": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "make_att_2d_masks": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "prepare_attention_masks_4d": "vvla_repro_pi05.models.pi05.modeling_pi05",
    "Pi05ImageProcessor": "vvla_repro_pi05.models.pi05.processor_pi05",
    "Pi05Processor": "vvla_repro_pi05.models.pi05.processor_pi05",
    "Pi05RelativeActions": "vvla_repro_pi05.models.pi05.processor_pi05",
    "build_pi05_prompt": "vvla_repro_pi05.models.pi05.processor_pi05",
    "discretize_state": "vvla_repro_pi05.models.pi05.processor_pi05",
    "apply_norm": "vvla_repro_pi05.models.pi05.processor_pi05",
    "build_norm_stats": "vvla_repro_pi05.models.pi05.processor_pi05",
    "pil_image_to_tensor": "vvla_repro_pi05.models.pi05.processor_pi05",
    "resize_with_pad": "vvla_repro_pi05.models.pi05.processor_pi05",
    "tokenize_prompt": "vvla_repro_pi05.models.pi05.processor_pi05",
    "Pi05Pipeline": "vvla_repro_pi05.models.pi05.pipeline_pi05",
    "get_pi05_post_process_func": "vvla_repro_pi05.models.pi05.pipeline_pi05",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
