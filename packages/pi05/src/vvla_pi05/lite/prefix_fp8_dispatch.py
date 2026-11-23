"""Explicit architecture dispatch; no author reports or weight identity gates."""
import torch


def install_prefix_fp8(model):
    device=next(model.parameters()).device
    capability=torch.cuda.get_device_capability(device)
    if capability==(12,0):
        from .prefix_fp8 import install_prefix_fp8 as install
    elif capability==(8,9):
        from .prefix_fp8_down import install_prefix_fp8_down as install
    else:
        raise ValueError('Select an explicit BF16 recipe on devices without the SM89/SM120 prefix FP8 backend')
    return install(model)
