"""Live math controls used by sealed owners; no experiment identity dependency."""
import torch


def math_settings():
    return (torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            str(torch.backends.cuda.preferred_blas_library()))
