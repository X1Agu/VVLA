"""Real constructors must join visual and text tokens for either declared prefix."""

import pytest
import torch

from vvla_repro_pi05.models.pi05.modeling_pi05 import PaliGemmaWithActionExpertPi05, get_gemma_config


@pytest.mark.parametrize("variant,width", [("gemma_2b", 2048), ("gemma_300m", 1024)])
def test_visual_and_text_prefix_widths(variant, width):
    # All production layers, but no weight storage or CUDA allocation.
    with torch.device("meta"):
        model = PaliGemmaWithActionExpertPi05(get_gemma_config(variant), get_gemma_config("gemma_300m"))
        projector = model.paligemma.model.multi_modal_projector
        visual = projector(torch.empty(2, 256, 1152))
        text = model.embed_language_tokens(torch.empty(2, 32, dtype=torch.long))
        prefix = torch.cat([visual, text], dim=1)
    assert prefix.shape == (2, 288, width)
    assert projector.linear.weight.shape == (width, 1152)
    assert len(model.paligemma.model.language_model.layers) == 18
    assert len(model.gemma_expert.model.layers) == 18
