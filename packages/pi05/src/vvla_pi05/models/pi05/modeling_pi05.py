# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only π0.5 VLA math kernel for vllm-omni.

Only the math that turns a robot observation into an action chunk; no serving or
request glue. Deliberately shaped like ``models/pi0/modeling_pi0.py`` so the two
can later be factored into a shared Pi-family module (RFC step 2) on a
"behaviour unchanged" review.

π0.5 = PaliGemma (SigLIP vision + Gemma 2B LM) prefix + Gemma 300M action expert
suffix + flow-matching head. Where π0 projects the robot state through
``state_proj``, π0.5 discretizes it into prompt tokens — so there is no
``state_proj`` layer, ``sample_actions`` takes no ``state`` argument, and the
suffix is action tokens only, which drops π0's leading state-token boundary from
the suffix attention mask and leaves ``[1] + [0] * (horizon - 1)``.

π0.5 is nonetheless the *larger* model: the 37 AdaRMS ``dense`` projections add
~116M parameters against the ~8K that ``state_proj`` saves. (LeRobot's README
says otherwise; the checkpoint disagrees.)

Reference implementations:
   - OpenPI: openpi/src/openpi/models_pytorch/pi0_pytorch.py, gemma_pytorch.py
   - LeRobot: lerobot/src/lerobot/policies/pi05/modeling_pi05.py
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
    apply_rotary_pos_emb,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)
from vvla_pi05.models.pi05.inference_cache_pi05 import Pi05InferenceCache, tensor_identity
from vvla_pi05.models.pi05.prefix_packing_pi05 import pack_prefix, prepare_prefix_packing

logger = logging.getLogger(__name__)


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load an unsharded tensor without importing the serving runtime."""
    if param.shape != loaded_weight.shape:
        raise ValueError(f"Weight shape mismatch: expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}")
    with torch.no_grad():
        param.copy_(loaded_weight)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_MAX_TOKEN_LEN = 200  # π0 uses 48
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_IMAGE_RESOLUTION = (224, 224)  # openpi/models/model.py IMAGE_RESOLUTION
DEFAULT_STATE_NUM_BINS = 256  # openpi PaliGemmaTokenizer.tokenize()

# Large negative value to fill masked-out positions in a float attention mask.
# Matches OpenPI's constant exactly so that numerics line up during parity.
# Ref: openpi/src/openpi/models/gemma.py
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ──────────────────────────────────────────────────────────────────────
# Gemma variant configs (matches openpi/models/gemma.py get_config)
# ──────────────────────────────────────────────────────────────────────
class GemmaVariantConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_2b":
        return GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)
    elif variant == "gemma_300m":
        return GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ──────────────────────────────────────────────────────────────────────
# Utility functions (match openpi/models_pytorch/pi0_pytorch.py)
# ──────────────────────────────────────────────────────────────────────
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute a sine/cosine positional embedding for scalar timesteps.

    Ref: openpi/models_pytorch/pi0_pytorch.py create_sinusoidal_pos_embedding
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")
    if device is None:
        device = time.device

    # Use float64 for the log-linear sweep and for the inner products, to
    # match the reference implementation's numerical behaviour exactly.
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build a 2D attention mask from a padding mask and an autoregressive mask.

    Ref: openpi/models_pytorch/pi0_pytorch.py make_att_2d_masks
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2-D, got {att_masks.ndim}-D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2-D, got {pad_masks.ndim}-D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, S, S)`` bool masks to ``(B, 1, S, S)`` float masks.

    ``True`` → 0.0 (attend), ``False`` → ``OPENPI_ATTENTION_MASK_VALUE``.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class Pi05AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm conditioned on the flow-matching timestep.

    π0 conditions on time by *concatenating* a time embedding onto each action
    embedding. π0.5 instead feeds the time embedding into every action-expert
    norm, which produces a per-layer ``(scale, shift, gate)`` triple::

        y    = norm(x) * (1 + scale) + shift
        out  = residual + gate * sublayer(y)

    ``dense`` is zero-initialized, so an untrained model starts as the identity
    modulation with a closed gate — matching OpenPI's parameterization.

    Note the shape of the unconditioned branch: ``normed * (1 + weight)`` with
    ``weight`` zero-initialized, which is exactly ``transformers``'
    ``GemmaRMSNorm``. That equivalence is why only the *expert* norms need
    replacing here and the PaliGemma prefix can keep stock Gemma layers.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(var + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(normed, gate)``; ``gate`` is ``None`` in the unconditioned case."""
        dtype = x.dtype
        normed = self._norm(x)
        if self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        if cond is None and modulation is None:
            # A conditioned norm silently falling back to an unconditioned one
            # would drop the entire timestep signal and still return a
            # well-shaped, finite tensor.
            raise ValueError(
                "Pi05AdaRMSNorm was built with cond_dim="
                f"{self.cond_dim} but called without an AdaRMS conditioning vector."
            )

        if modulation is None and cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected AdaRMS cond dim {self.cond_dim}, got {cond.shape[-1]}")

        if modulation is None:
            modulation = self.dense(cond.to(self.dense.weight.dtype))
        elif modulation.shape != (x.shape[0], 3 * self.dim):
            raise ValueError("Precomputed AdaRMS modulation has incompatible batch or width")
        if x.ndim == 3:
            # (B, 3*dim) → (B, 1, 3*dim), broadcast across the token axis: the
            # timestep is a per-sample scalar, identical for every action token.
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _gated_residual(residual: torch.Tensor, out: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return residual + out
    return residual + gate * out


# ──────────────────────────────────────────────────────────────────────
# Dual-backbone: PaliGemma + AdaRMS action expert
# ──────────────────────────────────────────────────────────────────────
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)``.
    """
    if n_rep == 1:
        return hidden_states
    b, nh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nh, n_rep, s, d)
    return hidden_states.reshape(b, nh * n_rep, s, d)


def _attend(
    query_states,
    key_states,
    value_states,
    attention_mask,
    num_kv_groups,
    scaling,
    attention=None,
    attention_metadata=None,
):
    """Manual eager attention: ``softmax(Q Kᵀ · scale + mask) · V``.

    The mask is sliced to ``key_states.shape[-2]`` so the same
    ``(B, 1, Q, prefix+suffix)`` mask works in both the prefix pass (K length =
    prefix) and the suffix pass (K length = prefix + suffix).
    """
    if attention is not None:
        return attention(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_metadata
        ).transpose(1, 2)
    k = _repeat_kv(key_states, num_kv_groups)
    v = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, v)


def _match(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast ``tensor`` to the dtype ``module``'s weight expects."""
    dtype = getattr(module, "input_dtype", None)
    if dtype is None:
        dtype = module.weight.dtype
    return tensor.to(dtype) if tensor.dtype != dtype else tensor


def _compute_layer_prefix_only(
    layer_idx,
    hidden_states,
    attention_mask,
    position_ids,
    paligemma,
    attention=None,
    attention_metadata=None,
    rotary=None,
    kv_only=False,
    fuse_rope=False,
    fuse_residual_square=False,
    squared=None,
):
    """Run one PaliGemma LM layer on the prefix, returning the layer output and
    the post-RoPE ``(k, v)`` for the suffix pass.

    Identical to π0: the prefix backbone is unchanged in π0.5 (no AdaRMS —
    there is no timestep in the prefix).
    """
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]
    residual = hidden_states
    normalized = (layer.input_layernorm(hidden_states, squared=squared)
                  if fuse_residual_square else layer.input_layernorm(hidden_states))
    x = _match(normalized, layer.self_attn.q_proj)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q = None if kv_only else layer.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(v, position_ids) if rotary is None else rotary
    # The old path computes a discarded dummy Q in KV-only mode; the fused path
    # rotates K alone with the same arithmetic.
    if fuse_rope:
        from vvla_runtime.pointwise import rotary_pos_emb
        q, k = rotary_pos_emb(None if kv_only else q, k, cos, sin)
    else:
        q, k = apply_rotary_pos_emb(k if kv_only else q, k, cos, sin, unsqueeze_dim=1)
    if kv_only:
        if attention_metadata is not None and "paged_cache" in attention_metadata.extra:
            attention_metadata.extra["paged_cache"].write(layer_idx, k.transpose(1, 2), v.transpose(1, 2), prefix=True)
            return (hidden_states, None, None) if fuse_residual_square else (hidden_states, None)
        return (hidden_states, (k, v), None) if fuse_residual_square else (hidden_states, (k, v))

    att = _attend(
        q,
        k,
        v,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
        attention=attention,
        attention_metadata=attention_metadata,
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    projected = layer.self_attn.o_proj(_match(att, layer.self_attn.o_proj))
    if fuse_residual_square:
        from vvla_runtime.pointwise import residual_square
        out, attention_square = residual_square(residual, projected)
        normed = layer.post_attention_layernorm(out, squared=attention_square)
        out, next_square = residual_square(out, layer.mlp(_match(normed, layer.mlp.up_proj)))
    else:
        out = projected + residual
        after_resid = out
        normed = layer.post_attention_layernorm(out)
        out = layer.mlp(_match(normed, layer.mlp.up_proj)) + after_resid
    # The paged path has already stored this layer's KV. Retaining dense KV
    # would duplicate the cache and defeat the storage ownership boundary.
    paged = attention_metadata is not None and "paged_cache" in attention_metadata.extra
    kv = None if paged else (k, v)
    return (out, kv, next_square) if fuse_residual_square else (out, kv)


def _compute_layer_suffix_only(
    layer_idx,
    hidden_states,
    prefix_kv,
    attention_mask,
    position_ids,
    gemma_expert,
    adarms_cond,
    attention=None,
    attention_metadata=None,
    rotary=None,
    modulations=None,
    fuse_rope=False,
):
    """Run one action-expert layer on the suffix with AdaRMS conditioning.

    This is where π0.5 diverges from π0. Both norms are
    :class:`Pi05AdaRMSNorm` and each returns a gate that scales its sublayer's
    contribution to the residual stream.
    """
    layer = gemma_expert.model.layers[layer_idx]
    gated_residual = getattr(layer, "_lite_gated_residual", _gated_residual)

    residual = hidden_states
    x, gate = layer.input_layernorm(
        hidden_states, adarms_cond, modulation=None if modulations is None else modulations[0]
    )
    x = _match(x, layer.self_attn.q_proj)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    packed_qkv = getattr(layer.self_attn, "_lite_expert_qkv", None)
    projections = (tuple(module(x) for module in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
                   if packed_qkv is None else packed_qkv.project(x))
    q, k_suf, v_suf = (value.view(hidden_shape).transpose(1, 2) for value in projections)

    # RoPE frequencies are shared between PaliGemma and the expert.
    cos, sin = gemma_expert.model.rotary_emb(v_suf, position_ids) if rotary is None else rotary
    if fuse_rope:
        from vvla_runtime.pointwise import rotary_pos_emb
        q, k_suf = rotary_pos_emb(q, k_suf, cos, sin)
    else:
        q, k_suf = apply_rotary_pos_emb(q, k_suf, cos, sin, unsqueeze_dim=1)

    split_attention = getattr(attention, "forward_split", None)
    if split_attention is not None:
        # Explicit opt-in suffix backend reads current separate KV after RoPE.
        # Prefix attention and the default concatenation path remain unchanged.
        att = split_attention(q, *prefix_kv, k_suf, v_suf, attention_metadata)
    else:
        # Concatenate cached prefix K/V (possibly different dtype) with suffix K/V.
        if attention_metadata is not None and "paged_cache" in attention_metadata.extra:
            # Write only this step's suffix; the attention kernel reads prefix KV
            # directly through its block table, without concatenating or gathering.
            k, v = k_suf, v_suf
        else:
            k_prefix, v_prefix = prefix_kv
            k = torch.cat([k_prefix.to(k_suf.dtype), k_suf], dim=2)
            v = torch.cat([v_prefix.to(v_suf.dtype), v_suf], dim=2)

        att = _attend(
            q,
            k,
            v,
            attention_mask,
            num_kv_groups=layer.self_attn.num_key_value_groups,
            scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
            attention=attention,
            attention_metadata=attention_metadata,
        )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    attention_output = layer.self_attn.o_proj(_match(att, layer.self_attn.o_proj))
    residual_adarms = getattr(layer, "_lite_residual_adarms", None)
    if residual_adarms is None:
        hidden_states = gated_residual(residual, attention_output, gate)
        residual = hidden_states
        x, gate = layer.post_attention_layernorm(
            hidden_states, adarms_cond, modulation=None if modulations is None else modulations[1]
        )
    else:
        # Independent opt-in: retain the FP32 residual for the following MLP
        # residual while fusing only this layer's attention-add and postnorm.
        residual, x, gate = residual_adarms(
            residual, attention_output, gate, adarms_cond,
            modulation=None if modulations is None else modulations[1],
        )
    return gated_residual(residual, layer.mlp(_match(x, layer.mlp.up_proj)), gate)


class PaliGemmaWithActionExpertPi05(nn.Module):
    """Dual-backbone transformer: PaliGemma (Gemma 2B) + AdaRMS expert (300M).

    Same two-mode dispatch as π0 (``prefix_only`` / ``suffix_only``), with one
    structural change: after building a stock ``GemmaForCausalLM`` expert, every
    norm in it is swapped for a :class:`Pi05AdaRMSNorm` carrying a ``dense``
    conditioning projection.

    Swapping in place — rather than subclassing ``GemmaModel`` as #4419 does —
    keeps the module tree, and therefore the checkpoint key layout, identical to
    the expert's stock layout apart from the norms themselves.
    """

    def __init__(self, vlm_config, action_expert_config, attention_backend="eager"):
        super().__init__()

        # PaliGemma prefix: identical to π0. It sees no timestep, so no AdaRMS.
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # The action expert doesn't embed tokens — it only consumes the
        # suffix action embeddings we feed in.
        self.gemma_expert.model.embed_tokens = None

        self.adarms_cond_dim = action_expert_config.width
        self._install_adarms_norms(action_expert_config)
        self.prefix_attention = None
        self.suffix_attention = None
        self.attention_backend = attention_backend
        self.paged_attention = attention_backend == "vllm_paged_fa2"
        if attention_backend in {"vllm_fa2", "vllm_paged_fa2", "flash_attn", "torch_fa2"}:
            if attention_backend in {"flash_attn", "torch_fa2"}:
                from vvla_pi05.lite.attention import make_pi05_flash_attention
            else:
                from vvla_pi05.models.pi05.attention_pi05 import make_pi05_flash_attention

            backend_options = {"backend": attention_backend} if attention_backend in {"flash_attn", "torch_fa2"} else {}
            self.prefix_attention = make_pi05_flash_attention(
                vlm_config.num_heads, vlm_config.num_kv_heads, vlm_config.head_dim, **backend_options
            )
            self.suffix_attention = make_pi05_flash_attention(
                action_expert_config.num_heads, action_expert_config.num_kv_heads, action_expert_config.head_dim, **backend_options
            )
        elif attention_backend != "eager":
            raise ValueError(f"Unsupported pi05 attention backend: {attention_backend}")

    def _install_adarms_norms(self, action_expert_config) -> None:
        """Replace every action-expert RMSNorm with a conditioned AdaRMS norm."""
        expert = self.gemma_expert.model
        eps = getattr(self.gemma_expert.config, "rms_norm_eps", 1e-6)
        width = action_expert_config.width
        for layer in expert.layers:
            layer.input_layernorm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)
            layer.post_attention_layernorm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)
        expert.norm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP vision tower + PaliGemma projector.

        The two steps are run explicitly rather than via
        ``PaliGemmaModel.get_image_features`` because that helper divides the
        projector output by ``sqrt(text_hidden_size)``. Being explicit keeps the
        scale unambiguous and matches π0 exactly (SigLIP is unchanged in π0.5).
        """
        # Shapes: pixel_values (B, 3, 224, 224) → SigLIP (B, 256, 1152)
        #                                      → projector (B, 256, 2048)
        vision = self.paligemma.model.vision_tower
        pixel_values = pixel_values.to(dtype=next(vision.parameters()).dtype)
        vision_outputs = vision(pixel_values)
        image_features = vision_outputs.last_hidden_state
        projected = self.paligemma.model.multi_modal_projector(image_features)
        # Isolate the vision precision experiment: the reference prefix
        # residual stream remains FP32 after concatenating image/language.
        output_dtype = getattr(self, "vision_output_dtype", None)
        return projected if output_dtype is None else projected.to(dtype=output_dtype)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens, returning the ``* sqrt(hidden)``-scaled embedding.

        The scaling location moved across transformers releases: at ≤5.3 the
        normalizer lives inside ``GemmaModel.forward`` (which we bypass), and at
        ≥5.4 ``GemmaTextScaledWordEmbedding`` self-applies it. Detect and avoid
        double-scaling.
        """
        embed_tokens = self.paligemma.model.language_model.embed_tokens
        lang_emb = embed_tokens(tokens)
        if getattr(embed_tokens, "embed_scale", None) is None:
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        return lang_emb

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        adarms_cond: torch.Tensor | None = None,
        attention_metadata=None,
        rotary=None,
        modulations=None,
        prefix_kv_only=False,
    ):
        """Dispatch to prefix_only / suffix_only and return
        ``([prefix_out, suffix_out], past_key_values_or_None)``.
        """
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        pali_lm = self.paligemma.model.language_model
        expert_lm = self.gemma_expert.model
        fuse_rope, fuse_residual_square = getattr(self, "_pointwise_fusion", (False, False))

        def layer_metadata(layer_idx, prefix):
            if not self.paged_attention:
                return attention_metadata
            if attention_metadata is None or "paged_cache" not in attention_metadata.extra:
                raise ValueError("Paged pi05 forward requires prepared request-owned KV metadata")
            from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

            return AttentionMetadata(extra={**attention_metadata.extra, "layer_index": layer_idx, "prefix": prefix})

        if inputs_embeds[1] is None:
            if prefix_kv_only and not use_cache:
                raise ValueError("prefix_kv_only requires use_cache=True")
            hidden_states = inputs_embeds[0]
            kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
            if self.prefix_attention is not None and attention_metadata is None:
                if self.attention_backend in {"flash_attn", "torch_fa2"}:
                    from vvla_pi05.lite.attention import prepare_pi05_attention_metadata
                else:
                    from vvla_pi05.models.pi05.attention_pi05 import prepare_pi05_attention_metadata

                attention_metadata = prepare_pi05_attention_metadata(attention_mask, hidden_states.shape[1])
            squared = None  # Request-local; never persists between observations/replays.
            for layer_idx in range(num_layers):
                layer_result = _compute_layer_prefix_only(
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    paligemma=self.paligemma,
                    attention=self.prefix_attention,
                    attention_metadata=layer_metadata(layer_idx, True),
                    rotary=rotary,
                    kv_only=prefix_kv_only and layer_idx == num_layers - 1,
                    fuse_rope=fuse_rope,
                    fuse_residual_square=fuse_residual_square,
                    squared=squared,
                )
                if fuse_residual_square:
                    hidden_states, kv, squared = layer_result
                else:
                    hidden_states, kv = layer_result
                kv_list.append(kv)
            if prefix_kv_only:
                return [None, None], kv_list
            hidden_states = pali_lm.norm(hidden_states)
            return [hidden_states, None], (kv_list if use_cache else None)

        if inputs_embeds[0] is not None:
            raise ValueError(
                "PaliGemmaWithActionExpertPi05.forward only supports prefix-only "
                "or suffix-only dispatch; got both inputs_embeds populated."
            )
        if prefix_kv_only:
            raise ValueError("prefix_kv_only is only valid for prefix encoding")
        if not isinstance(past_key_values, list):
            raise TypeError(
                "suffix_only forward expects past_key_values to be the "
                "list[(k, v)] produced by a previous prefix_only forward; "
                f"got {type(past_key_values)}"
            )
        hidden_states = inputs_embeds[1]
        if self.suffix_attention is not None and attention_metadata is None:
            if self.attention_backend in {"flash_attn", "torch_fa2"}:
                from vvla_pi05.lite.attention import prepare_pi05_attention_metadata
            else:
                from vvla_pi05.models.pi05.attention_pi05 import prepare_pi05_attention_metadata

            key_length = past_key_values[0][0].shape[-2] + hidden_states.shape[1]
            attention_metadata = prepare_pi05_attention_metadata(attention_mask, key_length)
        for layer_idx in range(num_layers):
            hidden_states = _compute_layer_suffix_only(
                layer_idx,
                hidden_states,
                past_key_values[layer_idx],
                attention_mask,
                position_ids,
                gemma_expert=self.gemma_expert,
                adarms_cond=adarms_cond,
                attention=self.suffix_attention,
                attention_metadata=layer_metadata(layer_idx, False),
                rotary=rotary,
                modulations=None if modulations is None else modulations[layer_idx],
                fuse_rope=fuse_rope,
            )
        hidden_states, _ = expert_lm.norm(
            hidden_states, adarms_cond, modulation=None if modulations is None else modulations[-1]
        )
        return [None, hidden_states], None


# ──────────────────────────────────────────────────────────────────────
# Main π0.5 Model
# ──────────────────────────────────────────────────────────────────────
class Pi05ForActionPrediction(nn.Module):
    """π0.5 VLA model for robot action prediction via flow matching.

    Inference flow:
      1. Embed prefix (images + language, where the language already carries the
         discretized state) → prefix tokens.
      2. Forward prefix through PaliGemma → layer-wise KV cache.
      3. For each denoising step ``t = 1.0, 1-dt, ..., 0``:
         a. Embed the timestep → an AdaRMS conditioning vector.
         b. Embed the suffix (action tokens only).
         c. Forward the suffix through the AdaRMS action expert.
         d. ``x_t = x_t + dt * v_t`` (Euler integration).
      4. Return ``x_0`` as the predicted action chunk.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Generic vLLM quant_config is unsupported for this HF-backed model; use an explicit execution_recipe")
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        self.max_state_dim = getattr(config, "max_state_dim", self.action_dim)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.vlm_width = vlm_config.width
        self.expert_width = expert_config.width

        # Dual backbone
        self.paligemma_with_expert = PaliGemmaWithActionExpertPi05(
            vlm_config, expert_config, getattr(config, "attention_backend", "eager")
        )

        # Action chunk projections.
        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        # π0.5 timestep MLP: (W → W → W) with SiLU, feeding AdaRMS.
        # π0 instead has action_time_mlp_{in,out} of shape (2W → W → W) because
        # it concatenates the time embedding onto the action embedding.
        # NOTE: there is deliberately **no** ``state_proj`` here — that is the
        # π0-only continuous-state path.
        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)
        self._inference_cache = Pi05InferenceCache()

    def invalidate_inference_caches(self, *, weights_changed=True):
        """Call after nonstandard weight writes; graph runners also observe this epoch."""
        self._inference_cache.clear()
        if weights_changed and getattr(self, "_execution_state", None) is not None:
            self._execution_state["status"] = "weights_invalidated"

    def optimization_flags(self):
        return tuple(
            getattr(self.config, name, False)
            for name in ("cache_empty_camera_features", "precompute_time_and_positions", "prefix_kv_only", "pack_prefix", "vision_bf16")
        )

    @torch.no_grad()
    def _prepare_inference_model_state(self, device_type):
        """Complete mutable-model lifecycle and packed-owner preparation.

        The fixed-checkpoint engine invokes this at its cold seal and explicit
        audit hooks. The ordinary model path invokes it on every request.
        """
        if getattr(self, "_execution_state", None) is not None:
            from vvla_pi05.models.pi05.precision_pi05 import validate_execution_recipe
            validate_execution_recipe(self)
        flags = self.optimization_flags()
        if torch.is_autocast_enabled(device_type):
            raise ValueError("Pi05 inference caches/graphs require explicit dtypes; autocast is unsupported")
        if any(flags) and self.training:
            raise ValueError("Pi05 inference optimizations require model.eval()")
        vision = self.paligemma_with_expert.paligemma.model.vision_tower
        if getattr(self, "_configured_vision_bf16", flags[4]) != flags[4]:
            raise ValueError("Vision precision changed: rebuild dtype layout and reload original weights before inference")
        if flags[4] and (next(vision.parameters()).dtype != torch.bfloat16
                         or getattr(self.paligemma_with_expert, "vision_output_dtype", None) != torch.float32):
            raise ValueError("vision_bf16 must be applied with _set_inference_dtype before weight loading")
        # Compute outside CUDA capture. Version checks cover normal in-place updates;
        # load_weights explicitly invalidates even when a loader uses .data.
        lifecycle = getattr(self, "_lite_lifecycle_signature_installation", None)
        native_lifecycle = getattr(self, "_lite_native_lifecycle_signature_installation", None)
        if native_lifecycle is not None:
            if lifecycle is not None:
                raise ValueError("Choose only one lifecycle signature collector")
            lifecycle = native_lifecycle
        request_identities = getattr(self, '_lite_request_identity_installation', None)
        if request_identities is not None and native_lifecycle is None:
            raise ValueError('Request identity lookup requires the complete native lifecycle collector')
        prefix_fp8 = getattr(self, "_lite_prefix_fp8_installation", None)
        prefix_inventory = getattr(self, "_lite_prefix_fp8_inventory_installation", None)
        if prefix_inventory is not None:
            # The opt-in owner consumes a freshly collected native v3 triple
            # inside a bounded callback. No retained signature is a freshness
            # proof; all other recipes keep the original path below.
            tensor_state, generation = prefix_inventory.prepare_context((
                flags, self.config.min_period, self.config.max_period,
                getattr(self.config, "prefix_pack_bucket_size", 64),
                torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
            ))
        elif lifecycle is None:
            tensor_state = (tuple(tensor_identity(p) for p in self.parameters()),
                            tuple(tensor_identity(b) for b in self.buffers()))
        else:
            # This still reads all live identity fields on every request and
            # additionally records module/parameter/buffer registration edges.
            tensor_state = tuple(
                lifecycle.signature(previous_full_signature=self._inference_cache.signature)
                if native_lifecycle is not None else lifecycle.signature()
            )
        if prefix_inventory is None:
            signature = (
                flags,
                self.config.min_period,
                self.config.max_period,
                getattr(self.config, "prefix_pack_bucket_size", 64),
                torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32,
                *tensor_state,
            )
            generation = self._inference_cache.refresh(signature)
            if prefix_fp8 is not None:
                prefix_fp8.prepare(generation)
        prefix_nvfp4 = getattr(self, "_lite_prefix_nvfp4_installation", None)
        if prefix_nvfp4 is not None:
            if prefix_fp8 is not None:
                raise RuntimeError("Conflicting active prefix precision installations")
            prefix_nvfp4.prepare(generation)
        vision_qkv = getattr(self, "_lite_vision_qkv_installation", None)
        if vision_qkv is not None:
            vision_qkv.prepare(generation)
        vision_fc2 = getattr(self, "_lite_vision_fc2_cublaslt_installation", None)
        if vision_fc2 is not None:
            vision_fc2.prepare(generation)
        vision_fp8 = getattr(self, "_lite_vision_fp8_installation", None)
        if vision_fp8 is not None:
            vision_fp8.prepare(generation)
        # Reuse only this call's freshly collected identities, while the native
        # owner lock is held. Never put the scope into a retained graph context.
        from contextlib import nullcontext
        identity_scope = (nullcontext(None) if request_identities is None else
                          request_identities.prepare_scope(tensor_state, generation))
        with identity_scope as identity_lookup:
            prepare_kwargs = {} if identity_lookup is None else {'identity_lookup': identity_lookup}
            packed_qkv = getattr(self, "_lite_expert_qkv_installation", None)
            if packed_qkv is not None:
                packed_qkv.prepare(generation, **prepare_kwargs)
            expert_mlp = getattr(self, "_lite_expert_mlp_installation", None)
            if expert_mlp is not None:
                expert_mlp.prepare(generation, **prepare_kwargs)
            expert_cublaslt = getattr(self, "_lite_expert_cublaslt_installation", None)
            if expert_cublaslt is not None:
                expert_cublaslt.prepare(generation, **prepare_kwargs)
        residual_adarms = getattr(self, "_lite_residual_adarms_installation", None)
        if residual_adarms is not None:
            residual_adarms.prepare(generation)
        dynamic_fp8 = getattr(self, "_lite_dynamic_fp8_installation", None)
        if dynamic_fp8 is not None:
            dynamic_fp8.prepare(generation)
        return flags, generation, vision_fp8

    @torch.no_grad()
    def prepare_inference_context(self, images, image_masks, lang_masks, num_steps, *, positions=True,
                                  prepared_layout=None):
        """Validate model state and prepare per-request inference constants.

        ``prepared_layout`` is an internal Lite optimization: its processor
        constructs masks on the host and owns the corresponding device inputs.
        It may reuse an immutable, previously validated packing/empty-camera
        template. Mutable-model lifecycle checks run per request; the explicit
        private fixed-checkpoint engine seals those checks at construction/audit.
        """
        fixed_engine = getattr(self, "_lite_fixed_checkpoint_engine", None)
        batch_engine = getattr(self, "_lite_batch_checkpoint_engine", None)
        if batch_engine is not None:
            from vvla_pi05.lite.batch_checkpoint_engine import BatchSealedModelState
            if type(batch_engine) is not BatchSealedModelState or fixed_engine is not None:
                raise RuntimeError("Invalid or conflicting B32 fixed checkpoint installation")
            flags, generation, vision_fp8 = batch_engine.prepare_model_state(self, lang_masks.device.type)
        else:
            if fixed_engine is not None:
                raise RuntimeError("Historical fixed-checkpoint owners are outside the release runtime")
            flags, generation, vision_fp8 = Pi05ForActionPrediction._prepare_inference_model_state(self, lang_masks.device.type)
        if prepared_layout is None:
            packing = self.prepare_prefix_packing(image_masks, lang_masks) if flags[3] else None
            pattern = tuple(
                flags[0] and not bool(mask.any()) and bool((img == -1).all())
                for img, mask in zip(images, image_masks, strict=True)
            )
        else:
            if set(prepared_layout) != {"packing", "empty_pattern"}:
                raise ValueError("Prepared layout must contain packing and empty_pattern")
            packing, pattern = prepared_layout["packing"], prepared_layout["empty_pattern"]
            if (not isinstance(pattern, tuple) or len(pattern) != len(images)
                    or any(type(value) is not bool for value in pattern)
                    or (not flags[0] and any(pattern))):
                raise ValueError("Prepared empty-camera pattern does not match inference flags")
            if flags[3] != (packing is not None):
                raise ValueError("Prepared prefix packing does not match inference flags")
            if packing is not None:
                if set(packing) != {"indices", "pads", "positions"}:
                    raise ValueError("Invalid prepared prefix packing fields")
                shape = packing["pads"].shape
                if (len(shape) != 2 or shape[0] != lang_masks.shape[0] or shape[1] < 1
                        or any(t.shape != shape or t.device != lang_masks.device for t in packing.values())
                        or packing["pads"].dtype != torch.bool
                        or packing["indices"].dtype != torch.int64
                        or packing["positions"].dtype != torch.int64):
                    raise ValueError("Invalid prepared prefix packing geometry")
        features = []
        for img, mask, eligible in zip(images, image_masks, pattern, strict=True):
            if not eligible:
                features.append(None)
                continue
            key = (tuple(img.shape), img.dtype, img.device)
            features.append(
                self._inference_cache.get_or_create(
                    self._inference_cache.empty_features,
                    key,
                    (lambda img=img, mask=mask: vision_fp8.encode_empty_camera(img, mask))
                    if vision_fp8 is not None else
                    (lambda img=img: self.paligemma_with_expert.embed_image(img).detach()),
                )
            )
        schedule = None
        if flags[1]:
            key = (lang_masks.shape[0], num_steps, lang_masks.device)

            def build_schedule():
                entries = []
                expert = self.paligemma_with_expert.gemma_expert.model
                for step in range(num_steps):
                    # Preserve reference Python arithmetic, dtype, and original B.
                    t = torch.full(
                        (lang_masks.shape[0],),
                        1.0 + step * (-1.0 / num_steps),
                        dtype=torch.float32,
                        device=lang_masks.device,
                    )
                    cond = self.embed_timestep(t)
                    per_layer = tuple(
                        tuple(
                            norm.dense(cond.to(norm.dense.weight.dtype))
                            for norm in (layer.input_layernorm, layer.post_attention_layernorm)
                        )
                        for layer in expert.layers
                    )
                    final = expert.norm.dense(cond.to(expert.norm.dense.weight.dtype))
                    entries.append((cond, (*per_layer, final)))
                return tuple(entries)

            schedule = self._inference_cache.get_or_create(self._inference_cache.time_schedules, key, build_schedule)
        return {
            "token": (flags, generation),
            "empty_pattern": pattern,
            "empty_features": tuple(features),
            "schedule": schedule,
            "packing": packing,
            "positions": self.prepare_position_context(image_masks, lang_masks, packing) if flags[1] and positions else None,
        }

    def prepare_prefix_packing(self, image_masks, lang_masks):
        vision = self.paligemma_with_expert.paligemma.config.vision_config
        tokens = (vision.image_size // vision.patch_size) ** 2
        return prepare_prefix_packing(image_masks, lang_masks, tokens, getattr(self.config, "prefix_pack_bucket_size", 64))

    @torch.no_grad()
    def prepare_position_context(self, image_masks, lang_masks, prefix_packing=None):
        if not self.optimization_flags()[1]:
            return None
        if prefix_packing is None and self.optimization_flags()[3]:
            prefix_packing = self.prepare_prefix_packing(image_masks, lang_masks)
        inner = self.paligemma_with_expert
        vision = inner.paligemma.config.vision_config
        n = (vision.image_size // vision.patch_size) ** 2
        pads = (torch.cat([m[:, None].expand(-1, n) for m in image_masks] + [lang_masks], dim=1)
                if prefix_packing is None else prefix_packing["pads"])
        prefix_positions = (torch.cumsum(pads, dim=1) - 1
                            if prefix_packing is None else prefix_packing["positions"])
        suffix_pads = torch.ones(pads.shape[0], self.action_horizon, dtype=torch.bool, device=pads.device)
        suffix_att = (torch.arange(self.action_horizon, device=pads.device) == 0)[None, :].expand(pads.shape[0], -1)
        suffix_positions = pads.sum(-1)[:, None] + suffix_pads.cumsum(-1) - 1
        suffix_mask = torch.cat(
            [pads[:, None, :].expand(-1, self.action_horizon, -1), make_att_2d_masks(suffix_pads, suffix_att)], dim=2
        )
        result = {
            "prefix_positions": prefix_positions,
            "suffix_positions": suffix_positions,
            "prefix_mask": prepare_attention_masks_4d(pads[:, :, None] & pads[:, None, :]),
            "suffix_mask": prepare_attention_masks_4d(suffix_mask),
        }
        # Gemma's rotary helper reads x's dtype/device, not its values. Keep its
        # original sequence geometry and use each backbone's own rotary module.
        for name, lm, pos in (
            ("prefix", inner.paligemma.model.language_model, prefix_positions),
            ("suffix", inner.gemma_expert.model, suffix_positions),
        ):
            if getattr(lm.rotary_emb, "rope_type", "default") != "default":
                raise ValueError("Position precomputation currently supports default Gemma RoPE only")
            attn = lm.layers[0].self_attn
            probe = torch.empty(
                pads.shape[0],
                attn.k_proj.out_features // attn.head_dim,
                pos.shape[1],
                attn.head_dim,
                device=pads.device,
                dtype=attn.v_proj.weight.dtype,
            )
            result[name + "_rotary"] = lm.rotary_emb(probe, pos)
        return result

    # ── Prefix embedding ─────────────────────────────────────────────
    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        empty_features=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prefix embeddings, per-token padding mask, and AR mask.

        Prefix tokens form ``[img_cam_0..., ..., lang_tokens...]`` with fully
        bidirectional attention (all-zero ``att_masks``). Identical to π0 — the
        state is inside ``lang_tokens``, so nothing here changes shape-wise.

        Cameras are embedded one at a time; each call is ``(B, 3, 224, 224)``.
        The number of slots is fixed by ``config.max_cameras`` for the deployed
        model; missing cameras occupy their slot with a false image mask.
        """
        num_views = len(images)
        if len(image_masks) != num_views:
            raise ValueError(
                f"images and image_masks must contain the same number of views, got {num_views} and {len(image_masks)}."
            )
        max_cameras = int(self.config.max_cameras)
        if num_views != max_cameras:
            raise ValueError(f"Expected exactly max_cameras={max_cameras} image views, got {num_views}.")

        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for index, (img, img_mask) in enumerate(zip(images, image_masks)):
            cached = None if empty_features is None else empty_features[index]
            img_emb = self.paligemma_with_expert.embed_image(img) if cached is None else cached
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.zeros(len(att_masks), dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)

        return embs, pad_masks, att_masks

    # ── Timestep + suffix embedding ──────────────────────────────────
    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Timestep → AdaRMS conditioning vector ``(B, expert_width)``.

        ``silu(time_mlp_out(silu(time_mlp_in(sinusoid(t)))))``. The trailing
        SiLU is part of the reference implementation — dropping it is a silent
        numerical error, not a crash.
        """
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=getattr(self.config, "min_period", 4e-3),
            max_period=getattr(self.config, "max_period", 4.0),
            device=timestep.device,
        ).to(dtype=model_dtype)
        time_cond = self.time_mlp_in(time_emb)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        return F.silu(time_cond)

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        time_cond=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the suffix: **action tokens only**, plus the AdaRMS condition.

        π0's suffix is ``[state_token, action_tokens×H]`` with an AR mask of
        ``[1, 1, 0...]``. π0.5 has no state token, so the suffix is
        ``[action_tokens×H]`` and the mask is ``[1] + [0]*(H-1)``: the first
        action token opens a causal block and the rest attend bidirectionally
        within it.
        """
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        if time_cond is None:
            time_cond = self.embed_timestep(timestep)
        action_emb = self.action_in_proj(noisy_actions)  # (B, H, W)

        bsize, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_len, dtype=torch.bool, device=action_emb.device)
        att_masks = (
            (torch.arange(self.action_horizon, device=action_emb.device) == 0)
            .to(action_emb.dtype)[None, :]
            .expand(bsize, -1)
        )
        return action_emb, pad_masks, att_masks, time_cond

    # ── Denoising step ───────────────────────────────────────────────
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        attention_metadata=None,
        position_context=None,
        time_entry=None,
    ) -> torch.Tensor:
        """Apply one flow-matching denoising step: predict ``v_t`` from ``x_t``.

        Signature differs from π0's by exactly one argument: no ``state``.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, time_cond = self.embed_suffix(
            x_t, timestep, time_cond=None if time_entry is None else time_entry[0]
        )

        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        if position_context is None:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        else:
            position_ids = position_context["suffix_positions"]
            full_att_2d_masks_4d = position_context["suffix_mask"]

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=time_cond,
            attention_metadata=attention_metadata,
            rotary=None if position_context is None else position_context["suffix_rotary"],
            modulations=None if time_entry is None else time_entry[1],
        )

        # Every suffix token is an action token in π0.5 (no state token to drop).
        suffix_out = outputs_embeds[1][:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    # ── Full action generation ───────────────────────────────────────
    def prepare_attention_metadata(self, image_masks, lang_masks, *, fixed_capacity=False, paged_cache=None, prefix_packing=None):
        """Prepare request-local FA2 indices once, outside any graph capture."""
        if self.paligemma_with_expert.prefix_attention is None:
            return None, None
        if prefix_packing is None and self.optimization_flags()[3]:
            prefix_packing = self.prepare_prefix_packing(image_masks, lang_masks)
        if self.paligemma_with_expert.attention_backend in {"flash_attn", "torch_fa2"}:
            from vvla_pi05.lite.attention import prepare_pi05_attention_metadata, prepare_pi05_static_metadata
        else:
            from vvla_pi05.models.pi05.attention_pi05 import prepare_pi05_attention_metadata, prepare_pi05_static_metadata

        vision = self.paligemma_with_expert.paligemma.config.vision_config
        image_tokens = (vision.image_size // vision.patch_size) ** 2
        pads = (torch.cat([m[:, None].expand(-1, image_tokens) for m in image_masks] + [lang_masks], dim=1)
                if prefix_packing is None else prefix_packing["pads"])
        suffix_keys = torch.cat(
            [pads, torch.ones(pads.shape[0], self.action_horizon, device=pads.device, dtype=torch.bool)], dim=1
        )
        if fixed_capacity or self.paligemma_with_expert.paged_attention:
            queries = torch.ones(pads.shape[0], self.action_horizon, device=pads.device, dtype=torch.bool)
            metadata = (
                prepare_pi05_static_metadata(pads, pads),
                prepare_pi05_static_metadata(queries, suffix_keys, all_queries_valid=True),
            )
            if self.paligemma_with_expert.paged_attention:
                from vvla_pi05.models.pi05.paged_pi05 import Pi05PagedKVCache

                lm = self.paligemma_with_expert.paligemma.config.text_config
                if paged_cache is None:
                    paged_cache = Pi05PagedKVCache(
                        pads,
                        self.action_horizon,
                        lm.num_hidden_layers,
                        lm.num_key_value_heads,
                        lm.head_dim,
                        self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.k_proj.weight.dtype,
                        block_size=self.config.paged_block_size,
                    )
                else:
                    paged_cache.update(pads)
                for item in metadata:
                    item.extra["paged_cache"] = paged_cache
            return metadata
        prefix = pads[:, :, None] & pads[:, None, :]
        suffix = suffix_keys[:, None, :].expand(-1, self.action_horizon, -1)
        return (
            prepare_pi05_attention_metadata(prepare_attention_masks_4d(prefix), pads.shape[1]),
            prepare_pi05_attention_metadata(prepare_attention_masks_4d(suffix), suffix_keys.shape[1]),
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        attention_metadata=None,
        inference_context=None,
    ) -> torch.Tensor:
        """Generate an action chunk via iterative flow-matching denoising.

        Convention: ``t=1`` is noise, ``t=0`` is the target — opposite of the
        published π0 paper but matching both OpenPI and LeRobot.

        Takes no ``state``: π0.5's state rides inside ``lang_tokens``.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if noise is None:
            noise_shape = (self.action_horizon, self.action_dim)
            if isinstance(generator, list):
                if len(generator) != bsize:
                    raise ValueError(f"Expected {bsize} generators, got {len(generator)}.")
                noise = torch.stack(
                    [
                        torch.randn(noise_shape, dtype=torch.float32, device=device, generator=item)
                        for item in generator
                    ]
                )
            else:
                noise = torch.randn(
                    bsize,
                    *noise_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )

        if inference_context is None:
            inference_context = self.prepare_inference_context(images, image_masks, lang_masks, num_steps)
        packing = inference_context.get("packing")
        if attention_metadata is None:
            attention_metadata = self.prepare_attention_metadata(image_masks, lang_masks, prefix_packing=packing)
        if packing is not None and attention_metadata[0] is not None:
            if attention_metadata[0].extra["q_length"] != packing["pads"].shape[1]:
                raise ValueError("Packed prefix and attention metadata geometry disagree")
        position_context = inference_context["positions"]

        # 1. Prefix embeddings + mask building.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks, empty_features=inference_context["empty_features"]
        )
        if packing is not None:
            prefix_embs = pack_prefix(prefix_embs, packing)
            prefix_pad_masks = packing["pads"]
            prefix_att_masks = torch.zeros_like(prefix_pad_masks)
        if position_context is None:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        else:
            prefix_position_ids = position_context["prefix_positions"]
            prefix_att_2d_masks_4d = position_context["prefix_mask"]

        # 2. Forward prefix through PaliGemma LM, producing a list[(k, v)] cache.
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            attention_metadata=attention_metadata[0],
            rotary=None if position_context is None else position_context["prefix_rotary"],
            prefix_kv_only=self.optimization_flags()[2],
        )

        # 3. Euler-integrated denoising from t=1 down to t=0.
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
                attention_metadata=attention_metadata[1],
                position_context=position_context,
                time_entry=None if inference_context["schedule"] is None else inference_context["schedule"][step],
            )
            x_t = x_t + dt * v_t
        return x_t

    # ── Weight loading ───────────────────────────────────────────────
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        strict: bool = True,
    ):
        """Load and audit a LeRobot π0.5 safetensors checkpoint.

        Same remap rules as π0 (strip the ``model.`` prefix, flatten→nested
        PaliGemma submodules, tied ``lm_head`` → ``embed_tokens``, version-robust
        SigLIP nesting), plus two π0.5-specific ones:

          * ``action_time_mlp_{in,out}`` → ``time_mlp_{in,out}``: some
            checkpoints were exported under the π0 parameter names.
          * ``state_proj.*`` is reported, not silently dropped. A π0.5
            checkpoint should not contain it; its presence usually means a π0
            checkpoint was pointed at the π0.5 model class, which would
            otherwise run happily with a randomly-initialized action expert.

        The action-expert norms are AdaRMS here, so they expose ``dense.weight``
        / ``dense.bias`` and no plain ``weight``. A checkpoint that carries a
        plain expert-norm ``weight`` is a π0-shaped checkpoint; that too is
        rejected rather than skipped. ``strict=False`` exists only for focused
        remapping unit tests that intentionally provide a partial state dict;
        the serving path always uses the strict default.
        """
        # Invalidate even a partial/failed load; custom loaders may use .data.
        self.invalidate_inference_caches()
        if getattr(self, "_execution_state", None) is not None:
            # Even .data loaders must never leave a valid-looking packed copy.
            self._execution_state["status"] = "weights_reloaded"
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        _PALIGEMMA_SUBMODULES = ("vision_tower", "multi_modal_projector", "language_model")
        _EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."

        def _remap(name: str) -> str:
            # Strip the leading "model." that LeRobot's PI05Policy wrapper adds.
            if name.startswith("model."):
                name = name[len("model.") :]

            # π0-style timestep MLP names → π0.5 names.
            if name.startswith("action_time_mlp_in."):
                name = "time_mlp_in." + name[len("action_time_mlp_in.") :]
            elif name.startswith("action_time_mlp_out."):
                name = "time_mlp_out." + name[len("action_time_mlp_out.") :]

            # Nested PaliGemma layout.
            for sub in _PALIGEMMA_SUBMODULES:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat) :]

            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

            return name

        def _fix_vision_tower(name: str) -> str:
            """Reconcile SigLIP nesting across transformers versions (≤5.3 wraps
            the encoder in ``vision_tower.vision_model.*``; ≥5.4 flattens it)."""
            vt = "paligemma_with_expert.paligemma.model.vision_tower."
            if not name.startswith(vt):
                return name
            rest = name[len(vt) :]
            if name in params_dict or name in buffers_dict:
                return name
            if rest.startswith("vision_model."):
                candidate = vt + rest[len("vision_model.") :]
            else:
                candidate = vt + "vision_model." + rest
            return candidate if (candidate in params_dict or candidate in buffers_dict) else name

        loaded = 0
        skipped: list[str] = []
        pi0_shaped: list[str] = []
        filled_params: set = set()

        for name, loaded_weight in weights:
            mapped = _fix_vision_tower(_remap(name))

            # Diagnose π0-shaped keys instead of dropping them quietly.
            is_expert_norm_weight = mapped.startswith(_EXPERT_PREFIX) and (
                mapped.endswith("input_layernorm.weight")
                or mapped.endswith("post_attention_layernorm.weight")
                or mapped == _EXPERT_PREFIX + "norm.weight"
            )
            if mapped.startswith("state_proj.") or is_expert_norm_weight:
                pi0_shaped.append(mapped)
                continue

            if mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            elif mapped in buffers_dict:
                buffers_dict[mapped].copy_(loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            else:
                skipped.append(mapped)

        # LeRobot stores PaliGemma's tied text embedding as lm_head.weight; keep
        # lm_head filled so the tied-weight state stays consistent.
        embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
        if embed_key in filled_params and lm_head_key in params_dict and lm_head_key not in filled_params:
            params_dict[lm_head_key].data.copy_(params_dict[embed_key].data)
            filled_params.add(lm_head_key)

        # Reverse audit: any model param that got no checkpoint tensor at all
        # would be running with random init.
        missing_params: list[str] = []
        for pname in params_dict:
            if pname in filled_params:
                continue
            if "rotary_emb" in pname or pname.endswith(".inv_freq"):
                continue
            missing_params.append(pname)

        parts: list[str] = []
        if pi0_shaped:
            parts.append(
                f"{len(pi0_shaped)} checkpoint key(s) are π0-shaped, not π0.5-shaped (first 5: {pi0_shaped[:5]})"
            )
        if skipped:
            parts.append(f"{len(skipped)} checkpoint key(s) had no model target (first 5: {skipped[:5]})")
        if missing_params:
            parts.append(f"{len(missing_params)} model param(s) received no weight (first 5: {missing_params[:5]})")

        if parts and strict:
            raise RuntimeError("Incomplete or incompatible π0.5 checkpoint: " + "; ".join(parts))
        if parts:
            logger.debug("π0.5 partial test load: %d tensors loaded — %s.", loaded, "; ".join(parts))
        else:
            logger.info("π0.5 load_weights: %d tensors loaded, 0 skipped, 0 missing.", loaded)
        return filled_params


EntryClass = Pi05ForActionPrediction
