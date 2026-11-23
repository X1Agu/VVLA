# SPDX-License-Identifier: Apache-2.0
"""B1 pi05: raw observation -> owned CPU physical actions, in one process."""

from copy import deepcopy
from dataclasses import asdict, fields
from functools import wraps
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch


# Transformers' no_init_weights patches process-global initialization functions.
# Serialize both ordinary and skipped Lite construction so these loaders cannot
# observe each other's temporary patch. Unrelated model constructors must also
# be kept outside this interval by the application.
_MODEL_CONSTRUCTION_LOCK = threading.Lock()

# Initialize all resources before any optional installation can fail. This also
# covers direct policy construction, not only the experiment controller's
# __new__/__init__ path. Foreign cuBLASLt plans require explicit retirement.
_POLICY_HANDLE_FIELDS = (
    "_adarms_installation", "_qkv_installation", "_mlp_installation",
    "_expert_cublaslt_installation", "_residual_installation", "_suffix_installation",
    "_suffix_fa2_installation", "_action_norm_cache", "_preserved_linears_installation",
    "_vision_installation", "_vision_qkv_installation", "_lifecycle_installation",
    "_native_lifecycle_installation", "_gpu_image_prepare_installation", "_residual_adarms_installation",
    "_request_identity_installation", "_packed_metadata_installation", "_vision_fc2_cublaslt_installation",
    "_prefix_fp8_installation", "_dynamic_fp8_installation",
)


def _construction_transaction(initialize):
    @wraps(initialize)
    def guarded(self, model, processor, *args, **kwargs):
        self.model, self.processor = model, processor
        self._lock, self._closed = threading.Lock(), False
        self.graph_runner = None
        for name in _POLICY_HANDLE_FIELDS:
            setattr(self, name, None)
        attention_flags = [(module, module.elide_identity_gathers) for module in model.modules()
                           if hasattr(module, "elide_identity_gathers")]
        try:
            initialize(self, model, processor, *args, **kwargs)
        except BaseException as error:
            try:
                self.close()
                for module, original in attention_flags:
                    module.elide_identity_gathers = original
            except BaseException as cleanup:
                # Preserve the construction failure. A failed retirement must
                # not be hidden or followed by unsafe descriptor destruction.
                error.add_note("Partial Lite policy cleanup failed: " + str(cleanup))
                error.pi05_policy=self
            raise
    return guarded


def _construct_pi05_model(config, *, skip_weight_init=False):
    from vvla_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction

    with _MODEL_CONSTRUCTION_LOCK:
        if not skip_weight_init:
            return Pi05ForActionPrediction(config)
        # The buffer constructors and tying behavior were audited for this
        # pinned release; do not silently extend that claim to another version.
        if package_metadata.version("transformers") != "5.19.0":
            raise RuntimeError("lite_skip_weight_init requires audited transformers==5.19.0")
        from transformers.initialization import no_init_weights

        with no_init_weights():
            model = Pi05ForActionPrediction(config)
        # no_init_weights also suppresses HF init_weights' tie_weights call.
        # Restore this alias before dtype conversion/device transfer. The expert
        # has already removed embed_tokens, so it must not be retied here.
        model.paligemma_with_expert.paligemma.tie_weights()
        return model


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_hash(tensor):
    """Actual contiguous tensor bytes; diagnostics only, outside timed calls."""
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _source_hash(paths):
    paths = [Path(p).resolve() for p in paths]
    root = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return _json_hash({str(path.relative_to(root)): _file_hash(path) for path in paths})


def _resolve_checkpoint(source, revision=None):
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=str(source), revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    ))


def _split_lite_overrides(config_overrides):
    overrides = dict(config_overrides or {})
    options = {}
    for name in ("elide_identity_gathers", "host_layout_cache", "fused_adarms", "skip_weight_init",
                 "fused_expert_qkv", "fused_expert_gate_up", "fused_expert_gelu",
                 "fused_gated_residual", "compile_sample_actions", "fused_adarms_reduction",
                 "compile_preserve_action_linears", "batched_vision", "fused_vision_qkv", "action_norm_cache",
                 "single_pass_lifecycle", "native_lifecycle", "gpu_uint8_prepare", "fused_residual_adarms",
                 "request_identity_lookup", "packed_input_metadata", "vision_fc2_cublaslt", "prefix_fp8"):
        enabled = overrides.pop("lite_" + name, False)
        if type(enabled) is not bool:
            raise ValueError(f"lite_{name} must be a boolean")
        options[name] = enabled
    options["native_lifecycle_variant"] = _validate_native_lifecycle_variant(
        overrides.pop("lite_native_lifecycle_variant", "v1"))
    options["suffix_attention"] = _validate_suffix_attention(overrides.pop("lite_suffix_attention", None))
    options["suffix_fa2"] = _validate_suffix_fa2(overrides.pop("lite_suffix_fa2", None))
    options["expert_cublaslt"] = _validate_expert_cublaslt(overrides.pop("lite_expert_cublaslt", None))
    options['batch_geometry'] = overrides.pop('lite_batch_geometry', None)
    options['dynamic_cublaslt'] = overrides.pop('lite_dynamic_cublaslt', None)
    if options['dynamic_cublaslt'] is not None and not isinstance(options['dynamic_cublaslt'],str):
        raise ValueError('lite_dynamic_cublaslt must be an explicit tactic asset path')
    if options['batch_geometry'] not in (None, 'sm89', 'sm120', 'portable'):
        raise ValueError('lite_batch_geometry must be sm89, sm120 or portable')
    return overrides, options


def _validate_native_lifecycle_variant(value):
    if type(value) is not str or value not in ("v1", "v2", "v3"):
        raise ValueError("lite_native_lifecycle_variant must be 'v1', 'v2' or 'v3'")
    return value


def _validate_suffix_attention(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"splits", "block_q"}:
        raise ValueError("lite_suffix_attention requires exactly splits and block_q, or None")
    from vvla_pi05.lite.suffix_attention import _validate_options
    _validate_options(value["splits"], value["block_q"])
    return dict(value)


def _validate_compilation_options(options, cuda_graph):
    if options.get('dynamic_cublaslt') is not None and (
            options.get('batch_geometry')!='sm120' or not options.get('fused_expert_qkv')
            or not options.get('fused_expert_gate_up') or not options.get('fused_expert_gelu')):
        raise ValueError('Dynamic Lt requires the complete SM120 batch and packed projection owners')
    if options.get('batch_geometry') is not None:
        if (not cuda_graph or not options.get('host_layout_cache') or options.get('batched_vision')
                or options.get('expert_cublaslt') is not None or options.get('suffix_fa2') is not None
                or options.get('vision_fc2_cublaslt')):
            raise ValueError('Batch geometry requires host layouts and its own camera/attention/Lt owners')
    if options.get("vision_fc2_cublaslt") and (not cuda_graph or not all(options.get(key) for key in
            ("batched_vision", "compile_sample_actions"))):
        raise ValueError("Vision fc2 cuBLASLt requires batched vision, compiled sampling and native CUDA graphs")
    variant = _validate_native_lifecycle_variant(options.get("native_lifecycle_variant", "v1"))
    if variant != "v1" and not options.get("native_lifecycle"):
        raise ValueError(f"lite_native_lifecycle_variant='{variant}' requires lite_native_lifecycle")
    if options.get("request_identity_lookup") and (not options.get("native_lifecycle") or not cuda_graph):
        raise ValueError("lite_request_identity_lookup requires lite_native_lifecycle and native CUDA graphs")
    if options.get("fused_residual_adarms") and not all(options.get(key) for key in
            ("fused_adarms", "fused_adarms_reduction", "fused_gated_residual")):
        raise ValueError("Residual-AdaRMS fusion requires full-reduction AdaRMS and gated residual")
    if options.get("native_lifecycle") and options.get("single_pass_lifecycle"):
        raise ValueError("Choose only one lifecycle signature collector")
    if options.get("expert_cublaslt") is not None:
        if not cuda_graph or not options["fused_expert_qkv"] or not options["fused_expert_gelu"]:
            raise ValueError("Expert cuBLASLt requires native CUDA graphs, expert QKV and fused expert GELU")
        packed = options["expert_cublaslt"]["profile"] == "g4_best_three"
        if options["fused_expert_gate_up"] != packed:
            raise ValueError("Expert cuBLASLt profile requires matching packed/separate gate-up")
    if options["suffix_fa2"] is not None and options["suffix_attention"] is not None:
        raise ValueError("Choose only one suffix attention candidate")
    if options["fused_adarms_reduction"] and not options["fused_adarms"]:
        raise ValueError("lite_fused_adarms_reduction requires lite_fused_adarms")
    if options["compile_preserve_action_linears"] and not options["compile_sample_actions"]:
        raise ValueError("lite_compile_preserve_action_linears requires compiled sampling")
    if options["compile_sample_actions"]:
        if not cuda_graph:
            raise ValueError("Compiled sampling requires native CUDA graphs")
        if options["suffix_attention"] is not None:
            raise ValueError("Compiled sampling is not validated with split suffix attention")


def _validate_suffix_fa2(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"splits"}:
        raise ValueError("lite_suffix_fa2 requires exactly splits, or None")
    from vvla_pi05.lite.suffix_fa2 import _validate_splits
    _validate_splits(value["splits"])
    return dict(value)


def _validate_expert_cublaslt(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"profile", "include_down"}:
        raise ValueError("lite_expert_cublaslt requires exactly profile and include_down, or None")
    from vvla_pi05.lite.cublaslt_runtime import profile_specs
    profile_specs(value["profile"], value["include_down"])
    return dict(value)


def load_lite_config(checkpoint_dir):
    """Consume already-materialized LeRobot empty camera declarations once.

    A saved LeRobot config may contain both empty_cameras=N and the N explicit
    input features created by its constructor. The standalone core intentionally
    rejects schema mutation. Validate that mutation is already represented, then
    disable only the redundant constructor count in a private dictionary.
    """
    from vvla_pi05.models.pi05.config import Pi05Config, load_lerobot_norm_stats

    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint config must be a JSON object")
    count = raw.get("empty_cameras", 0)
    if type(count) is not int or count < 0:
        raise ValueError("empty_cameras must be a nonnegative integer")
    adjustments = []
    if count:
        features = raw.get("input_features") or {}
        expected_keys = {f"observation.images.empty_camera_{i}" for i in range(count)}
        actual_keys = {key for key in features if key.startswith("observation.images.empty_camera_")}
        if actual_keys != expected_keys:
            raise ValueError("empty_cameras count does not match explicitly declared empty camera slots")
        resolution = raw.get("image_resolution", [224, 224])
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError("Empty camera validation requires [height,width] image_resolution")
        for key in sorted(expected_keys):
            feature = features[key]
            if (not isinstance(feature, dict) or feature.get("type") != "VISUAL"
                    or feature.get("shape") != [3, *resolution]):
                raise ValueError(f"Explicit empty camera {key} must be VISUAL with shape {[3, *resolution]}")
        if raw.get("image_feature_keys") is not None and not expected_keys.issubset(raw["image_feature_keys"]):
            raise ValueError("Explicit image_feature_keys omit declared empty camera slots")
        raw["empty_cameras"] = 0
        adjustments.append({
            "field": "empty_cameras", "original": count, "loader_value": 0,
            "reason": "constructor-generated empty slots already explicitly present in input_features",
            "validated_features": sorted(expected_keys),
        })
    config = Pi05Config.from_model_config(raw)
    if len(config.image_feature_keys or []) > config.max_cameras:
        raise ValueError("Declared camera features exceed max_cameras; refusing silent slot truncation")
    if config.norm_stats is None:
        stats = load_lerobot_norm_stats(checkpoint_dir)
        if stats:
            config.norm_stats = stats
            config._validate_norm_stats()
    return config, {
        "original_config_sha256": _file_hash(path), "schema_normalization": adjustments,
        "checkpoint_files_modified": False,
    }


class Pi05LitePolicy:
    """One serial policy owner; no engine, scheduler, IPC, or network overhead.

    ``infer(obs, seed)`` uses a request-local CUDA generator. ``predict(obs,
    noise)`` consumes an explicit [1,H,32] FP32 noise tensor for cross-framework
    parity. Each result owns its CPU float32 [H,action_dim] action storage.
    """

    @_construction_transaction
    def __init__(self, model, processor, *, device, cuda_graph=True, max_graphs=16, provenance=None,
                 elide_identity_gathers=False, host_layout_cache=False, fused_adarms=False,
                 skip_weight_init=False, fused_expert_qkv=False,
                 fused_expert_gate_up=False, fused_expert_gelu=False,
                 fused_gated_residual=False, compile_sample_actions=False,
                 fused_adarms_reduction=False, suffix_attention=None,
                 compile_preserve_action_linears=False, batched_vision=False, fused_vision_qkv=False,
                 suffix_fa2=None, action_norm_cache=False, single_pass_lifecycle=False, native_lifecycle=False,
                 expert_cublaslt=None, gpu_uint8_prepare=False, fused_residual_adarms=False,
                 native_lifecycle_variant="v1", request_identity_lookup=False, packed_input_metadata=False,
                 vision_fc2_cublaslt=False, prefix_fp8=False, batch_geometry=None, dynamic_cublaslt=None,
                 experimental_precision=None):
        self.model = model.eval()
        self.processor = processor
        self.config = processor.config
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._closed = False
        self._calls = 0
        self._episode_id = None
        self._provenance = dict(provenance or {})
        if type(elide_identity_gathers) is not bool:
            raise ValueError("elide_identity_gathers must be a boolean")
        if type(host_layout_cache) is not bool:
            raise ValueError("host_layout_cache must be a boolean")
        if host_layout_cache and not cuda_graph:
            raise ValueError("Host layout cache currently requires native CUDA graphs")
        if type(fused_adarms) is not bool:
            raise ValueError("fused_adarms must be a boolean")
        if type(skip_weight_init) is not bool:
            raise ValueError("skip_weight_init must be a boolean")
        if type(fused_expert_qkv) is not bool:
            raise ValueError("fused_expert_qkv must be a boolean")
        for name, value in (("fused_expert_gate_up", fused_expert_gate_up),
                            ("fused_expert_gelu", fused_expert_gelu),
                            ("fused_gated_residual", fused_gated_residual),
                            ("compile_sample_actions", compile_sample_actions),
                            ("fused_adarms_reduction", fused_adarms_reduction),
                            ("compile_preserve_action_linears", compile_preserve_action_linears),
                            ("batched_vision", batched_vision), ("fused_vision_qkv", fused_vision_qkv),
                            ("action_norm_cache", action_norm_cache), ("single_pass_lifecycle", single_pass_lifecycle),
                            ("native_lifecycle", native_lifecycle), ("gpu_uint8_prepare", gpu_uint8_prepare),
                            ("fused_residual_adarms", fused_residual_adarms),
                            ("request_identity_lookup", request_identity_lookup),
                            ("packed_input_metadata", packed_input_metadata),
                            ("vision_fc2_cublaslt", vision_fc2_cublaslt)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        suffix_attention = _validate_suffix_attention(suffix_attention)
        expert_cublaslt = _validate_expert_cublaslt(expert_cublaslt)
        suffix_fa2 = _validate_suffix_fa2(suffix_fa2)
        if cuda_graph and self.device.type != "cuda":
            raise ValueError("cuda_graph=True requires a CUDA device")
        if cuda_graph and (type(max_graphs) is not int or max_graphs < 1):
            raise ValueError("max_graphs must be a positive integer")
        self._lite_optimizations = {"elide_identity_gathers": elide_identity_gathers,
                                    "host_layout_cache": host_layout_cache, "fused_adarms": fused_adarms,
                                    "skip_weight_init": skip_weight_init, "fused_expert_qkv": fused_expert_qkv,
                                    "fused_expert_gate_up": fused_expert_gate_up,
                                    "fused_expert_gelu": fused_expert_gelu,
                                    "fused_gated_residual": fused_gated_residual,
                                    "compile_sample_actions": compile_sample_actions,
                                    "fused_adarms_reduction": fused_adarms_reduction,
                                    "suffix_attention": suffix_attention,
                                    "compile_preserve_action_linears": compile_preserve_action_linears,
                                    "batched_vision": batched_vision, "fused_vision_qkv": fused_vision_qkv,
                                    "suffix_fa2": suffix_fa2, "action_norm_cache": action_norm_cache,
                                    "single_pass_lifecycle": single_pass_lifecycle, "native_lifecycle": native_lifecycle,
                                    "native_lifecycle_variant": native_lifecycle_variant,
                                    "expert_cublaslt": expert_cublaslt, "gpu_uint8_prepare": gpu_uint8_prepare,
                                    "fused_residual_adarms": fused_residual_adarms,
                                    "request_identity_lookup": request_identity_lookup,
                                    "packed_input_metadata": packed_input_metadata,
                                    "vision_fc2_cublaslt": vision_fc2_cublaslt,
                                    "prefix_fp8": prefix_fp8}
        if type(prefix_fp8) is not bool or (prefix_fp8 and (not cuda_graph or not compile_sample_actions)):
            raise ValueError('Historical prefix FP8 requires a boolean switch and compiled native CUDA graphs')
        _validate_compilation_options(self._lite_optimizations, cuda_graph)
        if elide_identity_gathers:
            from vvla_pi05.lite.attention import Pi05FlashAttention
            attention_modules = [module for module in model.modules() if isinstance(module, Pi05FlashAttention)]
            if not attention_modules:
                raise ValueError("Identity gather elision requires standalone FA2 attention")
            for module in attention_modules:
                module.elide_identity_gathers = True
        self._adarms_installation = None
        if fused_adarms:
            from vvla_pi05.lite.fused_adarms import install_fused_adarms
            self._adarms_installation = install_fused_adarms(model, fuse_reduction=fused_adarms_reduction)
            self._lite_optimizations["fused_adarms_manifest"] = self._adarms_installation.manifest()
        self._qkv_installation = None
        if fused_expert_qkv:
            from vvla_pi05.lite.fused_qkv import install_expert_qkv
            self._qkv_installation = install_expert_qkv(model)
            self._lite_optimizations["fused_expert_qkv_manifest"] = self._qkv_installation.manifest()
        self._mlp_installation = None
        if fused_expert_gate_up or fused_expert_gelu:
            from vvla_pi05.lite.fused_expert_mlp import install_expert_mlp
            self._mlp_installation = install_expert_mlp(
                model, concatenate=fused_expert_gate_up, fuse_gelu_mul=fused_expert_gelu,
            )
            self._lite_optimizations["fused_expert_mlp_manifest"] = self._mlp_installation.manifest()
        self._expert_cublaslt_installation = None
        if expert_cublaslt is not None:
            from vvla_pi05.lite.expert_cublaslt import install_expert_cublaslt
            self._expert_cublaslt_installation = install_expert_cublaslt(model, **expert_cublaslt)
            self._lite_optimizations["expert_cublaslt_manifest"] = self._expert_cublaslt_installation.manifest()
        self._residual_installation = None
        if fused_gated_residual:
            from vvla_pi05.lite.fused_gated_residual import install_gated_residual
            self._residual_installation = install_gated_residual(model)
            self._lite_optimizations["fused_gated_residual_manifest"] = self._residual_installation.manifest()
        if fused_residual_adarms:
            from vvla_pi05.lite.fused_residual_adarms import install_residual_adarms
            self._residual_adarms_installation = install_residual_adarms(model)
            self._lite_optimizations["fused_residual_adarms_manifest"] = self._residual_adarms_installation.manifest()
        self._suffix_installation = None
        if suffix_attention is not None:
            from vvla_pi05.lite.suffix_attention import install_suffix_attention
            self._suffix_installation = install_suffix_attention(model, **suffix_attention)
            self._lite_optimizations["suffix_attention_manifest"] = self._suffix_installation.manifest()
        self._suffix_fa2_installation = None
        if suffix_fa2 is not None:
            from vvla_pi05.lite.suffix_fa2 import install_suffix_fa2
            self._suffix_fa2_installation = install_suffix_fa2(model, **suffix_fa2)
            self._lite_optimizations["suffix_fa2_manifest"] = self._suffix_fa2_installation.manifest()
        self._action_norm_cache = None
        if action_norm_cache:
            from vvla_pi05.lite.action_norm_cache import ActionNormDeviceCache
            self._action_norm_cache = ActionNormDeviceCache(processor)
            self._lite_optimizations["action_norm_cache_manifest"] = self._action_norm_cache.manifest()
        if gpu_uint8_prepare:
            from vvla_pi05.lite.gpu_image_prepare import install_gpu_uint8_prepare
            self._gpu_image_prepare_installation = install_gpu_uint8_prepare(processor)
            self._lite_optimizations["gpu_uint8_prepare_manifest"] = self._gpu_image_prepare_installation.manifest()
        self._packed_metadata_installation = None
        if packed_input_metadata:
            from vvla_pi05.lite.packed_input_metadata import install_packed_input_metadata
            self._packed_metadata_installation = install_packed_input_metadata(processor)
            self._lite_optimizations["packed_input_metadata_manifest"] = self._packed_metadata_installation.manifest()
        self._preserved_linears_installation = None
        if compile_preserve_action_linears:
            from vvla_pi05.lite.compilation import install_preserved_action_linears
            self._preserved_linears_installation = install_preserved_action_linears(model)
            self._lite_optimizations["preserved_action_linears_manifest"] = self._preserved_linears_installation.manifest()
        self._vision_installation = None
        self._vision_qkv_installation = None
        if fused_vision_qkv:
            from vvla_pi05.lite.fused_vision_qkv import install_vision_qkv
            self._vision_qkv_installation = install_vision_qkv(model)
            self._lite_optimizations["fused_vision_qkv_manifest"] = self._vision_qkv_installation.manifest()
        if batched_vision:
            from vvla_pi05.lite.batched_vision import install_batched_vision
            self._vision_installation = install_batched_vision(model)
            self._lite_optimizations["batched_vision_manifest"] = self._vision_installation.manifest()
        if vision_fc2_cublaslt:
            from vvla_pi05.lite.vision_fc2_cublaslt import install_vision_fc2_cublaslt
            self._vision_fc2_cublaslt_installation = install_vision_fc2_cublaslt(model)
            self._lite_optimizations["vision_fc2_cublaslt_manifest"] = self._vision_fc2_cublaslt_installation.manifest()
        if single_pass_lifecycle:
            from vvla_pi05.lite.lifecycle import install_lifecycle_signature
            self._lifecycle_installation = install_lifecycle_signature(model)
            self._lite_optimizations["single_pass_lifecycle_manifest"] = self._lifecycle_installation.manifest()
        self._native_lifecycle_installation = None
        if native_lifecycle:
            from vvla_pi05.lite.native_lifecycle import install_native_lifecycle_signature, load_native_lifecycle
            collector = load_native_lifecycle(variant=native_lifecycle_variant)
            self._native_lifecycle_installation = install_native_lifecycle_signature(model, collector=collector)
            self._lite_optimizations["native_lifecycle_manifest"] = self._native_lifecycle_installation.manifest()
        if request_identity_lookup:
            from vvla_pi05.lite.request_identity import install_request_identity_lookup
            self._request_identity_installation = install_request_identity_lookup(model)
            self._lite_optimizations["request_identity_lookup_manifest"] = self._request_identity_installation.manifest()
        if batch_geometry is not None:
            if batch_geometry not in ('sm89', 'sm120', 'portable'):
                raise ValueError('Unknown batch geometry')
            if not cuda_graph or not host_layout_cache or batched_vision:
                raise ValueError('Batch geometry requires native graphs, host layouts and its own camera owner')
            from .dynamic_batch_geometry import install_dynamic_geometry
            self._vision_installation, self._suffix_fa2_installation = install_dynamic_geometry(
                model, suffix=batch_geometry=='sm120')
            self._lite_optimizations.update(batch_geometry=batch_geometry,
                batched_vision_manifest=self._vision_installation.manifest())
            if self._suffix_fa2_installation is not None:
                self._lite_optimizations['suffix_fa2_manifest']=self._suffix_fa2_installation.manifest()
        if dynamic_cublaslt is not None:
            if batch_geometry!='sm120':raise ValueError('Dynamic Lt requires SM120 batch geometry')
            from .dynamic_cublaslt import install_dynamic_cublaslt
            self._expert_cublaslt_installation,self._vision_fc2_cublaslt_installation=install_dynamic_cublaslt(
                model,dynamic_cublaslt)
            self._lite_optimizations.update(dynamic_cublaslt=dynamic_cublaslt,
                expert_cublaslt_manifest=self._expert_cublaslt_installation.manifest(),
                vision_fc2_cublaslt_manifest=self._vision_fc2_cublaslt_installation.manifest())
        if prefix_fp8:
            from .prefix_fp8_dispatch import install_prefix_fp8
            try:
                self._prefix_fp8_installation = install_prefix_fp8(model)
            except BaseException:
                self._prefix_fp8_installation = getattr(model, '_lite_prefix_fp8_installation', None)
                raise
            self._lite_optimizations['prefix_fp8_manifest'] = self._prefix_fp8_installation.manifest()
        if experimental_precision is not None:
            if not cuda_graph:
                raise ValueError('Optimized FP8 requires its serialized native graph owner')
            from ..experimental.optimized_precision import install
            try:
                self._dynamic_fp8_installation=install(model,experimental_precision['stages'],
                    unsupported=experimental_precision.get('unsupported','error'))
            except BaseException:
                self._dynamic_fp8_installation=getattr(model,'_lite_dynamic_fp8_installation',None)
                raise
            self._provenance['experimental_precision']=self._dynamic_fp8_installation.manifest()
        if cuda_graph:
            from vvla_pi05.lite.graph import Pi05NativeGraphRunner
            self.graph_runner = Pi05NativeGraphRunner(
                model, max_graphs=max_graphs, elide_identity_gathers=elide_identity_gathers,
                host_layout_cache=host_layout_cache,
                compile_sample_actions=compile_sample_actions,
            )
            if batch_geometry is not None:
                from .dynamic_batch_prepare import DynamicHostLayoutCache
                self.graph_runner._host_layouts=DynamicHostLayoutCache(capacity=max_graphs*2)
            if self._expert_cublaslt_installation is not None:
                self._expert_cublaslt_installation.bind_runner(self.graph_runner)
            if self._vision_fc2_cublaslt_installation is not None:
                self._vision_fc2_cublaslt_installation.bind_runner(self.graph_runner)
            if self._prefix_fp8_installation is not None:
                self._prefix_fp8_installation.bind_runner(self.graph_runner)
            if self._dynamic_fp8_installation is not None:
                self._dynamic_fp8_installation.bind_runner(self.graph_runner)
            if compile_sample_actions:
                self._lite_optimizations["compilation_manifest"] = self.graph_runner.compilation_manifest()

    @classmethod
    def from_pretrained(
        cls, checkpoint, *, device="cuda", tokenizer=None, revision=None,
        config_overrides=None, attention_backend="torch_fa2", cuda_graph=True, max_graphs=16,
        profile="full-bf16", pointwise_fusions=None,
        experimental_precision=None,
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from vvla_pi05.lite.dtype import set_bfloat16_layout
        from vvla_pi05.models.pi05.config import Pi05Config
        from vvla_pi05.lite.processor import Pi05LiteProcessor

        started = time.perf_counter()
        checkpoint_dir = _resolve_checkpoint(checkpoint, revision)
        config, config_audit = load_lite_config(checkpoint_dir)
        values = {item.name: getattr(config, item.name) for item in fields(config) if item.init}
        if profile not in ("bf16", "optimized-bf16", "full-bf16"):
            raise ValueError("Lite accepts BF16 profiles only")
        from vvla_pi05.profiles import apply_profile
        deploy = {"stages": [{"model_config": {}, "custom_pipeline_args": {}}]}
        apply_profile(deploy, profile)
        values.update(deploy["stages"][0]["model_config"])
        if pointwise_fusions is None:
            pointwise_fusions = deploy["stages"][0]["custom_pipeline_args"].get("pi05_pointwise_fusions")
        overrides, lite_options = _split_lite_overrides(config_overrides)
        _validate_compilation_options(lite_options, cuda_graph)
        if lite_options["host_layout_cache"] and not cuda_graph:
            raise ValueError("Host layout cache currently requires native CUDA graphs")
        unknown = set(overrides) - values.keys()
        if unknown:
            raise ValueError(f"Unknown pi05 configuration overrides: {sorted(unknown)}")
        values.update(overrides)
        values["attention_backend"] = attention_backend
        if attention_backend not in ("eager", "flash_attn", "torch_fa2"):
            raise ValueError("Lite supports only independent eager/flash_attn/torch_fa2 backends")
        if lite_options["elide_identity_gathers"] and attention_backend == "eager":
            raise ValueError("Identity gather elision requires standalone FA2 attention")
        config = Pi05Config(**values)
        if config.execution_artifact_path:
            raise ValueError("Lite BF16 loads original checkpoint weights, not quantized execution artifacts")
        if not all((config.norm_stats or {}).get(key) for key in ("state", "action")):
            raise ValueError("Physical-action inference requires checkpoint state and action normalization statistics")
        if config.execution_recipe:
            for name in ("prefix_mlp", "vision_mlp", "expert_mlp"):
                if config.execution_recipe.get(name, "none") != "none":
                    raise ValueError("This Lite benchmark supports BF16 recipes only")
        weight_path = checkpoint_dir / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing complete original checkpoint: {weight_path}")
        tokenizer_source = tokenizer or (
            str(checkpoint_dir) if (checkpoint_dir / "tokenizer_config.json").is_file()
            else "google/paligemma-3b-pt-224"
        )
        if config.input_contract=='openpi-libero-pi05':
            from vvla_pi05.openpi import OpenPITokenizer
            if tokenizer is None:raise ValueError('OpenPI requires the explicit original SentencePiece model asset')
            tok=OpenPITokenizer(tokenizer)
        else:
            tok = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="right")
        model = _construct_pi05_model(config, skip_weight_init=lite_options["skip_weight_init"])
        set_bfloat16_layout(model)
        model.to(device=torch.device(device))
        state = load_file(str(weight_path))
        model.load_weights(state.items(), strict=True)
        del state
        model.eval()
        if config.execution_recipe:
            from vvla_pi05.models.pi05.precision_pi05 import install_execution_recipe
            install_execution_recipe(model, config.execution_recipe)
        if pointwise_fusions:
            from vvla_pi05.models.pi05.precision_pi05 import set_pointwise_fusions
            set_pointwise_fusions(model, **pointwise_fusions)
        if config.input_contract=='openpi-libero-pi05':
            from vvla_pi05.openpi import OpenPiLiberoProcessor
            if any(lite_options[k] for k in ('action_norm_cache','gpu_uint8_prepare','packed_input_metadata')):
                raise ValueError('OpenPI preprocessing cannot use LeRobot-specific processor caches')
            processor=OpenPiLiberoProcessor(config,tok,torch.device(device))
        else:
            processor = Pi05LiteProcessor(config, tok, torch.device(device))
        core = Path(__file__).resolve().parents[1] / "models" / "pi05"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        packages = {}
        for name in ("torch", "transformers", "numpy", "safetensors", "flash-attn", "triton"):
            try:
                packages[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                pass
        tokenizer_files = {}
        local_tokenizer = Path(tokenizer_source)
        if local_tokenizer.is_file():
            tokenizer_files[local_tokenizer.name] = _file_hash(local_tokenizer)
        if local_tokenizer.is_dir():
            tokenizer_files = {
                p.name: _file_hash(p) for p in local_tokenizer.iterdir()
                if p.is_file() and (p.name.startswith("tokenizer") or p.name.endswith(".model"))
            }
        preprocessing = {
            "sources_sha256": _source_hash([
                core / "processor_pi05.py", core / "config.py", Path(__file__).parent / "processor.py",
            ]),
            "image_preprocess_version": processor.image_preprocess_version,
            "resize_device": str(device),
            "tokenizer": str(tokenizer_source), "tokenizer_files": tokenizer_files,
            "normalization_and_config": asdict(config),
            "checkpoint_config_audit": config_audit,
        }
        backend_tokenizer = getattr(tok, "backend_tokenizer", None)
        tokenizer_model = (backend_tokenizer.to_str() if backend_tokenizer is not None
                           else json.dumps(tok.get_vocab(), sort_keys=True))
        preprocessing["tokenizer_model_sha256"] = hashlib.sha256(tokenizer_model.encode()).hexdigest()
        provenance = {
            "framework": "vvla-lite", "upstream_commit": commit,
            "profile": profile,
            "checkpoint": str(checkpoint), "checkpoint_revision": revision,
            "checkpoint_sha256": _file_hash(weight_path),
            "checkpoint_config_sha256": config_audit["original_config_sha256"],
            "checkpoint_config_audit": config_audit,
            "preprocessing_sha256": _json_hash(preprocessing),
            "recipe_sha256": _json_hash({"config": asdict(config), "lite_optimizations": lite_options}),
            "source_sha256": _source_hash(sorted(core.glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))),
            "packages": packages, "resolved_config": asdict(config),
            "weight_initialization": {
                "skipped": lite_options["skip_weight_init"],
                "context": "transformers.initialization.no_init_weights" if lite_options["skip_weight_init"] else None,
                "paligemma_tying": "explicit_after_constructor" if lite_options["skip_weight_init"] else "native_constructor",
                "strict_checkpoint_load": True,
            },
        }
        policy = cls(model, processor, device=device, cuda_graph=cuda_graph, max_graphs=max_graphs,
                     provenance=provenance, experimental_precision=experimental_precision, **lite_options)
        try:
            policy._provenance["recipe_sha256"] = _json_hash({
                "config": asdict(config), "lite_optimizations": policy._lite_optimizations,
            })
        except BaseException as error:
            try:
                policy.close()
            except BaseException as cleanup:
                error.add_note("Lite provenance-finalization cleanup failed: " + str(cleanup))
            raise
        # Construction timing is process diagnostics, never part of stable
        # identity metadata used by separate preflight/formal processes.
        policy.load_seconds = time.perf_counter() - started
        return policy

    def metadata(self):
        if self._dynamic_fp8_installation is not None:
            self._provenance['experimental_precision']=self._dynamic_fp8_installation.manifest()
        if self.graph_runner is not None and self.graph_runner.compile_sample_actions:
            self._lite_optimizations['compilation_manifest']=self.graph_runner.compilation_manifest()
        blas_preference = getattr(torch.backends.cuda, "preferred_blas_library", None)
        result = deepcopy({
            **self._provenance, "precision": ("dynamic-fp8-experimental" if self._provenance.get("experimental_precision") else "bf16"), "batch": 1,
            "horizon": self.config.chunk_size, "denoise_steps": self.config.num_inference_steps,
            "action_shape": [self.config.chunk_size, self.config.action_dim],
            "action_space": ("libero_delta_no_additional_state_shift" if self.config.input_contract=='openpi-libero-pi05'
                             else "checkpoint_inverse_normalization_then_absolute"),
            "input_contract": self.config.input_contract, "attention_backend": self.config.attention_backend,
            "image_preprocess_version": getattr(self.processor, "image_preprocess_version", "historical_core"),
            "sampler_dtype": "float32",
            "cuda_graph": self.graph_runner is not None, "runtime": "in_process",
            "lite_optimizations": self._lite_optimizations,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_blas_preference": str(blas_preference()) if blas_preference is not None else "unavailable",
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "noise": {"algorithm": "torch.randn", "generator_device": self.device.type,
                      "dtype": "float32", "shape": [1, self.model.action_horizon, self.model.action_dim]},
            "latency_boundary": "raw_observation_to_owned_cpu_physical_actions",
        })
        if self._prefix_fp8_installation is not None:
            result['precision'] = 'mixed_bf16_fp8_prefix'
            result['quantization'] = self._prefix_fp8_installation.validate_precision_manifest()
        return result

    def reset(self, episode_id=None):
        # Pi05 has no temporal KV or action queue. Preserve executable graphs;
        # all request-dependent image/token/mask/noise buffers are overwritten.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot reset during an active request")
        try:
            if self._closed:
                raise RuntimeError("Policy is closed")
            self._episode_id = episode_id
        finally:
            self._lock.release()

    @torch.inference_mode()
    def _prepared_tensors(self, raw_obs, noise):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot inspect inputs during an active request")
        try:
            if self._closed:
                raise RuntimeError("Policy is closed")
            images, image_masks, tokens, masks = self.processor.build_model_inputs(raw_obs)
            # embed_image applies exactly this cast before its first vision op.
            # Report actual FP32 for a reference profile; never disguise it as
            # BF16 solely to make a cross-framework hash pass.
            vision_dtype = torch.bfloat16 if self.config.vision_bf16 else torch.float32
            inner = getattr(self.model, "paligemma_with_expert", None)
            if inner is not None:
                vision_dtype = next(inner.paligemma.model.vision_tower.parameters()).dtype
            images = [image.to(vision_dtype) for image in images]
            noise = torch.as_tensor(noise)
            shape = (1, self.model.action_horizon, self.model.action_dim)
            if tuple(noise.shape) != shape or noise.dtype != torch.float32:
                raise ValueError(f"Preflight noise must be FP32 {shape}")
            return images, image_masks, tokens, masks, noise
        finally:
            self._lock.release()

    def prepared_fixture(self, raw_obs, noise):
        """Actual pre-pack model inputs as NumPy arrays, outside timed calls."""
        images, image_masks, tokens, masks, noise = self._prepared_tensors(raw_obs, noise)
        values = {
            "tokens": tokens.cpu().numpy().copy(), "masks": masks.cpu().numpy().copy(),
            "noise_fp32": noise.detach().cpu().numpy().copy(),
            "noise_bf16_bits": noise.to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().copy(),
        }
        for index, image in enumerate(images):
            if image.dtype == torch.bfloat16:
                values[f"image_{index}_bf16_bits"] = image.cpu().contiguous().view(torch.uint16).numpy().copy()
            else:
                values[f"image_{index}_float32"] = image.float().cpu().numpy().copy()
            values[f"image_mask_{index}"] = image_masks[index].cpu().numpy().copy()
        return values

    def preflight(self, raw_observation, noise, *, repeat=3):
        """Hash real boundary tensors, then check deterministic owned outputs.

        Cross-framework hashes/tolerance diagnostics are evaluated separately by
        the harness. Passing this method is not a claim of numerical equivalence
        to another framework or to an independent LeRobot implementation.
        """
        if type(repeat) is not int or repeat < 2:
            raise ValueError("Preflight requires at least two replay checks")
        images, image_masks, tokens, masks, prepared_noise = self._prepared_tensors(raw_observation, noise)
        first = self.predict(raw_observation, noise)["actions"]
        retained = first.copy()
        warm = []
        for _ in range(repeat):
            result = self.predict(raw_observation, noise)
            if not np.array_equal(first, result["actions"]):
                raise ValueError("Same raw observation and noise produced a different action chunk")
            if not np.array_equal(first, retained):
                raise ValueError("A later call mutated the earlier CPU action chunk")
            if result["cold"]:
                raise ValueError("Repeated identical request still captures or falls back")
            warm.append(result["diagnostics"])
        from vvla_pi05.models.pi05.processor_pi05 import apply_norm, as_state_vector, build_pi05_prompt
        if self.config.input_contract=='openpi-libero-pi05':
            prompt=raw_observation.get('prompt','').strip().replace('_',' ').replace('\n',' ')+'\n'
        else:
            state = as_state_vector(raw_observation.get("state"), self.config.state_dim)
            prompt = build_pi05_prompt(
                task=raw_observation.get("prompt", "") or "",
                normalized_state=apply_norm(torch.from_numpy(state), self.processor._state_norm).numpy(),
                state_num_bins=self.config.state_num_bins,
            )
        return {
            "passed": True, "prompt": prompt,
            "prepared_shapes": {
                "images": [list(image.shape) for image in images], "tokens": list(tokens.shape),
                "masks": list(masks.shape), "noise": list(prepared_noise.shape),
            },
            "prepared_dtypes": {"images": [str(image.dtype) for image in images], "sampler_noise": "float32"},
            "prepared_sha256": {
                "images": [_tensor_hash(image) for image in images],
                "image_masks": [_tensor_hash(mask) for mask in image_masks],
                "tokens": _tensor_hash(tokens), "masks": _tensor_hash(masks),
                "noise_fp32": _tensor_hash(prepared_noise),
                "noise_bf16": _tensor_hash(prepared_noise.to(torch.bfloat16)),
            },
            "actions_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "deterministic_replays": repeat, "warm_diagnostics": warm,
            "physical_action_transform": self.metadata()['action_space'],
            "cross_framework_prepared_fixture_check": "required separately",
            "metadata": self.metadata(),
        }

    def infer(self, raw_obs, seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        return self._predict(raw_obs, seed=int(seed), noise=None)

    def predict(self, raw_obs, noise):
        return self._predict(raw_obs, seed=None, noise=noise)

    @torch.inference_mode()
    def _predict(self, raw_obs, *, seed, noise):
        started = time.perf_counter_ns()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("One Lite policy requires serialized B1 requests")
        try:
            if self._closed:
                raise RuntimeError("Policy is closed")
            if not isinstance(raw_obs, dict):
                raise TypeError("raw_obs must be a dictionary")
            host_layout = None
            if self._lite_optimizations["host_layout_cache"]:
                images, image_masks, tokens, masks, host_layout = self.processor.build_model_inputs(
                    raw_obs, return_host_layout=True,
                )
            else:
                images, image_masks, tokens, masks = self.processor.build_model_inputs(raw_obs)
            shape = (1, self.model.action_horizon, self.model.action_dim)
            if noise is None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn(shape, dtype=torch.float32, device=self.device, generator=generator)
            else:
                if isinstance(noise, torch.Tensor):
                    if noise.dtype != torch.float32 or tuple(noise.shape) != shape:
                        raise ValueError(f"noise must have float32 dtype and shape {shape}")
                    noise = noise.to(self.device)
                else:
                    noise = np.asarray(noise)
                    if noise.dtype != np.float32 or noise.shape != shape or not np.isfinite(noise).all():
                        raise ValueError(f"noise must be finite float32 with shape {shape}")
                    noise = torch.from_numpy(np.ascontiguousarray(noise)).to(self.device)
            prepared = time.perf_counter_ns()
            runner = self.graph_runner or self.model.sample_actions
            result = runner(
                images=images, image_masks=image_masks, lang_tokens=tokens, lang_masks=masks,
                noise=noise, num_steps=self.config.num_inference_steps,
                **({"host_layout": host_layout} if host_layout is not None else {}),
            )
            submitted = time.perf_counter_ns()
            # The processor performs inverse normalization, then absolute action
            # conversion, then a blocking CPU copy. Copy once more to own storage
            # even on CPU and when a mock/alternate runner returns reused buffers.
            output_processor = self._action_norm_cache or self.processor
            actions = np.array(output_processor.build_model_outputs(result, raw_obs), dtype=np.float32, copy=True)
            if actions.shape != (self.config.chunk_size, self.config.action_dim) or not np.isfinite(actions).all():
                raise ValueError("Model produced an invalid physical action chunk")
            graph_info = dict(self.graph_runner.last_call) if self.graph_runner else {
                "cold": self._calls == 0, "graph": "disabled", "metadata_updated": False,
            }
            self._calls += 1
            completed = time.perf_counter_ns()
            return {
                "actions": actions, "cold": graph_info["cold"],
                "diagnostics": {
                    **graph_info, "seed": seed, "request_index": self._calls - 1,
                    "total_ms": (completed - started) / 1e6,
                    "host_prepare_ms": (prepared - started) / 1e6,
                    "host_submit_ms": (submitted - prepared) / 1e6,
                    "postprocess_and_wait_ms": (completed - submitted) / 1e6,
                    "graph_stats": dict(self.graph_runner.stats) if self.graph_runner else {},
                },
            }
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close during an active request")
        try:
            if self._closed:
                return
            if self.graph_runner:
                self.graph_runner.close()
            if self._prefix_fp8_installation is not None:
                self._prefix_fp8_installation.uninstall()
                self._prefix_fp8_installation = None
            if self._dynamic_fp8_installation is not None:
                self._dynamic_fp8_installation.uninstall()
                self._dynamic_fp8_installation = None
            if self._request_identity_installation is not None:
                self._request_identity_installation.uninstall()
                self._request_identity_installation = None
            if self._expert_cublaslt_installation is not None:
                self._expert_cublaslt_installation.uninstall()
                self._expert_cublaslt_installation = None
            if self._vision_fc2_cublaslt_installation is not None:
                self._vision_fc2_cublaslt_installation.uninstall()
                self._vision_fc2_cublaslt_installation = None
            if self._action_norm_cache is not None:
                self._action_norm_cache.close()
                self._action_norm_cache = None
            if self._gpu_image_prepare_installation is not None:
                self._gpu_image_prepare_installation.close()
                self._gpu_image_prepare_installation = None
            if self._packed_metadata_installation is not None:
                self._packed_metadata_installation.close()
                self._packed_metadata_installation = None
            if self._vision_installation is not None:
                self._vision_installation.uninstall()
                self._vision_installation = None
            if self._vision_qkv_installation is not None:
                self._vision_qkv_installation.uninstall()
                self._vision_qkv_installation = None
            if self._preserved_linears_installation is not None:
                self._preserved_linears_installation.uninstall()
                self._preserved_linears_installation = None
            if self._suffix_installation is not None:
                self._suffix_installation.uninstall()
                self._suffix_installation = None
            if self._suffix_fa2_installation is not None:
                self._suffix_fa2_installation.uninstall()
                self._suffix_fa2_installation = None
            if self._residual_adarms_installation is not None:
                self._residual_adarms_installation.uninstall()
                self._residual_adarms_installation = None
            if self._residual_installation is not None:
                self._residual_installation.uninstall()
                self._residual_installation = None
            if self._mlp_installation is not None:
                self._mlp_installation.uninstall()
                self._mlp_installation = None
            if self._qkv_installation is not None:
                self._qkv_installation.uninstall()
                self._qkv_installation = None
            if self._adarms_installation is not None:
                self._adarms_installation.uninstall()
                self._adarms_installation = None
            if self._lifecycle_installation is not None:
                self._lifecycle_installation.uninstall()
                self._lifecycle_installation = None
            if self._native_lifecycle_installation is not None:
                self._native_lifecycle_installation.uninstall()
                self._native_lifecycle_installation = None
            self._closed = True
        finally:
            self._lock.release()


def create_adapter(config):
    """Benchmark factory. Settings are explicit and saved in adapter metadata."""
    values = dict(config)
    checkpoint = values.pop("checkpoint", values.pop("model", "lerobot/pi05_libero_finetuned"))
    values.pop("factory", None)
    values.pop("name", None)
    return Pi05LitePolicy.from_pretrained(checkpoint, **values)
