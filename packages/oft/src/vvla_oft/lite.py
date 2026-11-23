"""In-process continuous-L1 OFT boundary for the dedicated locked OFT interpreter.

The output is checkpoint-unnormalized, NOT LIBERO's final gripper convention.
No ML package is imported until construction. See workers/oft/LITE.md.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np

from .recipe import recipe_from_config


_NATIVE_OPTIONS = frozenset({
    "backend", "reference_compatible", "fuse_rope", "fuse_silu_mul",
    "fuse_rmsnorm", "fuse_residual_square", "fuse_qkv", "qkvo_layout",
    "mlp_precision", "fp8_scope", "share_mlp_quant", "batch_camera_views",
    "skip_unused_vision_block", "action_only_final", "packed_single_kv", "fuse_residual_rms", "pack_gate_up",
    "vision_compile", "mlp_compile", "vision_compile_scope", "parallel_vision",
    "pack_bucket", "cache_token_layout",
})
_OWNER_LEASE = threading.Lock()
_PROCESS_ORIGINS = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _dependencies():
    # Deliberately separate from the pi05 package and its Transformers runtime.
    import torch
    import transformers
    from . import reference
    from .adapter import OFTAdapter
    from .native_adapter import OFTNativeAdapter
    from vvla_runtime.execution.graph import TensorGraphRunner
    from .service import OFTService
    return SimpleNamespace(torch=torch, transformers=transformers, reference=reference,
                           official=OFTAdapter, native=OFTNativeAdapter,
                           runner=TensorGraphRunner, service=OFTService)


def _native_state_dependencies():
    # Explicit enabled-only cold import. No signature()/request can build code.
    from vvla_runtime.execution.native_state import load_native_state_comparator
    from vvla_runtime.execution.state import ModelStateContract
    return SimpleNamespace(load=load_native_state_comparator, contract=ModelStateContract)


def _check_runtime(deps):
    torch = deps.torch
    if torch.__version__.split("+")[0] != "2.13.0" or not torch.cuda.is_available():
        raise RuntimeError("OFT Lite requires the separate pinned CUDA Torch 2.13.0 environment")
    if deps.transformers.__version__ != "4.40.1":
        raise RuntimeError("OFT Lite requires its Transformers 4.40.1 bidirectional fork")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Disable CUDA matmul and cuDNN TF32 before constructing OFT Lite")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("OFT Lite must not be constructed under CUDA/CPU autocast")


def _math_state(torch):
    """Actual global math controls; read live, never reset them for the caller."""
    cuda, matmul, cudnn = torch.backends.cuda, torch.backends.cuda.matmul, torch.backends.cudnn
    return {"cuda_matmul_allow_tf32": bool(matmul.allow_tf32),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_reduced_precision_reduction": bool(matmul.allow_bf16_reduced_precision_reduction),
            "fp16_reduced_precision_reduction": bool(matmul.allow_fp16_reduced_precision_reduction),
            "blas_library": str(cuda.preferred_blas_library()),
            "sdpa_flash": bool(cuda.flash_sdp_enabled()),
            "sdpa_mem_efficient": bool(cuda.mem_efficient_sdp_enabled()),
            "sdpa_math": bool(cuda.math_sdp_enabled()),
            "sdpa_cudnn": bool(cuda.cudnn_sdp_enabled()),
            "cudnn_enabled": bool(cudnn.enabled), "cudnn_benchmark": bool(cudnn.benchmark),
            "cudnn_deterministic": bool(cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "default_dtype": str(torch.get_default_dtype())}


def _source_manifest():
    root = Path(__file__).parent
    import vvla_runtime
    runtime = Path(vvla_runtime.__file__).parent
    result = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob("*")) if path.is_file() and path.suffix in (".py", ".cpp")}
    result.update({"vvla_runtime/"+str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(runtime.rglob("*")) if path.is_file() and path.suffix in (".py", ".cpp")})
    return result


def _origin_modules():
    return sys.modules


def _pin_runtime_origins(source, native_path, *, native_requested, require_loaded=False, require_native=False):
    """Cold-only namespace/ABI origin checks; successful close keeps this pin.

    Upstream namespace packages and the vLLM native loader survive an owner.
    They cannot be redirected merely by changing a later constructor's paths.
    """
    source = Path(source).resolve()
    fork = Path(os.environ.get("VVLA_OFT_TRANSFORMERS_SOURCE",source.parent / "transformers-openvla-oft")).resolve()
    wanted = {"source": str(source), "fork": str(fork)}
    if native_requested:
        import importlib.metadata
        selected = native_path or os.environ.get("VLA_VLLM_NATIVE_PATH")
        root = Path(selected if selected else importlib.metadata.distribution('vllm').locate_file('')).resolve()
        wanted["native_root"] = str(root)
    for key, value in wanted.items():
        if key in _PROCESS_ORIGINS and _PROCESS_ORIGINS[key] != value:
            raise RuntimeError("OFT interpreter origin is already pinned for " + key + "; use a new process")
    modules, observed = _origin_modules(), {}
    def tree(prefix, root):
        for name, module in list(modules.items()):
            if module is None or (name != prefix and not name.startswith(prefix + ".")):
                continue
            paths = []
            filename = getattr(module, "__file__", None)
            if filename is not None:
                paths.append(str(Path(filename).resolve()))
            paths.extend(str(Path(path).resolve()) for path in getattr(module, "__path__", ()))
            if not paths or any(not Path(path).is_relative_to(root) for path in paths):
                raise RuntimeError("Already imported OFT module has a different origin: " + name)
            observed[name] = sorted(set(paths))
    tree("prismatic", (source / "prismatic").resolve())
    llama = modules.get("transformers.models.llama.modeling_llama")
    if llama is not None:
        installed = Path(llama.__file__).resolve()
        actual = hashlib.sha256(installed.read_bytes()).hexdigest()
        expected = hashlib.sha256((fork / "src/transformers/models/llama/modeling_llama.py").read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Already imported Llama source differs from the selected OFT fork")
        wanted["llama_module"] = {"path": str(installed), "sha256": actual}
        if "llama_module" in _PROCESS_ORIGINS and _PROCESS_ORIGINS["llama_module"] != wanted["llama_module"]:
            raise RuntimeError("Installed OFT Llama origin changed; use a new process")
    if require_loaded and (llama is None or modules.get("prismatic.extern.hf.modeling_prismatic") is None):
        raise RuntimeError("The locked OFT loader did not publish its actual model module origins")
    native_root = wanted.get("native_root", _PROCESS_ORIGINS.get("native_root"))
    if native_root is not None:
        tree("vllm", (Path(native_root) / "vllm").resolve())
        native = modules.get("vvla_runtime.execution.native")
        loaded = bool(getattr(native, "_loaded", False))
        if loaded and any(modules.get(name) is None for name in
                         ("vllm", "vllm._C_stable_libtorch", "vllm.vllm_flash_attn")):
            raise RuntimeError("Native OFT loader reports loaded without its actual ABI modules")
        if loaded and (getattr(modules["vllm"], "__version__", None) != "0.32.0" or
                       getattr(native, "_flash", None) is not
                       getattr(modules["vllm.vllm_flash_attn"], "flash_attn_varlen_func", None)):
            raise RuntimeError("Native OFT ABI/Flash callable no longer matches its imported module")
        if require_native and not loaded:
            raise RuntimeError("Requested native OFT backend was not actually installed")
    _PROCESS_ORIGINS.update(wanted)
    return {"pinned": copy.deepcopy(_PROCESS_ORIGINS), "observed_modules": observed}


class OFTLitePolicy:
    """Serialized raw CPU observations -> owned CPU FP32 checkpoint actions.

    ``official`` uses get_vla_action for B1 and the existing official_batch
    generalization for a configured batch >1. ``native`` is an explicit choice;
    only existing BF16 options are exposed in this first direct-entry adapter.
    Raw inputs must remain unchanged until predict/predict_batch returns.
    """

    def __init__(self, source, checkpoint, *, execution="official", batch_size=1,
                 norm_key=None, native_options=None, cuda_graph=False, native_path=None,
                 native_state_comparator=False, paired_image_preprocessing=False, gpu_image_geometry=False,
                 versioned_state_checks=False, native_jpeg=False):
        if execution not in ("official", "native"):
            raise ValueError("execution must be official or native")
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an integer in [1, 32]")
        if type(cuda_graph) is not bool:
            raise ValueError("cuda_graph must be boolean")
        if type(native_state_comparator) is not bool:
            raise ValueError("native_state_comparator must be boolean")
        if native_state_comparator and execution != "native":
            raise ValueError("native_state_comparator requires native execution")
        if type(versioned_state_checks) is not bool or (versioned_state_checks and not native_state_comparator):
            raise ValueError("Versioned registration checks require the native state comparator")
        if type(paired_image_preprocessing) is not bool or (paired_image_preprocessing and execution != "native"):
            raise ValueError("paired_image_preprocessing requires a boolean and native execution when enabled")
        if type(gpu_image_geometry) is not bool or (gpu_image_geometry and not paired_image_preprocessing):
            raise ValueError("GPU geometry requires paired image preprocessing")
        if type(native_jpeg) is not bool or (native_jpeg and not gpu_image_geometry):
            raise ValueError("Native JPEG requires GPU image geometry")
        if norm_key is not None and (not isinstance(norm_key, str) or not norm_key):
            raise ValueError("norm_key must be a nonempty string or None")
        if native_options is not None and not isinstance(native_options, dict):
            raise ValueError("native_options must be an object or None")
        options = dict(native_options or {})
        if options.keys() - _NATIVE_OPTIONS:
            raise ValueError("Unknown or out-of-scope native_options: " + repr(sorted(map(str, options.keys() - _NATIVE_OPTIONS))))
        if execution == "official" and (options or cuda_graph or native_path is not None):
            raise ValueError("official execution does not accept native options or CUDA graphs")
        recipe = recipe_from_config({"backend": "sdpa", **options})
        # This entry exposes only the existing MLP FP8 implementation. Keep
        # quantization explicit and bind its native provider during cold load.
        if recipe.mlp_precision not in ("bf16", "fp8"):
            raise ValueError("Lite supports BF16 or explicitly selected MLP FP8")
        if recipe.mlp_precision == "fp8" and recipe.attention == "sdpa":
            raise ValueError("MLP FP8 requires an explicitly pinned native attention backend")
        if recipe.reference_compatible and (batch_size != 1 or cuda_graph):
            raise ValueError("reference_compatible requires B1 SDPA without CUDA graphs")
        if native_path is not None and recipe.attention == "sdpa":
            raise ValueError("native_path is only used by an explicitly selected native attention backend")
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._deps = None
        self._runner = self._service = self._adapter = None
        self._native_state_comparator = self._native_state_contract = None
        self._native_state_manifest = None
        self._model = self._head = self._proprio = self._processor = self._utils = None
        self._metadata = None
        self._lease_held = self._load_started = False
        self._math = None
        self._runtime_origins = None
        self._counts = dict(calls=0, successes=0, failures=0, first_calls=0, cold_calls=0, prepared_fixtures=0)
        self._last_call = None
        self._retired_runner_stats = {}
        self._execution, self._batch_size = execution, batch_size
        self._paired_image_preprocessing = paired_image_preprocessing
        self._gpu_image_geometry = gpu_image_geometry
        self._jpeg_manifest = None
        self._default_key, self._recipe = norm_key, recipe
        self._episode_id, self._sequence = None, 0
        self._device = "cuda:0"
        if not _OWNER_LEASE.acquire(blocking=False):
            raise RuntimeError("OFT Lite requires one live owner in this interpreter; close the previous owner")
        self._lease_held = True
        try:
            deps = self._deps = _dependencies()
            _check_runtime(deps)
            self._device = 'cuda:' + str(deps.torch.cuda.current_device())
            self._math = _math_state(deps.torch)
            if self._math["default_dtype"] != "torch.float32":
                raise RuntimeError("OFT Lite's original preprocessing requires default FP32 dtype")
            native_requested = execution == "native" and recipe.attention != "sdpa"
            self._runtime_origins = _pin_runtime_origins(source, native_path,
                                                         native_requested=native_requested)
            # Persistent model/packed state must retain version counters even if
            # a caller constructs this object from an inference-mode context.
            with deps.torch.inference_mode(False), deps.torch.no_grad():
                self._load_started = True
                loaded = deps.reference.load_reference(source, checkpoint, device=self._device,
                                                       resource_publisher=self._publish_resource)
                (self._model, self._head, self._proprio, self._processor,
                 self._utils, provenance) = loaded
                self._runtime_origins = _pin_runtime_origins(source, native_path,
                    native_requested=native_requested, require_loaded=True)
                if norm_key is not None and norm_key not in self._model.norm_stats:
                    raise ValueError("Configured normalization key is absent from checkpoint")
                if execution == "native":
                    previous_path = os.environ.get("VLA_VLLM_NATIVE_PATH")
                    try:
                        if native_path is not None:
                            os.environ["VLA_VLLM_NATIVE_PATH"] = os.fspath(native_path)
                        self._adapter = deps.native(self._model, self._head, self._proprio, recipe, None).eval()
                    finally:
                        if native_path is not None:
                            if previous_path is None:
                                os.environ.pop("VLA_VLLM_NATIVE_PATH", None)
                            else:
                                os.environ["VLA_VLLM_NATIVE_PATH"] = previous_path
                elif batch_size > 1:
                    self._adapter = deps.official(self._model, self._head, self._proprio,
                                                  execution="official_batch").eval()
                if self._adapter is not None:
                    revision = hashlib.sha256((Path(checkpoint) / "config.json").read_bytes()).hexdigest()
                    runner_options = {}
                    if native_state_comparator:
                        state_deps = _native_state_dependencies()
                        self._native_state_comparator = (state_deps.load(verbose=False,versioned=True)
                            if versioned_state_checks else state_deps.load(verbose=False))
                        self._native_state_manifest = self._native_state_comparator.manifest()
                        self._native_state_contract = state_deps.contract(
                            self._adapter, native_comparator=self._native_state_comparator)
                        runner_options["state_contract"] = self._native_state_contract
                    self._runner = deps.runner(self._adapter, revision=revision, use_graph=cuda_graph,
                                               **runner_options)
                    prepare = None
                    if execution == "native":
                        if self._recipe.cache_token_layout:
                            prepare = lambda tensors: self._adapter._prepare_inputs(
                                tensors, device=self._device, validate=False, reuse_layout=True)
                        else:
                            prepare = lambda tensors: self._adapter.prepare_inputs(
                                tensors, device=self._device, validate=False)
                    self._service = deps.service(self._model, self._processor, self._utils,
                        self._runner, capacity=128, batch_size=batch_size, prepare_inputs=prepare)
                    if paired_image_preprocessing:
                        if gpu_image_geometry:
                            from .spatial_images import SpatialCameraProcessor
                            self._service.paired_images = SpatialCameraProcessor(self._processor,self._utils,self._device,native_jpeg=native_jpeg)
                            self._jpeg_manifest = self._service.paired_images.jpeg_manifest
                        else:
                            from .paired_images import PairedCameraProcessor
                            self._service.paired_images = PairedCameraProcessor(self._processor,self._utils,self._device)
                self._runtime_origins = _pin_runtime_origins(source, native_path,
                    native_requested=native_requested, require_loaded=True, require_native=native_requested)
            self._metadata = self._build_metadata(checkpoint, provenance, cuda_graph, native_path)
        except BaseException as exc:
            # A failed retirement must not drop model/graph ownership. Expose
            # this owner on the original error so callers can retry close().
            try:
                self._close_resources()
            except BaseException as cleanup:
                self._poisoned = True
                exc.oft_lite_owner = self
                exc.add_note("OFT Lite cleanup failed; retry exc.oft_lite_owner.close(): " + repr(cleanup))
            raise

    def _publish_resource(self, name, value):
        if name not in ("model", "head", "proprio", "processor", "utils"):
            raise ValueError("Unexpected loader ownership publication")
        setattr(self, "_" + name, value)

    def _check_live_math(self):
        if self._native_state_manifest is not None and (
                self._native_state_comparator is None or
                self._runner.state_contract is not self._native_state_contract or
                self._native_state_contract.native_comparator is not self._native_state_comparator or
                self._native_state_contract.model is not self._adapter or
                self._runner.forward is not self._adapter):
            raise RuntimeError("OFT native state comparator binding changed; rebuild with a new identity")
        torch = self._deps.torch
        if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("OFT Lite prediction does not permit CUDA/CPU autocast")
        if torch.cuda.current_device() != 0 or _math_state(torch) != self._math:
            raise RuntimeError("OFT Lite math/device state changed; rebuild with a new metadata identity")

    def _build_metadata(self, checkpoint, provenance, cuda_graph, native_path):
        ref = self._deps.reference
        files = {}
        # Small input/config identity only. Do not claim these hashes identify
        # the multi-GB weights; an external campaign must hash those separately.
        for name in ("config.json", "dataset_statistics.json", "preprocessor_config.json",
                     "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.model"):
            path = Path(checkpoint) / name
            if path.is_file():
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        recipe = {"execution": self._execution, "batch_size": self._batch_size,
                  "cuda_graph": cuda_graph, "native_options": self._recipe.asdict()
                  if self._execution == "native" else None,
                  "native_path": None if native_path is None else os.fspath(native_path)}
        recipe["math_state"] = copy.deepcopy(self._math)
        if self._paired_image_preprocessing:
            recipe["paired_image_preprocessing"] = "tf-paired-crop-prismatic-bf16-v1"
        if self._gpu_image_geometry:
            recipe["paired_image_preprocessing"] = "tf-jpeg-device-lanczos-crop-v1"
        extra = {}
        if self._jpeg_manifest is not None:
            recipe["paired_image_preprocessing"] = "native-jpeg-device-lanczos-crop-v1"
            extra["jpeg_operator"] = copy.deepcopy(self._jpeg_manifest)
        precision = "bf16"
        if self._recipe.mlp_precision == "fp8":
            precision = "mixed_bf16_fp8_mlp"
            installed = copy.deepcopy(self._adapter.quantization_manifest)
            depth = self._model.language_model.config.num_hidden_layers
            if type(depth) is not int or depth < 1 or len(self._model.language_model.model.layers) != depth:
                raise RuntimeError('Checkpoint Llama depth does not match the loaded module topology')
            expected = {f"llama.{i}.{name}" for i in range(depth)
                        for name in ("gate_proj", "up_proj", "down_proj")}
            scope = self._recipe.fp8_scope
            categories = {"llama_mlp"}
            if scope != "mlp":
                categories.add("llama_attention")
                expected.update(f"llama.{i}.attention.{name}" for i in range(depth)
                                for name in ("q_proj","k_proj","v_proj","o_proj"))
                precision = "mixed_bf16_fp8_" + scope
            if scope in ("llama_vision","all_linear"):
                from vvla_runtime.precision import QuantizedLinear
                roots = {"vision":self._model.vision_backbone}
                if scope == "all_linear":
                    roots.update(multimodal_projector=self._model.projector,action_head=self._head,proprio_projector=self._proprio)
                for prefix,root in roots.items():
                    projections = {prefix+"."+name for name,mod in root.named_modules() if isinstance(mod,QuantizedLinear)}
                    if not projections or any(isinstance(mod,self._deps.torch.nn.Linear) for mod in root.modules()):
                        raise RuntimeError("Incomplete FP8 installation in "+prefix)
                    expected.update(projections)
                    categories.add(prefix)
            if (len(installed) != len(expected) or {row["name"] for row in installed} != expected
                    or any(row["precision"] != "fp8" or row["category"] not in categories
                           or row["reference_weight_device"] != "cpu" for row in installed)):
                raise RuntimeError("MLP FP8 installation does not cover every declared Llama MLP projection")
            extra["quantization"] = {"schema": 1, "scope": f"llama_mlp_{3*depth}_linears" if scope=="mlp" else scope,
                "layers": depth, "projection_count": len(installed), "historical_full_depth_projection_count": 96,
                "weights": "per-output-channel e4m3fn", "activations": "per-token e4m3fn",
                "other_components": "bf16", "manifest": installed,
                "manifest_sha256": _digest(installed)}
        if self._native_state_manifest is not None:
            recipe["native_state_comparator"] = {
                "version": self._native_state_manifest["version"],
                "manifest_sha256": _digest(self._native_state_manifest)}
            extra["native_state_comparator"] = copy.deepcopy(self._native_state_manifest)
        return {"schema": 1, "framework": "openvla-oft-lite", "precision": precision, **extra,
                "model": "openvla-oft-continuous-l1", "action_shape": [8, 7],
                "action_space": "physical_oft_checkpoint", "output_dtype": "float32",
                "gripper": {"output": "checkpoint unnormalized; no environment transform",
                            "libero_transform": "-np.sign(2 * g - 1)",
                            "transform_input_dtype": "float32; matches existing worker boundary",
                            "transform_applied": False,
                            "timing_contract": "shared harness applies once inside final env-ready timer; simulator must not repeat"},
                "noise": "none; deterministic continuous L1",
                "capacity": None if self._execution == "official" and self._batch_size == 1 else 128,
                "norm_keys": sorted(self._model.norm_stats), "default_norm_key": self._default_key,
                "norm_stats_sha256": _digest(self._model.norm_stats),
                "math_state": copy.deepcopy(self._math),
                "runtime_origins": copy.deepcopy(self._runtime_origins),
                "action_normalization": "checkpoint q01/q99 mask, +1e-8; existing official/service arithmetic",
                "preprocessing": copy.deepcopy(provenance), "input_config_files_sha256": files,
                "weights_content_verified": False,
                "checkpoint_revision_expected": ref.CHECKPOINT_REVISION,
                "upstream_commit": ref.REPOSITORY_REVISION,
                "transformers_commit": ref.TRANSFORMERS_REVISION,
                "torch": str(self._deps.torch.__version__),
                "transformers": str(self._deps.transformers.__version__),
                "reference": "upstream get_vla_action" if self._execution == "official" and self._batch_size == 1
                    else "official_batch generalization" if self._execution == "official" else "existing OFTNativeAdapter",
                "source_sha256": _source_manifest(), "recipe": recipe, "recipe_sha256": _digest(recipe)}

    @contextmanager
    def _request(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("OFT Lite requires serialized calls")
        try:
            if self._closed:
                raise RuntimeError("OFT Lite is closed")
            if self._poisoned:
                raise RuntimeError("OFT Lite is poisoned; close it before rebuilding")
            yield
        finally:
            self._lock.release()

    def metadata(self):
        """Stable construction provenance; caller mutations never modify it."""
        return copy.deepcopy(self._metadata)

    def _runner_counts(self):
        stats = getattr(self._runner, "stats", {}) if self._runner is not None else self._retired_runner_stats
        return {name: int(stats.get(name, 0)) for name in
                ("captures", "replays", "eager_calls", "capacity_fallbacks", "state_invalidations",
                 "state_rejections", "stream_handoffs")}

    def diagnostics(self):
        """Untimed cumulative counters; includes real prepared_fixture calls."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot read diagnostics during an active OFT call")
        try:
            return {"schema": 1, "counters": {**self._counts, **self._runner_counts(), "compiles": 0},
                    "last_call": copy.deepcopy(self._last_call), "closed": self._closed,
                    "poisoned": self._poisoned}
        finally:
            self._lock.release()

    def reset(self, episode_id=None):
        """Record a new episode; no action/input/KV cache is carried across calls."""
        if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
            raise ValueError("episode_id must be a nonempty string or None")
        with self._request():
            self._episode_id, self._sequence = episode_id, 0

    def _observation(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("observation must be an object")
        # get_vla_action selects every key containing 'wrist', whereas service
        # deliberately selects wrist_image. Reject ambiguous extra cameras.
        if any(isinstance(key, str) and "wrist" in key and key != "wrist_image" for key in obs):
            raise ValueError("Only the full_image and wrist_image camera contract is supported")
        for name in ("full_image", "wrist_image"):
            image = obs.get(name)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(name + " must be an HWC uint8 RGB image")
            if not 1 <= min(image.shape[:2]) <= max(image.shape[:2]) <= 4096:
                raise ValueError(name + " dimensions are invalid")
        state = np.asarray(obs.get("state"))
        if state.shape != (8,) or state.dtype.kind not in "fi" or not np.isfinite(state).all():
            raise ValueError("state must contain eight finite numeric values")
        if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
            raise ValueError("prompt must be a nonempty string")
        key = obs.get("norm_key", self._default_key)
        if not isinstance(key, str) or key not in self._model.norm_stats:
            raise ValueError("An explicit known norm_key is required")
        return key

    @staticmethod
    def _owned(value, shape):
        result = np.array(value, dtype=np.float32, copy=True, order="C")
        if result.shape != shape or not np.isfinite(result).all():
            raise RuntimeError("OFT returned invalid action values")
        return result

    def predict(self, raw_observation):
        # Copy the row too: a view into a batch is not an owned [8,7] result.
        return self.predict_batch([raw_observation])[0].copy()

    def predict_batch(self, raw_observations):
        """Return owned [real_batch,8,7]; static dummy rows never escape."""
        with self._request():
            return self._predict_locked(raw_observations)

    def _predict_locked(self, raw_observations, capture=None):
        if not isinstance(raw_observations, (list, tuple)) or not 1 <= len(raw_observations) <= self._batch_size:
            raise ValueError("Invalid request batch size")
        keys = [self._observation(obs) for obs in raw_observations]
        self._check_live_math()
        before = self._runner_counts()
        first = self._counts["calls"] == 0
        unwarmed = self._counts["successes"] == 0
        self._counts["calls"] += 1
        self._counts["first_calls"] += int(first)
        success = False
        tickets = []
        actual_tensor_batch = None
        try:
            if self._execution == "official" and self._batch_size == 1:
                with self._deps.torch.inference_mode():
                    obs = copy.deepcopy(raw_observations[0])
                    obs["norm_key"] = keys[0]
                    cfg = SimpleNamespace(num_images_in_input=2, use_proprio=True,
                                          unnorm_key=keys[0], center_crop=True)
                    value = self._utils.get_vla_action(cfg, self._model, self._processor,
                        obs, obs["prompt"], action_head=self._head, proprio_projector=self._proprio)
                    result = self._owned([value], (1, 8, 7))
                    actual_tensor_batch = result.shape[0]
            else:
                # Match Worker: persistent layout preparation is outside
                # inference mode; TensorGraphRunner owns inference-mode entry.
                with self._deps.torch.inference_mode(False):
                    rows = [self._service.prepare_row(obs, key)
                            for obs, key in zip(raw_observations, keys, strict=True)]
                    for _ in rows:
                        self._sequence += 1
                        tickets.append(self._service.ledger.start("lite:" + str(self._sequence)))
                    tensors, count = self._service.collate(rows,
                        pad=not (self._execution == "native" and self._recipe.reference_compatible))
                actual_tensor_batch = int(tensors['pixel_values'].shape[0])
                if capture is not None:
                    capture.native(tensors)
                values = self._service.predict_prepared(tickets, tensors, count, keys)
                result = self._owned([values[ticket] for ticket in tickets], (len(rows), 8, 7))
            success = True
            self._counts["successes"] += 1
            return result
        except BaseException as exc:
            self._counts["failures"] += 1
            # Covers preprocessing/transfer errors before runner entry too.
            try:
                self._deps.torch.cuda.synchronize(self._device)
            except BaseException as cleanup:
                self._poisoned = True
                exc.add_note("OFT Lite synchronization failed; owner retained: " + repr(cleanup))
            raise
        finally:
            for ticket in tickets:
                self._service.ledger.finish(ticket)
            after = self._runner_counts()
            delta = {name: after[name] - before[name] for name in before}
            cold = unwarmed or delta["captures"] > 0 or delta["state_invalidations"] > 0
            self._counts["cold_calls"] += int(cold)
            self._last_call = dict(index=self._counts["calls"], first_call=first,
                cold=cold, success=success, prepared_fixture=capture is not None,
                batch=len(raw_observations), actual_tensor_batch=actual_tensor_batch, graph_delta=delta)

    def prepared_fixture(self, raw_observation):
        """Execute one real B1 prediction and capture its actual consumer inputs.

        Cold diagnostic only; hooks/D2H copies are never part of formal timing.
        The real call warms execution and increments diagnostics, not a fake
        preflight certificate. Full numerical comparison belongs to the harness.
        """
        from .lite_prepared import PreparedCapture
        with self._request():
            if self._batch_size != 1:
                raise ValueError("prepared_fixture currently requires configured B1")
            key = self._observation(raw_observation)
            self._check_live_math()
            capture = PreparedCapture(self, key)
            try:
                with capture:
                    actions = self._predict_locked([raw_observation], capture=capture)[0]
                result = capture.finish(actions)
                self._counts["prepared_fixtures"] += 1
                return result
            except BaseException:
                # A hook installation/restoration failure must never permit
                # later production calls against a partially instrumented model.
                if capture.hooks:
                    self._poisoned = True
                raise

    def _close_resources(self):
        if self._runner is not None:
            self._runner.close()
        if self._deps is not None and self._load_started:
            # Also drain if the loader raised before publishing any model. We
            # cannot claim to release objects internal to a failed allocator;
            # those remain owned by the original loader exception/traceback.
            self._deps.torch.cuda.synchronize(self._device)
        # Release only after successful retirement. A failed close is retryable.
        self._retired_runner_stats = self._runner_counts()
        self._service = self._runner = self._adapter = None
        self._native_state_contract = self._native_state_comparator = None
        self._model = self._head = self._proprio = self._processor = self._utils = None
        self._closed = True
        if self._lease_held:
            self._lease_held = False
            _OWNER_LEASE.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close OFT Lite during an active call")
        try:
            if not self._closed:
                try:
                    self._close_resources()
                except BaseException:
                    self._poisoned = True
                    raise
        finally:
            self._lock.release()
