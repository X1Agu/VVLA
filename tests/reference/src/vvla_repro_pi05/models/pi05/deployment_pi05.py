"""Versioned packed deployment artifacts and explicit hardware fallback.

Artifacts contain remaining model tensors and native packed weights/scales, not
lossy reconstructions of the original checkpoint. No pickle is used. A fallback
to another precision requires the ORIGINAL checkpoint, never dequantized FP4.
"""
from dataclasses import asdict, is_dataclass, replace
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import torch

from vvla_repro_pi05.models.pi05.precision_pi05 import (
    ExecutionRecipe, QuantizedLinear, _identity, install_execution_recipe,
    selected_linears, validate_execution_recipe,
)


def resolve_hardware_recipe(recipe, device, fallback="error"):
    recipe = recipe if isinstance(recipe, ExecutionRecipe) else ExecutionRecipe(**recipe)
    if fallback not in ("error", "bf16", "fp8"):
        raise ValueError("Hardware fallback must be error, bf16 or fp8")
    device = torch.device(device)
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    needs_fp4 = recipe.prefix_mlp in ("nvfp4_full", "nvfp4_selective")
    needs_fp8 = (recipe.prefix_mlp in ("fp8", "nvfp4_selective")
                 or recipe.vision_mlp == "fp8" or recipe.expert_mlp == "fp8")
    reason = None
    if needs_fp4 and capability != (12, 0):
        reason = "This native NVFP4 artifact/backend is validated on SM120 only"
    elif needs_fp8 and capability is not None and capability < (8, 9):
        reason = "The selected native FP8 GEMM requires SM89 or newer"
    elif needs_fp4:
        from vllm import _custom_ops as ops
        if not all(hasattr(ops, name) for name in ("scaled_fp4_quant", "cutlass_scaled_fp4_mm")):
            reason = "Required native NVFP4 operations are unavailable"
        if recipe.nvfp4_backend.startswith("flashinfer_") and importlib.util.find_spec("flashinfer") is None:
            reason = "Requested FlashInfer backend is unavailable"
    effective = recipe
    if reason:
        if fallback == "error":
            raise RuntimeError(reason + "; select an explicit fallback with the original checkpoint")
        if fallback == "fp8" and (capability is None or capability < (8, 9)):
            raise RuntimeError("FP8 hardware fallback is unsupported on this device")
        effective = replace(recipe, prefix_mlp=("none" if fallback == "bf16" else "fp8") if recipe.prefix_mlp != "none" else "none",
                            vision_mlp="none" if fallback == "bf16" else recipe.vision_mlp,
                            expert_mlp="none" if fallback == "bf16" else recipe.expert_mlp,
                            nvfp4_layers=(), fuse_prefix_down_quant=False, nvfp4_backend="vllm_cutlass")
    return effective, {"requested": asdict(recipe), "effective": asdict(effective),
                       "policy": fallback, "reason": reason, "capability": capability,
                       "execution_kind": "native_cuda" if capability is not None else "cpu_reference"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _semantic_config(model, recipe):
    config = asdict(model.config) if is_dataclass(model.config) else dict(vars(model.config))
    for name in ("execution_artifact_path", "execution_artifact_sha256", "quantization_calibration_path", "precision_hardware_fallback"):
        config.pop(name, None)
    config["execution_recipe"] = asdict(recipe)
    # JSON canonicalization normalizes tuple/list representation after reload.
    return json.loads(json.dumps(config, sort_keys=True))


def _runtime_versions():
    result = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for package in ("vllm", "triton", "flashinfer-python", "transformers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _storage_view(tensor):
    return (tensor.device, tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype)


@torch.no_grad()
def save_execution_artifact(model, directory, provenance):
    """Export a fresh directory atomically. Existing artifacts are immutable."""
    from safetensors.torch import save_file
    validate_execution_recipe(model)
    state = getattr(model, "_execution_state", None)
    if not state or state["status"] != "ready":
        raise ValueError("Install and validate an execution recipe before exporting")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    paths = {id(module): name for name, module in model.named_modules()}
    quantized = [m for m in state["modules"] if isinstance(m, QuantizedLinear)]
    if not quantized or len(quantized) != len(state["manifest"]):
        raise ValueError("Artifact requires a complete quantized module manifest")
    omitted = {paths[id(m)] + ".weight" for m in quantized}
    tensors, aliases, seen = {}, {}, {}
    for key, value in model.state_dict().items():
        if key in omitted:
            continue
        identity = _storage_view(value)
        if value.numel() and identity in seen:
            aliases[key] = seen[identity]
        else:
            seen[identity] = key
            tensors["model." + key] = value.detach().cpu().contiguous().clone()
    modules = []
    for row, module in zip(state["manifest"], quantized, strict=True):
        attrs = {}
        for attr in ("packed_weight", "weight_scale", "input_global_scale", "alpha"):
            value = getattr(module, attr)
            if value is not None:
                key = "packed." + row["name"] + "." + attr
                tensors[key] = value.detach().cpu().contiguous().clone()
                attrs[attr] = key
        modules.append(dict(**row, path=paths[id(module)], tensors=attrs))
    manifest = {"format": "pi05-native-packed-v1", "recipe": asdict(state["recipe"]),
                "model_config": _semantic_config(model, state["recipe"]),
                "model_aliases": aliases,
                "runtime_versions": _runtime_versions(),
                "nvfp4_layout": "cutlass_swizzle128x4_e2m1_e4m3",
                "optimization_flags": list(model.optimization_flags()), "modules": modules,
                "provenance": provenance, "torch_version": torch.__version__,
                "original_quantized_weights_included": False,
                "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                "omitted_bf16_bytes": sum(m.in_features * m.out_features * 2 for m in quantized),
                "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}}
    with tempfile.TemporaryDirectory(prefix=".pi05-export-", dir=directory.parent) as tmp:
        tmp = Path(tmp)
        save_file(tensors, str(tmp / "weights.safetensors"))
        manifest["weights_sha256"] = _sha(tmp / "weights.safetensors")
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_hash = _sha(tmp / "manifest.json")
        (tmp / "manifest.sha256").write_text(manifest_hash + "\n", encoding="ascii")
        os.rename(tmp, directory)
    return manifest


@torch.no_grad()
def load_execution_artifact(model, directory, *, expected_manifest_sha256=None):
    """Load into a fresh model with the expected dtype/config and no live graphs."""
    from safetensors.torch import load_file
    if model.training or getattr(model, "_execution_state", None) is not None:
        raise ValueError("Packed load requires a fresh eval model")
    directory = Path(directory)
    manifest_hash = _sha(directory / "manifest.json")
    if manifest_hash != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Artifact manifest checksum mismatch")
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Artifact provenance hash does not match expected manifest")
    m = json.loads((directory / "manifest.json").read_text())
    if m["format"] != "pi05-native-packed-v1" or _sha(directory / "weights.safetensors") != m["weights_sha256"]:
        raise ValueError("Artifact format or tensor checksum mismatch")
    recipe = ExecutionRecipe(**m["recipe"])
    if recipe.prefix_mlp in ("nvfp4_full", "nvfp4_selective"):
        if m.get("nvfp4_layout") != "cutlass_swizzle128x4_e2m1_e4m3" or m.get("runtime_versions") != _runtime_versions():
            raise ValueError("Native packed ABI/runtime mismatch; revalidate and re-export from the original checkpoint")
    requested = getattr(model.config, "execution_recipe", None)
    if requested is not None and ExecutionRecipe(**requested) != recipe:
        raise ValueError("Requested recipe conflicts with the packed artifact")
    saved_config = dict(m["model_config"])
    # New opt-in fields default off when reading older immutable artifacts.
    saved_config["execution_recipe"] = json.loads(json.dumps(asdict(ExecutionRecipe(**saved_config["execution_recipe"]))))
    if _semantic_config(model, recipe) != saved_config:
        raise ValueError("Artifact model configuration mismatch (including normalization/action semantics)")
    device = next(model.parameters()).device
    resolve_hardware_recipe(recipe, device, "error")
    if list(model.optimization_flags()) != m["optimization_flags"]:
        raise ValueError("Artifact optimization flags mismatch")
    entries = selected_linears(model, recipe)
    expected = {name: (parent, leaf, precision) for name, parent, leaf, precision in entries}
    if len(m["modules"]) != len(expected) or {x["name"] for x in m["modules"]} != set(expected):
        raise ValueError("Artifact quantized module coverage mismatch")
    tensors = load_file(str(directory / "weights.safetensors"), device="cpu")
    if set(tensors) != set(m["tensors"]):
        raise ValueError("Artifact tensor keys mismatch")
    for key, tensor in tensors.items():
        spec = m["tensors"][key]
        if list(tensor.shape) != spec["shape"] or str(tensor.dtype) != spec["dtype"]:
            raise ValueError("Artifact tensor metadata mismatch: " + key)
    paths = {id(module): name for name, module in model.named_modules()}
    omitted, packed = set(), {}
    used = set()
    for row in m["modules"]:
        parent, leaf, precision = expected[row["name"]]
        original = getattr(parent, leaf)
        if row["path"] != paths[id(original)] or row["precision"] != precision or row["shape"] != list(original.weight.shape):
            raise ValueError("Artifact module geometry/precision/path mismatch")
        omitted.add(row["path"] + ".weight")
        values = {attr: tensors[key] for attr, key in row["tensors"].items()}
        # Validate everything before mutating model state. Validate geometry on CPU.
        temporary = torch.nn.Linear(original.in_features, original.out_features,
                                    bias=original.bias is not None, dtype=torch.bfloat16, device="meta").eval()
        QuantizedLinear.from_packed(temporary, precision, values, recipe.nvfp4_backend)
        packed[row["name"]] = values
        used.update(row["tensors"].values())
    remaining = {k.removeprefix("model."): v for k, v in tensors.items() if k.startswith("model.")}
    current = model.state_dict()
    model_keys = {"model." + k for k in remaining}
    for alias, canonical in m.get("model_aliases", {}).items():
        if alias in remaining or alias not in current or canonical not in remaining or canonical not in current:
            raise ValueError("Artifact model alias coverage mismatch")
        if _storage_view(current[alias]) != _storage_view(current[canonical]):
            raise ValueError("Artifact tied-weight structure differs from fresh model")
        remaining[alias] = remaining[canonical]
    if set(remaining) != set(current) - omitted or used | model_keys != set(tensors):
        raise ValueError("Artifact remaining model coverage mismatch")
    for key, value in remaining.items():
        if value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError("Artifact remaining model layout mismatch: " + key)
    try:
        model.load_state_dict(remaining, strict=False)
        model.config.execution_recipe = asdict(recipe)
        install_execution_recipe(model, recipe, packed=packed)
        release_original_quantized_weights(model)
    except BaseException:
        # Never let a caller continue with partly restored/random model weights.
        model._execution_state = {"status": "failed"}
        model.invalidate_inference_caches(weights_changed=False)
        raise
    return m


@torch.no_grad()
def release_original_quantized_weights(model):
    """Call only with no live GraphRunner; explicitly retires inference caches."""
    validate_execution_recipe(model)
    if any(r.entries or r._call_lock.locked() for r in getattr(model, "_graph_runners", ())):
        raise RuntimeError("Close/clear all graph runners before releasing original weights")
    state = model._execution_state
    if state["recipe"].fuse_gate_up:
        raise ValueError("Concatenated original weights cannot be released")
    released = 0
    for module in state["modules"]:
        if isinstance(module, QuantizedLinear) and module.weight is not None:
            released += module.weight.numel() * module.weight.element_size()
            module.weight = None
            module._source_identity = (_identity(None), _identity(module.bias))
    state["original_weights_retained"] = False
    model.invalidate_inference_caches(weights_changed=False)
    return released
