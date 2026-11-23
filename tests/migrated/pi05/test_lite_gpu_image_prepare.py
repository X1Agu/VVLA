"""Remote-only execution prepared for the optional GPU uint8 image helper."""

import numpy as np
from PIL import Image
import pytest
import torch

from vvla_repro_pi05.lite.gpu_image_prepare import HOOK, INSTALLATION, install_gpu_uint8_prepare
from vvla_repro_pi05.lite.processor import Pi05LiteProcessor, resize_unit_then_siglip, unit_image
from vvla_repro_pi05.models.pi05.config import Pi05Config


class Tokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        self.last_text = text
        return {"input_ids": list(range(max_length)), "attention_mask": [1] * (max_length-1) + [0]}


def processor(device="cpu", resolution=(224, 224)):
    config = Pi05Config(image_resolution=resolution, max_cameras=3, tokenizer_max_length=8,
        input_features={"observation.state": {"shape": [8]},
                        "observation.images.image": {"shape": [3, 256, 256]},
                        "observation.images.image2": {"shape": [3, 256, 256]},
                        "observation.images.empty_camera_0": {"shape": [3, 224, 224]}},
        output_features={"action": {"shape": [7]}},
        norm_stats={"state": {"mode": "mean_std", "mean": [0] * 8, "std": [1] * 8}})
    return Pi05LiteProcessor(config, Tokenizer(), torch.device(device))


def observation(shape=(256, 256, 3)):
    rng = np.random.default_rng(17)
    return {"observation.images.image": rng.integers(0, 256, shape, np.uint8),
            "observation.images.image2": rng.integers(0, 256, shape, np.uint8),
            "state": np.arange(8, dtype=np.float32), "prompt": "pick cube"}


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
    if a.dtype == torch.float32:
        assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
    else:
        assert torch.equal(a, b)


def exact_inputs(a, b):
    for left, right in zip(a[:4], b[:4], strict=True):
        if isinstance(left, list):
            for x, y in zip(left, right, strict=True):
                exact(x, y)
        else:
            exact(left, right)
    if len(a) == 5:
        # Each layout binds to its own freshly allocated tensors; those pointer
        # identities must differ, while mask values/order remain identical.
        assert a[4].cameras == b[4].cameras and a[4].tokens == b[4].tokens
        assert a[4].mask_bytes == b[4].mask_bytes
        a[4].validate(a[0], a[1], a[3])
        b[4].validate(b[0], b[1], b[3])


def test_install_cpu_native_dispatch_manifest_and_close_restore_real_processor(monkeypatch):
    from vvla_repro_pi05.lite import gpu_image_prepare as helper
    proc, obs = processor(resolution=(16, 16)), observation((32, 32, 3))
    before = proc.build_model_inputs(obs, return_host_layout=True)
    build = proc.build_model_inputs.__func__
    handle = install_gpu_uint8_prepare(proc)
    manifest = handle.manifest()
    assert manifest["actual_selected"] == "native" and manifest["image_cache_bytes"] == 0
    assert getattr(proc, HOOK) is handle and getattr(proc, INSTALLATION) is handle
    calls = []
    def native(*args):
        calls.append(args[0])
        return resize_unit_then_siglip(*args)
    monkeypatch.setattr(helper, "_native_resize", native)
    after = proc.build_model_inputs(obs, return_host_layout=True)
    assert len(calls) == 2  # Proves the explicit processor hook was called.
    exact_inputs(before, after)
    assert proc.build_model_inputs.__func__ is build and handle.manifest() == manifest
    with pytest.raises(ValueError, match="already installed"):
        install_gpu_uint8_prepare(proc)
    handle.close(); handle.close()
    assert not hasattr(proc, HOOK) and not hasattr(proc, INSTALLATION)
    exact_inputs(before, proc.build_model_inputs(obs, return_host_layout=True))
    with pytest.raises(RuntimeError, match="closed"):
        handle(obs["observation.images.image"], 16, 16, "cpu")


@pytest.mark.parametrize("kind", ["numpy_float", "tensor_uint8", "tensor_float", "pil"])
def test_unsupported_inputs_use_exact_original_path(kind, monkeypatch):
    from vvla_repro_pi05.lite import gpu_image_prepare as helper
    proc = processor(resolution=(16, 16))
    image = observation((32, 32, 3))["observation.images.image"]
    variants = {"numpy_float": image.astype(np.float32)/255,
                "tensor_uint8": torch.from_numpy(image).permute(2, 0, 1),
                "tensor_float": torch.from_numpy(image).permute(2, 0, 1).float()/255,
                "pil": Image.fromarray(image)}
    value = variants[kind]
    expected = resize_unit_then_siglip(value, 16, 16, "cpu")
    calls = []
    monkeypatch.setattr(helper, "_native_resize", lambda *args: (calls.append(args[0]), resize_unit_then_siglip(*args))[1])
    handle = install_gpu_uint8_prepare(proc)
    exact(handle(value, 16, 16, "cpu"), expected)
    assert len(calls) == 1 and calls[0] is value
    handle.close()


def test_lifecycle_lock_foreign_hook_builder_change_and_error_recovery():
    proc = processor(resolution=(16, 16))
    image = observation((32, 32, 3))["observation.images.image"]
    handle = install_gpu_uint8_prepare(proc)
    handle._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="request"):
            handle(image, 16, 16, "cpu")
        with pytest.raises(RuntimeError, match="request"):
            handle.close()
    finally:
        handle._lock.release()
    setattr(proc, HOOK, object())
    with pytest.raises(RuntimeError, match="hook changed"):
        handle.close()
    setattr(proc, HOOK, handle)
    proc.build_model_inputs = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="processor changed"):
        handle(image, 16, 16, "cpu")
    del proc.build_model_inputs
    with pytest.raises(ValueError):
        handle(image[..., 0], 16, 16, "cpu")
    exact(handle(image, 16, 16, "cpu"), resize_unit_then_siglip(image, 16, 16, "cpu"))
    handle.close()


@pytest.mark.parametrize("value", [1, "true", None])
def test_policy_rejects_nonboolean_gpu_image_option_before_model_allocation(value, monkeypatch, tmp_path):
    from vvla_repro_pi05.lite import pi05
    monkeypatch.setattr(pi05, "_resolve_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(pi05, "load_lite_config", lambda path: (Pi05Config(), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid GPU uint8 option reached model allocation")
    monkeypatch.setattr(pi05, "_construct_pi05_model", forbidden)
    with pytest.raises(ValueError, match="boolean"):
        pi05.Pi05LitePolicy.from_pretrained(tmp_path, config_overrides={"lite_gpu_uint8_prepare": value})


def test_policy_default_off_actual_backend_metadata_owned_actions_and_close_order(monkeypatch):
    from test_lite_pi05 import _Model, _obs, _policy
    from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides
    core, options = _split_lite_overrides({})
    assert core == {} and options["gpu_uint8_prepare"] is False
    baseline = _policy()
    assert not hasattr(baseline.processor, INSTALLATION)
    noise = np.ones((1, 2, 4), np.float32)
    expected = baseline.predict(_obs(), noise)["actions"]
    candidate = Pi05LitePolicy(_Model(), baseline.processor, device="cpu", cuda_graph=False,
                               gpu_uint8_prepare=True, provenance={"checkpoint_sha256": "a" * 64})
    handle = getattr(candidate.processor, INSTALLATION)
    metadata = candidate.metadata()
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["lite_optimizations"]["gpu_uint8_prepare"] is True
    assert metadata["lite_optimizations"]["gpu_uint8_prepare_manifest"] == handle.manifest()
    assert handle.manifest()["actual_selected"] == "native"
    actual = candidate.predict(_obs(), noise)["actions"]
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.owndata and actual.dtype == np.float32
    saved = actual.copy()
    candidate.predict(_obs(), noise * 2)
    np.testing.assert_array_equal(actual, saved)
    assert candidate.metadata() == metadata
    events = []
    class Graph:
        def close(self): events.append("graph")
    candidate.graph_runner = Graph()
    close = handle.close
    def close_hook():
        assert events == ["graph"]
        events.append("image")
        close()
    monkeypatch.setattr(handle, "close", close_hook)
    candidate.close()
    assert events == ["graph", "image"] and handle._closed
    assert not hasattr(candidate.processor, HOOK)
    candidate.close()
    baseline.close()


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="root remote GPU queue only")


@CUDA
def test_cuda_all_uint8_values_unit_fp32_bits_strides_and_measured_diagnostic_kernel():
    from vvla_repro_pi05.lite.gpu_image_prepare_kernel import uint8_to_unit
    with torch.inference_mode():
        sweep = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (1, 256, 3)).copy()
        raw = torch.from_numpy(sweep).permute(2, 0, 1).unsqueeze(0).to("cuda")
        actual, expected = uint8_to_unit(raw), unit_image(sweep).to("cuda")
        exact(actual, expected)
        # Every conversion is FP32; BF16 agreement cannot satisfy this assertion.
        with pytest.raises(ValueError, match="dense HWC"):
            uint8_to_unit(raw.contiguous())


@CUDA
@pytest.mark.parametrize("shape", [(256, 256, 3), (224, 224, 3), (255, 301, 3)])
def test_cuda_whole_processor_strict_parity_noncontiguous_missing_camera_and_dynamic_prompt(shape):
    proc, obs = processor("cuda"), observation(shape)
    obs["observation.images.image2"] = obs["observation.images.image2"][:, ::-1]
    state = obs["state"].copy()
    with torch.inference_mode():
        expected = proc.build_model_inputs(obs, return_host_layout=True)
    handle = install_gpu_uint8_prepare(proc)
    manifest = handle.manifest()
    assert manifest["actual_selected"] == "triton_div_rn"
    with torch.inference_mode():
        actual = proc.build_model_inputs(obs, return_host_layout=True)
        exact_inputs(actual, expected)
        retained = [x.clone() for x in actual[0]]
        handle.close()
        # Mix one eligible camera with a float camera, then change camera/prompt.
        obs["observation.images.image2"] = np.zeros(shape, dtype=np.float32)
        obs["prompt"] = "place block"
        expected_mixed = proc.build_model_inputs(obs, return_host_layout=True)
        handle = install_gpu_uint8_prepare(proc)
        exact_inputs(proc.build_model_inputs(obs, return_host_layout=True), expected_mixed)
        handle.close()
        del obs["observation.images.image2"]
        expected_missing = proc.build_model_inputs(obs, return_host_layout=True)
        handle = install_gpu_uint8_prepare(proc)
        exact_inputs(proc.build_model_inputs(obs, return_host_layout=True), expected_missing)
        for old, saved in zip(actual[0], retained, strict=True):
            exact(old, saved)
        np.testing.assert_array_equal(obs["state"], state)
        assert handle.manifest() == manifest
    handle.close()


@CUDA
def test_cuda_fallback_float_tensor_pil_autocast_and_other_stream_owned_outputs():
    proc = processor("cuda")
    image = observation()["observation.images.image"]
    handle = install_gpu_uint8_prepare(proc)
    with torch.inference_mode():
        for value in (image.astype(np.float32)/255, Image.fromarray(image),
                      torch.from_numpy(image).permute(2, 0, 1)):
            exact(handle(value, 224, 224, "cuda"), resize_unit_then_siglip(value, 224, 224, "cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            exact(handle(image, 224, 224, "cuda"), resize_unit_then_siglip(image, 224, 224, "cuda"))
        first = handle(image, 224, 224, "cuda")
        saved = first.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            later = handle(np.zeros_like(image), 224, 224, "cuda")
        stream.synchronize()
        assert first.data_ptr() != later.data_ptr()
        assert torch.all(later == -1)
        exact(first, saved)
    handle.close()
