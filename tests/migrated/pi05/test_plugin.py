"""Exercise installed entry points and distinguish plugin work from upstream probes."""

import os
import subprocess
import sys

import pytest


def test_importing_plugin_does_not_import_torch_or_models():
    code = """
import sys
import vvla_repro_pi05.plugin
from vvla_repro_pi05.models.pi05 import Pi05Config
assert Pi05Config().chunk_size > 0
assert "torch" not in sys.modules
assert "vvla_repro_pi05.pi05_pipeline" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.parametrize("hide_cuda", [True, pytest.param(False, marks=pytest.mark.gpu)])
def test_discovery_preserves_builtin_and_registers_models_lazily(hide_cuda):
    code = """
from importlib.metadata import entry_points
import faulthandler
import sys
faulthandler.dump_traceback_later(60, repeat=True)
import torch
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.plugins import load_omni_general_plugins
from vllm.plugins import load_general_plugins
# Pinned upstream FA discovery itself probes CUDA. Registration must not
# load model code or allocate tensors beyond that upstream baseline.
initialized = torch.cuda.is_initialized()
allocated = torch.cuda.memory_allocated() if initialized else 0
assert any(p.name == "vvla_repro_pi05" for p in entry_points(group="vllm_omni.general_plugins"))
assert any(p.name == "vvla_repro_pi05" for p in entry_points(group="vllm.general_plugins"))
builtin = DiffusionModelRegistry.models["Pi05Pipeline"]
load_general_plugins()
assert "vla_pi05" in OMNI_PIPELINES
load_omni_general_plugins()
assert DiffusionModelRegistry.models["Pi05Pipeline"] == builtin
assert "VlaPi05Pipeline" in DiffusionModelRegistry.models
assert "VlaOpenVlaOftPipeline" in DiffusionModelRegistry.models
assert "vla_openvla_oft" in OMNI_PIPELINES
assert "vvla_repro_pi05.oft_pipeline" not in sys.modules
assert "vvla_oft" not in sys.modules
assert OMNI_PIPELINES["vla_pi05"].model_arch == "VlaPi05Pipeline"
assert "vvla_repro_pi05.pi05_pipeline" not in sys.modules
assert "vvla_repro_pi05.models.pi05.modeling_pi05" not in sys.modules
assert torch.cuda.is_initialized() == initialized
assert (torch.cuda.memory_allocated() if initialized else 0) == allocated
if sys.argv[1] == "hidden":
    assert not initialized
faulthandler.cancel_dump_traceback_later()
print("plugin discovery passed")
"""
    env = dict(os.environ, VLLM_OMNI_USE_QUACK_FP8="0", OMP_NUM_THREADS="4")
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", code, "hidden" if hide_cuda else "visible"],
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "plugin discovery passed" in result.stdout
