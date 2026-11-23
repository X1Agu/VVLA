"""The migration must preserve the archived evidence and relocated numerical source."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_relocated_frozen_source_is_unchanged():
    manifest = json.loads((ROOT / "provenance/pi05-source.json").read_text())
    for row in manifest["files"]:
        relocated = ROOT / "src/vvla_repro_pi05/models/pi05" / row["file"]
        assert digest(relocated) == row["relocated_sha256"]
        original = Path(manifest["source_release"]) / "pi05" / row["file"]
        if original.exists():
            assert digest(original) == row["frozen_sha256"]
            if row["file"] != "__init__.py":
                expected = original.read_bytes().replace(b"vllm_omni.diffusion.models.pi05.", b"vvla_repro_pi05.models.pi05.")
                if row["file"] == "modeling_pi05.py":
                    expected = expected.replace(
                        b"        vlm_config_hf.vision_config.projection_dim = 2048\n",
                        b"        vlm_config_hf.projection_dim = vlm_config.width\n"
                        b"        vlm_config_hf.vision_config.projection_dim = vlm_config.width\n",
                    )
                assert relocated.read_bytes() == expected


def test_local_archived_evidence_is_unchanged():
    rows = json.loads((ROOT / "provenance/preserved-evidence.json").read_text())
    if not all(Path(row["path"]).exists() for row in rows):
        pytest.skip("Historical local archive is not present on this machine")
    for row in rows:
        assert digest(Path(row["path"])) == row["sha256"]


def test_oft_relocation_preserves_numerical_source():
    manifest = json.loads((ROOT / "provenance/oft-source.json").read_text())
    for row in manifest["files"]:
        target = ROOT / row["relocated"]
        assert digest(target) == row["relocated_sha256"]
        original = Path(row["original"])
        if original.exists():
            assert digest(original) == row["source_sha256"]
            expected = original.read_bytes().replace(b"from vla_execution", b"from vvla_runtime.execution")
            expected = expected.replace(b"from openvla_oft", b"from vvla_oft")
            if target.name == 'reference.py':
                expected = expected.replace(
                    b"    model=OpenVLAForActionPrediction.from_pretrained(checkpoint,config=cfg,torch_dtype=torch.bfloat16,\n"
                    b"                                                   attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()\n",
                    b"    model,loading_info=OpenVLAForActionPrediction.from_pretrained(checkpoint,config=cfg,torch_dtype=torch.bfloat16,\n"
                    b"                                                   attn_implementation='sdpa',low_cpu_mem_usage=True,output_loading_info=True)\n"
                    b"    loading_info['unmaterialized_keys']=[name for name,tensor in model.state_dict().items() if tensor.is_meta]\n"
                    b"    if any(loading_info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','unmaterialized_keys')):\n"
                    b"        raise ValueError('OFT checkpoint does not exactly match the supported model: '+repr(loading_info))\n"
                    b"    model=model.to(device).eval()\n",
                )
            assert target.read_bytes() == expected
