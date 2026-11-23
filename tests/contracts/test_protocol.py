import sys
import numpy as np
import pytest

from vvla.protocol.wire import dumps, loads
from vvla.workers.device import resolve_device
from vvla.errors import InputError


def test_arrays_roundtrip_own_storage_and_preserve_layout_values():
    original = np.arange(60, dtype=np.float32).reshape(10, 6)[:, ::2]
    decoded = loads(dumps({"camera": np.zeros((3, 4, 3), np.uint8), "state": original}))
    np.testing.assert_array_equal(decoded["state"], original)
    original[:] = -1
    assert decoded["state"][0, 1] == 2
    assert decoded["state"].flags.owndata


@pytest.mark.parametrize("value", [np.array([object()], dtype=object), float("nan"), {1: "bad"}, {"$array": "user"}])
def test_wire_rejects_executable_and_ambiguous_values(value):
    with pytest.raises(ValueError):
        dumps(value)


def test_wire_rejects_shape_payload_mismatch_and_huge_allocation():
    for shape in ([4], [10**20]):
        with pytest.raises(ValueError):
            loads('{"$array":true,"dtype":"<f4","shape":'+str(shape)+',"data":"AAAAAA=="}')


def test_visible_index_maps_without_mutating_parent_environment():
    env = {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}
    assert resolve_device("cuda:1", env) == {"requested":"cuda:1", "visible_token":"GPU-b", "worker_device":"cuda:0"}
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert resolve_device("cuda:3", {})["visible_token"] == "3"
    with pytest.raises(InputError):
        resolve_device("cuda:2", env)
    with pytest.raises(InputError):
        resolve_device("cuda:0", {"CUDA_VISIBLE_DEVICES": "-1"})

