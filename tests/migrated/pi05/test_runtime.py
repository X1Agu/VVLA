"""The plugin must reject an unvalidated ABI before importing model code."""

import pytest

from vvla_repro_pi05 import runtime


def test_runtime_accepts_recorded_torch_cuda_local_version(monkeypatch):
    packages = dict(runtime.PINNED_VERSIONS, torch="2.13.0+cu130")
    monkeypatch.setattr(runtime, "version", packages.__getitem__)
    assert runtime.check_runtime() == packages


@pytest.mark.parametrize("package", list(runtime.PINNED_VERSIONS))
def test_runtime_rejects_unknown_versions(monkeypatch, package):
    packages = dict(runtime.PINNED_VERSIONS)
    packages[package] = "0.0.0"
    monkeypatch.setattr(runtime, "version", packages.__getitem__)
    with pytest.raises(RuntimeError, match="other versions have not been validated"):
        runtime.check_runtime()


def test_runtime_missing_package_is_actionable(monkeypatch):
    def missing(package):
        raise runtime.PackageNotFoundError(package)

    monkeypatch.setattr(runtime, "version", missing)
    with pytest.raises(RuntimeError, match="install the pinned runtime first"):
        runtime.check_runtime()
