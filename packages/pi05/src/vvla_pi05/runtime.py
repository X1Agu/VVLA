"""The tested runtime boundary. This module does not import torch or initialize CUDA."""

from importlib.metadata import PackageNotFoundError, version

OMNI_COMMIT = "98c20414423d0c19af7fa220e5a0244bae7734dc"
OMNI_VERSION = "0.32.0.dev0+g98c204144"
PINNED_VERSIONS = {
    "vllm": "0.32.0",
    "vllm-omni": OMNI_VERSION,
    "torch": "2.13.0",
    "transformers": "5.19.0",
}


def check_runtime() -> dict[str, str]:
    """Fail with an actionable error instead of using an untested native ABI."""
    found = {}
    for package, expected in PINNED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"vllm-vla requires {package}=={expected}; install the pinned runtime first") from exc
        # PyTorch wheels can include the CUDA build in their local version.
        comparable = actual.split("+", 1)[0] if package == "torch" else actual
        if comparable != expected:
            raise RuntimeError(
                f"vllm-vla requires {package}=={expected}, found {actual}. "
                "Use this project's pinned runtime; other versions have not been validated."
            )
        found[package] = actual
    return found

