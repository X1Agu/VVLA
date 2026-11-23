"""Resolve backend assets relative to a checkpoint's portable vvla.json file."""
import json
from pathlib import Path

from .errors import InputError


def resolve_assets(checkpoint, explicit, *, required=(), defaults=None, offline=False, optional_missing=()):
    base = Path(checkpoint).expanduser().resolve()
    manifest = base / "vvla.json"
    assets = dict(defaults or {})
    if manifest.is_file():
        value = json.loads(manifest.read_text())
        if value.get("schema") != 1 or type(value.get("assets")) is not dict:
            raise InputError("vvla.json must contain schema=1 and an assets object")
        assets.update(value["assets"])
    assets.update(explicit)
    result = {}
    for name, value in assets.items():
        if type(value) is not str or not value:
            raise InputError(f"Asset {name} must be a nonempty local path")
        if value.startswith("hf://"):
            from .hub import fetch
            value = fetch(value, offline=offline)
        path = Path(value).expanduser()
        path = (base / path) if not path.is_absolute() else path
        if not path.exists() and name in optional_missing:
            continue
        if not path.exists():
            raise InputError(f"Missing asset {name}: {path}")
        result[name] = str(path.resolve())
    missing = set(required) - result.keys()
    if missing:
        raise InputError(f"Provide assets {sorted(missing)} in checkpoint/vvla.json or assets=; no historical reports are needed")
    return result

