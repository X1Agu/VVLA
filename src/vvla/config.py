from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import re
from .environments import worker_python as locate_worker

from .errors import InputError

MODELS = {"pi05", "oft", "lingbot"}


@dataclass(frozen=True)
class LoadConfig:
    model: str
    checkpoint: str
    device: str = "cuda:0"
    recipe: str | None = None
    assets: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    worker_python: str | None = None
    timeout_s: float = 600.0

    offline: bool = False
    mode: str | None = None
    max_batch: int | None = None

    def __post_init__(self):
        if self.model not in MODELS:
            raise InputError(f"model must be one of {sorted(MODELS)}")
        if type(self.checkpoint) is not str or not self.checkpoint:
            raise InputError("checkpoint must be a local directory or a backend-supported model ID")
        if type(self.device) is not str or not re.fullmatch(r"cuda:\d+", self.device):
            raise InputError("device must be a visible CUDA index, for example cuda:1")
        if self.recipe is None:
            object.__setattr__(self, "recipe", "auto")
        if type(self.recipe) is not str or not self.recipe:
            raise InputError("recipe must be a nonempty string")
        if type(self.assets) is not dict or type(self.options) is not dict:
            raise InputError("assets and options must be dictionaries")
        if type(self.timeout_s) not in (int, float) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise InputError("timeout_s must be finite and positive")
        if self.mode not in (None, "direct", "fixed", "serve"):
            raise InputError("mode must be direct, fixed or serve")
        if self.max_batch is not None and (type(self.max_batch) is not int or self.max_batch < 1):
            raise InputError("max_batch must be a positive integer")
        object.__setattr__(self, "worker_python", locate_worker(self.model, self.worker_python))
        if not Path(self.worker_python).is_file():
            raise InputError(f"Worker Python does not exist: {self.worker_python}")

    def to_dict(self):
        return asdict(self)

