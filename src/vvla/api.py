from copy import deepcopy
import uuid

from .config import LoadConfig
from .errors import InputError, WorkerFatalError
from .hub import checkpoint_path
from .workers.client import WorkerClient


class Engine:
    """Owned CPU actions in input order; every request has an explicit seed."""

    owns_outputs = True

    def __init__(self, config):
        self._worker = WorkerClient(config)
        self.last_execution = {}

    def infer_batch(self, observations, seeds, *, request_ids=None):
        if type(observations) is not list or not observations or any(type(o) is not dict for o in observations):
            raise InputError("observations must be a nonempty list of model-specific dictionaries")
        if type(seeds) is not list or len(seeds) != len(observations) or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
            raise InputError("Provide one independent uint63 seed per observation")
        ids = request_ids if request_ids is not None else [uuid.uuid4().hex for _ in observations]
        if len(ids) != len(observations) or any(type(i) is not str or not i for i in ids) or len(set(ids)) != len(ids):
            raise InputError("request_ids must be unique nonempty strings, one per observation")
        reply = self._worker.call("infer_with_receipt", observations=observations, seeds=seeds, request_ids=ids)
        outputs = reply["actions"]
        self.last_execution = reply["execution"]
        if type(outputs) is not list or len(outputs) != len(ids):
            self._worker.close()
            raise WorkerFatalError("Backend returned an invalid output count")
        return outputs

    def infer_requests(self, requests):
        return self.infer_batch([r.observation for r in requests], [r.seed for r in requests],
                                request_ids=[r.request_id for r in requests])

    def reset(self, episode_id=None):
        return self._worker.call("reset", episode_id=episode_id)

    def metadata(self):
        result = self._worker.call("metadata")
        result["device_mapping"] = self._worker.device_mapping.copy()
        return result

    def capabilities(self):
        return deepcopy(self._worker._capabilities)

    def close(self):
        self._worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load(model, *, checkpoint, device="cuda:0", recipe=None, assets=None, options=None,
         worker_python=None, timeout_s=600., mode=None, max_batch=None, revision=None, offline=False):
    """Start a model worker in its installed environment, with no implicit installs."""
    checkpoint = checkpoint_path(str(checkpoint), revision=revision, offline=offline)
    kwargs = dict(offline=offline, model=model, checkpoint=checkpoint, device=device, recipe=recipe,
                  assets=deepcopy(assets or {}), options=deepcopy(options or {}), timeout_s=timeout_s, mode=mode, max_batch=max_batch)
    if worker_python is not None:
        kwargs["worker_python"] = str(worker_python)
    return Engine(LoadConfig(**kwargs))

