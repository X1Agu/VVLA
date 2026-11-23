import re
from ..errors import InputError


def resolve_device(requested, environ):
    if type(requested) is not str or not re.fullmatch(r"cuda:\d+", requested):
        raise InputError("Use a visible CUDA index")
    index = int(requested.split(":")[1])
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        token = str(index)
    else:
        tokens = [s.strip() for s in visible.split(",") if s.strip()]
        if index >= len(tokens) or tokens[index] == "-1":
            raise InputError(f"{requested} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        token = tokens[index]
    return {"requested": requested, "visible_token": token, "worker_device": "cuda:0"}

