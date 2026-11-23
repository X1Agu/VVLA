import base64
import json
import math
import numpy as np

VERSION = 2
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_ARRAY_BYTES = 64 * 1024 * 1024


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif" or value.nbytes > MAX_ARRAY_BYTES:
            raise ValueError("Only bounded numeric/bool arrays can cross the worker boundary")
        array = np.ascontiguousarray(value)
        return {"$array": True, "dtype": array.dtype.str, "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        if any(type(k) is not str for k in value) or "$array" in value:
            raise ValueError("Protocol dictionary keys must be strings; $array is reserved")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported wire value: {type(value).__name__}")


def decode(value):
    if isinstance(value, dict):
        if "$array" in value:
            if set(value) != {"$array", "dtype", "shape", "data"} or value["$array"] is not True:
                raise ValueError("Invalid array envelope")
            dtype, shape = np.dtype(value["dtype"]), value["shape"]
            if dtype.kind not in "buif" or type(shape) is not list or len(shape) > 8:
                raise ValueError("Invalid array dtype or rank")
            if any(type(x) is not int or x < 0 or x > MAX_ARRAY_BYTES for x in shape):
                raise ValueError("Invalid array shape")
            size = math.prod(shape) * dtype.itemsize
            if size > MAX_ARRAY_BYTES:
                raise ValueError("Array exceeds transport limit")
            raw = base64.b64decode(value["data"], validate=True)
            if len(raw) != size:
                raise ValueError("Array shape does not match bytes")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def dumps(value):
    data = json.dumps(encode(value), allow_nan=False, separators=(",", ":"))
    if len(data.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("Message exceeds transport limit")
    return data


def loads(data):
    if len(data.encode("utf-8") if isinstance(data, str) else data) > MAX_MESSAGE_BYTES:
        raise ValueError("Message exceeds transport limit")
    return decode(json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))

