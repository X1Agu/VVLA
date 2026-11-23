"""Local worker entry point. The protocol stream is never used for model logs."""
import contextlib
import importlib
import sys
import traceback

from .. import __version__
from ..dispatch import serving_defaults
from ..errors import InputError, UnsupportedConfiguration
from ..protocol.wire import VERSION
from ..protocol.binary import encode_parts, write_parts, read_frame, END_OF_STREAM

BACKENDS = {"pi05": "vvla_pi05.backend", "oft": "vvla_oft.backend", "lingbot": "vvla_lingbot.backend"}


def metadata(engine):
    result=engine.metadata()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        free,total=torch.cuda.mem_get_info()
        properties=torch.cuda.get_device_properties(torch.cuda.current_device())
        result['worker_cuda_memory']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            free=free,total=total,peak_scope='worker lifetime including model loading')
        result['worker_hardware']=dict(name=properties.name,capability=[properties.major,properties.minor],
            uuid=str(getattr(properties,'uuid','unavailable')),torch=str(torch.__version__),cuda_build=torch.version.cuda)
    return result


def main():
    output = sys.stdout.buffer
    engine = None
    dispatch = {}
    while True:
        request_id, fatal = None, False
        try:
            request = read_frame(sys.stdin.buffer)
            if request is END_OF_STREAM:break
            request_id = request["id"]
            if request["version"] != VERSION:
                raise RuntimeError("Worker protocol version mismatch")
            operation = request["operation"]
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "load":
                    if engine is not None:
                        raise RuntimeError("One model owner per worker")
                    config = request["config"]
                    backend = importlib.import_module(BACKENDS[config["model"]])
                    if hasattr(backend, "prepare_config"):
                        config, dispatch = backend.prepare_config(config)
                    engine = backend.load_backend(config)
                    info = metadata(engine)
                    public = info.get("public_recipe") or info.get("provenance", {}).get("public_recipe", {})
                    if dispatch:
                        dispatch["resolved_recipe"] = public.get("resolved", config["recipe"])
                        if public.get("fallback"):
                            dispatch["fallback_reasons"].append(public["fallback"]["reason"])
                        loader = public.get("loader", {})
                        dispatch["fallback_reason"] = "; ".join(dispatch["fallback_reasons"]) or None
                        dispatch["attention_backend"] = (info.get("attention_backend") or
                            loader.get("native_options", {}).get("backend") or loader.get("attention_backend"))
                        torch = sys.modules.get("torch")
                        if torch is not None and torch.cuda.is_initialized():
                            dispatch["blas_library"] = str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]
                        dispatch["sdk_version"] = __version__
                    caps = engine.capabilities()
                    if dispatch:
                        caps["serving_defaults"] = serving_defaults(config["model"], dispatch)
                    result = {"sdk_version": __version__, "metadata": dict(info, dispatch=dispatch), "capabilities": caps}
                elif engine is None:
                    raise RuntimeError("Worker is not loaded")
                elif operation == "infer_batch":
                    result = engine.infer_batch(request["observations"], request["seeds"])
                elif operation == "infer_with_receipt":
                    actions = engine.infer_batch(request["observations"], request["seeds"])
                    receipt = engine.execution_receipt() if hasattr(engine, "execution_receipt") else {}
                    result = {"actions": actions, "execution": receipt}
                elif operation == "reset":
                    result = engine.reset(request.get("episode_id"))
                elif operation == "metadata":
                    result = dict(metadata(engine), dispatch=dispatch)
                elif operation == "close":
                    engine.close()
                    engine = None
                    result = None
                else:
                    raise InputError(f"Unknown operation: {operation}")
            response = {"version": VERSION, "id": request_id, "result": result}
        except Exception as error:
            category = "unsupported" if isinstance(error, UnsupportedConfiguration) else "input" if isinstance(error, InputError) else "fatal"
            fatal = category == "fatal"
            traceback.print_exc(file=sys.stderr)
            response = {"version": VERSION, "id": request_id,
                        "error": {"category": category, "message": f"{type(error).__name__}: {error}"}}
        write_parts(output,encode_parts(response))
        if fatal or (engine is None and request_id is not None):
            break
    if engine is not None:
        with contextlib.redirect_stdout(sys.stderr):
            engine.close()


if __name__ == "__main__":
    main()

