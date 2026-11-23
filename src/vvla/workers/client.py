from collections import deque
import os
import queue
import subprocess
import threading
import time
import uuid

from ..errors import BackendLoadError, ClosedError, InputError, UnsupportedConfiguration, WorkerFatalError
from ..protocol.wire import VERSION
from ..protocol.binary import encode_parts, write_parts, read_frame, END_OF_STREAM
from .device import resolve_device


class WorkerClient:
    """Serialized owner process. A fatal error poisons this instance permanently."""

    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self._closed = self._failed = False
        self._responses = queue.Queue(maxsize=2)
        self._writes = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.stderr_tail = deque(maxlen=80)
        env = os.environ.copy()
        self.device_mapping = resolve_device(config.device, env)
        env["CUDA_VISIBLE_DEVICES"] = self.device_mapping["visible_token"]
        env.pop("PYTHONPATH", None)
        if config.offline:
            env["HF_HUB_OFFLINE"] = "1"
        # Preserve the venv entrypoint: resolving a python symlink would select
        # the base interpreter's bin directory and lose the worker's tools.
        worker_python = os.path.abspath(config.worker_python)
        env["PATH"] = os.path.dirname(worker_python) + os.pathsep + env.get("PATH", os.defpath)
        self._process = subprocess.Popen(
            [worker_python, "-I", "-m", "vvla.workers.entry"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=env,
        )
        self._threads = [threading.Thread(target=target,daemon=True,name=name) for target,name in
                         ((self._read,'vvla-worker-reader'),(self._write,'vvla-worker-writer'),
                          (self._read_logs,'vvla-worker-logs'))]
        for thread in self._threads:thread.start()
        try:
            payload = config.to_dict()
            payload["device"] = "cuda:0"
            info = self.call("load", config=payload)
            from .. import __version__
            if info.get("sdk_version") != __version__:
                found = info.get("sdk_version", "an older release")
                raise UnsupportedConfiguration(f"Application SDK {__version__} requires matching worker SDK, found {found}")
            self._metadata, self._capabilities = info["metadata"], info["capabilities"]
            self._metadata["device_mapping"] = self.device_mapping.copy()
        except BaseException:
            self._terminate()
            raise

    def _read(self):
        try:
            while True:
                response = read_frame(self._process.stdout)
                if response is END_OF_STREAM:
                    self._emit_response(EOFError("Worker exited"))
                    return
                self._emit_response(response)
        except BaseException as exc:
            self._emit_response(exc)

    def _emit_response(self,value):
        while not self._stop.is_set():
            try:
                self._responses.put(value,timeout=.1)
                return
            except queue.Full:
                continue

    def _write(self):
        while not self._stop.is_set():
            try:message,ack=self._writes.get(timeout=.1)
            except queue.Empty:continue
            try:
                write_parts(self._process.stdin,message)
                ack.put(None)
            except BaseException as error:
                ack.put(error)
                return

    def _read_logs(self):
        try:
            while not self._stop.is_set():
                line = self._process.stderr.readline(8192)
                if not line:return
                self.stderr_tail.append(line.decode('utf-8',errors='replace').rstrip())
        except (OSError,ValueError):
            if not self._stop.is_set():raise

    def _terminate(self):
        self._stop.set()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._closed = True
        for thread in self._threads:thread.join(timeout=1)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            stream.close()

    def call(self, operation, **payload):
        with self._lock:
            if self._failed:
                raise WorkerFatalError("Worker is poisoned; load a fresh owner")
            if self._closed:
                raise ClosedError("Worker is closed")
            request_id = uuid.uuid4().hex
            # Encode before writing so invalid user input cannot corrupt the stream.
            try:
                message = encode_parts(dict(version=VERSION, id=request_id, operation=operation, **payload))
            except (TypeError, ValueError) as error:
                raise InputError(str(error)) from error
            try:
                deadline=time.monotonic()+self.config.timeout_s
                ack=queue.Queue(maxsize=1)
                self._writes.put((message,ack),timeout=self.config.timeout_s)
                written=ack.get(timeout=max(0,deadline-time.monotonic()))
                if isinstance(written,BaseException):raise written
                result = self._responses.get(timeout=max(0,deadline-time.monotonic()))
                if isinstance(result, BaseException):
                    raise result
                if result.get("version") != VERSION or result.get("id") != request_id:
                    raise ValueError("Worker protocol version/response ID mismatch")
                if "error" in result:
                    record = result["error"]
                    if record["category"] == "input":
                        raise InputError(record["message"])
                    if record["category"] == "unsupported":
                        raise UnsupportedConfiguration(record["message"])
                    raise BackendLoadError(record["message"]) if operation == "load" else WorkerFatalError(record["message"])
                return result["result"]
            except InputError:
                raise
            except Exception as error:
                self._failed = True
                self._terminate()
                if isinstance(error, WorkerFatalError):
                    raise
                raise WorkerFatalError(f"Worker communication failed: {error}; logs={list(self.stderr_tail)[-5:]}") from error

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self.call("close")
            except WorkerFatalError as error:
                # Retirement succeeds if the child is actually dead, even when
                # its graceful-close response was fatal. _terminate still raises
                # if process disposal itself fails.
                self.close_diagnostic=str(error)
            finally:
                self._terminate()

