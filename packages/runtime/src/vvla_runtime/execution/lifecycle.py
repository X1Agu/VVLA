"""Serialized ownership of reusable CUDA storage across streams and requests."""
from collections import Counter
from contextlib import contextmanager
import threading
import torch

class ExecutionLifecycle:
    def _init_lifecycle(self, guard_lifecycle=True):
        if type(guard_lifecycle) is not bool:
            raise ValueError("guard_lifecycle must be a boolean")
        self.entries = {}
        self.stats = Counter()
        self._guard_lifecycle = guard_lifecycle
        self._call_lock = threading.Lock()
        self._closed = False
        self._last_event = self._last_stream = self._device = None

    @property
    def guard_lifecycle(self):
        return self._guard_lifecycle

    def _retire(self):
        if self._last_event is not None:
            self._last_event.synchronize()
        else:
            devices = {t.device for entry in self.entries.values() for t in entry[0] if t.is_cuda}
            for device in devices:
                torch.cuda.synchronize(device)
        self.entries.clear()
        self._last_event = self._last_stream = self._device = None

    def clear(self):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Cannot clear a graph runner during an active request")
        try:
            self._retire()
            self._optimization_token = None
        finally:
            self._call_lock.release()

    def close(self):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Cannot close a graph runner during an active request")
        try:
            self._retire()
            self._closed = True
        finally:
            self._call_lock.release()

    @contextmanager
    def request_scope(self, device):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Graph runner requires serialized requests")
        stream = None
        try:
            if self._closed:
                raise RuntimeError("Graph runner is closed")
            if self.guard_lifecycle and device.type == 'cuda':
                if self._device is not None and self._device != device:
                    raise ValueError("Clear graph runner before changing CUDA device")
                stream = torch.cuda.current_stream(device)
                if self._last_event is not None and stream != self._last_stream:
                    stream.wait_event(self._last_event)
                    self.stats['stream_handoffs'] += 1
                self._device = device
            yield
        finally:
            try:
                if stream is not None:
                    if self._last_event is None:
                        self._last_event = torch.cuda.Event()
                    self._last_event.record(stream)
                    self._last_stream = stream
                    self._device = device
            finally:
                self._call_lock.release()
