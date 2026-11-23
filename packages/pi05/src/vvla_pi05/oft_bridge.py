"""Owned OFT child with deadline-bounded, private MessagePack IPC.

Frames match multiprocessing.connection.Connection (signed big-endian length,
then payload), with the existing 64 MiB limit. A single monotonic deadline covers
queueing, sending, and the entire reply. Serialization is synchronous Python;
the deadline bounds I/O, not arbitrary CPU scheduling or OS process reaping.
"""

import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import msgpack
import msgpack_numpy

MAX_MESSAGE = 64 * 1024 * 1024


def positive_timeout(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class OFTBridge:
    CLOSE_TIMEOUT = 2.0
    EXIT_GRACE = 1.0
    TERMINATE_GRACE = 2.0
    KILL_GRACE = 2.0

    def __init__(self, config):
        self.timeout = positive_timeout(config.get("request_timeout", 120), "request_timeout")
        load_timeout = positive_timeout(config.get("load_timeout", 600), "load_timeout")
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._closed = False
        self.process = None
        self.connection = None
        log_path = Path(config["worker_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("ab", buffering=0)
        parent, child = socket.socketpair()
        try:
            environment = dict(
                os.environ,
                TOKENIZERS_PARALLELISM="false",
                USE_TF="0",
                TF_NUM_INTRAOP_THREADS="2",
                TF_NUM_INTEROP_THREADS="2",
            )
            environment.pop("PYTHONPATH", None)
            self.process = subprocess.Popen(
                [config["worker_python"], "-m", "vvla_oft.worker", "--fd", str(child.fileno())],
                pass_fds=(child.fileno(),),
                env=environment,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
            child.close()
            self.connection = parent
            deadline = time.monotonic() + load_timeout
            self._send(config, deadline)
            ready = self._recv(deadline)
            if ready.get("ready") is not True or ready.get("pid") != self.process.pid:
                raise RuntimeError(f"OFT worker did not become ready: {ready}")
        except BaseException:
            parent.close()
            child.close()
            self._shutdown()
            raise

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OFT worker timed out; the worker is not restarted")
        return remaining

    def _send(self, value, deadline):
        data = msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)
        if len(data) > MAX_MESSAGE:
            raise ValueError("OFT IPC message exceeds 64 MiB")
        # sendall uses a total timeout, not a fresh timeout for each short write.
        for part in (struct.pack("!i", len(data)), data):
            self.connection.settimeout(self._remaining(deadline))
            self.connection.sendall(part)

    def _read_exact(self, length, deadline):
        data = bytearray(length)
        view = memoryview(data)
        offset = 0
        while offset < length:
            self.connection.settimeout(self._remaining(deadline))
            count = self.connection.recv_into(view[offset:])
            if count == 0:
                raise EOFError("OFT worker closed a reply frame")
            offset += count
        return data

    def _recv(self, deadline):
        length = struct.unpack("!i", self._read_exact(4, deadline))[0]
        if not 0 <= length <= MAX_MESSAGE:
            raise ValueError("Invalid OFT IPC frame length (limit: 64 MiB)")
        value = msgpack.unpackb(self._read_exact(length, deadline), object_hook=msgpack_numpy.decode, raw=False)
        if not isinstance(value, dict):
            raise ValueError("OFT IPC reply must be an object")
        if "fatal" in value:
            raise RuntimeError(value["fatal"])
        return value

    def infer(self, requests):
        deadline = time.monotonic() + self.timeout
        if not self._lock.acquire(timeout=self._remaining(deadline)):
            # Nothing was sent; do not poison the other request's active channel.
            raise TimeoutError("OFT request timed out waiting for the IPC channel")
        try:
            if self._closed:
                raise RuntimeError("OFT worker is closed")
            try:
                self._send({"op": "infer", "requests": requests}, deadline)
                outputs = self._recv(deadline)["outputs"]
                if self._closed:
                    raise RuntimeError("OFT worker closed during inference")
                if not isinstance(outputs, list) or len(outputs) != len(requests):
                    raise RuntimeError("OFT response batch size differs")
                return outputs
            except BaseException:
                # A partial write/read or late response can never be reused.
                self._shutdown()
                raise
        finally:
            self._lock.release()

    def _shutdown(self):
        # Independent of the inference lock: close() must interrupt blocked I/O.
        # Serializing cleanup prevents competing waits/signals on the owned PID.
        with self._shutdown_lock:
            self._closed = True
            if self.connection is not None:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
            try:
                if self.process is not None:
                    try:
                        self.process.wait(timeout=self.EXIT_GRACE)
                    except subprocess.TimeoutExpired:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=self.TERMINATE_GRACE)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=self.KILL_GRACE)
            finally:
                self._log.close()

    def close(self):
        # An idle worker gets a bounded graceful close. An in-flight worker is
        # cancelled immediately; never wait for its inference lock or send a
        # second operation into the same stream.
        acquired = self._lock.acquire(blocking=False)
        try:
            if acquired and not self._closed:
                if self.connection is not None and self.process is not None and self.process.poll() is None:
                    try:
                        deadline = time.monotonic() + self.CLOSE_TIMEOUT
                        self._send({"op": "close"}, deadline)
                        self._recv(deadline)
                    except (OSError, EOFError, RuntimeError, ValueError):
                        pass
        finally:
            try:
                self._shutdown()
            finally:
                if acquired:
                    self._lock.release()
