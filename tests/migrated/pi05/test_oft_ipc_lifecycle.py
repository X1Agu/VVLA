"""Fault injection with real private sockets and owned children; no GPU model."""

import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from vvla_repro_pi05.oft_bridge import OFTBridge


@pytest.fixture
def bridges():
    created = []

    def connect(child_code, timeout=0.15):
        parent, child = socket.socketpair()
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        bridge = object.__new__(OFTBridge)
        bridge._lock = threading.Lock()
        bridge._shutdown_lock = threading.Lock()
        bridge._closed = False
        bridge.timeout = timeout
        bridge.EXIT_GRACE = 0.05
        bridge.TERMINATE_GRACE = 0.5
        bridge.KILL_GRACE = 0.5
        bridge.CLOSE_TIMEOUT = 0.15
        bridge._log = open(os.devnull, "ab")
        # Keep interpreter startup out of the short fault-injection deadline.
        bootstrap = "import os,sys; os.write(int(sys.argv[1]), b'R'); "
        bridge.process = subprocess.Popen(
            [sys.executable, "-c", bootstrap + child_code, str(child.fileno())],
            pass_fds=(child.fileno(),),
            stdout=bridge._log,
            stderr=bridge._log,
        )
        child.close()
        bridge.connection = parent
        created.append(bridge)
        parent.settimeout(5)
        assert parent.recv(1) == b"R"
        return bridge

    yield connect
    for bridge in created:
        bridge.close()
        assert bridge.process.poll() is not None
        assert bridge.connection.fileno() == -1
        assert bridge._log.closed


def assert_poisoned(bridge):
    assert bridge.process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        bridge.infer([{}])


@pytest.mark.parametrize("partial", ["b''", "b'\\x00'", "b'\\x00\\x00\\x00\\x20X'"])
def test_silent_and_partial_replies_obey_deadline(bridges, partial):
    bridge = bridges(f"""
import time
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
c.recv_bytes()
os.write(c.fileno(), {partial})
time.sleep(10)
""")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        bridge.infer([{}])
    assert time.monotonic() - start < 1.5
    assert_poisoned(bridge)


def test_trickle_does_not_refresh_total_deadline(bridges):
    bridge = bridges("""
import time, struct
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
c.recv_bytes()
os.write(c.fileno(), struct.pack('!i', 100))
for i in range(100):
    os.write(c.fileno(), b'x')
    time.sleep(.04)
""")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        bridge.infer([{}])
    assert time.monotonic() - start < 1.5
    assert_poisoned(bridge)


def test_send_backpressure_has_deadline(bridges):
    bridge = bridges("import time; time.sleep(10)")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        bridge.infer([{"payload": b"x" * (4 * 1024 * 1024)}])
    assert time.monotonic() - start < 1.5
    assert_poisoned(bridge)


def test_concurrent_close_interrupts_inference(bridges):
    bridge = bridges("import time; time.sleep(10)", timeout=30)
    entered = threading.Event()
    original = bridge._read_exact

    def reading(*args):
        entered.set()
        return original(*args)

    bridge._read_exact = reading
    errors = []

    def infer():
        try:
            bridge.infer([{}])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=infer, daemon=True)
    thread.start()
    assert entered.wait(2)
    start = time.monotonic()
    bridge.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert time.monotonic() - start < 2
    assert len(errors) == 1
    assert_poisoned(bridge)
    bridge.close()  # idempotent


def test_late_response_is_never_reused(bridges):
    bridge = bridges("""
import time
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
c.recv_bytes()
time.sleep(.3)
c.send_bytes(b'\\x81\\xa7outputs\\x90')
""")
    with pytest.raises(TimeoutError):
        bridge.infer([{}])
    assert_poisoned(bridge)


def test_crash_is_reaped_without_restart(bridges):
    bridge = bridges("os._exit(7)")
    pid = bridge.process.pid
    with pytest.raises((EOFError, OSError)):
        bridge.infer([{}])
    assert bridge.process.returncode == 7
    assert bridge.process.pid == pid
    assert_poisoned(bridge)


@pytest.mark.parametrize(
    "frame",
    [
        "struct.pack('!i', -1)",
        "struct.pack('!i', 64 * 1024 * 1024 + 1)",
        "struct.pack('!i', 1) + b'\\xc1'",  # invalid MessagePack
        "struct.pack('!i', 1) + b'\\x90'",  # not an object
        "struct.pack('!i', 10) + b'\\x81\\xa7outputs\\x90'",  # wrong batch size
    ],
)
def test_bad_frames_poison_channel(bridges, frame):
    bridge = bridges(
        f"""
import struct
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
c.recv_bytes()
os.write(c.fileno(), {frame})
time.sleep(10)
""".replace("import struct", "import struct, time")
    )
    with pytest.raises((ValueError, RuntimeError)):
        bridge.infer([{}])
    assert_poisoned(bridge)


def test_queue_timeout_does_not_poison_unsent_request(bridges):
    bridge = bridges("import time; time.sleep(10)")
    bridge._lock.acquire()
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="waiting"):
            bridge.infer([{}])
        assert time.monotonic() - start < 1
        assert not bridge._closed
        assert bridge.process.poll() is None
    finally:
        bridge._lock.release()


def test_connection_wire_compatibility_and_graceful_close(bridges):
    bridge = bridges(
        """
import msgpack
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
request = msgpack.unpackb(c.recv_bytes(), raw=False)
assert request == {'op': 'infer', 'requests': [{'id': 3}]}
c.send_bytes(msgpack.packb({'outputs': [{'actions': [1, 2, 3]}]}))
assert msgpack.unpackb(c.recv_bytes(), raw=False) == {'op': 'close'}
c.send_bytes(msgpack.packb({'closed': True}))
c.close()
""",
        timeout=2,
    )
    assert bridge.infer([{"id": 3}]) == [{"actions": [1, 2, 3]}]
    bridge.close()
    assert bridge.process.returncode == 0


@pytest.mark.parametrize("name", ["request_timeout", "load_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts_rejected_before_spawn(name, value):
    with pytest.raises(ValueError, match=name):
        OFTBridge({name: value})


def test_loading_partial_handshake_has_deadline(monkeypatch, tmp_path):
    popen = subprocess.Popen
    children = []

    def start(command, **kwargs):
        code = """
import os,sys,time
from multiprocessing.connection import Connection
c = Connection(int(sys.argv[1]))
c.recv_bytes()
os.write(c.fileno(), b'\\x00')
time.sleep(10)
"""
        process = popen([sys.executable, "-c", code, command[-1]], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(OFTBridge, "EXIT_GRACE", 0.05)
    start_time = time.monotonic()
    with pytest.raises(TimeoutError):
        OFTBridge({"worker_python": sys.executable, "worker_log": str(tmp_path / "worker.log"), "load_timeout": 0.3})
    assert time.monotonic() - start_time < 2
    assert children[0].poll() is not None
