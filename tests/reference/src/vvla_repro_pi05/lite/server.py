"""Optional serialized MessagePack HTTP transport around the in-process API.

The benchmark imports the policy directly; this transport is not in its timer.
Run: python -m vvla_repro_pi05.lite.server --config /path/to/adapter.json
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


def serve(policy, *, host="127.0.0.1", port=8001):
    import msgpack
    import msgpack_numpy

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, result):
            data = msgpack.packb(result, default=msgpack_numpy.encode, use_bin_type=True)
            self.send_response(status)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/metadata":
                self._send(200, policy.metadata())
            else:
                self._send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path not in ("/infer", "/predict", "/reset"):
                self._send(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 32 * 1024 * 1024:
                    raise ValueError("Request length must be between 1 byte and 32 MiB")
                body = msgpack.unpackb(self.rfile.read(length), object_hook=msgpack_numpy.decode, raw=False)
                if self.path == "/reset":
                    policy.reset(body.get("episode_id"))
                    result = {"ok": True}
                elif self.path == "/infer":
                    result = policy.infer(body["raw_obs"], body["seed"])
                else:
                    result = policy.predict(body["raw_obs"], body["noise"])
            except (ValueError, TypeError, KeyError) as error:
                self._send(400, {"error": str(error)})
                return
            self._send(200, result)

    # HTTPServer is deliberately single-threaded: each GPU gets one request
    # owner and no concurrent model execution or graph-buffer races.
    with HTTPServer((host, port), Handler) as server:
        try:
            server.serve_forever()
        finally:
            policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON kwargs for create_adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    from vvla_repro_pi05.lite.pi05 import create_adapter
    with open(args.config, encoding="utf-8") as stream:
        policy = create_adapter(json.load(stream))
    serve(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
