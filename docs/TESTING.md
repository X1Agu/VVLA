# Run and verify the beta

Run the local CPU contracts, source-distribution round trip and GPU mechanism checks with a compatible ML environment. The packaging check builds each sdist, builds its wheel, creates a new venv outside the source directories, verifies imports/resources, exercises worker IPC and can compile native extensions. It reuses installed ML dependencies. The beta also supports a standalone application install from the SDK wheel and its serve and hub extras. Full target-card dependency installation is tracked separately.

~~~bash
python -m pip install '.[test]'
python tools/run_cpu_checks.py
python tools/check_release.py --native
python tools/run_gpu_checks.py
~~~

The CPU suite includes real TCP HTTP/WebSocket clients, request IDs, reordered rows, cancellation, timeouts, overload and restarting a failed worker. CUDA checks exercise tiny model geometry. To check full checkpoints, record the card, driver, worker packages, revisions and engine metadata with each run.

## Run VVLA benchmarks

Save real application observations and explicit seeds in a corpus. Use at least four observations for fixed B4 and varied observations for C8. Arrays should use the production shape and dtype.

~~~python
from pathlib import Path
from vvla.protocol.binary import encode_frame

Path("observations.vvla").write_bytes(encode_frame({
    "observations": observations,
    "seeds": list(range(len(observations)))
}))
~~~

~~~bash
vvla bench --mode b1 --config engine.json --input observations.vvla
vvla bench --mode fixed --batch-size 4 --config engine.json --input observations.vvla
vvla bench --mode c8 --url http://127.0.0.1:8000 --transport ws \
    --input observations.vvla --warmup 4 --iterations 20 --output c8.json
~~~

The report's RTX PRO 6000 π0.5 Prefix FP8 loopback HTTP settings were FIFO, cap 16 and a 2 ms collection window. Use the frozen H50 checkpoint, tokenizer and observations for a comparable run. Start the server and run the client in a second terminal.

~~~bash
vvla serve --model pi05 --checkpoint /models/pi05 --recipe prefix-fp8 --max-batch 16 --wait-ms 2
vvla bench --mode c32 --url http://127.0.0.1:8000 --transport http \
    --input observations.vvla --warmup 4 --timeout 600 --duration 60 --output c32.json
~~~

The online modes are `c1`, `c8`, `c16` and `c32`. Each client has at most one request in flight. With `--duration`, throughput and latency use replies received during the window. Later replies appear in `drain_actions` and `drain_s`. An empty window reports zero throughput and null percentiles. Warm the widths used by the workload before timing. The report's G4 HTTP C8 comparison measured 83.27 chunks/s at 0 ms and 94.31 at 2 ms. Its C32 result at 2 ms was 107.73. For this recipe and cap, the binary FIFO server defaults to 2 ms and the in-process queue to 0 ms. Single-client HTTP reached 42.55 chunks/s with explicit `--wait-ms 0` and 39.05 at 2 ms.

B1 and fixed measure direct SDK calls. Without --url, C8 uses the local asynchronous service. With --url, it opens eight independent closed-loop client connections to the running service. Warmup is excluded from the measured duration and actual call histogram. Results include action count, wall time, actions per second and request latency P50/P95. Compare timings only within the same entry point, model assets, workload, precision and warmup. For full-checkpoint acceptance also check actions on frozen observations, reorder four rows, replace one row, verify grippers and rerun the mixed client load. [Release acceptance](RELEASE_REQUIREMENTS.zh-CN.md)
