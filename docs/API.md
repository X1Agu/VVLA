# Inference and service API

## Direct and fixed batches

~~~python
import vvla

with vvla.load("oft", checkpoint="/models/oft") as engine:
    action = engine.infer_batch([obs], [17], request_ids=["arm-a"])[0]
    print(engine.metadata()["dispatch"])
    print(engine.last_execution)

with vvla.load("oft", checkpoint="/models/oft", mode="fixed", max_batch=4) as engine:
    actions = engine.infer_batch(observations, seeds, request_ids=ids)
~~~

Direct mode defaults to a dedicated B1 worker. Fixed mode processes the caller's real rows in order. Each returned action is an independently owned CPU float32 array. Use engine.capabilities() to read supported widths, action shape, observation contract and seed semantics. The direct SDK uses worker IPC. The network server and FIFO queue are optional.

| Model | Observation | Action |
| --- | --- | --- |
| π0.5 | images dictionary keyed by checkpoint cameras, state vector and prompt string | Checkpoint horizon and action dimension, commonly 50 × 7 |
| OFT | full_image and wrist_image as HWC uint8 RGB, state of length 8, prompt, optional norm_key | 8 × 7, checkpoint normalization and gripper convention |
| LingBot | task, observation.state of length 14, observation.images.cam_high, observation.images.cam_left_wrist, observation.images.cam_right_wrist as HWC uint8 RGB | 50 × 14 in physical RoboTwin coordinates |

Pass an integer seed for every row. LingBot uses uint32, π0.5 accepts uint63, and deterministic OFT accepts and records the seed contract. Published state/action conventions for OpenPI H10 differ from the LeRobot H50 input contract. [OpenPI](OPENPI_CONTRACT.md)

## Serve and call

~~~bash
vvla serve --model oft --checkpoint /models/oft --port 8000
vvla serve --model lingbot --checkpoint /models/lingbot --recipe split-stage --max-batch 4
~~~

~~~python
from vvla import VvlaClient

async def infer_one(observation):
    async with VvlaClient("http://127.0.0.1:8000", transport="ws") as client:
        reply = await client.infer(observation, seed=17, request_id="arm-1-turn-8", timeout_s=5)
        return reply["result"], reply["scheduler"]
~~~

POST /infer accepts application/vnd.vvla.frame. GET /ws carries the same binary format and multiplexes live request IDs over one connection. Each message encodes a request_id, observation, seed, optional timeout_s and target_slo_ms. Array dtype, shape and bytes are preserved. The default network input limit is 16 MiB. GET /health, /metadata and /metrics return JSON. The service exposes dispatch, FIFO width/wait, model calls by executed width, graph builds/rebuilds/replays, outcomes and recent queue/execution P95. Metrics have an owner generation. Request receipts identify the model call within that generation.

For a controller that can use an action after its soft SLO, pass `valid_for_ms` with `target_slo_ms` to `VvlaClient.infer`. The validity budget starts when the server handles the decoded request. The scheduler reports `slo_met` and `action_valid` separately. A robot that timestamps the observation before sending it can also check the chunk age when applying it. Local callers can pass an absolute `valid_until_at` on the server's `time.perf_counter()` clock.

RemoteError exposes code, status and request_id. The FIFO service rejects an exhausted queue with HTTP 429 and reports expired calls with 504. Cancelling a queued client drops its pending work. An in-flight GPU call drains before the worker can be reused. After a fatal owner error, the next request can start a new generation. The failed action is never resubmitted automatically.

## Scheduler and configuration

The measured per-card settings apply when --max-batch and --wait-ms are omitted. FIFO is the default. EDF, Slack-FIFO, Slack-EDF and cohort Slack-EDF are explicit --policy choices. Predictive policies take --service-profile with measured positive service_ms values for each supported batch width, for example {"service_ms":{"1":40,"2":60,"4":90}}. Set per-request target_slo_ms for deadline-aware use.

The SDK includes two frozen service-cost tables for RTX PRO 6000 Blackwell Server Edition. The π0.5 H50 Prefix FP8 multi-arm run used the first table with cap 16 and a 2 ms wait. The OFT BF16 soup-can run loaded a B4-capable engine and limited the scheduler to B2 with zero wait. The native server accepts `--dispatch-cap` (Python `dispatch_cap`) for this distinction. These costs include worker execution and IPC on this card. Other devices or recipes can supply their own JSON file.

~~~bash
vvla serve --model pi05 --checkpoint /models/pi05 --recipe prefix-fp8 --max-batch 16 --wait-ms 2 \
    --policy cohort-slack-edf --service-profile pi05-g4-prefix-fp8
vvla serve --model oft --checkpoint /models/oft --recipe bf16 --max-batch 4 --dispatch-cap 2 --wait-ms 0 \
    --policy cohort-slack-edf --service-profile oft-g4-bf16
~~~

Use --config file.json for SDK arguments, including model, checkpoint, assets, options and worker_python. Command-line model fields override the file. Local async callers can use DynamicBatchServer.from_model(...) with the same settings. The optional --backend omni integrates with upstream AsyncOmni in a separate controller environment containing the exact pinned vLLM and vLLM-Omni versions and the vvla[omni] extra. The pipeline calls the installed model worker and exposes upstream co-batching through the same binary client. The tested upstream controller commit is 98c20414423d0c19af7fa220e5a0244bae7734dc. Install its development wheel in the controller environment. [Plugin](../src/vvla/integrations/omni/plugin.py)
