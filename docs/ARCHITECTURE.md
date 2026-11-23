# Worker and service architecture

Each Engine owns one process started with its selected interpreter and python -I. The application imports NumPy. Torch, model code and native operators load inside the model-specific worker. The worker maps the chosen device to its local cuda:0 and selects the recipe before loading weights. Worker process lifetime bounds CUDA graphs, packed weights and model state.

The direct SDK sends binary protocol-v2 frames on private pipes. The network client and server use the same typed frame format. Frames carry bounded JSON structure followed by raw contiguous CPU array bytes. Received arrays own their storage. Local asynchronous submissions snapshot caller-owned observations once. A decoded network request already has private storage. Returning a published action never exposes a CUDA replay buffer.

The local server owns one execution thread and a finite ingress queue. Model capabilities specify compatible widths and independent text fields. FIFO selects only real rows with compatible observations. The worker receipt reports actual tensor width. A disagreement with the dispatch width fails the owner. Cancellation can discard queued work while an in-flight model call drains. Each request receipt records an owner generation and a call number. Inspect /metrics to distinguish the first capture of a layout, a capture replacing an existing graph, a replay and capacity fallback.

OFT pads tokens where the model requires a stable layout. The batch axis remains equal to the number of real requests. LingBot cache32 serializes dedicated calls. TrueBatch uses fixed layouts; split-stage reuses a separate action graph for each supported width. The latter can be requested explicitly.

The pip distributions contain the SDK, model code and runtime operators. Test-only numerical references stay in the repository. Upstream model sources, checkpoint weights and native vLLM/FlashAttention binaries are provisioned as installation assets. [Source lineage](SOURCE_LINEAGE.json)
