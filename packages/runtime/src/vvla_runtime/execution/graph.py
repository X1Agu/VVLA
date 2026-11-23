"""Bounded native CUDA Graph executor for tensor-only model adapters.

The cache contains executables and storage, never semantic request state or KV.
Adapters own normalization, attention masks, and output geometry. Model mutation
is checked before every call; supported edits retire stale captures. Use
mutation_scope for writes. Quantized recipes can reject stale packing before replay.
"""
import torch
import os
import time
from contextlib import contextmanager
from .state import ModelStateContract
from .lifecycle import ExecutionLifecycle

class TensorGraphRunner(ExecutionLifecycle):
    def __init__(self, forward, *, revision, use_graph=True, max_graphs=4, state_contract=None, prepare_inputs=None, diagnostics_path=None):
        if type(use_graph) is not bool or type(max_graphs) is not int or max_graphs < 1:
            raise ValueError('invalid graph options')
        if not isinstance(revision,str) or not revision:
            raise ValueError('an immutable model revision is required')
        self.forward = forward
        self.revision = revision
        self.use_graph = use_graph
        self.max_graphs = max_graphs
        self._init_lifecycle(True)
        self.state_contract = state_contract or (ModelStateContract(forward) if isinstance(forward,torch.nn.Module) else None)
        self.prepare_inputs = prepare_inputs
        self._state_token = None
        self._state_generation = 0
        diagnostics_path = diagnostics_path or os.environ.get('VLA_GRAPH_DIAGNOSTICS')
        self.graph_diagnostics = None
        if diagnostics_path:
            from .graph_diagnostics import GraphDiagnostics
            self.graph_diagnostics = GraphDiagnostics(diagnostics_path, model='openvla-oft')

    def _record_graph(self, key, outcome, **fields):
        if self.graph_diagnostics is not None:
            from .graph_diagnostics import describe_oft_key
            self.graph_diagnostics.observe(key, lambda: describe_oft_key(key),
                outcome=outcome, cache_size=len(self.entries), capacity=self.max_graphs, **fields)

    def _validate_model_state(self):
        try:
            token = self.state_contract.signature() if self.state_contract is not None else None
        except BaseException:
            self._retire()
            self._state_token = None
            self.stats['state_rejections'] += 1
            raise
        if token != self._state_token:
            changed = self._state_token is not None
            self._retire()
            self._state_token = token
            self._state_generation += 1
            if changed:
                self.stats['state_invalidations'] += 1

    @contextmanager
    def mutation_scope(self):
        # Retire submitted work before changing weights, scales or recipe.
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Cannot mutate model during an active request')
        try:
            if self._closed:
                raise RuntimeError('Graph runner is closed')
            self._retire()
            self._state_token = None
            self.stats['explicit_mutations'] += 1
            yield
        finally:
            self._call_lock.release()

    @torch.inference_mode()
    def __call__(self, **inputs):
        if not inputs or any(not isinstance(t,torch.Tensor) for t in inputs.values()):
            raise ValueError('nonempty tensor-only inputs required')
        names = tuple(sorted(inputs))
        tensors = [inputs[n] for n in names]
        device = tensors[0].device
        if any(t.device != device for t in tensors):
            raise ValueError('input devices differ')
        with self.request_scope(device):
            self._validate_model_state()
            if self.prepare_inputs is not None:
                inputs = self.prepare_inputs(inputs)
                if not inputs or any(not isinstance(t,torch.Tensor) or t.device != device for t in inputs.values()):
                    raise ValueError('Prepared inputs must be tensors on the input device')
                names = tuple(sorted(inputs))
                tensors = [inputs[n] for n in names]
            key = (self.revision,self._state_generation,names,tuple((tuple(t.shape),t.dtype,t.device) for t in tensors))
            if not self.use_graph or device.type != 'cuda':
                self.stats['eager_calls'] += 1
                self._record_graph(key, 'eager', eager_reason='disabled' if not self.use_graph else 'non_cuda')
                return self.forward(**inputs).clone()
            entry = self.entries.get(key)
            if entry is None:
                if len(self.entries) >= self.max_graphs:
                    self.stats['capacity_fallbacks'] += 1
                    self._record_graph(key, 'capacity_fallback')
                    return self.forward(**inputs).clone()
                build_start = time.perf_counter() if self.graph_diagnostics is not None else None
                static = [t.clone() for t in tensors]
                args = dict(zip(names,static))
                # Finish lazy initialization outside capture. No data-dependent
                # Python branches or device-to-host transfers are allowed in forward.
                for _ in range(2):
                    self.forward(**args)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        output = self.forward(**args)
                    entry = (static,graph,output)
                    self.entries[key] = entry
                    self.stats['captures'] += 1
                    self._record_graph(key, 'capture', capture_ms=(time.perf_counter()-build_start)*1000 if build_start is not None else None)
                except BaseException:
                    torch.cuda.synchronize(device)
                    self._record_graph(key, 'capture_error')
                    raise
            else:
                for dst,src in zip(entry[0],tensors):
                    dst.copy_(src)
                self.stats['replays'] += 1
                self._record_graph(key, 'hit')
            try:
                entry[1].replay()
                return entry[2].clone()
            except BaseException:
                # Captured storage must outlive any work already submitted.
                torch.cuda.synchronize(device)
                self.entries.pop(key,None)
                self._record_graph(key, 'replay_error')
                raise
