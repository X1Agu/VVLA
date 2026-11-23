"""Complete, independently captured B1 graph entries over shared frozen weights.

Only one entry executes at a time. Each entry owns its static input/output, KV,
positions, constants, capture stream, graph pool, and upstream MoE workspaces.
Eviction and failed captures drain before releasing any of those references.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time

import torch

from .engine import TunedEngine


ENTRY_FIELDS = (
    'template', 'prefix_pad', 'prefix_pos', 'prefix_indices', 'visual_indices', 'p', 'packed_pos',
    'num_images', 'patch_count', 'boundaries', 'image_token_len', 'image_end',
    'language_end', 'prefix_causal', 'h', 's', 'suffix_pos',
    'cache_prefix_indices', 'kp', 'rope', 'page_size', 'kcache', 'vcache',
    'block_table', 'prefix_slots', 'suffix_slots', 'scale', 'cu', 'lengths',
    'static', 'noise', '_signature', '_constant_tensors', '_constant_token',
    '_graph', '_meta_key', '_capture_stream', 'output',
)
VISION_FIELDS = ('pos_embeds', 'position_embeddings', 'cu_seqlens',
                 'visual_split_sizes', 'visual_max_seqlen')
MOE_FIELDS = ('_robby_moe_workspace', '_robby_moe_workspace_key')


@dataclass(frozen=True)
class GraphEntry:
    key: tuple
    key_sha256: str
    fields: dict
    vision: tuple
    experts: tuple

    @classmethod
    def retain(cls, engine, key):
        from .graph_diagnostics import describe_lingbot_key
        digest = hashlib.sha256(json.dumps(describe_lingbot_key(key),sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        return cls(key, digest, {name: getattr(engine, name) for name in ENTRY_FIELDS},
            tuple(getattr(engine.qwe, name) for name in VISION_FIELDS),
            tuple(tuple(getattr(layer.mlp.experts, name) for name in MOE_FIELDS)
                  for layer in engine.action.layers))

    def activate(self, engine):
        for name, value in self.fields.items():
            setattr(engine, name, value)
        for name, value in zip(VISION_FIELDS, self.vision, strict=True):
            setattr(engine.qwe, name, value)
        for layer, values in zip(engine.action.layers, self.experts, strict=True):
            for name, value in zip(MOE_FIELDS, values, strict=True):
                setattr(layer.mlp.experts, name, value)

    def reset(self):
        # The owner has already synchronized the physical device. Keep all
        # storage reachable if reset raises so close/eviction can be retried.
        self.fields['_graph'].reset()


class PooledTunedEngine(TunedEngine):
    def __init__(self, *args, graph_cache_capacity=8, graph_memory_budget_bytes=None,
                 capture_reserve_bytes=64*1024*1024, **kwargs):
        if type(graph_cache_capacity) is not int or not 1 <= graph_cache_capacity <= 64:
            raise ValueError('graph_cache_capacity must be an integer in [1,64]')
        self._entries = OrderedDict()
        self._capacity = graph_cache_capacity
        self._entry_lock = threading.Lock()
        self._pool_failed = False
        self._pool_closed = False
        self._evictions = 0
        super().__init__(*args, **kwargs)
        self._memory_budget = None
        if graph_memory_budget_bytes is not None:
            from vvla_runtime.memory import GraphMemoryBudget, cuda_snapshot
            self._memory_budget = GraphMemoryBudget(graph_memory_budget_bytes, cuda_snapshot(),
                                                     capture_reserve_bytes=capture_reserve_bytes)
        self.recipe['graph_cache'] = dict(policy='LRU',capacity=graph_cache_capacity,
            entry_ownership='static_io_kv_positions_constants_stream_graph_moe_workspace',
            independent_capture_pools=True, shared_weights=True)
        self.recipe['graph_cache']['memory_budget_bytes'] = graph_memory_budget_bytes

    def _measure_memory(self):
        budget = getattr(self, '_memory_budget', None)
        if budget is not None:
            from vvla_runtime.memory import cuda_snapshot, tensor_storage_bytes
            values = [(entry.fields, entry.vision, entry.experts) for entry in self._entries.values()]
            budget.observe(cuda_snapshot(), tensor_storage_bytes(values))
        return budget

    def _evict_oldest(self):
        # Caller has drained the device and holds the entry lock.
        key = next(iter(self._entries))
        victim = self._entries[key]
        victim.reset()
        if getattr(self, '_graph', None) is victim.fields['_graph']:
            for name in ENTRY_FIELDS:
                self.__dict__.pop(name, None)
            for name in VISION_FIELDS:
                setattr(self.qwe, name, None)
            for layer in self.action.layers:
                for name in MOE_FIELDS:
                    setattr(layer.mlp.experts, name, None)
            self._graph = self._meta_key = self._signature = None
        del self._entries[key]
        self._evictions += 1

    def _admit_capture(self):
        budget = self._measure_memory()
        while self._entries and (len(self._entries) >= self._capacity or
                                 (budget is not None and not budget.fits(budget.reserve))):
            self._evict_oldest()
            if budget is not None:
                torch.cuda.empty_cache()
                self._measure_memory()
        if budget is not None and not budget.fits(budget.reserve):
            raise MemoryError('Capture reserve cannot fit graph budget; rebuild with a larger budget')

    def _check_capture_budget(self, key):
        budget = self._measure_memory()
        if budget is None:
            return
        while not budget.fits() and len(self._entries) > 1:
            self._evict_oldest()
            torch.cuda.empty_cache()
            self._measure_memory()
        if not budget.fits():
            # Keep the candidate reachable until its drained reset succeeds.
            self._evict_oldest()
            torch.cuda.empty_cache()
            self._measure_memory()
            raise MemoryError('One captured graph exceeds the resident memory budget; owner must close')

    def _validate_active(self):
        if self._signature is not None and self._state_signature() != self._signature:
            raise RuntimeError('Model parameters changed; rebuild engine and graphs')
        if self._signature is not None and tuple((id(t),t.data_ptr(),t._version)
                for t in self._constant_tensors) != self._constant_token:
            raise RuntimeError('Cached metadata or modulation changed; rebuild engine')

    @torch.no_grad()
    def __call__(self, images, img_masks, tokens, lang_masks, state, noise=None, image_grid_thw=None):
        if not self._entry_lock.acquire(blocking=False):
            raise RuntimeError('Graph entries require serialized ownership')
        try:
            if self._pool_closed or self._pool_failed:
                raise RuntimeError('Graph pool closed or failed; close the owner before rebuilding')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('External capture unsupported')
            vals = [images,img_masks,tokens,lang_masks,state,image_grid_thw]
            key = tuple((tuple(v.shape),str(v.dtype),str(v.device)) for v in vals) + (
                bytes(img_masks.cpu().numpy()),bytes(lang_masks.cpu().numpy()),
                bytes(image_grid_thw.cpu().numpy()))
            entry = self._entries.get(key)
            tick = time.perf_counter()
            if entry is not None:
                entry.activate(self)
            self._validate_active()
            signature_ms = (time.perf_counter()-tick)*1000
            previous_evictions = self._evictions
            built, capture_ms = entry is None, 0.
            if built:
                tick = time.perf_counter()
                torch.cuda.synchronize()
                self._admit_capture()
                for name in ENTRY_FIELDS:
                    self.__dict__.pop(name, None)
                self._graph = self._meta_key = self._signature = None
                for layer in self.action.layers:
                    for name in MOE_FIELDS:
                        setattr(layer.mlp.experts, name, None)
                self._prepare(*vals)
                self._meta_key = key
                self.noise.zero_()
                stream = torch.cuda.Stream()
                self._capture_stream = stream
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._execute()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                self._graph = graph
                # Do not share a graph memory pool: replay order is arbitrary.
                with torch.cuda.graph(graph,stream=stream):
                    self.output = self._execute()
                entry = GraphEntry.retain(self,key)
                self._entries[key] = entry
                torch.cuda.synchronize()
                self._check_capture_budget(key)
                self.recipe['graph_builds'] += 1
                capture_ms = (time.perf_counter()-tick)*1000
            self._entries.move_to_end(key)
            for dst, src in zip(self.static,vals,strict=True):
                dst.copy_(src)
            if noise is None:
                noise = torch.randn(self.noise.shape,device=state.device,dtype=state.dtype)
            self.noise.copy_(noise)
            self._graph.replay()
            self.last_call = dict(graph_build=built,graph_rebuild=self._evictions > previous_evictions,capture_ms=capture_ms,
                signature_ms=signature_ms,graph_replay=True,cache_size=len(self._entries),
                capacity=self._capacity,evictions=self._evictions,
                graph_key_sha256=entry.key_sha256,prefix_valid=self.p,
                language_valid=int(sum(key[-2])))
            if getattr(self, '_memory_budget', None) is not None:
                self.last_call['memory'] = self._measure_memory().report()
            if self.graph_diagnostics is not None:
                from .graph_diagnostics import describe_lingbot_key
                digest = self.graph_diagnostics.observe(key,lambda:describe_lingbot_key(key),
                    outcome='capture' if built else 'hit',capture_ms=capture_ms,
                    cache_size=len(self._entries),capacity=self._capacity)
                self.last_call['graph_key_sha256'] = digest
            return self.output.clone()
        except BaseException:
            # A failed drain/capture retains both old entries and the incomplete
            # active state. Never replay after an uncertain device operation.
            self._pool_failed = True
            raise
        finally:
            self._entry_lock.release()

    def close(self):
        if not self._entry_lock.acquire(blocking=False):
            raise RuntimeError('Cannot close an executing graph pool')
        try:
            if self._pool_closed:
                return
            torch.cuda.synchronize()
            active = getattr(self,'_graph',None)
            owned = {id(entry.fields['_graph']) for entry in self._entries.values()}
            if active is not None and id(active) not in owned:
                active.reset()
            # Delete each entry only after successful reset. A later reset
            # failure keeps every remaining owner available for a close retry.
            for key in list(self._entries):
                self._entries[key].reset()
                del self._entries[key]
            self._graph = None
            for name in ENTRY_FIELDS:
                if name != '_graph':
                    self.__dict__.pop(name,None)
            for name in VISION_FIELDS:
                if hasattr(self,'qwe'):
                    setattr(self.qwe,name,None)
            for layer in getattr(getattr(self,'action',None),'layers',()):
                for name in MOE_FIELDS:
                    setattr(layer.mlp.experts,name,None)
            self._pool_closed = True
        finally:
            self._entry_lock.release()
