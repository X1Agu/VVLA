"""Ragged eager vision/text prefill followed by one fixed action graph per real width."""
import time

import torch

from .engine import CachedAdaNorm
from .lite_tuning.engine import TunedBatchedEngine


class SplitStageBatchedEngine(TunedBatchedEngine):
    def _allocate_action_buffers(self, state, noise, capacity, suffix_rope):
        device = state.device
        self._allocate_pages(capacity, device)
        self.static = [torch.empty_like(state)]
        self.noise = torch.empty_like(noise)
        self._suffix_rope = tuple(torch.empty_like(t) for t in suffix_rope)
        self.suffix_slots = torch.empty(self.batch*self.s, dtype=torch.int64, device=device)
        self.lengths = {n: torch.empty(self.batch, dtype=torch.int32, device=device) for n in (1, self.s)}
        self._finished = torch.cuda.Event()
        # These constants must stay unchanged. Slots, seqused_k, state, noise,
        # KV and suffix rotary values are deliberately updated before replay.
        self._constant_tensors = [self.dt, *self.time_embs, self.block_table, self.scale, *self.cu.values()]
        for module in self.action.modules():
            if isinstance(module, CachedAdaNorm):
                self._constant_tensors.extend(t for pair in module.modulations for t in pair)
        self._constant_token = self._constant_versions()
        self._signature = self._state_signature()

    def _constant_versions(self):
        return tuple((id(t), t.data_ptr(), t._version) for t in self._constant_tensors)

    def _capture_actions(self):
        stream = torch.cuda.Stream(device=self.noise.device)
        self._capture_stream = stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._action_loop(self.static[0])
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=stream):
            self.output = self._action_loop(self.static[0])
        self.recipe['graph_builds'] += 1
        self._write_recipe()

    @torch.no_grad()
    def __call__(self, images, img_masks, tokens, lang_masks, state, noise=None, image_grid_thw=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('External capture unsupported')
        first = self._graph is None
        started = time.perf_counter()
        if not first:
            self._finished.wait()
            if self._state_signature() != self._signature or self._constant_versions() != self._constant_token:
                raise RuntimeError('Model or captured constants changed; rebuild the engine')
        guard_ms = (time.perf_counter() - started)*1000
        grid_key = tuple(image_grid_thw.flatten().tolist())
        prefix, deep, suffix_rope = self._prepare_prefix(
            images, img_masks, tokens, lang_masks, state, image_grid_thw,
            refresh_vision=grid_key != getattr(self, '_vision_key', None))
        self._vision_key = grid_key
        if first:
            # The public processor pads text to tokenizer_max_length. Reserving
            # the full budget also handles direct calls with unpadded tokens.
            text_budget = max(tokens.shape[1], self.flow.config.tokenizer_max_length)
            capacity = prefix.shape[1] - tokens.shape[1] + text_budget
            self._allocate_action_buffers(state, noise, capacity, suffix_rope)
            self._meta_key = ('split-stage', self.batch, capacity, self.s)
            self.recipe.update(batch_size=self.batch, prefix_capacity=capacity, suffix_tokens=self.s,
                               execution_path='eager-prefix/cuda-graph-action', kv_page_size=self.page_size)
        if max(self.kps) > self.kp:
            raise ValueError('Prefix exceeds the captured KV capacity for this camera/tokenizer geometry')
        slots, lengths = self._page_inputs(state.device)
        self.suffix_slots.copy_(slots)
        for n, value in lengths.items():
            self.lengths[n].copy_(value)
        for dest, src in zip(self._suffix_rope, suffix_rope, strict=True):
            dest.copy_(src)
        self.rope['suffix'] = self._suffix_rope
        self.static[0].copy_(state)
        self.noise.copy_(noise)
        self._run_prefix(prefix, deep)

        capture_ms = 0.0
        if first:
            tick = time.perf_counter()
            self._capture_actions()
            capture_ms = (time.perf_counter() - tick)*1000
        self._graph.replay()
        result = self.output.clone()
        self._finished.record()
        self.last_call = dict(graph_build=first, graph_rebuild=False, capture_ms=capture_ms, graph_replay=True,
                              signature_ms=guard_ms, execution_path='eager-prefix/cuda-graph-action',
                              prefix_valid_per_row=list(self.ps), kv_prefix_tokens_per_row=list(self.kps))
        return result
