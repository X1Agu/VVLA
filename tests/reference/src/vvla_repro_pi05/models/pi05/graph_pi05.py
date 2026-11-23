# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, fixed-capacity graphs using the main vLLM graph executor.

Only storage and executable graphs are reused across observations. Every image,
token and noise tensor is copied per call; prefix KV is recomputed inside the
graph. This runner belongs to one serialized diffusion worker.
"""

from __future__ import annotations

from collections import Counter
import threading
import weakref

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vvla_repro_pi05.models.pi05.inference_cache_pi05 import copy_position_context
from vvla_repro_pi05.models.pi05.prefix_packing_pi05 import copy_prefix_packing


class Pi05GraphRunner:
    def __init__(self, model, max_graphs: int = 16, *, guard_lifecycle: bool = False):
        if type(guard_lifecycle) is not bool:
            raise ValueError("guard_lifecycle must be a boolean")
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.model = model
        self.max_graphs = max_graphs
        self.entries = {}
        self.stats = Counter()
        self.vllm_config = VllmConfig()
        self._optimization_token = None
        self._guard_lifecycle = guard_lifecycle
        self._call_lock = threading.Lock()
        self._closed = False
        self._last_event = None
        self._last_stream = None
        self._device = None
        if not hasattr(model, "_graph_runners"):
            model._graph_runners = weakref.WeakSet()
        model._graph_runners.add(self)

    @property
    def guard_lifecycle(self):
        return self._guard_lifecycle

    def _retire(self):
        # Synchronize only when destroying captured storage, never on steady replay.
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

    @torch.inference_mode()
    def __call__(self, *, images, image_masks, lang_tokens, lang_masks, noise=None, num_steps=None, generator=None):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Graph runner requires serialized requests")
        stream = None
        try:
            if self._closed:
                raise RuntimeError("Graph runner is closed")
            if self.guard_lifecycle and lang_tokens.is_cuda:
                if self._device is not None and self._device != lang_tokens.device:
                    raise ValueError("Clear graph runner before changing CUDA device")
                stream = torch.cuda.current_stream(lang_tokens.device)
                if self._last_event is not None and stream != self._last_stream:
                    stream.wait_event(self._last_event)
                    self.stats["stream_handoffs"] += 1
                self._device = lang_tokens.device
            return self._call(images=images, image_masks=image_masks, lang_tokens=lang_tokens,
                              lang_masks=lang_masks, noise=noise, num_steps=num_steps, generator=generator)
        finally:
            try:
                if stream is not None:
                    if self._last_event is None:
                        self._last_event = torch.cuda.Event()
                    self._last_event.record(stream)
                    self._last_stream = stream
                    self._device = lang_tokens.device
            finally:
                self._call_lock.release()

    def _call(self, *, images, image_masks, lang_tokens, lang_masks, noise=None, num_steps=None, generator=None):
        steps = num_steps if num_steps is not None else self.model.num_inference_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("num_steps must be a positive integer")
        batch = lang_tokens.shape[0]
        if noise is None:
            shape = (self.model.action_horizon, self.model.action_dim)
            if isinstance(generator, list):
                if len(generator) != batch:
                    raise ValueError("One generator is required per request")
                noise = torch.stack(
                    [
                        torch.randn(shape, device=lang_tokens.device, dtype=torch.float32, generator=g)
                        for g in generator
                    ]
                )
            else:
                noise = torch.randn(
                    (batch, *shape), device=lang_tokens.device, dtype=torch.float32, generator=generator
                )
        tensors = [*images, *image_masks, lang_tokens, lang_masks, noise]
        # Masks update request-local compact metadata in fixed-capacity storage;
        # they do not create a new graph for every text length or row ordering.
        mask_bytes = torch.cat([m[:, None] for m in image_masks] + [lang_masks], 1).cpu().numpy().tobytes()
        # Prepare constants before capture, never run host inspections inside it.
        context = self.model.prepare_inference_context(images, image_masks, lang_masks, steps, positions=False)
        if context["token"] != self._optimization_token:
            if self.guard_lifecycle:
                self._retire()
            else:
                self.entries.clear()
            self._optimization_token = context["token"]
        packing = context.get("packing")
        capacity = None if packing is None else packing["pads"].shape[1]
        key = (steps, tuple((tuple(x.shape), x.dtype, x.device) for x in tensors), context["empty_pattern"], capacity)
        entry = self.entries.get(key)
        newly_captured = entry is None
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                self.stats["capacity_fallbacks"] += 1
                return self.model.sample_actions(
                    images, image_masks, lang_tokens, lang_masks, noise=noise, num_steps=steps
                )
            static = [x.clone() for x in tensors]
            views = len(images)
            static_images = static[:views]
            static_masks = static[views : 2 * views]
            tokens, masks, static_noise = static[-3:]
            metadata = self.model.prepare_attention_metadata(static_masks, masks, fixed_capacity=True, prefix_packing=packing)
            context["positions"] = self.model.prepare_position_context(static_masks, masks, packing)
            static_context = context

            def run():
                return self.model.sample_actions(
                    static_images,
                    static_masks,
                    tokens,
                    masks,
                    noise=static_noise,
                    num_steps=steps,
                    attention_metadata=metadata,
                    inference_context=static_context,
                )

            # Library/JIT initialization must finish before vLLM captures.
            for _ in range(2):
                run()
            torch.accelerator.synchronize()
            wrapper = CUDAGraphWrapper(
                run, self.vllm_config, CUDAGraphMode.FULL, CUDAGraphOptions(weak_ref_output=False)
            )
            descriptor = BatchDescriptor(num_tokens=batch * lang_tokens.shape[1], num_reqs=batch)
            entry = (static, wrapper, descriptor, metadata, mask_bytes, static_context)
            self.entries[key] = entry
            self.stats["captures"] += 1
        else:
            self.stats["replays"] += 1
            if mask_bytes != entry[4]:
                # Retire the entry before mutating captured buffers. If any
                # refresh fails, a later request must build a fresh graph.
                self.entries.pop(key)
                cache = entry[3][0].extra.get("paged_cache") if entry[3][0] is not None else None
                updated = self.model.prepare_attention_metadata(
                    image_masks, lang_masks, fixed_capacity=True, paged_cache=cache, prefix_packing=packing
                )
                for target, source in zip(entry[3], updated, strict=True):
                    if target is None and source is None:
                        continue
                    for name, value in source.extra.items():
                        if isinstance(value, torch.Tensor):
                            target.extra[name].copy_(value)
                        elif target.extra[name] != value:
                            raise ValueError("Graph metadata geometry changed without a new shape key")
                copy_position_context(
                    entry[5]["positions"], self.model.prepare_position_context(image_masks, lang_masks, packing)
                )
                copy_prefix_packing(entry[5].get("packing"), packing)
                entry = (*entry[:4], mask_bytes, entry[5])
                self.entries[key] = entry
                self.stats["metadata_updates"] += 1
            for target, source in zip(entry[0], tensors, strict=True):
                target.copy_(source)
        try:
            with set_forward_context(
                None, self.vllm_config, cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=entry[2]
            ):
                if newly_captured:
                    # Capture records work; replay before returning data.
                    entry[1]()
                # Caller owns its output; graph storage is overwritten on replay.
                return entry[1]().clone()
        except BaseException:
            self.entries.pop(key, None)
            raise
