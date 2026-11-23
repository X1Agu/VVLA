# SPDX-License-Identifier: Apache-2.0
"""Native torch CUDA Graphs; fixed buffers, fresh observations, owned outputs."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
import threading
import weakref

import torch

from vvla_pi05.models.pi05.inference_cache_pi05 import copy_position_context
from vvla_pi05.models.pi05.prefix_packing_pi05 import copy_prefix_packing


@dataclass
class _Entry:
    tensors: list
    graph: object
    output: torch.Tensor
    metadata: tuple
    mask_bytes: bytes
    context: dict
    # Keep Inductor-owned constants/workspaces alive with their captured graph.
    compiled_callable: object = None


class Pi05NativeGraphRunner:
    """A single serialized device owner. No prefix KV crosses requests.

    All inference and tensor copies use the caller's CUDA stream; a recorded
    event orders a subsequent request on another stream. Returned tensors are
    cloned before that event, so replay never overwrites a caller's result.
    """

    def __init__(self, model, max_graphs=16, *, elide_identity_gathers=False, host_layout_cache=False,
                 compile_sample_actions=False):
        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        self.model = model
        self.max_graphs = max_graphs
        self.elide_identity_gathers = elide_identity_gathers
        if type(compile_sample_actions) is not bool:
            raise ValueError("compile_sample_actions must be a boolean")
        self.compile_sample_actions = compile_sample_actions
        if compile_sample_actions:
            from vvla_pi05.lite.compilation import validate_compile_model
            validate_compile_model(model)
        if type(host_layout_cache) is not bool:
            raise ValueError("host_layout_cache must be a boolean")
        self._host_layouts = None
        if host_layout_cache:
            from vvla_pi05.lite.layout import HostLayoutCache
            self._host_layouts = HostLayoutCache(capacity=max_graphs * 2)
        self.entries = OrderedDict()
        self.stats = Counter()
        self.last_call = {}
        self._token = None
        self._lock = threading.Lock()
        self._call_lock = self._lock  # Shared-core lifecycle mutation hooks.
        self._closed = False
        self._event = self._stream = self._device = None
        if not hasattr(model, "_graph_runners"):
            model._graph_runners = weakref.WeakSet()
        model._graph_runners.add(self)

    def compilation_manifest(self):
        from vvla_pi05.lite.compilation import compilation_manifest
        return compilation_manifest(self.compile_sample_actions)

    def _make_run(self, images, image_masks, tokens, masks, noise, steps, metadata, context):
        def run():
            return self.model.sample_actions(
                images, image_masks, tokens, masks, noise=noise, num_steps=steps,
                attention_metadata=metadata, inference_context=context,
            )

        if self.compile_sample_actions:
            from vvla_pi05.lite.compilation import compile_prepared_run
            return compile_prepared_run(self.model, run, owner=self, context=context)
        return run

    def _retire(self):
        if self._event is not None:
            self._event.synchronize()
        elif self._device is not None:
            torch.cuda.synchronize(self._device)
        for entry in self.entries.values():
            self._retire_entry(entry)
        self.entries.clear()
        if self._host_layouts is not None:
            self._host_layouts.clear()
        self._event = self._stream = self._device = None

    @staticmethod
    def _retire_entry(entry):
        compiled = getattr(entry, "compiled_callable", None)
        if compiled is not None:
            compiled.retire()

    def _drop_entry(self, key):
        # A graph's private allocation pool must outlive its last replay on any
        # stream. Eviction is a cold operation; synchronize only that owner.
        if key in self.entries and self._event is not None:
            self._event.synchronize()
        entry = self.entries.pop(key, None)
        if entry is not None:
            self._retire_entry(entry)

    def clear(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot clear graphs during an active request")
        try:
            self._retire()
            self._token = None
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close graphs during an active request")
        try:
            self._retire()
            self._closed = True
        finally:
            self._lock.release()

    @torch.inference_mode()
    def __call__(self, *, images, image_masks, lang_tokens, lang_masks, noise, num_steps, host_layout=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Native graph runner requires serialized requests")
        stream = None
        try:
            if self._closed:
                raise RuntimeError("Graph runner is closed")
            if not lang_tokens.is_cuda:
                raise ValueError("Native graph execution requires CUDA")
            if type(num_steps) is not int or num_steps < 1:
                raise ValueError("num_steps must be a positive integer")
            if self._device is not None and self._device != lang_tokens.device:
                raise ValueError("Clear graphs before changing CUDA device")
            stream = torch.cuda.current_stream(lang_tokens.device)
            if self._event is not None and stream != self._stream:
                stream.wait_event(self._event)
                self.stats["stream_handoffs"] += 1
            self._device = lang_tokens.device
            return self._call(images, image_masks, lang_tokens, lang_masks, noise, num_steps, host_layout)
        finally:
            try:
                if stream is not None:
                    if self._event is None:
                        self._event = torch.cuda.Event()
                    self._event.record(stream)
                    self._stream = stream
                    self._device = lang_tokens.device
            finally:
                self._lock.release()

    def _call(self, images, image_masks, lang_tokens, lang_masks, noise, steps, host_layout=None):
        tensors = [*images, *image_masks, lang_tokens, lang_masks, noise]
        template = None
        if self._host_layouts is not None:
            from vvla_pi05.lite.layout import HostMaskLayout
            if not isinstance(host_layout, HostMaskLayout):
                raise ValueError("Host layout cache requires processor-owned CPU masks")
            template = self._host_layouts.get(self.model, host_layout, images, image_masks, lang_masks)
            mask_bytes = host_layout.mask_bytes
            # The shared core still performs its complete per-request lifecycle
            # validation before consuming this trusted processor layout.
            context = self.model.prepare_inference_context(
                images, image_masks, lang_masks, steps, positions=False, prepared_layout=template.prepared_layout,
            )
        else:
            if host_layout is not None:
                raise ValueError("Host layout metadata supplied while the option is disabled")
            masks = torch.cat([m[:, None] for m in image_masks] + [lang_masks], 1)
            mask_bytes = masks.cpu().numpy().tobytes()
            context = self.model.prepare_inference_context(images, image_masks, lang_masks, steps, positions=False)
        rebuild = False
        if context["token"] != self._token:
            if self.entries:
                rebuild = True
                self._retire()
                self.stats["invalidations"] += 1
            elif self._host_layouts is not None:
                self._host_layouts.clear()
            self._token = context["token"]
            if self._host_layouts is not None:
                # Rebuild all device-dependent metadata after normal parameter
                # writes, recipe/layout changes, or an explicit cache epoch.
                template = self._host_layouts.get(self.model, host_layout, images, image_masks, lang_masks)
                context["packing"] = template.packing
        if template is not None:
            template.materialize(self.model, image_masks, lang_masks,
                                 elide_identity_gathers=self.elide_identity_gathers)
        packing = context.get("packing")
        capacity = None if packing is None else packing["pads"].shape[1]
        key = (steps, tuple((tuple(x.shape), x.dtype, x.device) for x in tensors), context["empty_pattern"], capacity)
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        refreshed = None
        if self.elide_identity_gathers and entry is not None and mask_bytes != entry.mask_bytes:
            from vvla_pi05.lite.attention import annotate_identity_gathers, identity_gather_signature
            if template is not None:
                refreshed = template.metadata
            else:
                refreshed = self.model.prepare_attention_metadata(
                    image_masks, lang_masks, fixed_capacity=True, prefix_packing=packing
                )
                for item in refreshed:
                    annotate_identity_gathers(item)
            if identity_gather_signature(entry.metadata) != identity_gather_signature(refreshed):
                # A captured view cannot become a gather on replay. Rebuild only
                # this entry when mask geometry changes the proven identity.
                self._drop_entry(key)
                entry = None
                rebuild = True
                self.stats["identity_recaptures"] += 1
        cold = entry is None
        rebuild |= cold and len(self.entries) >= self.max_graphs
        updated = False
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                self._drop_entry(next(iter(self.entries)))
                self.stats['capacity_evictions'] += 1
            static = [x.clone() for x in tensors]
            views = len(images)
            static_images, static_masks = static[:views], static[views:2 * views]
            tokens, masks, static_noise = static[-3:]
            if template is not None:
                from vvla_pi05.lite.layout import copy_layout_tree
                # A graph's refreshable buffers must never be cache templates.
                metadata = copy_layout_tree(template.metadata)
                context["packing"] = copy_layout_tree(template.packing)
                context["positions"] = copy_layout_tree(template.positions)
            else:
                metadata = self.model.prepare_attention_metadata(
                    static_masks, masks, fixed_capacity=True, prefix_packing=packing
                )
                if self.elide_identity_gathers:
                    from vvla_pi05.lite.attention import annotate_identity_gathers
                    for item in metadata:
                        annotate_identity_gathers(item)
                context["positions"] = self.model.prepare_position_context(static_masks, masks, packing)

            run = self._make_run(static_images, static_masks, tokens, masks, static_noise, steps, metadata, context)

            # Complete lazy library/JIT initialization on a side stream before
            # capture. Only this cold path synchronizes the device.
            current = torch.cuda.current_stream(lang_tokens.device)
            warm = torch.cuda.Stream(device=lang_tokens.device)
            warm.wait_stream(current)
            with torch.cuda.stream(warm):
                for _ in range(2):
                    run()
            current.wait_stream(warm)
            torch.cuda.synchronize(lang_tokens.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=warm):
                output = run()
            entry = _Entry(static, graph, output, metadata, mask_bytes, context,
                           run if self.compile_sample_actions else None)
            self.entries[key] = entry
            self.stats["captures"] += 1
            if self.compile_sample_actions:
                self.stats["compiled_entries"] += 1
        else:
            if mask_bytes != entry.mask_bytes:
                # Fail closed if a metadata update changes geometry unexpectedly.
                # The same entry is reinserted after successful metadata copy;
                # its callable remains live until geometry validation succeeds.
                self.entries.pop(key)
                try:
                    if refreshed is None:
                        refreshed = template.metadata if template is not None else self.model.prepare_attention_metadata(
                            image_masks, lang_masks, fixed_capacity=True, prefix_packing=packing,
                        )
                    for target, source in zip(entry.metadata, refreshed, strict=True):
                        if target is None and source is None:
                            continue
                        if target is None or source is None or target.extra.keys() != source.extra.keys():
                            raise ValueError("Graph attention metadata fields changed")
                        for name, value in source.extra.items():
                            old = target.extra[name]
                            if isinstance(value, torch.Tensor):
                                if old.shape != value.shape or old.dtype != value.dtype or old.device != value.device:
                                    raise ValueError("Graph attention metadata geometry changed")
                                old.copy_(value)
                            elif old != value:
                                raise ValueError("Graph attention metadata changed without a new key")
                    copy_position_context(
                        entry.context["positions"], template.positions if template is not None
                        else self.model.prepare_position_context(image_masks, lang_masks, packing),
                    )
                    copy_prefix_packing(entry.context.get("packing"), packing)
                except BaseException:
                    self._retire_entry(entry)
                    raise
                entry.mask_bytes = mask_bytes
                self.entries[key] = entry
                self.stats["metadata_updates"] += 1
                updated = True
            for target, source in zip(entry.tensors, tensors, strict=True):
                target.copy_(source)
            self.stats["replays"] += 1
        try:
            # Capture records work; replay explicitly before consuming outputs.
            entry.graph.replay()
            output = entry.output.clone()
        except BaseException:
            self._drop_entry(key)
            raise
        self.last_call = {"cold": cold, "rebuild": rebuild, "actual_tensor_batch": lang_tokens.shape[0], "graph": "capture" if cold else "replay", "metadata_updated": updated}
        return output
