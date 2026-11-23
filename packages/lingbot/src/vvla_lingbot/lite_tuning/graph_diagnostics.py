"""Opt-in CPU-only graph-key diagnostics; never retain GPU tensors or observations.

Log captures, failures, fallbacks and key transitions. Unchanged warm hits update
only counters: no JSON encoding, file writes or extra device-to-host copies.
This describes cache behavior; it never selects a graph or alters a cache key.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import uuid


class GraphDiagnostics:
    def __init__(self, path, *, model, history_limit=4096):
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('history_limit must be a positive integer')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.session = uuid.uuid4().hex
        self.history_limit = history_limit
        self.seen = set()
        self.history_complete = True
        self.last_key = self.last_descriptor = self.last_digest = None
        self.counts = Counter()
        self.calls = 0

    def observe(self, key, describe, *, outcome, **fields):
        self.calls += 1
        self.counts[outcome] += 1
        same = self.last_key is not None and key == self.last_key
        if same and outcome in ('hit', 'eager'):
            return self.last_digest
        descriptor = self.last_descriptor if same else describe()
        encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        seen = digest in self.seen
        changed = [] if self.last_descriptor is None else sorted(
            k for k in set(descriptor) | set(self.last_descriptor)
            if descriptor.get(k) != self.last_descriptor.get(k))
        if self.last_descriptor is None:
            reason = 'first_request_in_diagnostic_session'
        elif 'state_generation' in changed or 'revision' in changed:
            reason = 'model_state_or_revision_changed'
        elif seen:
            reason = 'previously_seen_key'
        else:
            reason = 'new_metadata_key' if self.history_complete else 'key_not_in_bounded_history'
        event = dict(schema=1, session=self.session, model=self.model,
                     request_index=self.calls - 1, time=time.time(), outcome=outcome,
                     key_sha256=digest, previous_key_sha256=self.last_digest,
                     seen_before=seen, history_complete=self.history_complete,
                     reason=reason, changed_components=changed, key=descriptor,
                     cumulative_counts=dict(self.counts), **fields)
        # Fail closed if a requested audit cannot be persisted. Do not silently
        # produce a campaign that claims to have diagnostics but has no evidence.
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        if len(self.seen) < self.history_limit:
            self.seen.add(digest)
        elif digest not in self.seen:
            self.history_complete = False
        self.last_key, self.last_descriptor, self.last_digest = key, descriptor, digest
        return digest


def describe_lingbot_key(key):
    specs, (image_mask, language_mask, grid) = key[:-3], key[-3:]
    names = ('images', 'img_masks', 'tokens', 'lang_masks', 'state', 'image_grid_thw')
    if len(specs) != len(names):
        raise ValueError('Unknown LingBot graph key layout')
    result = {name: dict(shape=list(shape), dtype=str(dtype), device=str(device))
              for name, (shape, dtype, device) in zip(names, specs)}
    for name, value in [('image_mask', image_mask), ('language_mask', language_mask), ('grid', grid)]:
        result[name + '_sha256'] = hashlib.sha256(value).hexdigest()
    return result


def describe_oft_key(key):
    revision, generation, names, specs = key
    return dict(revision=revision, state_generation=generation,
                inputs={name: dict(shape=list(shape), dtype=str(dtype), device=str(device))
                        for name, (shape, dtype, device) in zip(names, specs)})
