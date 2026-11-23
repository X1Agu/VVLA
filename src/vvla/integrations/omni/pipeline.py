"""Group upstream requests into exact VVLA calls on one isolated owner."""
from collections import Counter
import time
import weakref

from torch import nn
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from vvla import load
from vvla.serve.server import observation_key


def get_post_process_func(od_config):
    return lambda value: {'actions': value}


class VvlaPipeline(nn.Module):
    supports_request_batch = True
    dummy_run_num_frames = 0

    def __init__(self, *, od_config, prefix=''):
        super().__init__()
        config = dict(od_config.custom_pipeline_args['engine'])
        self.owner = load(**config)
        self._finalizer = weakref.finalize(self, self.owner.close)
        caps = self.owner.capabilities()
        cap = min(od_config.max_num_seqs, caps['serving_defaults']['max_batch'])
        self.widths = tuple(b for b in caps['batch_sizes'] if b <= cap)
        self.text_fields = caps.get('batch_independent_text_fields', ())
        self.calls = 0
        self.width_counts = Counter()
        self.graph = Counter()

    def load_weights(self, weights=()):
        return set()

    def close(self):
        self._finalizer()

    def forward(self, request, **kwargs):
        batched = isinstance(request, DiffusionRequestBatch)
        rows = request.requests if batched else [request]
        commands = [row.sampling_params.extra_args['robot_obs'] for row in rows]
        outputs = {}
        active = []
        for index, command in enumerate(commands):
            operation = command['op']
            identity = command['request_id']
            if operation == 'infer':
                key = observation_key(command['observation'], per_request_text=self.text_fields)
                active.append((index, key))
            elif operation == 'metadata':
                outputs[index] = dict(request_id=identity, result=dict(
                    engine=self.owner.metadata(), capabilities=self.owner.capabilities()))
            elif operation == 'metrics':
                outputs[index] = dict(request_id=identity, result=dict(
                    model_batches=dict(self.width_counts), model_calls=self.calls, **self.graph))
            else:
                raise ValueError('Unknown Omni operation')
        while active:
            now = time.perf_counter()
            live = []
            for index, key in active:
                command = commands[index]
                valid_until = command.get('valid_until_at')
                if now >= command['deadline_at'] or (valid_until is not None and now >= valid_until):
                    outputs[index] = dict(request_id=command['request_id'], status='expired')
                else:
                    live.append((index, key))
            if not live:
                break
            compatible = [index for index, key in live if key == live[0][1]]
            width = max(b for b in self.widths if b <= len(compatible))
            indices = compatible[:width]
            group = [commands[index] for index in indices]
            active = [(i, key) for i, key in live if i not in indices]
            actions = self.owner.infer_batch([c['observation'] for c in group], [c['seed'] for c in group],
                                            request_ids=[c['request_id'] for c in group])
            receipt = self.owner.last_execution
            if receipt['actual_tensor_batch'] != width:
                raise RuntimeError('VVLA and AsyncOmni disagree on the actual batch width')
            self.calls += 1
            self.width_counts[width] += 1
            for key in ('graph_build', 'graph_rebuild', 'graph_replay', 'graph_fallback'):
                observed = receipt.get(key) and (key != 'graph_replay' or not receipt.get('graph_build'))
                self.graph[key+'s'] += int(bool(observed))
            ended = time.perf_counter()
            for i, command, action in zip(indices, group, actions, strict=True):
                outputs[i] = dict(status='completed', request_id=command['request_id'], result=action,
                                  scheduler=dict(generation=1, model_call=self.calls, model_batch=width,
                                      actual_batch=width, scheduler_batch=len(rows), policy='omni',
                                      queue_ms=(now-command['admitted_at'])*1000, executor_ms=(ended-now)*1000,
                                      deadline_met=ended<=command['deadline_at']))
        values = [DiffusionOutput(output=outputs[i]) for i in range(len(commands))]
        return values if batched else values[0]
