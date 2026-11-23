"""Private B1 BF16 owner, preserving each card's existing BF16 mathematics."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import torch

from .bf16_checkpoint_engine_evidence import LINEAGES, digest, load_origin, make_manifest, validate_base

_QUANTIZED_OWNERS = ('_prefix_fp8_installation', '_prefix_nvfp4_installation',
    '_prefix_fp8_inventory_installation', '_prefix_fp8_gelu_installation', '_vision_fp8_installation')


def process_math_settings():
    # Getters only. In Torch 2.13 an assignment of legacy allow_tf32=False can
    # change the newer API from none to ieee while returning the same boolean.
    return dict(default_dtype=str(torch.get_default_dtype()), default_device=str(torch.get_default_device()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        global_fp32_precision=getattr(torch.backends, 'fp32_precision', None),
        matmul_fp32_precision=getattr(torch.backends.cuda.matmul, 'fp32_precision', None),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        preferred_blas=str(torch.backends.cuda.preferred_blas_library()),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        environment={name: os.environ.get(name) for name in ('CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE',
            'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'TORCH_BLAS_PREFER_CUBLASLT')})


def _sources():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.json', '.cpp', '.cu', '.h', '.hpp')}


def _hardware(device, profile):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('BF16 checkpoint deployment requires its actual CUDA device')
    actual = torch.cuda.get_device_properties(device)
    expected = LINEAGES[profile]
    if (actual.name != expected['device_name'] or [actual.major, actual.minor] != expected['capability']
            or str(torch.__version__).split('+')[0] != '2.13.0' or torch.version.cuda != '13.0'
            or torch.version.git_version != 'cf30153c4c131c8164ee7798e5022d810682e2cb'):
        raise ValueError('BF16 private owner requires the separately restored original card/runtime family')
    return dict(name=actual.name, capability=[actual.major, actual.minor], uuid=str(actual.uuid),
                torch=str(torch.__version__), torch_git=torch.version.git_version, cuda=torch.version.cuda)


def _precision(policy):
    if any(getattr(policy, name, None) is not None for name in _QUANTIZED_OWNERS):
        raise RuntimeError('Quantized precision owner is outside the BF16 execution domain')
    recipe = policy.config.execution_recipe or {}
    if any(recipe.get(name, 'none') != 'none' for name in ('prefix_mlp', 'vision_mlp', 'expert_mlp')):
        raise RuntimeError('BF16 model execution recipe changed')
    inventory = {}
    for name, tensor in policy.model.named_parameters():
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError('BF16 model has an unexpected parameter dtype: ' + name)
        row = inventory.setdefault(str(tensor.dtype), dict(tensors=0, elements=0))
        row['tensors'] += 1
        row['elements'] += tensor.numel()
    if 'torch.bfloat16' not in inventory:
        raise RuntimeError('No actual BF16 model parameters were audited')
    return dict(precision='bf16', sampler_dtype='float32', quantized_owners=[],
                parameter_dtype_inventory=inventory, execution_recipe=deepcopy(recipe))


class Bf16CheckpointEngine:
    __slots__ = ('__policy', '__state', '__origin', '__hardware', '__sources', '__manifest', '__precision', '__closed', '__process_math')

    def __init__(self, *args, **kwargs):
        raise TypeError('Use the BF16 checkpoint factory; external models/processors are unsupported')

    def __setattr__(self, name, value):
        raise AttributeError('Private BF16 settings are immutable; reconstruct from checkpoint')

    def __reduce_ex__(self, protocol):
        raise TypeError('Private BF16 owners cannot be serialized')

    @classmethod
    def from_pretrained(cls, checkpoint, *, tokenizer, origin_identity, hardware_profile,
                        device='cuda:0', max_graphs=16, precision='bf16'):
        if cls is not Bf16CheckpointEngine or precision != 'bf16':
            raise ValueError('Use the exact independent BF16 checkpoint factory')
        if type(max_graphs) is not int or not 1 <= max_graphs <= 64:
            raise ValueError('max_graphs must be an integer in [1,64]')
        origin = load_origin(origin_identity, hardware_profile)
        hardware = _hardware(device, hardware_profile)
        from .pi05 import Pi05LitePolicy
        from .batch_checkpoint_engine import BatchSealedModelState, _CAPABILITY
        config = deepcopy(origin['config'])
        config.update(checkpoint=checkpoint, tokenizer=tokenizer, device=device, max_graphs=max_graphs, precision='bf16')
        policy = Pi05LitePolicy.from_pretrained(**config)
        state = None
        try:
            engine = object.__new__(cls)
            with policy._lock, policy.graph_runner._call_lock:
                state = BatchSealedModelState(policy, _CAPABILITY)
                state.audit()
                base = policy.metadata()
                validate_base(base, origin)
                sources, precision_manifest = _sources(), _precision(policy)
                process_math = process_math_settings()
                if process_math['default_dtype'] != 'torch.float32' or process_math['default_device'] != 'cpu':
                    raise ValueError('BF16 fresh preprocessing requires the original FP32/CPU defaults')
                manifest = make_manifest(origin=origin, hardware=hardware, source_sha256=sources,
                    base_metadata=base, precision=precision_manifest, process_math=process_math)
            for name, value in dict(policy=policy, state=state, origin=origin, hardware=hardware,
                    sources=sources, manifest=manifest, precision=precision_manifest, closed=False,
                    process_math=process_math).items():
                object.__setattr__(engine, '_Bf16CheckpointEngine__' + name, value)
            engine.metadata()
            return engine
        except BaseException as error:
            try:
                if state is not None:
                    with policy._lock, policy.graph_runner._call_lock:
                        state.detach(policy, policy.graph_runner)
                policy.close()
            except BaseException as cleanup:
                error.bf16_policy_owner = policy
                error.bf16_state_owner = state
                error.add_note('BF16 factory cleanup failed: ' + repr(cleanup))
            raise

    def _live_policy(self):
        if self.__closed:
            raise RuntimeError('Private BF16 owner is closed')
        if process_math_settings() != self.__process_math:
            raise RuntimeError('Private BF16 process math, defaults or numerical environment changed')
        return self.__state._assert_live()[0]

    def infer(self, raw_observation, seed):
        return self._live_policy().infer(raw_observation, seed)

    def predict(self, raw_observation, noise):
        return self._live_policy().predict(raw_observation, noise)

    def reset(self, episode_id=None):
        return self._live_policy().reset(episode_id)

    def prepared_fixture(self, raw_observation, noise):
        return self._live_policy().prepared_fixture(raw_observation, noise)

    def preflight(self, raw_observation, noise, *, repeat=3):
        result = self._live_policy().preflight(raw_observation, noise, repeat=repeat)
        result['metadata'] = self.metadata()
        return result

    def metadata(self):
        policy = self._live_policy()
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError('Cannot audit an executing private BF16 owner')
        runner = policy.graph_runner
        if not runner._call_lock.acquire(blocking=False):
            policy._lock.release()
            raise RuntimeError('Cannot audit an executing BF16 graph owner')
        try:
            self.__state.audit()
            if (_sources() != self.__sources or _hardware(policy.device, self.__origin['profile']) != self.__hardware
                    or load_origin(self.__origin['identity_path'], self.__origin['profile']) != self.__origin):
                raise RuntimeError('BF16 source, hardware or original recipe identity changed')
            base, precision = policy.metadata(), _precision(policy)
            current = make_manifest(origin=self.__origin, hardware=self.__hardware, source_sha256=self.__sources,
                                    base_metadata=base, precision=precision, process_math=process_math_settings())
            if current != self.__manifest or precision != self.__precision:
                raise RuntimeError('Actual BF16 source/precision/recipe seal changed')
            result = deepcopy(base)
            result['bf16_precision_manifest'] = precision
            result['bf16_deployment_manifest'] = current
            result['recipe_components'] = dict(actual_base_recipe_sha256=base['recipe_sha256'],
                                               bf16_deployment_manifest_sha256=digest(current))
            result['recipe_sha256'] = digest(result['recipe_components'])
            return result
        finally:
            runner._call_lock.release()
            policy._lock.release()

    def validate_precision_manifest(self):
        return self.metadata()['bf16_precision_manifest']

    def close(self):
        if self.__closed:
            return
        policy = self.__policy
        if policy._closed:
            self._release_retired_owners()
            return
        if not policy._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close an executing BF16 policy')
        runner = policy.graph_runner
        if runner is not None and not runner._call_lock.acquire(blocking=False):
            policy._lock.release()
            raise RuntimeError('Cannot close an executing BF16 graph')
        try:
            self.__state.detach(policy, runner)
        finally:
            if runner is not None:
                runner._call_lock.release()
            policy._lock.release()
        # Normal synchronous graph/native retirement remains unchanged. On
        # failure both policy and cold state remain reachable for a close retry.
        policy.close()
        self._release_retired_owners()

    def _release_retired_owners(self):
        # Retain no strong model/cache snapshot after successful synchronous
        # retirement, even when the caller keeps the closed facade alive.
        object.__setattr__(self, '_Bf16CheckpointEngine__closed', True)
        object.__setattr__(self, '_Bf16CheckpointEngine__state', None)
        object.__setattr__(self, '_Bf16CheckpointEngine__policy', None)


def create_adapter(config):
    return Bf16CheckpointEngine.from_pretrained(**deepcopy(dict(config)))
