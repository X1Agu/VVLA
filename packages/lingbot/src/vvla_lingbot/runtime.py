"""Cold LingBot construction and retryable ownership. No model imports at import time.

Managed Lite runtimes hold the process lease until a successful CUDA drain. Historical websocket construction is archived outside the distribution.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
import hashlib
import functools
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import ast
from types import MappingProxyType

_LEASE = threading.Lock()
_PROCESS_PIN = None
_NATIVE_PIN = None
_ENV_KEYS = ("QWEN3VL_PATH", "VLA_VLLM_NATIVE_PATH", "HF_HUB_OFFLINE",
             "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _absolute(value, name, *, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute local path")
    path = Path(value).resolve(strict=True)
    if directory != path.is_dir():
        raise ValueError(f"{name} has the wrong file/directory type")
    return path


from .config import LiteRuntimeConfig


def validate_assets(config):
    """Record the selected assets; historical author manifests are not admission rules."""
    files={str(p.relative_to(config.official_root)):file_sha256(p)
           for sub in ('deploy','lingbotvla') for p in sorted((config.official_root/sub).rglob('*.py'))}
    if not {'deploy/lingbot_vla_v2_policy.py','lingbotvla/data/vla_data/utils.py'} <= files.keys():
        raise ValueError('Official source asset lacks the supported policy/preprocessing modules')
    policy_tree=ast.parse((config.official_root/'deploy/lingbot_vla_v2_policy.py').read_text())
    server=next((n for n in policy_tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVLAv2Server'),None)
    constructor=None if server is None else next((n for n in server.body
        if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
    arguments=set() if constructor is None else {n.arg for n in (*constructor.args.args,*constructor.args.kwonlyargs)}
    if not {'training_config_path','robot_config_path'}<=arguments:
        raise ValueError('Prepare the upstream inference asset first with vvla-lingbot-prepare-source '
                         '--source ORIGINAL --output PREPARED; select PREPARED as official_root')
    shards=sorted(config.checkpoint.glob('*.safetensors'))
    if not shards:raise ValueError('Checkpoint must contain safetensors shards')
    def inventory(directory):
        result={}
        for path in sorted(directory.rglob('*')):
            if not path.is_file():continue
            stat=path.stat()
            small=path.suffix in ('.json','.yaml','.yml','.model','.txt') and stat.st_size<=32*1024**2
            result[str(path.relative_to(directory))]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                sha256=file_sha256(path) if small or config.asset_integrity=='sha256' else None)
        return result
    checkpoint=inventory(config.checkpoint)
    qwen=inventory(config.qwen)
    if not qwen:raise ValueError('Qwen processor/config asset is empty')
    # A preprocessing identity describes file contents, not download timestamps
    # or Hugging Face cache bookkeeping. Only processor assets are hashed here;
    # choosing a full Qwen directory must not reread its tensor weight shards.
    processor_suffixes={'.json','.yaml','.yml','.model','.txt','.jinja','.py'}
    qwen_processor={name:record['sha256'] or file_sha256(config.qwen/name)
                    for name,record in qwen.items()
                    if Path(name).suffix in processor_suffixes
                    and not {'.cache','__pycache__','.git'}.intersection(Path(name).parts)}
    package=Path(__file__).parent
    sources={str(p.relative_to(package)):file_sha256(p) for p in sorted(package.rglob('*.py'))}
    training=config.training_config
    return {'official_commit':None,'official_source_sha256':digest(files),
            'source_manifest_sha256':None,'package_files_sha256':sources,
            'combined_sources':{'kind':'installed package','files':sources},
            'checkpoint_inventory':checkpoint,'checkpoint_inventory_sha256':digest(checkpoint),
            'checkpoint_sha256':digest({k:v['sha256'] for k,v in checkpoint.items()}) if config.asset_integrity=='sha256' else None,
            'asset_integrity':config.asset_integrity,'weights_content_hashed':config.asset_integrity=='sha256',
            'training_config_sha256':file_sha256(training),
            'robot_config_sha256':file_sha256(config.robot_config),
            'norm_stats_sha256':file_sha256(config.norm_stats),
            'calibration_sha256':file_sha256(config.calibration) if config.calibration else None,
            'packed_weights_sha256':file_sha256(config.packed_weights) if config.packed_weights else None,
            'qwen_inventory':qwen,'qwen_processor_files_sha256':qwen_processor,
            'qwen_metadata_sha256':digest(qwen_processor)}


@contextmanager
def local_environment(config):
    """Only called while the process lease is held; restore even on failed import."""
    cwd, paths = Path.cwd(), list(sys.path)
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        os.environ.update(QWEN3VL_PATH=str(config.qwen),
                          VLA_VLLM_NATIVE_PATH=str(config.native_path),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path[:0] = [str(config.official_root), *(str(p) for p in config.python_paths)]
        os.chdir(config.official_root)
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = paths
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _check_import_origins(root):
    for name, module in tuple(sys.modules.items()):
        if name == "deploy" or name.startswith("deploy.") or name == "lingbotvla" or name.startswith("lingbotvla."):
            source = getattr(module, "__file__", None)
            locations = ([source] if source else []) + list(getattr(module, "__path__", ()))
            if not locations or any(not Path(path).resolve().is_relative_to(root) for path in locations):
                raise RuntimeError(f"Conflicting loaded official module: {name}; use a fresh process")


def dependency_origins(config, *, before_load=False):
    """Bind actual loaded native/dependency modules, not just an environment label."""
    global _NATIVE_PIN
    package_root = Path(__file__).resolve().parent
    records = {}
    for name, module in tuple(sys.modules.items()):
        prefix = name.split(".", 1)[0]
        if prefix == "vvla_lingbot":
            allowed = (package_root,)
        elif config.mode != "official-bf16" and prefix == "vllm":
            allowed = (config.native_path,)
        elif config.mode != "official-bf16" and prefix == "vvla_runtime":
            import vvla_runtime
            allowed = (Path(vvla_runtime.__file__).resolve().parent,)
        else:
            continue
        source = getattr(module, "__file__", None)
        locations = ([source] if source else []) + list(getattr(module, "__path__", ()))
        if not locations or any(not any(Path(path).resolve().is_relative_to(root) for root in allowed)
                                for path in locations):
            raise RuntimeError(f"Conflicting loaded dependency module/namespace: {name}")
        if source:
            path = Path(source).resolve(strict=True)
            records[name] = {"path": str(path), "sha256": file_sha256(path)}
    if config.mode != "official-bf16":
        native = sys.modules.get("vvla_runtime.execution.native")
        loaded = native is not None and bool(getattr(native, "_loaded", False))
        if loaded and before_load and _NATIVE_PIN is None:
            raise RuntimeError("Native operators were loaded before this verified LingBot owner; use a fresh process")
        if _NATIVE_PIN is not None:
            for name, original in _NATIVE_PIN.items():
                if records.get(name) != original:
                    raise RuntimeError(f"Previously loaded native/dependency bytes changed: {name}")
        if not before_load:
            required = {"vllm", "vllm._C_stable_libtorch", "vvla_runtime.execution.native"}
            if not loaded or not required <= records.keys():
                raise RuntimeError("Expected actual pinned native operator modules were not loaded")
            _NATIVE_PIN = dict(records)
    return records


class RuntimeBundle:
    """Own partial initialization as well as successful engines until CUDA retires."""
    def __init__(self, config=None):
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError("Only one active LingBot runtime is allowed per process")
        self.config, self.policy, self.engine = config, None, None
        self.torch, self.device = None, config.device if config else "cuda"
        self.cuda_started, self.closed, self._lease_held = False, False, True
        self.sources, self.actual_runtime = {}, {}
        self.set_seed_everywhere = None

    def close(self):
        if self.closed:
            return
        # Do not clear graphs, closures, streams, model parameters or the lease
        # before this succeeds. A caller can retry close after a drain error.
        with self.torch.cuda.device(self.device) if self.cuda_started else nullcontext():
            if self.cuda_started:
                self.torch.cuda.synchronize(self.device)
            pool_close = getattr(self.engine, 'close', None)
            graph = getattr(self.engine, "_graph", None)
            if pool_close is not None:
                pool_close()
            elif graph is not None:
                graph.reset()
                self.engine._graph = None
        self.policy = self.engine = None
        self.closed = True
        if self._lease_held:
            self._lease_held = False
            _LEASE.release()

    def reset_robot(self):
        if self.closed or self.config is None:
            raise RuntimeError("reset_robot requires a live managed runtime")
        # Official reset re-reads its relative YAML. Keep its exact method and
        # FeatureTransform construction, while pinning both files absolutely.
        if file_sha256(self.config.robot_config) != self.sources["robot_config_sha256"] or \
                file_sha256(self.config.norm_stats) != self.sources["norm_stats_sha256"]:
            raise RuntimeError("Robot or normalization bytes changed; construct a new process")
        with local_environment(self.config), self.torch.cuda.device(self.device):
            self.policy.reset(self.config.robot, path_to_pi_model=None)


def _construction_failed(bundle, error):
    # Retain an explicit retry handle even if cleanup itself raises.
    error.lingbot_runtime = bundle
    try:
        bundle.close()
    except BaseException as cleanup:
        error.add_note(f"LingBot cleanup did not retire CUDA; retry error.lingbot_runtime.close(): {cleanup!r}")


def _construct_engine(bundle, factory, *args, **kwargs):
    # Own a partially initialized class before any CUDA work can fail.
    while isinstance(factory, functools.partial):
        args = (*factory.args, *args)
        kwargs = {**(factory.keywords or {}), **kwargs}
        factory = factory.func
    if not isinstance(factory, type):
        raise TypeError("Engine factory must be a class or functools.partial of a class")
    bundle.engine = factory.__new__(factory)
    factory.__init__(bundle.engine, *args, **kwargs)


def _native_distribution_version(path):
    # native_path may be a separate pip --target directory which intentionally
    # is not put ahead of the selected Torch/Transformers environment.
    matches = [dist for dist in importlib.metadata.distributions(path=[str(path)])
               if str(dist.metadata.get("Name", "")).lower().replace("-", "_") == "vllm"]
    if len(matches) != 1:
        raise RuntimeError("native_path must contain exactly one installed vLLM distribution")
    return str(matches[0].version)


def build_managed_runtime(config):
    global _PROCESS_PIN
    config = LiteRuntimeConfig.from_mapping(config.to_mapping() if isinstance(config, LiteRuntimeConfig) else config)
    bundle = RuntimeBundle(config)
    try:
        bundle.sources = validate_assets(config)
        process_pin = (str(config.official_root), str(config.qwen), str(config.native_path),
                       tuple(map(str, config.python_paths)), config.device,
                       digest(bundle.sources), digest(dict(config.expected_runtime)))
        if _PROCESS_PIN is not None and _PROCESS_PIN != process_pin:
            raise RuntimeError("LingBot module/env source pin changed; use a fresh process")
        with local_environment(config):
            import torch
            bundle.torch = torch
            actual = {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                      "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                      "transformers": importlib.metadata.version("transformers")}
            try:actual['flash_attn']=importlib.metadata.version('flash-attn')
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError('LingBot construction requires the compatible FlashAttention 2.8.3 wheel; '
                                   'prepare it with tools/build_lingbot_flash_attn.py before loading weights') from error
            if config.mode != "official-bf16":
                actual["vllm"] = _native_distribution_version(config.native_path)
            if any(actual.get(key)!=expected for key,expected in config.expected_runtime.items()):
                raise RuntimeError(f"Runtime differs from explicit pin: {actual!r}")
            if (str(torch.__version__).split('+')[0]!='2.13.0' or actual['transformers']!='4.57.3'
                    or actual['flash_attn']!='2.8.3'
                    or (config.mode!='official-bf16' and actual['vllm']!='0.32.0')):
                raise RuntimeError('LingBot requires Torch 2.13, Transformers 4.57.3, FlashAttention 2.8.3 and native vLLM 0.32 ABI')
            if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
                raise RuntimeError("Construct LingBot outside external autocast")
            if config.mode=='framework-nvfp4' and torch.cuda.get_device_capability(config.device)!=(12,0):
                raise ValueError('The historical VLM NVFP4 recipe requires SM120')
            if config.mode=='framework-fp8' and torch.cuda.get_device_capability(config.device)<(8,9):
                raise ValueError('The historical VLM FP8 recipe requires SM89 or newer')
            bundle.actual_runtime = actual
            _check_import_origins(config.official_root)
            dependency_origins(config, before_load=True)
            # Official import seeds CUDA and installs shared Transformer patches.
            # Once imported, a different process pin is never silently reused.
            _PROCESS_PIN = process_pin
            bundle.cuda_started = True
            with torch.cuda.device(config.device), torch.inference_mode(False), torch.no_grad():
                module = importlib.import_module("deploy.lingbot_vla_v2_policy")
                _check_import_origins(config.official_root)
                bundle.set_seed_everywhere = module.set_seed_everywhere
                policy_type = module.LingbotVLAv2Server
                bundle.policy = policy_type.__new__(policy_type)
                policy_type.__init__(bundle.policy, str(config.checkpoint),
                                     robot_norm_path=str(config.norm_stats), use_length=50,
                                     training_config_path=str(config.training_config),
                                     robot_config_path=str(config.robot_config),
                                     chunk_ret=True, use_bf16=True, use_fp32=False,
                                     use_compile=config.mode == "official-bf16")
                if config.mode == "framework-bf16":
                    from .lite_tuning.engine import TunedEngine
                    factory, options = TunedEngine, {}
                    if config.denoise_precision != 'bf16':
                        options['denoise_precision'] = config.denoise_precision
                    restored = None
                    if config.denoise_precision in ('fp32-heads', 'fp16-expert', 'fp16-prefix-expert'):
                        from .precision_weights import restore_action_precision
                        torch.set_float32_matmul_precision('highest')
                        torch.backends.cuda.matmul.allow_tf32 = False
                        restored = restore_action_precision(bundle.policy.vla.model, config.checkpoint,
                            half_expert=config.denoise_precision in ('fp16-expert', 'fp16-prefix-expert'),
                            half_text=config.denoise_precision == 'fp16-prefix-expert')
                    if config.true_batch:
                        if config.split_stage:
                            options['split_stage'] = True
                        if config.posttopk:
                            options['posttopk_sum_order'] = config.posttopk_sum_order
                        from .experimental.true_batch_v2 import TrueBatchEnginePool
                        factory = TrueBatchEnginePool
                    elif config.graph_cache_capacity > 1 or config.graph_memory_budget_bytes is not None:
                        from .lite_tuning.graph_pool import PooledTunedEngine
                        factory = PooledTunedEngine
                        options['graph_cache_capacity'] = config.graph_cache_capacity
                        options['graph_memory_budget_bytes'] = config.graph_memory_budget_bytes
                        options['capture_reserve_bytes'] = config.capture_reserve_bytes
                    _construct_engine(bundle, factory, bundle.policy.vla.model,
                                      "framework-bf16", audit=None, variant="combined",
                                      graph_diagnostics_path=None, **options)
                    if restored is not None:
                        bundle.engine.recipe['restored_original_fp32'] = restored
                    if config.routed_down_fp8:
                        from .experimental.routed_down_fp8_v1 import install_overlay
                        install_overlay(bundle.engine,posttopk=config.posttopk,sum_order=config.posttopk_sum_order)
                    elif config.posttopk and not config.true_batch:
                        from .routing import install_post_topk
                        install_post_topk(bundle.engine,sum_order=config.posttopk_sum_order)
                    bundle.policy.sample_actions_fn = bundle.engine
                elif config.mode in ('framework-fp8','framework-nvfp4'):
                    from .engine import Engine
                    calibration=json.loads(config.calibration.read_text()) if config.calibration else {}
                    if 'maxima' in calibration:calibration=calibration['maxima']
                    _construct_engine(bundle,Engine,bundle.policy.vla.model,config.mode,
                                      calibration=calibration,audit=None,packed_weights=config.packed_weights)
                    bundle.policy.sample_actions_fn=bundle.engine
                bundle.reset_robot()
                bundle.sources["loaded_dependency_files"] = dependency_origins(config)
        return bundle
    except BaseException as error:
        _construction_failed(bundle, error)
        raise
