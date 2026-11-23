from dataclasses import dataclass, field, fields
import re
from pathlib import Path
import sysconfig


@dataclass(frozen=True)
class LiteRuntimeConfig:
    checkpoint: Path
    qwen: Path
    official_root: Path
    norm_stats: Path
    robot_config: Path
    native_path: Path
    mode: str = 'framework-bf16'
    device: str = 'cuda:0'
    robot: str = 'robotwin'
    python_paths: tuple = ()
    expected_runtime: dict = field(default_factory=dict)
    graph_cache_capacity: int = 1
    graph_memory_budget_bytes: int | None = None
    capture_reserve_bytes: int = 64*1024*1024
    denoise_precision: str = 'bf16'
    posttopk: bool = False
    posttopk_sum_order: str = 'pairwise_02_13'
    routed_down_fp8: bool = False
    true_batch: bool = False
    split_stage: bool = False
    training_config: Path | None = None
    asset_integrity: str = 'metadata'
    calibration: Path | None = None
    packed_weights: Path | None = None

    def to_mapping(self):
        result={f.name:getattr(self,f.name) for f in fields(self)}
        for name in ('checkpoint','qwen','official_root','norm_stats','robot_config','native_path'):
            result[name]=str(result[name])
        result['python_paths']=list(map(str,result['python_paths']))
        result['expected_runtime']=dict(result['expected_runtime'])
        result['training_config']=None if self.training_config is None else str(self.training_config)
        for name in ('calibration','packed_weights'):
            result[name]=None if result[name] is None else str(result[name])
        return result

    @classmethod
    def from_mapping(cls,value):
        if type(value) is not dict:raise TypeError('Lite config must be a dictionary')
        required={'checkpoint','qwen','official_root','norm_stats'}
        if not required<=value.keys() or value.keys()-{f.name for f in fields(cls)}:
            raise ValueError('Unknown or missing LingBot config keys')
        values=dict(value)
        for name in required:
            values[name]=Path(values[name]).expanduser().resolve(strict=True)
        values['robot_config']=Path(values.get('robot_config',values['official_root']/'configs/robot_configs/robotwin.yaml')).resolve(strict=True)
        values['native_path']=Path(values.get('native_path',sysconfig.get_paths()['purelib'])).resolve(strict=True)
        for name in ('checkpoint','qwen','official_root','native_path'):
            if not values[name].is_dir():raise ValueError(name+' must be a directory')
        for name in ('norm_stats','robot_config'):
            if not values[name].is_file():raise ValueError(name+' must be a file')
        training=values.get('training_config')
        if training is None:
            candidates=(values['checkpoint']/'lingbotvla_cli.yaml',
                        values['checkpoint'].parent.parent.parent/'lingbotvla_cli.yaml')
            training=next((p for p in candidates if p.is_file()),None)
        if training is None:
            raise ValueError('Provide training_config or checkpoint/lingbotvla_cli.yaml')
        values['training_config']=Path(training).expanduser().resolve(strict=True)
        if not values['training_config'].is_file():raise ValueError('training_config must be a file')
        if values.get('asset_integrity','metadata') not in ('metadata','sha256'):
            raise ValueError('asset_integrity must be metadata or sha256')
        values['python_paths']=tuple(Path(p).resolve(strict=True) for p in values.get('python_paths',()))
        mode=values.get('mode','framework-bf16')
        if mode not in ('framework-bf16','official-bf16','framework-fp8','framework-nvfp4'):raise ValueError('Unsupported LingBot mode')
        for name in ('calibration','packed_weights'):
            if values.get(name) is not None:
                values[name]=Path(values[name]).expanduser().resolve(strict=True)
                if not values[name].is_file():raise ValueError(name+' must be an existing file')
        if mode=='framework-nvfp4' and values.get('calibration') is None:
            raise ValueError('The historical VLM NVFP4 recipe requires its real-observation calibration asset')
        if mode in ('official-bf16','framework-bf16') and any(values.get(k) is not None for k in ('calibration','packed_weights')):
            raise ValueError('VLM precision assets require an explicit VLM precision recipe')
        if values.get('robot','robotwin')!='robotwin':raise ValueError('Only RoboTwin contract is implemented')
        if re.fullmatch(r'cuda:\d+',values.get('device','cuda:0')) is None:raise ValueError('Explicit CUDA index required')
        precision = values.get('denoise_precision', 'bf16')
        if precision not in ('bf16', 'fp32-clock', 'fp32-clock-accum', 'fp32-heads', 'fp16-expert', 'fp16-prefix-expert'):
            raise ValueError('Unsupported LingBot denoising precision')
        if precision != 'bf16' and (mode != 'framework-bf16' or not values.get('posttopk') or values.get('routed_down_fp8')):
            raise ValueError('Mixed denoising requires the framework BF16 PostTopK engine')
        capacity=values.get('graph_cache_capacity',1)
        if type(capacity) is not int or not 1<=capacity<=64:raise ValueError('graph_cache_capacity must be in [1,64]')
        if mode=='official-bf16' and (capacity!=1 or values.get('graph_memory_budget_bytes') is not None
                or any(values.get(k,False) for k in ('posttopk','routed_down_fp8','true_batch'))):
            raise ValueError('Custom graph pools and PostTopK require framework-bf16')
        if mode in ('framework-fp8','framework-nvfp4') and (capacity!=1 or values.get('graph_memory_budget_bytes') is not None
                or any(values.get(k,False) for k in ('posttopk','routed_down_fp8','true_batch'))):
            raise ValueError('Historical VLM precision uses its original B1 engine; overlays are separate recipes')
        for name in ('posttopk','routed_down_fp8','true_batch'):
            if type(values.get(name,False)) is not bool:raise ValueError(name+' must be boolean')
        if type(values.get('split_stage', False)) is not bool:
            raise ValueError('split_stage must be boolean')
        if values.get('split_stage') and (not values.get('true_batch') or precision != 'fp16-prefix-expert'):
            raise ValueError('Split-stage execution requires the mixed FP16 prefix/expert true-batch recipe')
        if values.get('true_batch',False) and (capacity!=1 or values.get('graph_memory_budget_bytes') is not None
                or (values.get('posttopk',False) and precision == 'bf16') or values.get('routed_down_fp8',False)):
            raise ValueError('True-batch BF16 owns one graph per B; B1 pool/precision overlays are separate candidates')
        budget=values.get('graph_memory_budget_bytes')
        reserve=values.get('capture_reserve_bytes',64*1024*1024)
        if type(reserve) is not int or reserve<0:raise ValueError('Invalid capture reserve')
        if budget is not None and (type(budget) is not int or budget<1 or reserve>budget):raise ValueError('Invalid graph memory budget/reserve')
        versions=values.get('expected_runtime',{})
        if type(versions) is not dict or versions.keys()-{'python','torch','cuda','transformers','vllm'} or any(type(v) is not str for v in versions.values()):
            raise ValueError('Invalid optional runtime pin')
        return cls(**values)

