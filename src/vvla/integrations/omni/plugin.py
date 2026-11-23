"""Optional registration for the exact upstream scheduler used by the control."""
from importlib.metadata import PackageNotFoundError, version

PINNED={'vllm':'0.32.0','vllm-omni':'0.32.0.dev0+g98c204144','torch':'2.13.0','transformers':'5.19.0'}
_registered=False


def register(*, strict=False):
    global _registered
    if _registered:return
    try:
        for package,expected in PINNED.items():
            actual=version(package)
            if (actual.split('+')[0] if package=='torch' else actual)!=expected:
                raise RuntimeError('Pinned Omni controller differs: '+package+' '+actual)
    except (PackageNotFoundError,RuntimeError):
        if strict:raise
        return  # Ordinary OFT/LingBot interpreters do not activate this plugin.
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.config.stage_config import PipelineConfig,StageExecutionType,StagePipelineConfig
    from vllm_omni.diffusion.registry import register_diffusion_model
    register_diffusion_model(model_arch='VvlaPipeline',module_name='vvla.integrations.omni.pipeline',
        class_name='VvlaPipeline',post_process_func_name='get_post_process_func')
    register_pipeline(PipelineConfig(model_type='vvla_dynamic',model_arch='VvlaPipeline',
        stages=(StagePipelineConfig(stage_id=0,model_stage='diffusion',execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),final_output=True,final_output_type='action',model_arch='VvlaPipeline'),)))
    _registered=True
