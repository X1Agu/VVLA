"""Lightweight topology, separate from model imports and CUDA initialization."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

PI05_PIPELINE = PipelineConfig(
    model_type="vla_pi05",
    model_arch="VlaPi05Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="VlaPi05Pipeline",
        ),
    ),
)

OFT_PIPELINE = PipelineConfig(
    model_type="vla_openvla_oft",
    model_arch="VlaOpenVlaOftPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="VlaOpenVlaOftPipeline",
        ),
    ),
)
