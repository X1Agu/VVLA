"""Omni entry point: register names lazily in the API, engine, and worker processes."""

from vvla_pi05.runtime import check_runtime

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    check_runtime()
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.diffusion.registry import register_diffusion_model

    from vvla_pi05.topology import PI05_PIPELINE, OFT_PIPELINE

    register_diffusion_model(
        model_arch="VlaPi05Pipeline",
        module_name="vvla_pi05.pi05_pipeline",
        class_name="VlaPi05Pipeline",
        post_process_func_name="get_pi05_post_process_func",
    )
    register_pipeline(PI05_PIPELINE)
    register_diffusion_model(
        model_arch="VlaOpenVlaOftPipeline",
        module_name="vvla_pi05.oft_pipeline",
        class_name="VlaOpenVlaOftPipeline",
        post_process_func_name="get_oft_post_process_func",
    )
    register_pipeline(OFT_PIPELINE)
    _registered = True
