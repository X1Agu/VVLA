def test_installed_helper_binds_libdevice_and_restores_precision_context():
    import torch
    from torch._inductor import config
    from vvla_pi05.lite.compile_environment import ensure_compile_environment
    from vvla_pi05.lite.compilation import INDUCTOR_OPTIONS
    before=(config.emulate_precision_casts,config.eager_numerics.use_pytorch_libdevice,torch.cuda.is_initialized())
    first=ensure_compile_environment(INDUCTOR_OPTIONS)
    second=ensure_compile_environment(INDUCTOR_OPTIONS)
    assert first==second and len(first['sha256'])==64
    assert before==(config.emulate_precision_casts,config.eager_numerics.use_pytorch_libdevice,torch.cuda.is_initialized())
