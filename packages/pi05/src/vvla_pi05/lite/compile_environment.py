"""Initialize the installed Torch libdevice before fresh or cached compilation.

This uses the original Torch helper under the same numerical options as compile;
there is no old GPU/report/checkpoint admission, cache purge or helper patch.
"""
import hashlib
import os
from pathlib import Path
import threading

_LOCK=threading.Lock()
_RECEIPT=None


def ensure_compile_environment(options):
    global _RECEIPT
    import torch
    from torch._inductor import config
    from torch._inductor.runtime import compile_tasks
    from triton import knobs
    if not str(torch.__version__).startswith('2.13.'):
        raise RuntimeError('Compile environment helper requires the checked Torch 2.13 API')
    with _LOCK:
        # The pip CUDA toolkit has no CUDA_HOME. Select its installed NVVM
        # bitcode as input to Torch's helper, without patching the helper or
        # reusing an author-machine path/hash. Explicit caller settings win.
        if 'TRITON_LIBDEVICE_PATH' not in os.environ:
            selected=knobs.nvidia.libdevice_path
            if selected is None:
                from torch.utils.cpp_extension import CUDA_HOME
                if CUDA_HOME is None:
                    major=str(torch.version.cuda).split('.')[0]
                    candidate=Path(torch.__file__).resolve().parents[1]/'nvidia'/('cu'+major)/'nvvm/libdevice/libdevice.10.bc'
                    if not candidate.is_file():
                        raise RuntimeError('Compiled sampling needs CUDA libdevice: install the declared vvla-pi05 dependencies or set TRITON_LIBDEVICE_PATH')
                    selected=str(candidate)
            if selected is not None:os.environ['TRITON_LIBDEVICE_PATH']=str(selected)
        before=(config.emulate_precision_casts,config.eager_numerics.use_pytorch_libdevice)
        with config.patch(options):compile_tasks._set_triton_libdevice_path()
        if before!=(config.emulate_precision_casts,config.eager_numerics.use_pytorch_libdevice):
            raise RuntimeError('Inductor numerical options were not restored')
        path=os.environ.get('TRITON_LIBDEVICE_PATH')
        if not path or str(knobs.nvidia.libdevice_path)!=path or not Path(path).is_file():
            raise RuntimeError('Installed Torch helper did not bind an available libdevice')
        _RECEIPT=dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      torch=str(torch.__version__),options=dict(options),precision_context_restored=True,
                      initialization='before original compiler/cache entry; no cache invalidation',
                      performance_qualification=False)
        return dict(_RECEIPT)


def receipt():return None if _RECEIPT is None else dict(_RECEIPT)
