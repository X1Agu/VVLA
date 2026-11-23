"""Default-off host-only fresh bulk reads of immutable FP8 tensor epochs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

import torch

VERSION='fp8-stateless-fresh-tensor-epochs-v1'
SOURCE=Path(__file__).with_name('csrc')/'fp8_tensor_epochs.cpp'
FLAGS=['-O3']
REFERENCE=Path(__file__).resolve().parents[1]/'models/pi05/inference_cache_pi05.py'
_COLD_BUILDS={}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _core_manifest():
    return dict(version=VERSION,source_sha256={SOURCE.name:_sha(SOURCE),Path(__file__).name:_sha(__file__),
        REFERENCE.name:_sha(REFERENCE)},
        torch_version=str(torch.__version__),torch_git_version=torch.version.git_version,
        torch_cxx11_abi=getattr(torch._C,'_GLIBCXX_USE_CXX11_ABI',None),python_soabi=sysconfig.get_config_var('SOABI'),
        python_version=sys.version,extra_cflags=FLAGS,with_cuda=False,compiler_request=os.environ.get('CXX','default C++ compiler'),
        platform=platform.platform(),machine=platform.machine(),
        reads='each current tensor id,data_ptr,version,dtype,device,shape before comparison on every call',
        retained_tensor_impl=False,signature_cache=False,inventory_cache=False,model_references=False)


def load_native_epochs():
    if sys.implementation.name!='cpython' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('Native FP8 epochs require CPython with the GIL')
    from torch.utils.cpp_extension import load
    initialized=torch.cuda.is_initialized()
    manifest=_core_manifest()
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    extension=load(name='vvla_fp8_tensor_epochs_'+identity[:20],sources=[str(SOURCE)],extra_cflags=FLAGS,with_cuda=False)
    if torch.cuda.is_initialized()!=initialized:raise RuntimeError('Host epoch loading initialized CUDA')
    manifest.update(module_name=extension.__name__,binary_sha256=_sha(extension.__file__),compiler=extension.compiler())
    # Code/toolchain-only immutable receipt. Never retains tensors or a model.
    _COLD_BUILDS[id(extension)]=(extension,_canonical(manifest))
    return extension,deepcopy(manifest)


def enable_owner(owner,extension):
    if (owner._runner is not None or owner._generation is not None or not owner.enabled
            or owner._native_epoch_validation is not None or owner._native_epoch_binding is not None):
        raise RuntimeError('Native epoch validation is a cold immutable owner option')
    tensors=tuple(t for site in owner.sites for t in
                  ((site.weight,site.bias,*site._packed) if hasattr(site,'bias') else (site.weight,*site._packed)))
    expected=tuple(identity for site in owner.sites for identity in site._identities)
    extension.validate_tensor_epochs(tensors,expected)
    owner._native_epoch_validation=(extension,extension.validate_tensor_epochs,expected,
        tuple(site._identities for site in owner.sites))
    owner._native_epoch_binding=owner._native_epoch_validation


def validate_owner(owner,tensors):
    binding=owner._native_epoch_validation
    if (type(binding) is not tuple or binding is not owner._native_epoch_binding or len(binding)!=4
            or binding[0].validate_tensor_epochs is not binding[1]):
        raise RuntimeError('Native FP8 epoch callable/owner binding changed')
    if (len(owner.sites)!=len(binding[3]) or any(site._identities is not expected
            for site,expected in zip(owner.sites,binding[3],strict=True))):
        raise RuntimeError('Native FP8 immutable expected identity record changed')
    binding[1](tensors,binding[2])


def check_binding(owner):
    binding=owner._native_epoch_validation
    if (type(binding) is not tuple or binding is not owner._native_epoch_binding or len(binding)!=4
            or binding[0].validate_tensor_epochs is not binding[1]):
        raise RuntimeError('Native FP8 epoch callable/owner binding changed')


def current_manifest(extension,expected):
    cold=_COLD_BUILDS.get(id(extension))
    if cold is None or cold[0] is not extension:
        raise RuntimeError('Native FP8 epoch implementation has no actual cold build receipt')
    value=_core_manifest()
    value.update(module_name=extension.__name__,binary_sha256=_sha(extension.__file__),compiler=extension.compiler())
    if _canonical(value)!=cold[1] or _canonical(expected)!=cold[1]:
        raise RuntimeError('Native FP8 epoch actual runtime/source/binary or declared build identity changed')
    return value
