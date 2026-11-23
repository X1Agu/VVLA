"""Default-off policy gates and fail-safe cleanup, prepared for remote CPU."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy, _split_lite_overrides, _validate_compilation_options, _POLICY_HANDLE_FIELDS


def options():
    return _split_lite_overrides(dict(lite_prefix_fp8=True,lite_native_lifecycle=True,
        lite_native_lifecycle_variant='v3',lite_compile_sample_actions=True,lite_batched_vision=True,
        lite_vision_fc2_cublaslt=False,lite_vision_fp8=True))[1]


def test_default_absent_and_false_preserve_existing_option_identity():
    baseline=_split_lite_overrides({})
    assert _split_lite_overrides({'lite_vision_fp8':False})==baseline
    assert 'vision_fp8' not in baseline[1]
    assert options()['vision_fp8'] is True
    _validate_compilation_options(options(),True)
    for value in (1,0,None,'true'):
        with pytest.raises(ValueError,match='boolean'): _split_lite_overrides({'lite_vision_fp8':value})


@pytest.mark.parametrize('fault',['prefix_fp8','native_lifecycle','compile_sample_actions','batched_vision',
                                 'native_lifecycle_variant','vision_fc2_cublaslt','cuda_graph'])
def test_vision_dependencies_fail_before_constructing_model(fault):
    values=options(); cuda=True
    if fault=='cuda_graph': cuda=1 # Explicit strict True, not truthiness.
    elif fault=='native_lifecycle_variant': values[fault]='v2'
    elif fault=='vision_fc2_cublaslt': values[fault]=True
    else: values[fault]=False
    with pytest.raises(ValueError,match='Vision FP8'): _validate_compilation_options(values,cuda)


def test_evidence_requires_flag_and_selected_prefix():
    proof={'backend_report':'/actual/report.json','backend_report_sha256':'a'*64}
    Pi05LitePolicy._validate_vision_fp8_options(True,proof,True)
    for enabled,evidence,prefix in ((False,proof,True),(True,None,True),(True,proof,False)):
        with pytest.raises(ValueError): Pi05LitePolicy._validate_vision_fp8_options(enabled,evidence,prefix)


def test_close_graph_then_vision_then_prefix_and_failed_vision_retains_handle():
    policy=Pi05LitePolicy.__new__(Pi05LitePolicy)
    policy._lock=threading.Lock(); policy._closed=False
    for name in _POLICY_HANDLE_FIELDS: setattr(policy,name,None)
    calls=[]
    class Handle:
        def __init__(self,name): self.name=name; self.fail=False
        def uninstall(self):
            calls.append(self.name)
            if self.fail: raise RuntimeError('injected vision retirement failure')
    vision=Handle('vision'); prefix=Handle('prefix'); native=Handle('native')
    policy.graph_runner=SimpleNamespace(close=lambda:calls.append('graph'))
    policy._vision_fp8_installation=vision; policy._prefix_fp8_installation=prefix
    policy._native_lifecycle_installation=native
    vision.fail=True
    with pytest.raises(RuntimeError,match='retirement'): policy.close()
    assert calls==['graph','vision'] and policy._vision_fp8_installation is vision and not policy._closed
    vision.fail=False; policy.close()
    assert calls==['graph','vision','graph','vision','prefix','native'] and policy._closed
