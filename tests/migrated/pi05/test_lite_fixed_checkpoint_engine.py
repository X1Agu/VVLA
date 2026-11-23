"""CPU contracts for the new private API and real dynamic context path.

Tiny tensors exercise production context logic; these tests do not substitute
for checkpoint/SM120/closed-loop qualification or assert performance/quality.
"""
from copy import deepcopy
from types import SimpleNamespace
import pickle
import threading

import pytest
import torch

from vvla_repro_pi05.lite import fixed_checkpoint_engine as fixed
from vvla_repro_pi05.lite.pi05 import Pi05LitePolicy,_split_lite_overrides,_validate_compilation_options
from vvla_repro_pi05.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from test_lite_host_layout import _LayoutModel,_inputs


class _ContextModel(_LayoutModel):
    def __init__(self):
        super().__init__()
        self.complete_prepares=0

    def _prepare_inference_model_state(self,device_type):
        self.complete_prepares+=1
        return Pi05ForActionPrediction._prepare_inference_model_state(self,device_type)


class _Runner:
    def __init__(self,model):
        self.model=model;self.entries={};self._closed=False;self._call_lock=threading.Lock()


class _Policy:
    def __init__(self):
        self.model=_ContextModel();self.graph_runner=_Runner(self.model)
        self._closed=False;self._calls=0;self.device=torch.device('cpu');self._lock=threading.Lock()
        self._expert_cublaslt_installation=SimpleNamespace(_request_thread=None)
        self.closes=0

    def metadata(self):
        # Fictional build data tests declaration binding only, never hardware
        # support; real deployment acceptance uses the actual collector binary.
        source=fixed._sources()
        native=dict(collector_variant='v3',enabled=True,source_sha256={
            name:source[name] for name in ('native_lifecycle.py','modeling_pi05.py')},
            torch_version='fictional CPU test',torch_git_version='fictional CPU test',binary_sha256='c'*64)
        if not hasattr(self,'_lite_optimizations'):
            self._lite_optimizations=dict(native_lifecycle_manifest=native)
        quant=dict(precision='mixed_bf16_fp8_prefix',implementation='pi05-prefix-rowwise-fp8-v1',
            scope='prefix_mlp_layers_0_16',checkpoint_sha256='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae',
            backend=dict(quantizer_variant='triton-eager-rn-native-sm120-v1',torch_version='fictional CPU test',
                torch_git_version='fictional CPU test'),lifecycle=dict(source_sha256=source['prefix_fp8.py']))
        return deepcopy(dict(lite_optimizations=self._lite_optimizations,quantization_manifest=quant,
            recipe_components=dict(base_recipe_sha256='a'*64,quantization_manifest_sha256=fixed.digest(quant),bf16_control={})))

    def close(self):
        assert not hasattr(self.model,fixed._INSTALLATION)
        self.graph_runner._closed=True;self._closed=True;self.closes+=1


@pytest.fixture
def sealed():
    policy=_Policy()
    with policy._lock,policy.graph_runner._call_lock:
        state=fixed._SealedModelState(policy,fixed._CAPABILITY)
    engine=object.__new__(fixed.FixedCheckpointEngine)
    object.__setattr__(engine,'_FixedCheckpointEngine__policy',policy)
    object.__setattr__(engine,'_FixedCheckpointEngine__state',state)
    yield policy,state,engine
    if not policy._closed:engine.close()


def _context(policy,inputs):
    model=policy.model
    with policy._lock,policy.graph_runner._call_lock:
        return model.prepare_inference_context(inputs['images'],inputs['image_masks'],inputs['lang_masks'],
            inputs['num_steps'],positions=False)


def test_private_constructor_no_mutable_public_refs_or_serialization(sealed):
    policy,state,engine=sealed
    with pytest.raises(TypeError,match='from_pretrained'):fixed.FixedCheckpointEngine(policy.model,object())
    with pytest.raises(TypeError,match='checkpoint factory'):fixed._SealedModelState(policy,object())
    for name in ('model','policy','processor','graph_runner','config','device','state_dict','parameters','load_weights','train','to'):
        assert not hasattr(engine,name)
        with pytest.raises(AttributeError):setattr(engine,name,object())
    assert not hasattr(engine,'__dict__')
    with pytest.raises(TypeError,match='serialized'):pickle.dumps(engine)


def test_dynamic_context_never_reuses_observations_masks_positions_or_noise(sealed):
    policy,state,engine=sealed;mutable=_LayoutModel()
    saved=[]
    for valid,cameras,value in [(2,(True,False),1.),(5,(True,True),3.),(1,(False,True),-2.),(4,(True,False),.25)]:
        inputs=_inputs(valid=valid,cameras=cameras,value=value)
        actual=_context(policy,inputs)
        expected=mutable.prepare_inference_context(inputs['images'],inputs['image_masks'],inputs['lang_masks'],
            inputs['num_steps'],positions=False)
        assert actual['empty_pattern']==expected['empty_pattern']
        for key in ('indices','pads','positions'):assert torch.equal(actual['packing'][key],expected['packing'][key])
        assert actual['packing']['pads'].sum().item()==sum(cameras)*4+valid
        actual['positions']=policy.model.prepare_position_context(inputs['image_masks'],inputs['lang_masks'],actual['packing'])
        expected['positions']=mutable.prepare_position_context(inputs['image_masks'],inputs['lang_masks'],expected['packing'])
        args={key:inputs[key] for key in ('images','image_masks','lang_tokens','lang_masks','noise','num_steps')}
        result=policy.model.sample_actions(**args,inference_context=actual)
        reference=mutable.sample_actions(**args,inference_context=expected)
        assert torch.equal(result,reference)
        saved.append(result.clone())
    assert policy.model.complete_prepares==1
    assert all(not torch.equal(saved[0],other) for other in saved[1:])
    with policy._lock,policy.graph_runner._call_lock:state.audit_base()
    assert policy.model.complete_prepares==2 # Explicit audits still run full checks.


def test_hot_preparation_requires_both_locks_and_updates_actual_owner_thread(sealed):
    policy,state,_=sealed
    with pytest.raises(RuntimeError,match='serialized'):state.prepare_model_state(policy.model,'cpu')
    with policy._lock,policy.graph_runner._call_lock:
        result=state.prepare_model_state(policy.model,'cpu')
    assert result is state._state
    assert policy._expert_cublaslt_installation._request_thread==threading.get_ident()
    errors=[]
    def another_serial_thread():
        try:
            with policy._lock,policy.graph_runner._call_lock:
                state.prepare_model_state(policy.model,'cpu')
                assert policy._expert_cublaslt_installation._request_thread==threading.get_ident()
        except BaseException as error:errors.append(error)
    worker=threading.Thread(target=another_serial_thread);worker.start();worker.join()
    assert not errors and policy.model.complete_prepares==1


@pytest.mark.parametrize('fault',['cache_epoch','math','bf16_reduction','autocast','detached','runner_closed'])
def test_invalid_scope_rejects_and_cleanup_remains_available(sealed,fault,monkeypatch):
    policy,state,engine=sealed
    original_bf16=None
    if fault=='cache_epoch':policy.model._inference_cache.clear()
    elif fault=='math':monkeypatch.setattr(fixed,'_math_settings',lambda:('changed',))
    elif fault=='bf16_reduction':
        # cuBLASModule is a custom ModuleType proxy; pytest's generic module
        # monkeypatch can record this virtual attribute as absent on Linux.
        original_bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=not original_bf16
    elif fault=='autocast':monkeypatch.setattr(torch,'is_autocast_enabled',lambda *args:True)
    elif fault=='detached':delattr(policy.model,fixed._INSTALLATION)
    elif fault=='runner_closed':policy.graph_runner._closed=True
    try:
        with pytest.raises(RuntimeError):state._assert_live()
        engine.close();engine.close()
        assert policy.closes==1
        with pytest.raises(RuntimeError,match='closed'):engine._live_policy()
    finally:
        if original_bf16 is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original_bf16


def test_audit_rechecks_normal_weight_mutation_without_claiming_hot_support(sealed):
    policy,state,engine=sealed
    with torch.no_grad():policy.model.weight.add_(1.)
    with policy._lock,policy.graph_runner._call_lock:
        with pytest.raises(RuntimeError,match='epoch changed|generation or scope changed'):state.audit_base()
    engine.close()


def test_active_request_cannot_close(sealed):
    policy,_,engine=sealed
    with policy._lock:
        with pytest.raises(RuntimeError,match='active'):engine.close()
    assert not policy._closed


def test_factory_forwards_only_checkpoint_and_copies_options(monkeypatch):
    received=[]
    def checkpoint_factory(cls,checkpoint,**kwargs):received.append((checkpoint,kwargs));return 'checkpoint-built-engine'
    monkeypatch.setattr(Pi05LitePolicy,'from_pretrained',classmethod(checkpoint_factory))
    options={'lite_prefix_fp8':True};before=deepcopy(options)
    assert fixed.FixedCheckpointEngine.from_pretrained('/checkpoint',config_overrides=options)=='checkpoint-built-engine'
    assert options==before and received[0][1]['config_overrides']['lite_fixed_checkpoint_engine'] is True
    with pytest.raises(ValueError):fixed.FixedCheckpointEngine.from_pretrained('/checkpoint',config_overrides={'lite_fixed_checkpoint_engine':False})


def test_option_default_off_and_exact_static_scope():
    assert _split_lite_overrides({'lite_fixed_checkpoint_engine':False})==_split_lite_overrides({})
    for bad in (1,0,None,'true'):
        with pytest.raises(ValueError,match='boolean'):_split_lite_overrides({'lite_fixed_checkpoint_engine':bad})
    options=_split_lite_overrides({})[1]
    options.update(prefix_fp8=True,native_lifecycle=True,native_lifecycle_variant='v3',compile_sample_actions=True,
        host_layout_cache=True,vision_fc2_cublaslt=True,batched_vision=True,fixed_checkpoint_engine=True)
    _validate_compilation_options(options,True)
    for key in ('prefix_fp8','native_lifecycle','compile_sample_actions','host_layout_cache','vision_fc2_cublaslt'):
        with pytest.raises(ValueError):_validate_compilation_options(dict(options,**{key:False}),True)
    for key in ('vision_fp8','prefix_fp8_inventory','prefix_fp8_gelu','fp8_host_epochs','prefix_nvfp4'):
        with pytest.raises(ValueError):_validate_compilation_options(dict(options,**{key:True}),True)
    with pytest.raises(ValueError):_validate_compilation_options(options,False)
    from test_lite_pi05 import _Model,_policy
    baseline=_policy()
    try:
        with pytest.raises(TypeError,match='fixed_checkpoint_engine'):
            Pi05LitePolicy(_Model(),baseline.processor,device='cpu',cuda_graph=False,fixed_checkpoint_engine=True)
    finally:
        baseline.close()


def _declare_fixture(policy,state):
    with policy._lock,policy.graph_runner._call_lock:
        base=state.audit_base();manifest=state.manifest(base)
        policy._lite_optimizations.update(fixed_checkpoint_engine=True,fixed_checkpoint_engine_manifest=manifest)


def test_saved_manifest_and_returned_metadata_are_independent(sealed):
    policy,state,engine=sealed;_declare_fixture(policy,state)
    first=engine.metadata();second=engine.metadata()
    assert first==second and first is not second
    first['lite_optimizations']['fixed_checkpoint_engine_manifest']['public_api'].clear()
    assert engine.metadata()==second
    assert first['recipe_sha256']==second['recipe_sha256']
    policy._lite_optimizations['fixed_checkpoint_engine_manifest']['public_api']=[]
    with pytest.raises(RuntimeError,match='declaration changed'):engine.metadata()


def test_failed_close_can_retry_after_seal_detachment(sealed,monkeypatch):
    policy,state,engine=sealed;original=policy.close;attempts=[]
    def fail_first():
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('injected cleanup failure')
        original()
    monkeypatch.setattr(policy,'close',fail_first)
    with pytest.raises(RuntimeError,match='cleanup failure'):engine.close()
    assert state._closed and not policy._closed
    engine.close()
    assert policy._closed and len(attempts)==2


def test_metadata_math_guard_failure_can_still_retire(sealed,monkeypatch):
    policy,state,engine=sealed;_declare_fixture(policy,state)
    monkeypatch.setattr(fixed,'_math_settings',lambda:('changed',))
    with pytest.raises(RuntimeError,match='math settings'):engine.metadata()
    engine.close();assert policy._closed
