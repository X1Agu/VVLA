"""Experimental private LingBot owner with cold/full and cheap serving guards.

The facade never exports model, engine, processor or tensor references. External
model mutation and private-reference introspection/monkeypatching are outside
this API, as in the private pi05 checkpoint owner. Public mutable Lite keeps
the complete per-request guard. This candidate still requires separate GPU A/B.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import weakref

from .lite import LingBotLitePolicy
from .runtime import digest

_CAPABILITY=object()


class _SealedGuard:
    def __init__(self,adapter,full_guard,capability):
        if capability is not _CAPABILITY:raise TypeError('Private checkpoint construction only')
        self._adapter=weakref.ref(adapter)
        self._engine=weakref.ref(adapter._runtime.engine)
        self._model=weakref.ref(adapter._runtime.policy.vla.model)
        self._full=full_guard
        self._token=object()
        self._closed=False
        self._generation=0
        full_guard.check()

    def check(self):
        adapter,engine,model=self._adapter(),self._engine(),self._model()
        if (self._closed or self._generation!=0 or adapter is None or adapter._closed
                or adapter._failed or adapter._runtime is None or adapter._runtime.closed
                or adapter._runtime.engine is not engine or engine is None
                or engine._frozen_guard is not self or model is None
                or adapter._runtime.policy.vla.model is not model or engine.flow is not model):
            raise RuntimeError('Private LingBot owner closed, failed or detached; reconstruct')
        return self._token

    def audit(self):
        self.check();self._full.check();self.check()


class PrivateLingBot:
    __slots__=('__adapter','__guard','__manifest','__closed')

    def __init__(self,*args,**kwargs):raise TypeError('Use private_lite.factory')
    def __setattr__(self,name,value):raise AttributeError('Private LingBot settings are immutable; reconstruct')
    def __reduce_ex__(self,protocol):raise TypeError('Private LingBot owners cannot be serialized')

    def infer(self,raw_observation,seed):
        return self.__adapter.infer(raw_observation,seed)

    def predict(self,raw_observation,noise):
        return self.__adapter.predict(raw_observation,noise)

    def reset(self,episode_id):
        return self.__adapter.reset(episode_id)

    def initial_noise(self):
        return self.__adapter.initial_noise()

    def prepared_fixture(self,raw_observation,noise):
        return self.__adapter.prepared_fixture(raw_observation,noise)

    def metadata(self):
        adapter=self.__adapter
        with adapter._exclusive():
            adapter._check_live()
            if self.__guard is not None:self.__guard.audit()
            else:adapter._runtime.engine._state_signature()
        value=adapter.metadata()
        value['private_owner']=deepcopy(self.__manifest)
        value['recipe_sha256']=digest(dict(base_recipe_sha256=value['recipe_sha256'],private_owner=self.__manifest))
        return value

    def close(self):
        if self.__closed:return
        self.__adapter.close()
        if self.__guard is not None:
            self.__guard._closed=True;self.__guard._generation+=1
            # The cold guard owns all model tensors. Release it only after
            # successful device drain/graph retirement, even if the caller
            # retains this closed facade for later metadata/error handling.
            self.__guard._full=None
        object.__setattr__(self,'_PrivateLingBot__closed',True)


def factory(config,*,sealed_guard=True):
    if type(sealed_guard) is not bool:raise TypeError('sealed_guard must be explicit boolean')
    adapter=LingBotLitePolicy(config)
    try:
        engine=adapter._runtime.engine
        if engine is None:raise ValueError('Private guard candidate requires framework BF16')
        from .lite_tuning.state_guard import FrozenStateGuard
        guard=None
        if sealed_guard:
            guard=_SealedGuard(adapter,FrozenStateGuard(engine.flow),_CAPABILITY)
            engine._frozen_guard=guard
        result=object.__new__(PrivateLingBot)
        manifest=dict(schema=1,implementation='lingbot-private-owner-guard-v1',
            guard='private_generation_and_ownership' if sealed_guard else 'full_per_request',
            full_audit='construction and explicit metadata audit',
            mutable_model_exported=False,shared_graph_constants_guard_preserved=True,
            policy_preprocessor_math_rng_output_checks_preserved=True,
            unsupported='private reference access, monkeypatching, model/processor/precision mutation',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gpu_qualification_claimed=False)
        for name,value in dict(adapter=adapter,guard=guard,manifest=manifest,closed=False).items():
            object.__setattr__(result,'_PrivateLingBot__'+name,value)
        result.metadata()
        return result
    except BaseException:
        adapter.close();raise
