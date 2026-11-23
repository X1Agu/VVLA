"""Opt-in engine classes. Every candidate can be ablated independently."""
import types
import time
from vvla_lingbot.engine import Engine
from vvla_lingbot.batch_engine import BatchedEngine
from .state_guard import FrozenStateGuard

VARIANTS = {'baseline':frozenset(), 'guard':frozenset({'guard'}),
            'moe_grid':frozenset({'moe_grid'}), 'router_cache':frozenset({'router_cache'}),
            'rope':frozenset({'rope'}),
            'combined':frozenset({'guard','moe_grid','router_cache','rope'})}


class TuningMixin:
    def __init__(self,*args,variant='combined',graph_diagnostics_path=None,**kwargs):
        self.tuning_options=VARIANTS[variant]
        self._frozen_guard=None
        super().__init__(*args,**kwargs)
        self.graph_diagnostics=None
        if graph_diagnostics_path is not None or self.audit is not None:
            from .graph_diagnostics import GraphDiagnostics
            self.graph_diagnostics=GraphDiagnostics(graph_diagnostics_path or self.audit/'GRAPH_EVENTS.jsonl',model='lingbot')
        for layer in self.action.layers:
            mlp=layer.mlp
            if 'router_cache' in self.tuning_options:
                from .router import CachedRouterLinear
                mlp.gate=CachedRouterLinear(mlp.gate)
            if 'moe_grid' in self.tuning_options:
                from .moe import tight_moe_forward
                from ..routing import install_moe_forward
                install_moe_forward(mlp,routed_forward=tight_moe_forward)
        self.recipe['experimental_tuning']=dict(variant=variant,enabled=sorted(self.tuning_options),production_default=False)
        self._write_recipe()

    def __call__(self,*args,**kwargs):
        result=super().__call__(*args,**kwargs)
        if self.graph_diagnostics is not None:
            from .graph_diagnostics import describe_lingbot_key
            start=time.perf_counter()
            digest=self.graph_diagnostics.observe(self._meta_key,
                lambda:describe_lingbot_key(self._meta_key),
                outcome='capture' if self.last_call['graph_build'] else 'hit',
                capture_ms=self.last_call['capture_ms'],cache_size=1,capacity=1)
            self.last_call.update(graph_key_sha256=digest,graph_diagnostics_ms=(time.perf_counter()-start)*1000)
        return result

    def _state_signature(self):
        if 'guard' not in self.tuning_options:
            return super()._state_signature()
        if self._frozen_guard is None:
            self._frozen_guard=FrozenStateGuard(self.flow)
        return self._frozen_guard.check()

    def _rope(self,q,k,label):
        if 'rope' not in self.tuning_options:
            return super()._rope(q,k,label)
        from .rope import fused_rope
        return fused_rope(q,k,*self.rope[label])


class TunedEngine(TuningMixin,Engine):
    pass


class TunedBatchedEngine(TuningMixin,BatchedEngine):
    pass
