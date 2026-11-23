"""CPU topology checks; no CUDA algorithm or policy qualification implied."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'vllm-vla-plugin/src'))
from vvla_pi05.lite.dynamic_cublaslt import BATCH_SIZES,DynamicVisionInstallation


def owner():
    layers=[object() for _ in range(27)]
    model=SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(
        vision_tower=SimpleNamespace(encoder=SimpleNamespace(layers=layers))))))
    # SimpleNamespace is not weak-referenceable; retain a real Python owner.
    class Model:pass
    actual=Model();actual.paligemma_with_expert=model.paligemma_with_expert
    plans={b:[object() for _ in range(27)] for b in BATCH_SIZES}
    runtime=SimpleNamespace(batch_plans=plans,plans=[p for b in BATCH_SIZES for p in plans[b]],validate=lambda:None)
    installation=DynamicVisionInstallation(actual,runtime)
    installation.sites=[SimpleNamespace(layer=layer,plan=plans[1][i],validate=lambda index=i:('site',index)) for i,layer in enumerate(layers)]
    installation._installed_sites=tuple(installation.sites)
    return actual,installation


def test_all_six_plan_sets_and_all_27_sites_validate():
    model,installation=owner()
    assert len(installation._validate())==27
    assert len(installation.runtime.plans)==162


def test_an_inactive_batch_plan_cannot_be_replaced():
    model,installation=owner()
    installation.runtime.batch_plans[32][8]=object()
    with pytest.raises(RuntimeError,match='ownership changed'):installation._validate()


def test_inactive_geometry_cannot_lose_a_site():
    model,installation=owner()
    installation.runtime.batch_plans[16].pop()
    with pytest.raises(RuntimeError,match='ownership changed'):installation._validate()
