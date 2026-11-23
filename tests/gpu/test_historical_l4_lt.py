"""Execute the unchanged L4 semantic tactics on local Ada, including capture."""
import threading

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


def test_exact_l4_algorithms_execute_when_missing_from_heuristic_shortlist(monkeypatch):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip('L4/RTX Ada semantic tactic check')
    from vvla_pi05.lite import cublaslt_runtime as rt
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', True)
    # Exercise the public AlgoInit/Config/Check path for every tactic, regardless
    # of this card's heuristic ranking. It must retain every original field.
    monkeypatch.setattr(rt.Plan, 'heuristics', lambda self: [])
    owner = rt.ExpertLtRuntime('cuda:0', 'l4_exact_three', include_down=True)
    class Guard:
        lock = threading.Lock()
        def assert_execution_owner(self):
            assert self.lock.locked()
    guard = Guard()
    owner.bind_owner(guard)
    try:
        with guard.lock, torch.inference_mode():
            for label, bound in owner.plans.items():
                spec = bound.spec
                assert bound.rank is None
                assert bound.info == spec['algorithm']
                x = torch.randn(spec['m'], spec['k'], device='cuda', dtype=torch.bfloat16)
                weight = (torch.randn(spec['k'], spec['n'], device='cuda', dtype=torch.float32)
                          / spec['k']**.5).to(torch.bfloat16)
                expected = (x.float() @ weight.float()).to(torch.bfloat16)
                actual = bound.execute(x, weight, bound.workspace, spec['n'])
                torch.testing.assert_close(actual, expected, rtol=.008, atol=.004, msg=label)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = bound.execute(x, weight, bound.workspace, spec['n'])
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, actual, rtol=0, atol=0)
                graph.reset()
    finally:
        owner.close()
    assert not owner.plans
