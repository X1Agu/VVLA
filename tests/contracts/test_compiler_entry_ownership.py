import pytest
import torch
from vvla_pi05.lite.compilation import compile_prepared_run


def make_entry(model,width):
    tensor=torch.arange(width,dtype=torch.float32)
    def run():return model(tensor)+width
    return compile_prepared_run(model,run,backend='eager'),tensor+width


def test_many_graph_entries_do_not_exhaust_one_shared_dynamo_specialization_budget():
    model=torch.nn.Identity().eval()
    entries=[make_entry(model,width) for width in range(1,17)]
    try:
        for execute,expected in entries:torch.testing.assert_close(execute(),expected)
        entries[0][0].retire()
        with pytest.raises(RuntimeError,match='retired'):entries[0][0]()
        # Retiring one owner must not invalidate or recompile another entry.
        for execute,expected in entries[1:]:torch.testing.assert_close(execute(),expected)
    finally:
        for execute,_ in entries:execute.retire()
