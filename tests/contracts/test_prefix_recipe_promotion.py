"""Guard the selected historical arithmetic and formal-package installation path."""
import ast
import hashlib
import json
from pathlib import Path
import pytest

from vvla_pi05.recipes import resolve_recipe
from vvla.errors import UnsupportedConfiguration

ROOT=Path(__file__).resolve().parents[2]
LITE=ROOT/'packages/pi05/src/vvla_pi05/lite'


@pytest.mark.parametrize('sm,batch,resolved',[
    ((12,0),1,'prefix-fp8-sm120'),((12,0),32,'prefix-fp8-batch-sm120'),
    ((8,9),1,'prefix-fp8-down-sm89')])
def test_public_fp8_selects_the_preserved_prefix_recipe(sm,batch,resolved):
    actual=resolve_recipe('fp8',capability=sm,horizon=50,max_batch=batch,lt_tactics_available=True)
    original=resolve_recipe(resolved,capability=sm,horizon=50,max_batch=batch,lt_tactics_available=True)
    assert actual['requested']=='fp8' and actual['resolved']==resolved
    assert actual['loader']==original['loader'] and actual['required_assets']==original['required_assets']
    assert not actual['customized'] and not actual['full_weight_release_qualified']


@pytest.mark.parametrize('sm,horizon,batch', [((8,0),50,1),((8,9),50,2),((12,0),10,1)])
def test_public_fp8_never_silently_selects_bf16(sm,horizon,batch):
    with pytest.raises(UnsupportedConfiguration):resolve_recipe('fp8',capability=sm,horizon=horizon,max_batch=batch)


def test_selected_prefix_weight_quantizer_gemm_and_triton_kernel_preserved():
    records=json.loads((ROOT/'tests/fixtures/prefix-arithmetic.json').read_text())
    for name,record in records.items():
        path=LITE/name
        if not record['functions']:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['source_sha256']
        source=path.read_text();tree=ast.parse(source);lines=source.splitlines()
        # Source slices avoid Python 3.10/3.12 AST schema differences while
        # preserving the original function, including casts and constants.
        actual={node.name:hashlib.sha256(('\n'.join(lines[
                    min([node.lineno]+[d.lineno for d in node.decorator_list])-1:node.end_lineno])+'\n').encode()).hexdigest()
                for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in record['functions']}
        assert actual==record['functions']
        assert not any(isinstance(n,ast.ImportFrom) and
                       ('evidence' in (n.module or '') or 'repro' in (n.module or '')) for n in ast.walk(tree))


def test_prefix_recipe_preserves_complete_optimized_bf16_stack():
    bf16=resolve_recipe('bf16-sm120',capability=(12,0),horizon=50)
    fp8=resolve_recipe('prefix-fp8-sm120',capability=(12,0),horizon=50)
    options=fp8['loader']['config_overrides']
    assert options.pop('lite_prefix_fp8') is True
    assert options==bf16['loader']['config_overrides']


def test_prefix_operators_are_importable_from_formal_package_with_fake_dispatch():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vvla_pi05.lite import prefix_fp8_native as native
    with FakeTensorMode():
        for rows in (576,640,32*640):
            x=torch.empty((rows,2048),device='cuda',dtype=torch.bfloat16)
            q,s=native.quantize_rows(x)
            w=torch.empty((16384,2048),device='cuda',dtype=torch.float8_e4m3fn)
            ws=torch.empty((16384,1),device='cuda',dtype=torch.float32)
            result=native.native_scaled_mm(q,w,s,ws)
            assert result.shape==(rows,16384) and result.dtype==torch.bfloat16
