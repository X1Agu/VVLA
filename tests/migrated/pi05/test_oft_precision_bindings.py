"""Current model-path installation contracts without Torch or a GPU import."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class Quantized:
    def __init__(self, precision='fp8'):
        self.precision = precision


def helpers():
    path = Path(__file__).resolve().parents[1]/'workers/oft/src/vvla_oft/precision.py'
    tree = ast.parse(path.read_bytes())
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ('_current_quantized_module', '_validate_quantized_bindings')], type_ignores=[])
    namespace = {'QuantizedLinear': Quantized}
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


def adapter():
    gate, up, down = Quantized(), Quantized(), Quantized()
    layer = NS(mlp=NS(gate_proj=gate, up_proj=up, down_proj=down))
    value = NS(model=NS(language_model=NS(model=NS(layers=[layer]))),
        _mlp_quantized_bindings=(('llama.0.gate_proj', gate, 'fp8'),
            ('llama.0.up_proj', up, 'fp8'), ('llama.0.down_proj', down, 'fp8')))
    return value, layer


def test_unchanged_installed_quantized_objects_pass():
    value, _ = adapter()
    helpers()['_validate_quantized_bindings'](value)


@pytest.mark.parametrize('change', ['replace_linear', 'swap', 'replace_parent', 'precision', 'missing'])
def test_replacing_or_swapping_live_paths_never_reuses_former_installation(change):
    value, layer = adapter()
    if change == 'replace_linear':
        layer.mlp.gate_proj = NS(precision='fp8')
    elif change == 'swap':
        layer.mlp.gate_proj, layer.mlp.up_proj = layer.mlp.up_proj, layer.mlp.gate_proj
    elif change == 'replace_parent':
        layer.mlp = NS(gate_proj=Quantized(), up_proj=Quantized(), down_proj=Quantized())
    elif change == 'precision':
        layer.mlp.gate_proj.precision = 'bf16'
    else:
        del layer.mlp.gate_proj
    with pytest.raises(RuntimeError, match='Quantized'):
        helpers()['_validate_quantized_bindings'](value)
