import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'workers/oft/src/vvla_oft/calibration.py'
spec = importlib.util.spec_from_file_location('calibration_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_calibration_binds_bytes_revision_and_finite_maxima(tmp_path):
    p = tmp_path / 'cal.json'
    data = dict(status='complete', checkpoint_revision='frozen', maxima={'llama.0.gate_proj': 2.0})
    def config(value):
        p.write_text(json.dumps(value))
        return dict(mlp_precision='nvfp4_full', calibration_path=str(p), calibration_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    c = config(data)
    assert module.load_calibration(c, 'frozen') == data['maxima']
    with pytest.raises(ValueError): module.load_calibration(c, 'different')
    p.write_text(p.read_text() + ' ')
    with pytest.raises(ValueError): module.load_calibration(c, 'frozen')
    for bad in [0, -1, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError):
            module.load_calibration(config(data | dict(maxima={'x': bad})), 'frozen')
    with pytest.raises(ValueError): module.load_calibration(dict(mlp_precision='nvfp4_full'), 'frozen')
    assert module.load_calibration(dict(mlp_precision='bf16'), 'frozen') is None
