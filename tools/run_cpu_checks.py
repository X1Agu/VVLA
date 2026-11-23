"""Run migrated contracts and public SDK tests with explicit new-tree imports."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
paths = [root / "src", *(p / "src" for p in (root / "packages").iterdir()),
         root / "tests/reference/src"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=os.pathsep.join(map(str, paths)))
files = [
    "tests/migrated/pi05/test_lite_bf16_checkpoint_engine.py",
    "tests/migrated/pi05/test_lite_fixed_checkpoint_engine.py",
    "tests/migrated/pi05/test_lite_batch_checkpoint_engine.py",
    "tests/migrated/pi05/test_oft_lite.py",
    "tests/migrated/pi05/test_oft_lite_prepared.py",
    "tests/migrated/oft/test_bf16_operators.py",
    "tests/migrated/oft/test_precision.py",
    "tests/migrated/lingbot/test_graph_pool.py",
    "tests/migrated/lingbot/test_private_lite.py",
    "tests/migrated/p0/test_dynamic_batching.py",
    "tests/contracts",
    "tests/packaging",
]
(root / "artifacts").mkdir(exist_ok=True)
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", *files,
                                  "--junitxml=artifacts/cpu-contracts.xml", *sys.argv[1:]], cwd=root, env=env))
