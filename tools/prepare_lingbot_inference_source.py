"""Source-tree convenience entry for the packaged offline asset preparer."""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/
        'packages/lingbot/src/vvla_lingbot/prepare_source.py'),run_name='__main__')
