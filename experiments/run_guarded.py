"""Run an experiment script with the official-test guard active for the whole process.

    python experiments/run_guarded.py experiments/unsw_nested_cv.py --models xgboost --feature-sets full42

The guard (src/unsw/official_test_lock.py) is installed before the script or any of its imports run; the
script itself runs unchanged (its own file, so its module hash in run headers is unaffected).
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.unsw.official_test_lock import identity_locked, install_guard  # noqa: E402

install_guard()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_guarded.py <script.py> [args...]")
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    print(f"official-test guard active (identity lock: {identity_locked()})", flush=True)
    runpy.run_path(str(script), run_name="__main__")
