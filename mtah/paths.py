"""Where things live.

Importing this module resolves the data bundle location and exports it as MTAH_DATA_DIR *before*
`games` is imported anywhere, because `games.DATA_DIR` is fixed at import time. It also puts the
vendored organiser code (vendor/mtah2027: `games`, `possession_metrics`, `starter`) on sys.path.

Resolution order: the MTAH_DATA_DIR environment variable, then ./data in this repo, then the
organisers' repo checked out next to this one (../more-than-a-hack-2027/data).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("MTAH_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = ROOT / "results"                # our submissions
CONFIG_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"

# Unmodified copies of the organisers' code and public assets (MIT) -- see NOTICE.md.
VENDOR_DIR = ROOT / "vendor" / "mtah2027"
EVALUATION_DIR = VENDOR_DIR / "evaluation"
GT_DIR = EVALUATION_DIR / "gt"
BASELINES_DIR = EVALUATION_DIR / "results"

for _p in (VENDOR_DIR, EVALUATION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SIBLING_DATA = ROOT.parent / "more-than-a-hack-2027" / "data"


def resolve_data_dir() -> Path:
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return Path(env)
    for candidate in (ROOT / "data", _SIBLING_DATA):
        if (candidate / "tactical").is_dir():
            return candidate
    return ROOT / "data"


DATA_DIR = resolve_data_dir()
os.environ.setdefault("MTAH_DATA_DIR", str(DATA_DIR))
