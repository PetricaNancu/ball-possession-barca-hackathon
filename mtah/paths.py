"""Where things live.

Importing this module resolves the data bundle location and exports it as MTAH_DATA_DIR *before*
`games` is imported anywhere, because `games.DATA_DIR` is fixed at import time.

Resolution order: the MTAH_DATA_DIR environment variable, then ./data in this repo, then the
organisers' repo checked out next to this one (../more-than-a-hack-2027/data).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("MTAH_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = ROOT / "evaluation" / "results"
CONFIG_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"

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
