"""Shared bootstrap for scripts run as `python scripts/<name>.py` without installing the package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mtah  # noqa: E402,F401  adds vendor/mtah2027 (+ its evaluation/) to sys.path
