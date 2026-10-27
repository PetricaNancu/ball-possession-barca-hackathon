"""Our loader + baseline must reproduce the organisers' published baseline exactly.

Skipped when the data bundle is not available (CI without data).
"""

import os

import numpy as np
import pytest

import mtah  # noqa: F401
from mtah import paths
from mtah.baseline import majority_smooth, nearest_team_codes
from mtah.io import load_tracking
from mtah.labels import read_submission

DATA_PRESENT = (paths.DATA_DIR / "tactical").is_dir()


@pytest.mark.skipif(not DATA_PRESENT, reason="match data not available")
@pytest.mark.parametrize("window,name", [(0, "baseline_raw"), (50, "baseline_smooth50")])
def test_matches_reference_baseline(window, name):
    tracking = load_tracking()
    reference = read_submission(paths.BASELINES_DIR / f"{name}.csv")
    ours = majority_smooth(nearest_team_codes(tracking), window)
    assert len(ours) == len(reference) == 199319
    mismatches = int(np.sum(ours != reference))
    assert mismatches == 0, f"{mismatches} frames differ from {name}"
