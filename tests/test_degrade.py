"""Noise simulator: determinism, label safety, and that it moves the clean statistics toward
BAR-ATH's targets. Needs the corpus (skipped otherwise)."""

import numpy as np
import pytest

import mtah  # noqa: F401
from mtah import paths
from mtah.degrade import NoiseModel, degrade, moving_average_runs
from mtah.io import Tracking

MODEL = paths.MODELS_DIR / "noise_model.npz"
CLEAN = paths.CACHE_DIR / "J03WPY_sportec_tracking.npz"
needs_corpus = pytest.mark.skipif(not (MODEL.is_file() and CLEAN.is_file()),
                                  reason="noise model / corpus not built")


def test_moving_average_does_not_bleed_across_gaps():
    xy = np.array([[0, 0], [1, 0], [2, 0], [np.nan, np.nan], [10, 0], [11, 0], [12, 0]], float)
    present = ~np.isnan(xy[:, 0])
    out = moving_average_runs(xy, present, 3)
    assert np.isnan(out[3, 0])
    assert out[1, 0] == pytest.approx(1.0) and out[5, 0] == pytest.approx(11.0)


@needs_corpus
def test_degrade_is_deterministic_and_keeps_labels():
    clean = Tracking.load(CLEAN)
    model = NoiseModel.load(MODEL)
    a = degrade(clean, model, seed=3)
    b = degrade(clean, model, seed=3)
    c = degrade(clean, model, seed=4)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.players, b.players, equal_nan=True)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)
    assert np.array_equal(a.labels, clean.labels)
    assert np.array_equal(a.blocks, clean.blocks)


@needs_corpus
def test_degrade_matches_targets_roughly():
    clean = Tracking.load(CLEAN)
    model = NoiseModel.load(MODEL)
    d = degrade(clean, model, seed=0)
    g = d.graded_mask()
    assert d.n_slots == model.n_slots_home + model.n_slots_away == 46
    assert abs(d.ball_present[g].mean() - model.target_visible) < 0.03
    objects = d.player_present[g].sum(axis=1)
    assert abs(objects.mean() - model.target_objects_mean) < 0.5
    assert (objects > 22).mean() > 0
    assert not d.player_present[~g].any() and not d.ball_present[~g].any()
