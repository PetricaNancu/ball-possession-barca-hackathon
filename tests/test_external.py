"""Invariants of the external corpus. Skipped for matches not yet built (`make corpus`)."""

import numpy as np
import pytest

import mtah  # noqa: F401
from mtah import paths
from mtah.external import METRICA_MATCHES, SPORTEC_MATCHES
from mtah.io import Tracking
from mtah.labels import mask_by_blocks, spells

CACHED = sorted(p for p in paths.CACHE_DIR.glob("*_tracking.npz") if "BAR-ATH" not in p.name)


@pytest.mark.skipif(not CACHED, reason="no external match cached; run scripts/build_corpus.py")
@pytest.mark.parametrize("path", CACHED, ids=[p.name for p in CACHED])
def test_external_tracking_invariants(path):
    tr = Tracking.load(path)
    assert tr.game in SPORTEC_MATCHES + METRICA_MATCHES
    assert tr.fps == 25.0 and tr.pitch == (105.0, 68.0)
    assert tr.blocks.shape == (2, 3) and tr.blocks[0, 1] == 0 and tr.blocks[1, 2] == tr.n_frames - 1
    graded = tr.graded_mask()
    assert set(np.unique(tr.labels[graded])) <= {"H", "A", "D"}
    assert set(np.unique(tr.labels[~graded])) <= {"X"}
    assert (tr.player_present[graded].sum(axis=1) == 22).all()
    assert not tr.player_present[~graded].any()
    if tr.source == "sportec":
        assert tr.ball_present[graded].all()
    else:
        assert tr.ball_present[graded].mean() > 0.5
    # Positions in metres on the pitch (a little slack for the ball leaving the field).
    assert np.nanmin(tr.ball[:, 0]) > -10 and np.nanmax(tr.ball[:, 0]) < 115
    assert np.nanmin(tr.ball[:, 1]) > -10 and np.nanmax(tr.ball[:, 1]) < 78
    assert set(tr.team_of_slot.tolist()) == {0, 1}
    sp = spells(tr.labels)
    assert 200 < len(sp) < 700, f"{len(sp)} possessions is implausible for one match"
    assert (mask_by_blocks(tr.labels, tr.blocks) == tr.labels).all()
