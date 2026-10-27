"""The organisers' reference baseline, re-implemented on the canonical `Tracking` arrays.

Nearest tracked player to the ball -> that team; no ball -> nobody; then a centred majority vote.
Kept for two reasons: it is the published floor (spellF1 0.2433 on the full match), and matching
`starter/baseline.py` bit for bit is the test that our loader is right.
"""

from __future__ import annotations

import numpy as np

from .io import HOME, AWAY, Tracking

NOBODY = -1


def nearest_team_codes(tracking: Tracking) -> np.ndarray:
    """(N,) int8: HOME/AWAY of the player nearest the ball, NOBODY where there is no answer.

    Squared distances in metres, as in the reference; ties go to the lowest slot index.
    """
    ball, players = tracking.ball, tracking.players
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    d2 = dx * dx + dy * dy
    d2 = np.where(np.isnan(d2), np.inf, d2)
    best_slot = np.argmin(d2, axis=1)
    has_answer = np.isfinite(d2[np.arange(len(d2)), best_slot])
    codes = np.full(len(ball), NOBODY, dtype=np.int8)
    codes[has_answer] = tracking.team_of_slot[best_slot[has_answer]]
    return codes


def majority_smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred majority vote over +/- `window` frames. D where nobody has the ball; ties -> H."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype("<U1")
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def baseline_labels(tracking: Tracking, window: int = 50) -> np.ndarray:
    return majority_smooth(nearest_team_codes(tracking), window)
