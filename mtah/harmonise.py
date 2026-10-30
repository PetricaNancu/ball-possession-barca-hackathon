"""Label harmonisation: bring DFL's per-frame possession flags to the GT's conventions.

CHALLENGE.md: a change of possession requires *controlled* possession; "a simple touch of the
ball without control" and "a deflection of a pass that does not intercept it" do not count.
DFL's `BallPossession` flips on such touches (dozens of sub-second X->Y->X spells per match),
Metrica's event-derived labels -- same provider and conventions as the ground truth -- do not.

The rule here is deliberately minimal: a possession spell shorter than `max_frames` whose two
neighbours are the *same other team* is a touch, not a possession, and is relabelled to that
team. Spells next to a dead ball are left alone. The threshold is chosen by matching the
possession-duration distribution of the Metrica matches (scripts/analyze_conventions.py), never
by looking at BAR-ATH.
"""

from __future__ import annotations

import numpy as np

from .labels import spells


def merge_short_sandwiches(labels: np.ndarray, max_frames: int) -> np.ndarray:
    """Relabel X->Y->X spells shorter than max_frames (Y adjacent to X on both sides) to X.

    Applied iteratively so that merging one touch can expose another; converges in a few passes.
    """
    out = np.asarray(labels, dtype="<U1").copy()
    if max_frames <= 0:
        return out
    while True:
        sp = spells(out)
        changed = False
        i = 1
        while i < len(sp) - 1:
            team, start, end = sp[i]
            prev_team, _, prev_end = sp[i - 1]
            next_team, next_start, _ = sp[i + 1]
            if (end - start < max_frames and prev_team == next_team != team
                    and prev_end == start and next_start == end):
                out[start:end] = prev_team
                changed = True
                i += 2  # the next spell's left neighbour just changed; re-evaluate it next pass
            else:
                i += 1
        if not changed:
            return out


def harmonise(labels: np.ndarray, source: str, sandwich_max_frames: int) -> np.ndarray:
    """Source-specific harmonisation; Metrica labels are the reference and pass through."""
    if source == "sportec":
        return merge_short_sandwiches(labels, sandwich_max_frames)
    return np.asarray(labels, dtype="<U1").copy()
