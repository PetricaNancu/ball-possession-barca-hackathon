"""Decoder: per-frame probabilities -> one label sequence with few, long segments.

Viterbi over the three states {H, A, D}, run independently inside each graded block (half).
Staying costs nothing; switching between the teams costs `switch_team`, entering or leaving a
dead ball costs `switch_dead` (log-probability units). A final pass merges spells shorter than
`min_frames` into the longer neighbour. Both penalties and the minimum duration are tuned on the
official metric over held-out *external* matches (scripts/train.py); none of this touches
BAR-ATH labels.

The scorer forward-fills X inside a half before matching possessions, so there is nothing to gain
from abstaining on a graded frame; the decoder always commits.
"""

from __future__ import annotations

import numpy as np

from .features import CLASS_TO_LABEL

H, A, D = 0, 1, 2
LOG_FLOOR = 1e-4


def viterbi_block(log_emis: np.ndarray, switch_team: float, switch_dead: float,
                  prior: np.ndarray | None = None) -> np.ndarray:
    """Most probable state path for one block. log_emis (T, 3); optional additive prior (T, 3).

    Plain Python floats in the recursion: with three states that is several times faster than
    per-frame numpy calls, and the decoder grid search runs it thousands of times.
    """
    emis = log_emis if prior is None else log_emis + prior
    e = emis.tolist()
    T = len(e)
    back = np.zeros((T, 3), dtype=np.int8)
    sH, sA, sD = e[0]
    st, sd = float(switch_team), float(switch_dead)
    for t in range(1, T):
        eH, eA, eD = e[t]
        # into H
        fromA, fromD = sA - st, sD - sd
        if sH >= fromA and sH >= fromD:
            nH, bH = sH, 0
        elif fromA >= fromD:
            nH, bH = fromA, 1
        else:
            nH, bH = fromD, 2
        # into A
        fromH, fromD = sH - st, sD - sd
        if sA >= fromH and sA >= fromD:
            nA, bA = sA, 1
        elif fromH >= fromD:
            nA, bA = fromH, 0
        else:
            nA, bA = fromD, 2
        # into D
        fromH, fromA = sH - sd, sA - sd
        if sD >= fromH and sD >= fromA:
            nD, bD = sD, 2
        elif fromH >= fromA:
            nD, bD = fromH, 0
        else:
            nD, bD = fromA, 1
        back[t, 0], back[t, 1], back[t, 2] = bH, bA, bD
        sH, sA, sD = nH + eH, nA + eA, nD + eD
    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(np.argmax([sH, sA, sD]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short_spells(states: np.ndarray, min_frames: int) -> np.ndarray:
    """Relabel runs shorter than min_frames to the longer of their two neighbours, shortest
    first, until none remain. Runs at a block edge keep their only neighbour."""
    out = states.copy()
    if min_frames <= 1 or len(out) == 0:
        return out
    while True:
        edges = np.flatnonzero(np.diff(out)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(out)]))
        lengths = ends - starts
        short = np.flatnonzero(lengths < min_frames)
        if not len(short) or len(starts) == 1:
            return out
        i = short[np.argmin(lengths[short])]
        left = lengths[i - 1] if i > 0 else -1
        right = lengths[i + 1] if i + 1 < len(starts) else -1
        donor = i - 1 if left >= right else i + 1
        out[starts[i]:ends[i]] = out[starts[donor]]


def decode(proba: np.ndarray, blocks, switch_team: float, switch_dead: float,
           min_frames: int = 0, prior: np.ndarray | None = None) -> np.ndarray:
    """(N,) labels in {H, A, D, X}: X outside the blocks, decoded inside."""
    labels = np.full(len(proba), "X", dtype="<U1")
    log_p = np.log(np.clip(np.nan_to_num(proba, nan=1.0 / 3), LOG_FLOOR, 1.0))
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        path = viterbi_block(log_p[seg], switch_team, switch_dead,
                             None if prior is None else prior[seg])
        path = merge_short_spells(path, min_frames)
        labels[seg] = CLASS_TO_LABEL[path]
    return labels


def argmax_labels(proba: np.ndarray, blocks) -> np.ndarray:
    """No decoding at all -- the per-frame argmax, for ablation."""
    labels = np.full(len(proba), "X", dtype="<U1")
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        labels[seg] = CLASS_TO_LABEL[np.argmax(np.nan_to_num(proba[seg], nan=0.0), axis=1)]
    return labels
