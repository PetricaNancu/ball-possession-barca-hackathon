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

from .features import CLASS_TO_LABEL, _runs
from .noise import ball_speed_from_positions

H, A, D = 0, 1, 2
LOG_FLOOR = 1e-4

ATTACK_CODES = {
    "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES",
    "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
}
DEFEND_CODES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
}


def detect_kickoff_frame(tracking, start_frame: int, end_frame: int) -> int:
    """Detect the kickoff strike inside a Smart Tagging kickoff window [start_frame, end_frame].

    Finds the last frame where the ball rests near the centre spot (<= 3.0 m, speed < 1.5 m/s)
    before being put into play, returning the immediately following frame. Falls back to the
    tagged window midpoint when the ball is unobserved across the window.
    """
    s = max(0, int(start_frame))
    end = min(tracking.n_frames, int(end_frame))
    if end <= s:
        return s
    cx, cy = tracking.pitch[0] / 2.0, tracking.pitch[1] / 2.0
    bx = tracking.ball[s:end, 0]
    by = tracking.ball[s:end, 1]
    spd = ball_speed_from_positions(tracking.ball[s:end, :2], tracking.fps)
    last_stationary = None
    for idx in range(end - s):
        if np.isfinite(bx[idx]):
            d_c = float(np.hypot(bx[idx] - cx, by[idx] - cy))
            v = float(spd[idx]) if np.isfinite(spd[idx]) else 0.0
            if d_c <= 3.0 and v < 1.5:
                last_stationary = s + idx
    if last_stationary is not None and last_stationary + 1 < end:
        return last_stationary + 1
    return (s + end) // 2


def smart_tagging_prior(tracking, phase_weight: float = 0.05) -> np.ndarray | None:
    """Build log-probability prior (N, 3) from Smart Tagging when available for `tracking.game`.

    Enforces D before each half's kickoff ('Kick Off (Start)', detected algorithmically inside the
    tagged restart window), during post-goal stoppages before 'Kick Off (After Goal)', and adds a
    soft prior inside interior attacking/defending phase intervals.
    """
    try:
        import starter.load_smart_tagging as st
        events = st.load(game=tracking.game)
    except Exception:
        return None

    n = tracking.n_frames
    prior = np.zeros((n, 3), dtype=np.float32)
    vote_H = np.zeros(n, dtype=np.float32)
    vote_A = np.zeros(n, dtype=np.float32)

    for e in events:
        s = max(0, int(e["start_frame"]))
        end = min(n, int(e["end_frame"]))
        code = (e.get("code") or "").strip().upper()
        tags = e.get("tags") or {}
        team = (tags.get("Team") or "").strip()
        typ = (tags.get("Type") or "").strip().lower()

        if code in ATTACK_CODES:
            s_in, e_in = min(end, s + 25), max(s, end - 25)
            if team == "Barcelona":
                vote_H[s_in:e_in] += 1.0
            elif team == "Atletic Club":
                vote_A[s_in:e_in] += 1.0
        elif code in DEFEND_CODES:
            s_in, e_in = min(end, s + 25), max(s, end - 25)
            if team == "Barcelona":
                vote_A[s_in:e_in] += 1.0
            elif team == "Atletic Club":
                vote_H[s_in:e_in] += 1.0
        elif code == "SET PIECES":
            if "kick off (start)" in typ:
                ko_frame = detect_kickoff_frame(tracking, s, end)
                for _, first, last in tracking.blocks:
                    if first <= ko_frame <= last:
                        prior[first:ko_frame, D] += 25.0
            elif "kick off (after goal)" in typ:
                ko_frame = detect_kickoff_frame(tracking, s, end)
                prior[max(0, s - 400):ko_frame, D] += 12.0

    if phase_weight > 0:
        net = np.clip(vote_H - vote_A, -3.0, 3.0)
        prior[:, H] += phase_weight * np.maximum(0.0, net)
        prior[:, A] += phase_weight * np.maximum(0.0, -net)
    return prior


def postprocess_stoppages(labels: np.ndarray, X: np.ndarray, names: list[str], blocks,
                          tracking=None) -> np.ndarray:
    """Physics-based stoppage & boundary refinement:
    1. Pre-kickoff graded frames (up to the detected kickoff strike inside Smart Tagging 'Kick Off
       (Start)') are D.
    2. Out-of-bounds latched frames (ball crossed pitch boundary and stoppage confirmed) are D.
    3. Missing-ball gaps >= 100 frames where players decelerate into a stoppage (tail_cs < 1.40)
       are D from the frame where collective_speed < 2.60 m/s.
    4. Short H/A spells (< 225 frames = 9s) sandwiched between D and D while walking (cs < 1.60)
       are absorbed into D; sub-3s (< 75 frames) flips immediately preceding D become D if already
       in stoppage speed (< 1.35 m/s) or stay with the previous possessor (CHALLENGE.md rule).
    """
    idx = {n: i for i, n in enumerate(names)}
    coll_spd = X[:, idx["collective_speed"]]
    oob_latched = X[:, idx["oob_latched"]] if "oob_latched" in idx else np.zeros(len(X))
    ball_obs = X[:, idx["ball_observed"]] > 0.5
    out = labels.copy()

    # Enforce pre-kickoff D if Smart Tagging is present
    if tracking is not None:
        try:
            import starter.load_smart_tagging as st
            for e in st.load(game=tracking.game):
                code = (e.get("code") or "").strip().upper()
                typ = ((e.get("tags") or {}).get("Type") or "").strip().lower()
                if code == "SET PIECES" and "kick off (start)" in typ:
                    ko_frame = detect_kickoff_frame(tracking, int(e["start_frame"]), int(e["end_frame"]))
                    for _, first, last in blocks:
                        if first <= ko_frame <= last:
                            out[first:ko_frame] = "D"
        except Exception:
            pass

    for _, first, last in blocks:
        seg = out[first:last + 1]
        cs = coll_spd[first:last + 1]
        oob = oob_latched[first:last + 1]
        obs = ball_obs[first:last + 1]

        seg[oob > 0.5] = "D"

        for gs, ge in _runs(~obs):
            if ge - gs >= 100:
                tail_cs = float(np.nanmean(cs[gs + (ge - gs) // 3:ge]))
                if tail_cs < 1.40:
                    for t in range(gs, ge):
                        if cs[t] < 2.60:
                            seg[t:ge] = "D"
                            break

        for _ in range(4):
            edges = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            starts = np.concatenate(([0], edges))
            ends = np.concatenate((edges, [len(seg)]))
            changed = False
            for i in range(len(starts)):
                s, e = starts[i], ends[i]
                lbl = seg[s]
                if lbl == "D":
                    continue
                dur = e - s
                mean_cs = float(np.nanmean(cs[s:e])) if dur > 0 else 2.0
                prev_lbl = seg[starts[i - 1]] if i > 0 else "D"
                next_lbl = seg[starts[i + 1]] if i + 1 < len(starts) else "D"
                if prev_lbl == "D" and next_lbl == "D" and dur < 225 and mean_cs < 1.60:
                    seg[s:e] = "D"
                    changed = True
                elif next_lbl == "D" and dur < 75:
                    seg[s:e] = "D" if mean_cs < 1.35 else prev_lbl
                    changed = True
            if not changed:
                break
        out[first:last + 1] = seg
    return out


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
