"""Rule layer and per-frame features: tracking -> one feature vector per frame.

Everything here is generic football physics and works identically on BAR-ATH and on a degraded
external match, which is what makes training on other matches legitimate. The layer is offline
and bidirectional on purpose: for every frame it looks at what happened before *and* after
(the next touch, the next ball sighting), because the submission is produced from the complete
match, not live.

Feature naming convention (used by `swap_teams` for the H/A augmentation):
  - names containing "_H" / "_A" refer to the home / away team and are swapped pairwise;
  - names ending in "_sgn" are signed +home / -away quantities and are negated;
  - everything else is team-symmetric and left alone.
"""

from __future__ import annotations

import numpy as np

from .io import AWAY, HOME, Tracking
from .noise import (ball_speed_from_positions, collective_speed, moving_average_runs,
                    rolling_mean, slow_phases)

LABEL_TO_CLASS = {"H": 0, "A": 1, "D": 2}
CLASS_TO_LABEL = np.array(["H", "A", "D"])

# Rule-layer constants: generic physics / football, not fitted to any match.
BALL_SMOOTH_FRAMES = 5
MAX_FILL_FRAMES = 50            # interpolate ball gaps up to 2 s
TOUCH_ACCEL_MPS2 = 12.0         # acceleration spike that marks a touch
TOUCH_TURN_RAD = 0.6            # heading change (at speed) that marks a touch
TOUCH_MIN_SPEED = 1.5
TOUCH_RADIUS_M = 2.5            # a touch is attributed only if a player is this close
CONTROL_CARRY_FRAMES = 10       # ball stays with the team this long -> controlled
CONTROL_RADIUS_M = 2.0
CROWD_RADIUS_M = 3.0
STATIONARY_MPS = 0.5


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def _ffill(values: np.ndarray, valid: np.ndarray, fill):
    idx = np.where(valid, np.arange(len(values)), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, values[np.maximum(idx, 0)], fill)
    return out


def _bfill(values: np.ndarray, valid: np.ndarray, fill):
    n = len(values)
    idx = np.where(valid, np.arange(n), n)
    idx = np.minimum.accumulate(idx[::-1])[::-1]
    out = np.where(idx < n, values[np.minimum(idx, n - 1)], fill)
    return out


def _frames_since(valid: np.ndarray, cap: int) -> np.ndarray:
    """Frames since the last True (0 on a True frame), capped; cap where none before."""
    idx = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))
    out = np.where(idx >= 0, np.arange(len(valid)) - idx, cap)
    return np.minimum(out, cap)


def _frames_until(valid: np.ndarray, cap: int) -> np.ndarray:
    return _frames_since(valid[::-1], cap)[::-1]


def fill_ball_gaps(ball_xy: np.ndarray, blocks, max_fill: int = MAX_FILL_FRAMES):
    """Linear interpolation of short gaps inside a block. Returns (filled_xy, interpolated_mask)."""
    out = ball_xy.copy()
    interp = np.zeros(len(out), dtype=bool)
    present = ~np.isnan(out[:, 0])
    for _, first, last in blocks:
        for start, end in _runs(~present[first:last + 1]):
            s, e = first + start, first + end  # gap = [s, e)
            if e - s > max_fill or s == first or e > last:
                continue
            t = np.arange(1, e - s + 1) / (e - s + 1)
            out[s:e] = out[s - 1] + t[:, None] * (out[e] - out[s - 1])
            interp[s:e] = True
    return out, interp


# --------------------------------------------------------------------------------------------
# rule layer
# --------------------------------------------------------------------------------------------

def detect_touches(ball_xy, known, speed, accel, turn, d_nearest, team_nearest, d_opp_nearest):
    """Touch events: (frame, team, contested). A touch is an acceleration spike or a sharp turn
    while moving, attributed to the nearest player if close enough; consecutive touch frames
    within 3 frames collapse into one event."""
    candidate = known & ((accel > TOUCH_ACCEL_MPS2) | ((turn > TOUCH_TURN_RAD) & (speed > TOUCH_MIN_SPEED)))
    candidate &= d_nearest <= TOUCH_RADIUS_M
    events = []
    last = -10
    for f in np.flatnonzero(candidate):
        if f - last <= 3:
            last = f
            continue
        events.append((int(f), int(team_nearest[f]), bool(d_opp_nearest[f] <= TOUCH_RADIUS_M)))
        last = f
    return events


def controlled_possession(events, n_frames, d_team, blocks):
    """Team in controlled possession per frame from the touch sequence (CHALLENGE.md rules).

    A touch by team T flips possession to T only if T also makes the next touch, or the ball
    stays within CONTROL_RADIUS_M of a T player for CONTROL_CARRY_FRAMES after it (a carry /
    controlled reception). A lone touch (deflection, failed clearance) does not. Possession is
    unknown (-1) at the start of each block until the first controlled touch.
    """
    block_id = np.full(n_frames, -1, dtype=np.int64)
    for b, (_, first, last) in enumerate(blocks):
        block_id[first:last + 1] = b
    mark = np.full(n_frames, -1, dtype=np.int8)
    current, current_block = -1, -1
    for i, (f, team, _) in enumerate(events):
        if block_id[f] != current_block:
            current, current_block = -1, block_id[f]
        if team < 0 or team == current:
            continue
        next_same = i + 1 < len(events) and events[i + 1][1] == team
        window = d_team[team][f:f + CONTROL_CARRY_FRAMES]
        carry = len(window) == CONTROL_CARRY_FRAMES and bool(np.all(window <= CONTROL_RADIUS_M))
        if next_same or carry:
            current = team
            mark[f] = team
    state = np.full(n_frames, -1, dtype=np.int8)
    for _, first, last in blocks:
        seg = mark[first:last + 1]
        state[first:last + 1] = _ffill(seg, seg >= 0, -1)
    return state


# --------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------

def compute_features(tr: Tracking) -> tuple[np.ndarray, list[str]]:
    """(X (N, F) float32, feature names). NaN where undefined; LightGBM handles it natively."""
    n, fps = tr.n_frames, tr.fps
    L, W = tr.pitch
    blocks = tr.blocks
    graded = tr.graded_mask()
    feats: dict[str, np.ndarray] = {}

    # ---- ball track: smooth, fill short gaps, kinematics --------------------------------
    observed = tr.ball_present
    ball_xy = moving_average_runs(tr.ball[:, :2].copy(), observed, BALL_SMOOTH_FRAMES)
    ball_xy, interp = fill_ball_gaps(ball_xy, blocks)
    known = ~np.isnan(ball_xy[:, 0])
    vel = np.full((n, 2), np.nan)
    vel[1:-1] = (ball_xy[2:] - ball_xy[:-2]) * fps / 2
    speed = np.hypot(vel[:, 0], vel[:, 1])
    accel = np.full(n, np.nan)
    accel[1:-1] = np.hypot(vel[2:, 0] - vel[:-2, 0], vel[2:, 1] - vel[:-2, 1]) * fps / 2
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    turn = np.full(n, np.nan)
    turn[1:-1] = np.abs(np.angle(np.exp(1j * (heading[2:] - heading[:-2]))))

    feats["ball_observed"] = observed.astype(np.float32)
    feats["ball_known"] = known.astype(np.float32)
    feats["ball_interp"] = interp.astype(np.float32)
    feats["ball_speed"] = speed
    feats["ball_accel"] = accel
    feats["ball_turn"] = turn
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    feats["ball_to_sideline"] = np.minimum(by, W - by)            # negative when outside
    feats["ball_to_goalline"] = np.minimum(bx, L - bx)
    feats["ball_oob"] = (known & ((bx < 0) | (bx > L) | (by < 0) | (by > W))).astype(np.float32)
    stationary = known & (speed < STATIONARY_MPS)
    feats["ball_stationary_frames"] = _frames_since(~stationary | ~known, 250).astype(np.float32) * stationary

    # ---- ball vs players ------------------------------------------------------------------
    px, py = tr.players[:, :, 0], tr.players[:, :, 1]
    dist = np.hypot(px - bx[:, None], py - by[:, None])             # (N, S), NaN where missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_team = {}
    for team, tag in ((HOME, "H"), (AWAY, "A")):
        cols = tr.team_of_slot == team
        d = np.sort(dist[:, cols], axis=1)
        d1 = np.where(np.isfinite(d[:, 0]), d[:, 0], np.nan) if d.shape[1] else np.full(n, np.nan)
        d2 = np.where(np.isfinite(d[:, 1]), d[:, 1], np.nan) if d.shape[1] > 1 else np.full(n, np.nan)
        d_team[team] = np.where(np.isnan(d1), np.inf, d1)
        feats[f"d_ball_{tag}1"] = d1
        feats[f"d_ball_{tag}2"] = d2
        feats[f"n_{tag}_3m"] = (dist[:, cols] <= CROWD_RADIUS_M).sum(axis=1).astype(np.float32)
        feats[f"n_{tag}_8m"] = (dist[:, cols] <= 8.0).sum(axis=1).astype(np.float32)
    dH, dA = d_team[HOME], d_team[AWAY]
    nearest_team = np.where(dH <= dA, HOME, AWAY).astype(np.int8)
    nearest_team[~known | (~np.isfinite(dH) & ~np.isfinite(dA))] = -1
    d_nearest = np.minimum(dH, dA)
    d_opp = np.where(nearest_team == HOME, dA, dH)
    feats["d_margin_sgn"] = np.where(known, np.clip(dA - dH, -30, 30), np.nan)  # +: home closer
    feats["nearest_team_sgn"] = np.where(nearest_team >= 0, np.where(nearest_team == HOME, 1.0, -1.0), np.nan)
    feats["d_nearest"] = np.where(np.isfinite(d_nearest), d_nearest, np.nan)

    # ---- touches, last / next touch, controlled possession --------------------------------
    events = detect_touches(ball_xy, known, speed, accel, turn, d_nearest, nearest_team, d_opp)
    touch_frame = np.zeros(n, dtype=bool)
    touch_team = np.full(n, -1, dtype=np.int8)
    touch_contested = np.zeros(n, dtype=bool)
    for f, team, contested in events:
        touch_frame[f] = True
        touch_team[f] = team
        touch_contested[f] = contested
    team_sgn = np.where(touch_team == HOME, 1.0, np.where(touch_team == AWAY, -1.0, np.nan))
    feats["last_touch_sgn"] = _ffill(team_sgn, touch_frame, np.nan)
    feats["next_touch_sgn"] = _bfill(team_sgn, touch_frame, np.nan)
    feats["t_since_touch"] = _frames_since(touch_frame, 500).astype(np.float32)
    feats["t_to_touch"] = _frames_until(touch_frame, 500).astype(np.float32)
    feats["last_touch_contested"] = _ffill(touch_contested.astype(float), touch_frame, np.nan)
    feats["touch_rate_5s"] = rolling_mean(touch_frame.astype(float), 62) * 125
    state = controlled_possession(events, n, d_team, blocks)
    state_sgn = np.where(state == HOME, 1.0, np.where(state == AWAY, -1.0, np.nan))
    feats["control_sgn"] = state_sgn
    feats["control_next_sgn"] = _bfill(state_sgn, state >= 0, np.nan)
    # frames since / until the controlled team last changed (bidirectional transition context)
    change = np.zeros(n, dtype=bool)
    change[1:] = (state[1:] != state[:-1]) & (state[1:] >= 0)
    feats["t_since_control_change"] = _frames_since(change, 750).astype(np.float32)
    feats["t_to_control_change"] = _frames_until(change, 750).astype(np.float32)

    # ---- gap context (bidirectional) -----------------------------------------------------
    cap = 750
    feats["t_since_seen"] = _frames_since(observed, cap).astype(np.float32)
    feats["t_to_seen"] = _frames_until(observed, cap).astype(np.float32)
    gap_len = feats["t_since_seen"] + feats["t_to_seen"]
    feats["gap_len"] = np.where(observed, 0.0, gap_len)
    margin = np.where(observed, dA - dH, np.nan)
    feats["last_seen_margin_sgn"] = _ffill(margin, observed, np.nan)
    feats["next_seen_margin_sgn"] = _bfill(margin, observed, np.nan)
    sp_obs = np.where(observed, speed, np.nan)
    feats["last_seen_speed"] = _ffill(sp_obs, observed, np.nan)
    feats["next_seen_speed"] = _bfill(sp_obs, observed, np.nan)

    # ---- collective motion (works without the ball) --------------------------------------
    present = tr.player_present
    feats["n_objects"] = present.sum(axis=1).astype(np.float32)
    feats["collective_speed"] = collective_speed(tr.players, fps)
    feats["slow_phase"] = slow_phases(tr.players, fps).astype(np.float32)
    step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * fps
    walking = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        walking[1:] = np.nanmean(np.where(step <= 12, (step < 1.0).astype(float), np.nan), axis=1)
    feats["frac_players_walking"] = rolling_mean(walking, 12)
    cents = {}
    for team, tag in ((HOME, "H"), (AWAY, "A")):
        cols = tr.team_of_slot == team
        with np.errstate(invalid="ignore"):
            cx = np.nanmean(np.where(present[:, cols], px[:, cols], np.nan), axis=1)
            cy = np.nanmean(np.where(present[:, cols], py[:, cols], np.nan), axis=1)
            sx = np.nanstd(np.where(present[:, cols], px[:, cols], np.nan), axis=1)
            sy = np.nanstd(np.where(present[:, cols], py[:, cols], np.nan), axis=1)
        cents[team] = (cx, cy)
        feats[f"spread_x_{tag}"] = sx
        feats[f"spread_y_{tag}"] = sy
        feats[f"d_ball_centroid_{tag}"] = np.hypot(bx - cx, by - cy)
    feats["centroid_dist"] = np.hypot(cents[HOME][0] - cents[AWAY][0], cents[HOME][1] - cents[AWAY][1])
    feats["centroid_dx_sgn"] = cents[HOME][0] - cents[AWAY][0]
    # team in front of the other along x: which team's centroid is nearer the ball's goal line side
    feats["ball_x_rel_centroids"] = (bx - (cents[HOME][0] + cents[AWAY][0]) / 2)

    # ---- block context ----------------------------------------------------------------------
    since_block = np.full(n, np.nan)
    moved_yet = np.zeros(n, dtype=np.float32)
    for _, first, last in blocks:
        since_block[first:last + 1] = np.arange(last - first + 1)
        fast = np.nan_to_num(speed[first:last + 1], nan=0.0) > 3.0
        moved_yet[first:last + 1] = np.cumsum(fast) > 0
    feats["t_since_block_start"] = np.minimum(since_block, 5000)
    feats["ball_moved_in_block"] = moved_yet

    # ---- temporal context: rolling means over +/- 0.5 s, 2 s, 6 s -------------------------
    for name in ("d_ball_H1", "d_ball_A1", "d_margin_sgn", "nearest_team_sgn", "ball_speed",
                 "ball_observed", "collective_speed", "control_sgn", "ball_oob"):
        for half in (12, 50, 150):
            feats[f"{name}_rm{half}"] = rolling_mean(feats[name], half)

    names = list(feats)
    X = np.column_stack([np.asarray(feats[k], dtype=np.float32) for k in names])
    X[~graded] = np.nan  # never let ungraded frames leak structure; they are X anyway
    return X, names


def swap_teams(X: np.ndarray, names: list[str]) -> np.ndarray:
    """The same frames seen from the other team's side: swap *_H/*_A columns, negate *_sgn."""
    out = X.copy()
    index = {k: i for i, k in enumerate(names)}
    for k, i in index.items():
        if k.endswith("_sgn") or "_sgn_" in k:
            out[:, i] = -X[:, i]
        elif "_H" in k:
            partner = k.replace("_H", "_A")
            if partner in index:
                out[:, i] = X[:, index[partner]]
        elif "_A" in k:
            partner = k.replace("_A", "_H")
            if partner in index:
                out[:, i] = X[:, index[partner]]
    return out


def swap_labels(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out
