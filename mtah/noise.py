"""Small, label-free helpers shared by the noise statistics and the noise simulator."""

from __future__ import annotations

import numpy as np


def run_lengths(flag: np.ndarray, within: np.ndarray | None = None) -> np.ndarray:
    """Lengths of maximal runs of True in `flag`, optionally only counting frames in `within`.

    A run that crosses the edge of `within` is cut there, so the halftime gap never joins two
    runs.
    """
    f = np.asarray(flag, dtype=bool)
    if within is not None:
        f = f & np.asarray(within, dtype=bool)
    if not f.any():
        return np.zeros(0, dtype=np.int64)
    padded = np.concatenate(([False], f, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[0::2]


def ball_speed_from_positions(ball: np.ndarray, fps: float) -> np.ndarray:
    """(N,) speed in m/s from consecutive visible positions; NaN where either frame is missing.

    Provider speed fields are not trusted (units and plausibility vary); this is the one
    definition used everywhere.
    """
    d = np.full(len(ball), np.nan)
    step = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1])) * fps
    d[1:] = step
    return d


def crowding(ball: np.ndarray, players: np.ndarray, radius_m: float = 3.0) -> np.ndarray:
    """(N,) number of tracked players within radius of the ball; 0 where the ball is missing."""
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    d2 = dx * dx + dy * dy
    close = np.nan_to_num(d2, nan=np.inf) <= radius_m * radius_m
    return close.sum(axis=1)


def second_difference_std(xy: np.ndarray, present: np.ndarray) -> float:
    """Std of the second difference of a trajectory over frames where three consecutive points
    exist -- a unit-free proxy for measurement jitter, in metres per frame^2."""
    ok = present[:-2] & present[1:-1] & present[2:]
    if ok.sum() < 100:
        return float("nan")
    acc = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    return float(np.std(np.hypot(acc[ok, 0], acc[ok, 1])))


def moving_average_runs(xy: np.ndarray, present: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average of each visible run separately (no bleeding across gaps)."""
    if window <= 1:
        return xy
    out = xy.copy()
    padded = np.concatenate(([False], present, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    kernel = np.ones(window) / window
    half = window // 2
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start < window:
            continue
        for c in range(xy.shape[1]):
            seg = xy[start:end, c]
            ext = np.concatenate((np.full(half, seg[0]), seg, np.full(window - 1 - half, seg[-1])))
            out[start:end, c] = np.convolve(ext, kernel, mode="valid")
    return out


def rolling_mean(values: np.ndarray, half_window: int) -> np.ndarray:
    """Centred NaN-aware rolling mean over +/- half_window frames, O(n) via cumulative sums."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(v))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(v), idx + half_window + 1)
    count = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / count, np.nan)


def collective_speed(players: np.ndarray, fps: float, smooth_frames: int = 25,
                     max_mps: float = 12.0) -> np.ndarray:
    """(N,) mean speed of the tracked players, m/s, smoothed over a centred window.

    Per-slot steps faster than `max_mps` are slot re-assignments, not sprints, and are ignored.
    Walking players (~1 m/s) are the label-free signature of a stoppage or a very slow phase.
    """
    step = np.hypot(np.diff(players[:, :, 0], axis=0), np.diff(players[:, :, 1], axis=0)) * fps
    step = np.where(step <= max_mps, step, np.nan)
    speed = np.full(len(players), np.nan)
    with np.errstate(invalid="ignore"):
        speed[1:] = np.nanmean(step, axis=1)
    if smooth_frames > 1:
        filled = np.where(np.isnan(speed), np.nanmean(speed), speed)
        kernel = np.ones(smooth_frames) / smooth_frames
        speed = np.convolve(np.pad(filled, smooth_frames // 2, mode="edge"), kernel,
                            mode="valid")[:len(players)]
    return speed


MOVING_MPS = 2.0  # collective speed above which the game is clearly being played


def slow_phases(players: np.ndarray, fps: float, threshold_mps: float = MOVING_MPS,
                min_frames: int = 50) -> np.ndarray:
    """(N,) bool: frames inside a run of at least `min_frames` with collective speed below the
    threshold. Catches stoppages (and some very slow build-up) without any label."""
    slow = collective_speed(players, fps) < threshold_mps
    padded = np.concatenate(([False], slow, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out = np.zeros(len(players), dtype=bool)
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start >= min_frames:
            out[start:end] = True
    return out
