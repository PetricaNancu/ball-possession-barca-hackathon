"""Noise simulator: make clean, corrected tracking look like the automatic BAR-ATH ATD.

Fitted only to label-free statistics of BAR-ATH (scripts/noise_stats.py -> corpus/noise_targets.*)
and applied to the external matches before training. Four effects, in this order:

1. Smoothing. The ATD is heavily filtered (second-difference std 0.015 m for the ball, 0.002 m
   for players, against 0.48 / 0.07 m in the corrected sources), so external trajectories are
   smoothed with a moving average whose window reproduces those numbers.
2. Ball dropout, by regime. In BAR-ATH 91 % of the frames without a ball fall in "slow" phases
   (collective player speed < 2 m/s for >= 2 s: stoppages and very slow play) where the ball is
   missing 55 % of the time; in "moving" phases it is missing 11 %. Gaps are long (median ~10 s).
   For each regime the simulator *tops up* the source's own missing rate to the target by starting
   gaps at a calibrated per-frame hazard and drawing their lengths from BAR-ATH's empirical,
   regime-conditional gap distribution. Metrica's corrected tracking already lacks the ball in
   most stoppages, so it receives little extra dropout there; Sportec receives it all.
   Ball excursions outside the pitch are dropped with a probability that reproduces the share of
   visible out-of-bounds frames.
3. Player tracks. Each player's track is cut into fragments whose lengths follow BAR-ATH's slot
   lifetimes, separated by absences whose lengths are scaled to reproduce the mean number of
   tracked objects per frame; every fragment is re-assigned to an anonymous slot of its team
   (46 slots: 24 home, 22 away, as in the ATD), so identities are gone.
4. Ghost duplicates of present players, at the rate that reproduces frames with > 22 objects.

Labels are untouched. Everything is seeded and deterministic; calibrations are bisections on the
realised statistics of the match being degraded, so one model serves every source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .io import AWAY, HOME, Tracking
from .noise import ball_speed_from_positions, second_difference_std, slow_phases

TARGETS_JSON = paths.ROOT / "corpus" / "noise_targets.json"
TARGETS_NPZ = paths.ROOT / "corpus" / "noise_targets.npz"

MAX_ABSENCE_FRAMES = 750       # longer slot holes are unused slots, not player absences
GHOST_LENGTH_FRAMES = 60
GHOST_OFFSET_STD_M = 0.7
# The ATD's filtering is heavier than any centred moving average can mimic without lagging fast
# trajectories (a 41-frame window costs the reference baseline ~0.03 spellF1); windows are capped.
MAX_SMOOTH_WINDOW = 15


@dataclass
class NoiseModel:
    target_visible: float
    target_missing_slow: float
    target_missing_moving: float
    start_rate_slow: float                # BAR-ATH gap starts per visible slow frame
    start_rate_moving: float
    gap_slow: np.ndarray                  # empirical gap lengths (frames) starting in slow phases
    gap_moving: np.ndarray
    target_oob_visible: float
    target_objects_mean: float
    target_over_22: float
    lifetimes: np.ndarray                 # empirical fragment lengths (frames)
    absences: np.ndarray                  # empirical absence lengths (frames), truncated
    n_slots_home: int
    n_slots_away: int
    target_ball_jitter: float
    target_player_jitter: float
    ball_smooth_window: int = 1
    player_smooth_window: int = 1
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_targets(cls, json_path: Path = TARGETS_JSON, npz_path: Path = TARGETS_NPZ) -> "NoiseModel":
        t = json.loads(Path(json_path).read_text())
        z = np.load(npz_path)
        b, p, r = t["ball"], t["players"], t["ball"]["regimes"]
        holes = z["slot_holes"]
        return cls(
            target_visible=float(b["visible_graded"]),
            target_missing_slow=float(r["p_missing_given_slow"]),
            target_missing_moving=float(r["p_missing_given_moving"]),
            start_rate_slow=float(r["p_gap_start_per_slow_frame"]),
            start_rate_moving=float(r["p_gap_start_per_moving_frame"]),
            gap_slow=z["ball_gap_lengths_slow"], gap_moving=z["ball_gap_lengths_moving"],
            target_oob_visible=float(b["out_of_bounds_when_visible"]),
            target_objects_mean=float(p["objects_per_graded_frame_mean"]),
            target_over_22=float(p["share_frames_over_22"]),
            lifetimes=z["slot_lifetimes"][z["slot_lifetimes"] > 0],
            absences=holes[(holes > 0) & (holes < MAX_ABSENCE_FRAMES)],
            n_slots_home=int(p["n_slots_home"]), n_slots_away=int(p["n_slots_away"]),
            target_ball_jitter=float(b["jitter_second_diff_m"]),
            target_player_jitter=float(p["jitter_second_diff_m_median_over_slots"]),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scalars = {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)}
        np.savez_compressed(path, gap_slow=self.gap_slow, gap_moving=self.gap_moving,
                            lifetimes=self.lifetimes, absences=self.absences,
                            scalars=np.array(json.dumps(scalars)))
        return path

    @classmethod
    def load(cls, path: Path) -> "NoiseModel":
        z = np.load(path)
        return cls(gap_slow=z["gap_slow"], gap_moving=z["gap_moving"], lifetimes=z["lifetimes"],
                   absences=z["absences"], **json.loads(str(z["scalars"])))


# --------------------------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------------------------

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


def fit_smoothing_window(xy: np.ndarray, present: np.ndarray, target_jitter: float,
                         candidates=(1, 3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 51),
                         max_window: int = MAX_SMOOTH_WINDOW) -> int:
    best, best_err = 1, float("inf")
    for w in candidates:
        if w > max_window:
            continue
        j = second_difference_std(moving_average_runs(xy, present, w), present)
        err = abs(np.log(max(j, 1e-9)) - np.log(target_jitter))
        if err < best_err:
            best, best_err = w, err
    return best


# --------------------------------------------------------------------------------------------
# Ball dropout
# --------------------------------------------------------------------------------------------

def _drop_runs(ball: np.ndarray, mask: np.ndarray, prob: float, rng: np.random.Generator):
    """Set whole runs of `mask` to NaN, each with probability `prob`."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[0::2], edges[1::2]):
        if rng.random() < prob:
            ball[start:end] = np.nan


def _start_gaps(present: np.ndarray, hazard: np.ndarray, slow: np.ndarray, model: NoiseModel,
                block_end: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """New presence mask after starting gaps with a per-frame hazard; lengths drawn from the
    regime-conditional pool of the frame where the gap starts."""
    out = present.copy()
    candidates = np.flatnonzero(present & (hazard > 0))
    starts = candidates[rng.random(len(candidates)) < hazard[candidates]]
    skip_until = -1
    for s in starts:
        if s < skip_until:
            continue
        pool = model.gap_slow if slow[s] else model.gap_moving
        e = min(int(block_end[s]), s + int(rng.choice(pool)))
        out[s:e] = False
        skip_until = e
    return out


def _calibrate_gaps(present, graded, slow, model, block_end, seed, iters=12):
    """Bisection on one multiplier of BAR-ATH's per-regime start rates so that the overall
    visible share on graded frames reaches the target."""
    if present[graded].mean() <= model.target_visible:
        return present
    base_hazard = np.where(slow, model.start_rate_slow, model.start_rate_moving) * graded
    lo, hi = 0.01, 50.0
    trial = present
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        trial = _start_gaps(present, mid * base_hazard, slow, model, block_end,
                            np.random.default_rng(seed))
        if trial[graded].mean() > model.target_visible:
            lo = mid
        else:
            hi = mid
    return trial


def drop_ball(ball: np.ndarray, players: np.ndarray, blocks, pitch, fps: float,
              model: NoiseModel, seed: int) -> np.ndarray:
    out = ball.copy()
    rng = np.random.default_rng(seed)
    graded = np.zeros(len(out), dtype=bool)
    block_end = np.zeros(len(out), dtype=np.int64)
    for _, first, last in blocks:
        graded[first:last + 1] = True
        block_end[first:last + 1] = last + 1
    present = ~np.isnan(out[:, 0]) & graded

    oob = present & ((out[:, 0] < 0) | (out[:, 0] > pitch[0]) | (out[:, 1] < 0) | (out[:, 1] > pitch[1]))
    oob_share = oob[present].mean() if present.any() else 0.0
    if oob_share > model.target_oob_visible:
        _drop_runs(out, oob, 1.0 - model.target_oob_visible / oob_share, rng)
        present = ~np.isnan(out[:, 0]) & graded

    slow = slow_phases(players, fps) & graded
    present = _calibrate_gaps(present, graded, slow, model, block_end, seed + 1)
    out[~present] = np.nan
    return out


# --------------------------------------------------------------------------------------------
# Player tracks -> anonymous slots
# --------------------------------------------------------------------------------------------

def _fragments(players, blocks, model, absence_scale, rng):
    """[(player, start, end)] after cutting every track into lifetimes separated by absences."""
    segs = []
    for p in range(players.shape[1]):
        present = ~np.isnan(players[:, p, 0])
        for _, first, last in blocks:
            t = first
            while t <= last:
                if not present[t]:
                    t += 1
                    continue
                length = int(rng.choice(model.lifetimes))
                end = t
                while end <= last and present[end] and end - t < length:
                    end += 1
                segs.append((p, t, end))
                t = end + int(round(absence_scale * rng.choice(model.absences)))
    return segs


def _object_count(items, n_frames):
    count = np.zeros(n_frames, dtype=np.int32)
    for it in items:
        count[it[1]:it[2]] += 1
    return count


def fragment_players(players: np.ndarray, team_of_player: np.ndarray, blocks,
                     model: NoiseModel, seed: int):
    """(players_slots (N, S, 3), team_of_slot (S,)) with breaks, absences and ghosts."""
    n = players.shape[0]
    graded = np.zeros(n, dtype=bool)
    block_end = np.zeros(n, dtype=np.int64)
    for _, first, last in blocks:
        graded[first:last + 1] = True
        block_end[first:last + 1] = last + 1
    clean_mean = float((~np.isnan(players[graded, :, 0])).sum(axis=1).mean())
    # A red card leaves fewer than 22 in the source; scale the target proportionally.
    target_mean = model.target_objects_mean * min(1.0, clean_mean / 22.0)

    lo, hi = 0.0, 4.0
    segs = []
    for _ in range(8):
        mid = (lo + hi) / 2
        segs = _fragments(players, blocks, model, mid, np.random.default_rng(seed))
        realised = float(_object_count(segs, n)[graded].mean())
        if realised > target_mean:
            lo = mid
        else:
            hi = mid
    count = _object_count(segs, n)

    # Ghosts: duplicates appear when the tracker already sees everyone, so they start on frames
    # that are full; bisection on their number so the share of frames with > 22 objects matches.
    full_frames = np.flatnonzero(graded & (count >= 22))
    if not len(full_frames):
        full_frames = np.flatnonzero(graded)

    def make_ghosts(k):
        g_rng = np.random.default_rng(seed + 11)
        out = []
        for start in g_rng.choice(full_frames, size=k):
            start = int(start)
            end = min(int(block_end[start]), start + GHOST_LENGTH_FRAMES)
            cand = np.flatnonzero(~np.isnan(players[start, :, 0]))
            if len(cand):
                out.append((int(g_rng.choice(cand)), start, end,
                            g_rng.normal(0, GHOST_OFFSET_STD_M, 2)))
        return out

    lo_g, hi_g = 0, int(model.target_over_22 * graded.sum() / GHOST_LENGTH_FRAMES * 3) + 1
    ghosts = []
    for _ in range(8):
        k = (lo_g + hi_g) // 2
        ghosts = make_ghosts(k)
        over = float(((count + _object_count(ghosts, n))[graded] > 22).mean())
        if over < model.target_over_22:
            lo_g = k + 1
        else:
            hi_g = k

    rng = np.random.default_rng(seed + 7)
    slot_team = np.array([t for t in (HOME, AWAY)
                          for _ in range({HOME: model.n_slots_home, AWAY: model.n_slots_away}[t])],
                         dtype=np.int8)
    free_at = np.zeros(len(slot_team), dtype=np.int64)
    out = np.full((n, len(slot_team), 3), np.nan)
    items = [(s, e, p, None) for p, s, e in segs] + [(s, e, p, off) for p, s, e, off in ghosts]
    items.sort(key=lambda it: it[0])
    for start, end, p, offset in items:
        cands = np.flatnonzero((slot_team == int(team_of_player[p])) & (free_at <= start))
        if not len(cands):
            continue  # every slot of this team busy: the tracker would miss this fragment
        s = int(rng.choice(cands))
        out[start:end, s] = players[start:end, p]
        if offset is not None:
            out[start:end, s, 0] += offset[0]
            out[start:end, s, 1] += offset[1]
        free_at[s] = end + int(rng.integers(0, 25))
    return out, slot_team


# --------------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------------

def degrade(tracking: Tracking, model: NoiseModel, seed: int) -> Tracking:
    ball = tracking.ball.copy()
    players = tracking.players.copy()

    ball[:, :2] = moving_average_runs(ball[:, :2], ~np.isnan(ball[:, 0]), model.ball_smooth_window)
    for p in range(players.shape[1]):
        pres = ~np.isnan(players[:, p, 0])
        players[:, p, :2] = moving_average_runs(players[:, p, :2], pres, model.player_smooth_window)

    ball = drop_ball(ball, players, tracking.blocks, tracking.pitch, tracking.fps, model, seed)
    ball[:, 2] = ball_speed_from_positions(ball, tracking.fps)
    slots, team_of_slot = fragment_players(players, tracking.team_of_slot, tracking.blocks,
                                           model, seed)
    return Tracking(
        ball=ball, players=slots, team_of_slot=team_of_slot, fps=tracking.fps,
        pitch=tracking.pitch, game=f"{tracking.game}_s{seed}", source=tracking.source,
        labels=None if tracking.labels is None else tracking.labels.copy(),
        blocks=tracking.blocks.copy(),
        meta={**tracking.meta, "degraded_from": tracking.game, "seed": seed},
    )


def fit(model: NoiseModel, reference: Tracking) -> NoiseModel:
    """Only the smoothing windows need fitting; everything else self-calibrates per match."""
    model.ball_smooth_window = fit_smoothing_window(reference.ball[:, :2], reference.ball_present,
                                                    model.target_ball_jitter)
    windows = []
    for p in range(min(6, reference.n_slots)):
        pres = reference.player_present[:, p]
        if pres.sum() > 1000:
            windows.append(fit_smoothing_window(reference.players[:, p, :2], pres,
                                                model.target_player_jitter))
    model.player_smooth_window = int(np.median(windows)) if windows else 1
    model.meta["fit_reference"] = reference.game
    return model
