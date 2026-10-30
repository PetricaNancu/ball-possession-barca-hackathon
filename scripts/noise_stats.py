"""Unlabelled tracking statistics: what the noise simulator must reproduce.

    python scripts/noise_stats.py                 # BAR-ATH vs the clean external matches
    python scripts/noise_stats.py --json corpus/noise_targets.json

Everything here is computed from the *input* (positions, presence) of graded frames -- never
from any label -- so it is legitimate to fit the simulator to it (RULES.md). Reported for BAR-ATH
and, for contrast, for one Sportec and one Metrica match, which are what the simulator starts
from.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from mtah import paths
from mtah.io import Tracking, load_tracking
from mtah.noise import (MOVING_MPS, ball_speed_from_positions, collective_speed, crowding,
                        run_lengths, second_difference_std, slow_phases)

Q = (10, 25, 50, 75, 90, 99)


def quantiles(a):
    return {str(q): float(np.percentile(a, q)) for q in Q} if len(a) else {}


def ball_stats(tr: Tracking) -> dict:
    g = tr.graded_mask()
    present = tr.ball_present & g
    gaps = run_lengths(~tr.ball_present, g)
    runs = run_lengths(tr.ball_present, g)
    speed = ball_speed_from_positions(tr.ball, tr.fps)
    crowd = crowding(tr.ball, tr.players, radius_m=3.0)

    # Hazard of losing the ball on the next frame, by speed and crowding at the last seen frame.
    seen = np.flatnonzero(present[:-1])
    lost_next = ~tr.ball_present[seen + 1]
    s = speed[seen]
    speed_bins = [0, 1, 3, 6, 10, 15, 60]
    hazard_speed = {}
    for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
        m = (s >= lo) & (s < hi) & np.isfinite(s)
        if m.sum() > 200:
            hazard_speed[f"{lo}-{hi}"] = {"p_lost": float(lost_next[m].mean()), "n": int(m.sum())}
    c = crowd[seen]
    hazard_crowd = {}
    for k in (0, 1, 2, 3):
        m = (c == k) if k < 3 else (c >= 3)
        if m.sum() > 200:
            hazard_crowd[str(k) if k < 3 else "3+"] = {"p_lost": float(lost_next[m].mean()),
                                                        "n": int(m.sum())}
    # Gap length conditional on the speed at the last seen frame (slow balls vanish for longer).
    gap_starts = seen[lost_next]
    gap_len_at_start = {}
    for i in gap_starts:
        j = i + 1
        while j < len(tr.ball) and not tr.ball_present[j] and g[j]:
            j += 1
        gap_len_at_start[int(i)] = j - (i + 1)
    slow_start_gaps = np.array([gap_len_at_start[int(i)] for i in gap_starts if speed[i] < 3], dtype=np.int64)
    fast = np.array([gap_len_at_start[int(i)] for i in gap_starts if speed[i] >= 3], dtype=np.int64)
    in_bounds = ((tr.ball[:, 0] >= 0) & (tr.ball[:, 0] <= tr.pitch[0])
                 & (tr.ball[:, 1] >= 0) & (tr.ball[:, 1] <= tr.pitch[1]))

    # Two regimes, label-free: "slow" = collective player speed below MOVING_MPS for a while
    # (stoppages and very slow phases), "moving" = the game is clearly being played.
    slow = slow_phases(tr.players, tr.fps) & g
    moving = g & ~slow
    gap_start = np.zeros(len(tr.ball), dtype=bool)
    gap_start[seen[lost_next] + 1] = True
    gaps_moving = np.array([gap_len_at_start[int(i)] for i in gap_starts if moving[i + 1]],
                           dtype=np.int64)
    gaps_slow = np.array([gap_len_at_start[int(i)] for i in gap_starts if slow[i + 1]],
                         dtype=np.int64)
    regimes = {
        "share_slow_frames": float(slow.sum() / g.sum()),
        "p_missing_given_slow": float((~tr.ball_present)[slow].mean()),
        "p_missing_given_moving": float((~tr.ball_present)[moving].mean()),
        "share_missing_frames_in_slow": float((~tr.ball_present & slow).sum()
                                              / max(1, (~tr.ball_present & g).sum())),
        "n_gaps_starting_moving": int(len(gaps_moving)),
        "n_gaps_starting_slow": int(len(gaps_slow)),
        "gap_frames_moving_quantiles": quantiles(gaps_moving),
        "gap_frames_slow_quantiles": quantiles(gaps_slow),
        "p_gap_start_per_moving_frame": float(len(gaps_moving) / max(1, (tr.ball_present & moving).sum())),
        "p_gap_start_per_slow_frame": float(len(gaps_slow) / max(1, (tr.ball_present & slow).sum())),
    }
    return {
        "visible_graded": float(tr.ball_present[g].mean()),
        "n_gaps": int(len(gaps)), "gap_frames_quantiles": quantiles(gaps),
        "gap_share_over_1s": float(np.sum(gaps[gaps > 25]) / max(1, np.sum(gaps))),
        "n_runs": int(len(runs)), "run_frames_quantiles": quantiles(runs),
        "p_lost_next_frame": float(lost_next.mean()),
        "hazard_by_speed_mps": hazard_speed, "hazard_by_crowding_3m": hazard_crowd,
        "speed_mps_quantiles_when_visible": quantiles(speed[present & np.isfinite(speed)]),
        "out_of_bounds_when_visible": float((~in_bounds)[present].mean()),
        "jitter_second_diff_m": second_difference_std(tr.ball[:, :2], present),
        "regimes": regimes,
        "_arrays": {"ball_gap_lengths": gaps, "ball_gap_lengths_slow_start": slow_start_gaps,
                    "ball_gap_lengths_fast_start": fast, "ball_run_lengths": runs,
                    "ball_gap_lengths_moving": gaps_moving, "ball_gap_lengths_slow": gaps_slow},
    }


def player_stats(tr: Tracking) -> dict:
    g = tr.graded_mask()
    pp = tr.player_present
    n_obj = pp[g].sum(axis=1)
    lifetimes = np.concatenate([run_lengths(pp[:, s], g) for s in range(tr.n_slots)])
    holes = np.concatenate([run_lengths(~pp[:, s], g) for s in range(tr.n_slots)])
    # Holes at the very start/end of a block are absences, not breaks; run_lengths includes them,
    # which is fine for a distribution dominated by the many short ones.
    jit = [second_difference_std(tr.players[:, s, :2], pp[:, s] & g) for s in range(tr.n_slots)]
    return {
        "objects_per_graded_frame_mean": float(n_obj.mean()),
        "objects_per_graded_frame_hist": {str(k): int(v) for k, v in
                                          zip(*np.unique(n_obj, return_counts=True))},
        "share_frames_over_22": float((n_obj > 22).mean()),
        "share_frames_under_20": float((n_obj < 20).mean()),
        "n_slots": tr.n_slots,
        "n_slots_home": int((tr.team_of_slot == 0).sum()),
        "n_slots_away": int((tr.team_of_slot == 1).sum()),
        "slot_lifetime_frames_quantiles": quantiles(lifetimes),
        "slot_lifetime_frames_mean": float(lifetimes.mean()),
        "slot_hole_frames_quantiles": quantiles(holes),
        "breaks_per_graded_minute": float(len(lifetimes) / (g.sum() / tr.fps / 60)),
        "jitter_second_diff_m_median_over_slots": float(np.nanmedian(jit)),
        "_arrays": {"slot_lifetimes": lifetimes, "slot_holes": holes},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the BAR-ATH targets here")
    parser.add_argument("--contrast", nargs="*", default=["J03WPY_sportec", "metrica3_metrica"])
    args = parser.parse_args()

    tracks = {"BAR-ATH": load_tracking()}
    for name in args.contrast:
        p = paths.CACHE_DIR / f"{name}_tracking.npz"
        if p.is_file():
            tracks[name] = Tracking.load(p)

    report = {}
    for name, tr in tracks.items():
        report[name] = {"ball": ball_stats(tr), "players": player_stats(tr)}

    def row(label, fn, fmt="{:.3f}"):
        print(f"{label:<44}" + "".join(f"{fmt.format(fn(report[n])):>18}" for n in tracks))

    print(f"{'':<44}" + "".join(f"{n:>18}" for n in tracks))
    row("ball visible (graded)", lambda r: r["ball"]["visible_graded"])
    row("ball gaps: count", lambda r: r["ball"]["n_gaps"], "{:d}")
    for q in ("50", "90", "99"):
        row(f"ball gap frames q{q}", lambda r, q=q: r["ball"]["gap_frames_quantiles"].get(q, float("nan")), "{:.0f}")
    row("share of missing frames in gaps > 1 s", lambda r: r["ball"]["gap_share_over_1s"])
    for q in ("50", "90"):
        row(f"ball visible-run frames q{q}", lambda r, q=q: r["ball"]["run_frames_quantiles"].get(q, float("nan")), "{:.0f}")
    row("P(ball lost next frame | visible)", lambda r: r["ball"]["p_lost_next_frame"], "{:.4f}")
    for k in ("0-1", "1-3", "3-6", "6-10", "10-15", "15-60"):
        row(f"  hazard | speed {k} m/s", lambda r, k=k: r["ball"]["hazard_by_speed_mps"].get(k, {}).get("p_lost", float("nan")), "{:.4f}")
    for k in ("0", "1", "2", "3+"):
        row(f"  hazard | players within 3 m = {k}", lambda r, k=k: r["ball"]["hazard_by_crowding_3m"].get(k, {}).get("p_lost", float("nan")), "{:.4f}")
    row("ball speed q50 (m/s, from positions)", lambda r: r["ball"]["speed_mps_quantiles_when_visible"].get("50", float("nan")))
    row("ball out of bounds when visible", lambda r: r["ball"]["out_of_bounds_when_visible"])
    row("ball jitter (2nd diff std, m)", lambda r: r["ball"]["jitter_second_diff_m"], "{:.4f}")
    row(f"share of graded frames 'slow' (< {MOVING_MPS} m/s)", lambda r: r["ball"]["regimes"]["share_slow_frames"])
    row("P(ball missing | slow)", lambda r: r["ball"]["regimes"]["p_missing_given_slow"])
    row("P(ball missing | moving)", lambda r: r["ball"]["regimes"]["p_missing_given_moving"])
    row("share of missing frames in slow phases", lambda r: r["ball"]["regimes"]["share_missing_frames_in_slow"])
    row("gaps starting while moving / slow", lambda r: f"{r['ball']['regimes']['n_gaps_starting_moving']} / {r['ball']['regimes']['n_gaps_starting_slow']}", "{}")
    row("gap frames q50 when starting moving", lambda r: r["ball"]["regimes"]["gap_frames_moving_quantiles"].get("50", float("nan")), "{:.0f}")
    row("gap frames q50 when starting slow", lambda r: r["ball"]["regimes"]["gap_frames_slow_quantiles"].get("50", float("nan")), "{:.0f}")
    row("objects / graded frame (mean)", lambda r: r["players"]["objects_per_graded_frame_mean"], "{:.2f}")
    row("frames with > 22 objects", lambda r: r["players"]["share_frames_over_22"])
    row("frames with < 20 objects", lambda r: r["players"]["share_frames_under_20"])
    row("slot lifetime frames q50", lambda r: r["players"]["slot_lifetime_frames_quantiles"].get("50", float("nan")), "{:.0f}")
    row("slot lifetime frames q90", lambda r: r["players"]["slot_lifetime_frames_quantiles"].get("90", float("nan")), "{:.0f}")
    row("track breaks per graded minute", lambda r: r["players"]["breaks_per_graded_minute"], "{:.1f}")
    row("player jitter (2nd diff std, m, median)", lambda r: r["players"]["jitter_second_diff_m_median_over_slots"], "{:.4f}")

    if args.json:
        target = report["BAR-ATH"]
        arrays = {**target["ball"].pop("_arrays"), **target["players"].pop("_arrays")}
        for r in report.values():
            r["ball"].pop("_arrays", None)
            r["players"].pop("_arrays", None)
        with open(args.json, "w") as fh:
            json.dump(target, fh, indent=2)
        npz = args.json.rsplit(".", 1)[0] + ".npz"
        np.savez_compressed(npz, **arrays)
        print(f"\ntargets -> {args.json}, {npz}")


if __name__ == "__main__":
    main()
