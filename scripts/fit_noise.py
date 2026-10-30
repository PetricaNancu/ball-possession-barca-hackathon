"""Fit the noise simulator to BAR-ATH's label-free statistics and validate it.

    python scripts/fit_noise.py                # fit, save models/noise_model.npz, validate
    python scripts/fit_noise.py --no-fit       # validate the saved model only

Validation = the one assumption of the plan, made testable: the organisers' baseline algorithm,
run on the *degraded* external matches, should score about what it scores on the full BAR-ATH
(spellF1 0.2433 / macroF1 0.6724, published in EVALUATION.md), and the degraded tracking should
reproduce BAR-ATH's visibility, gap, object-count and jitter statistics. The fit uses only
unlabelled BAR-ATH statistics (corpus/noise_targets.*); no label of BAR-ATH is read anywhere.
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

import possession_metrics as pm
from mtah import paths
from mtah.baseline import baseline_labels
from mtah.config import load_config
from mtah.degrade import NoiseModel, degrade, fit
from mtah.external import corpus_labels
from mtah.io import Tracking
from mtah.labels import mask_by_blocks, write_submission
from mtah.noise import run_lengths, second_difference_std, slow_phases

MODEL_PATH = paths.MODELS_DIR / "noise_model.npz"
PUBLISHED_FULL_MATCH_BASELINE = {"spell_f1": 0.2433, "macro_f1": 0.6724}


def tracking_stats(tr: Tracking) -> dict:
    g = tr.graded_mask()
    present = tr.ball_present & g
    gaps = run_lengths(~tr.ball_present, g)
    n_obj = tr.player_present[g].sum(axis=1)
    lifetimes = np.concatenate([run_lengths(tr.player_present[:, s], g) for s in range(tr.n_slots)])
    inb = ((tr.ball[:, 0] >= 0) & (tr.ball[:, 0] <= tr.pitch[0]) & (tr.ball[:, 1] >= 0)
           & (tr.ball[:, 1] <= tr.pitch[1]))
    jit = [second_difference_std(tr.players[:, s, :2], tr.player_present[:, s] & g)
           for s in range(tr.n_slots)]
    slow = slow_phases(tr.players, tr.fps) & g
    return {
        "visible": float(present.sum() / g.sum()),
        "miss_slow": float((~tr.ball_present)[slow].mean()) if slow.any() else float("nan"),
        "miss_moving": float((~tr.ball_present)[g & ~slow].mean()),
        "gap_q50": float(np.percentile(gaps, 50)) if len(gaps) else 0.0,
        "gap_q90": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
        "n_gaps_per_hour": float(len(gaps) / (g.sum() / tr.fps / 3600)),
        "oob_visible": float((~inb)[present].mean()),
        "objects_mean": float(n_obj.mean()), "over_22": float((n_obj > 22).mean()),
        "under_20": float((n_obj < 20).mean()),
        "lifetime_q50": float(np.percentile(lifetimes, 50)),
        "breaks_per_min": float(len(lifetimes) / (g.sum() / tr.fps / 60)),
        "ball_jitter": second_difference_std(tr.ball[:, :2], present),
        "player_jitter": float(np.nanmedian(jit)),
    }


def score_baseline(tr: Tracking, labels_gt: np.ndarray, window: int) -> dict:
    pred = mask_by_blocks(baseline_labels(tr, window=window), tr.blocks)
    tmp_pred = write_submission(paths.ROOT / "corpus" / "tmp" / f"{tr.game}_pred.csv", pred)
    tmp_gt = write_submission(paths.ROOT / "corpus" / "tmp" / f"{tr.game}_gt.csv", labels_gt)
    gt_codes = pm.label_codes(pm.read_label_csv(str(tmp_gt)))
    r = pm.evaluate(gt_codes, pm.label_codes(pm.read_label_csv(str(tmp_pred))))
    return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "recall_D": r["per_class"]["D"]["recall"], "n_pred_spells": r["n_pred_spells"],
            "n_gt_spells": r["n_gt_spells"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--fit-on", default="J03WPY_sportec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sandwich = int(cfg["corpus"]["sportec_sandwich_max_frames"])
    window = int(cfg["baseline"]["window"])

    bar = Tracking.load(paths.CACHE_DIR / "BAR-ATH_tactical_tracking.npz")
    bar_stats = tracking_stats(bar)

    if args.no_fit and MODEL_PATH.is_file():
        model = NoiseModel.load(MODEL_PATH)
    else:
        t0 = time.time()
        model = NoiseModel.from_targets()
        model = fit(model, Tracking.load(paths.CACHE_DIR / f"{args.fit_on}_tracking.npz"))
        model.save(MODEL_PATH)
        print(f"fitted in {time.time() - t0:.0f}s: smoothing ball {model.ball_smooth_window} / "
              f"players {model.player_smooth_window} frames; targets missing|slow "
              f"{model.target_missing_slow:.3f}, missing|moving {model.target_missing_moving:.3f}, "
              f"objects {model.target_objects_mean:.2f}  -> {MODEL_PATH}")

    cols = ["visible", "miss_slow", "miss_moving", "gap_q50", "gap_q90", "n_gaps_per_hour",
            "oob_visible", "objects_mean", "over_22", "under_20", "lifetime_q50", "breaks_per_min",
            "ball_jitter", "player_jitter"]
    fmt = {"visible": "{:.3f}", "miss_slow": "{:.3f}", "miss_moving": "{:.3f}", "gap_q50": "{:.0f}",
           "gap_q90": "{:.0f}", "n_gaps_per_hour": "{:.0f}", "oob_visible": "{:.3f}",
           "objects_mean": "{:.2f}", "over_22": "{:.3f}", "under_20": "{:.3f}", "lifetime_q50": "{:.0f}",
           "breaks_per_min": "{:.1f}", "ball_jitter": "{:.4f}", "player_jitter": "{:.4f}"}
    w = 12
    header = f"{'match':<20}" + "".join(f"{c:>{w}}" for c in cols) + f"{'spellF1':>9}{'macroF1':>9}{'recD':>6}{'pred':>6}{'gt':>5}"
    print("\n" + header)
    print(f"{'BAR-ATH (target)':<20}" + "".join(f"{fmt[c].format(bar_stats[c]):>{w}}" for c in cols)
          + f"{PUBLISHED_FULL_MATCH_BASELINE['spell_f1']:>9.3f}{PUBLISHED_FULL_MATCH_BASELINE['macro_f1']:>9.3f}{'~0.2':>6}{'':>6}{'':>5}")

    results = {}
    for p in sorted(paths.CACHE_DIR.glob("*_tracking.npz")):
        if "BAR-ATH" in p.name:
            continue
        clean = Tracking.load(p)
        d = degrade(clean, model, seed=args.seed)
        st = tracking_stats(d)
        sc = score_baseline(d, corpus_labels(clean, sandwich), window)
        results[clean.game] = {**st, **sc}
        print(f"{d.game:<20}" + "".join(f"{fmt[c].format(st[c]):>{w}}" for c in cols)
              + f"{sc['spell_f1']:>9.3f}{sc['macro_f1']:>9.3f}{sc['recall_D']:>6.2f}{sc['n_pred_spells']:>6}{sc['n_gt_spells']:>5}")

    for src in ("sportec", "metrica"):
        rows = [v for k, v in results.items() if (k.startswith("metrica")) == (src == "metrica")]
        if rows:
            print(f"{src + ' mean':<20}" + "".join(f"{fmt[c].format(np.mean([r[c] for r in rows])):>{w}}" for c in cols)
                  + f"{np.mean([r['spell_f1'] for r in rows]):>9.3f}{np.mean([r['macro_f1'] for r in rows]):>9.3f}"
                  + f"{np.mean([r['recall_D'] for r in rows]):>6.2f}")

    out = paths.ROOT / "corpus" / "noise_validation.json"
    out.write_text(json.dumps({"bar_ath": bar_stats, "published_baseline": PUBLISHED_FULL_MATCH_BASELINE,
                               "model": {"ball_smooth_window": model.ball_smooth_window,
                                         "player_smooth_window": model.player_smooth_window, **model.meta},
                               "degraded": results}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
