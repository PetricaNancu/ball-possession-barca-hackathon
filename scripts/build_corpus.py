"""Download the external training corpus and convert it to the canonical format.

    python scripts/build_corpus.py                      # all Sportec + Metrica sample matches
    python scripts/build_corpus.py --matches J03WPY metrica3
    python scripts/build_corpus.py --force              # re-convert even if cached

For every match: cache/<id>_<source>_tracking.npz (Tracking + labels), corpus/gt/<id>_gt.csv
(frame,label for the official scorer), and -- as a reference point -- the organisers' baseline
algorithm run on the *clean* match and scored with the official metric. Downloads go through
kloppy and are cached in ~/kloppy_cache (override with KLOPPY_CACHE_DIR).

No frame of BAR-ATH is involved here; this is other matches only (RULES.md).
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
from mtah.external import (CORPUS_GT_DIR, METRICA_MATCHES, SPORTEC_MATCHES, corpus_labels,
                           load_external, write_corpus_gt)
from mtah.labels import mask_by_blocks, spells, write_submission

SUMMARY_PATH = paths.ROOT / "corpus" / "summary.json"


def summarize(tr, baseline_window: int, sandwich_max_frames: int) -> dict:
    graded = tr.graded_mask()
    labels = corpus_labels(tr, sandwich_max_frames)
    counts = {k: int(((labels == k) & graded).sum()) for k in ("H", "A", "D")}
    runs = spells(labels)
    durations = np.array([e - s for _, s, e in runs]) / tr.fps
    n_raw = len(spells(tr.labels))

    pred = mask_by_blocks(baseline_labels(tr, window=baseline_window), tr.blocks)
    pred_path = write_submission(paths.ROOT / "corpus" / "baseline" / f"{tr.game}_smooth{baseline_window}.csv", pred)
    gt_codes = pm.label_codes(pm.read_label_csv(str(CORPUS_GT_DIR / f"{tr.game}_gt.csv")))
    result = pm.evaluate(gt_codes, pm.label_codes(pm.read_label_csv(str(pred_path))))

    return {
        "game": tr.game, "source": tr.source, "home": tr.meta.get("home_name"),
        "away": tr.meta.get("away_name"), "n_frames": tr.n_frames, "n_graded": int(graded.sum()),
        "fps": tr.fps, "pitch": list(tr.pitch), "n_slots": tr.n_slots,
        "label_counts_graded": counts, "d_share_graded": counts["D"] / max(1, graded.sum()),
        "ball_present_graded": float(tr.ball_present[graded].mean()),
        "players_per_frame_graded": float(tr.player_present[graded].sum(axis=1).mean()),
        "sandwich_max_frames": sandwich_max_frames if tr.source == "sportec" else 0,
        "n_possessions_raw": n_raw,
        "n_possessions": len(runs), "mean_possession_s": float(durations.mean()),
        "median_possession_s": float(np.median(durations)),
        "possessions_under_1s": int((durations < 1.0).sum()),
        "possessions_under_2s": int((durations < 2.0).sum()),
        f"baseline_smooth{baseline_window}": {
            "spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_pred_spells": result["n_pred_spells"],
            "recall_D": result["per_class"]["D"]["recall"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+",
                        default=list(SPORTEC_MATCHES) + list(METRICA_MATCHES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window", type=int, default=50, help="baseline half-width")
    parser.add_argument("--config", help="YAML overriding configs/default.yaml")
    args = parser.parse_args()
    sandwich = int(load_config(args.config)["corpus"]["sportec_sandwich_max_frames"])

    summaries = {}
    if SUMMARY_PATH.is_file():
        summaries = json.loads(SUMMARY_PATH.read_text())

    for match_id in args.matches:
        t0 = time.time()
        tr = load_external(match_id, use_cache=not args.force)
        write_corpus_gt(tr, sandwich_max_frames=sandwich)
        info = summarize(tr, args.window, sandwich)
        summaries[match_id] = info
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summaries, indent=2))
        b = info[f"baseline_smooth{args.window}"]
        print(f"{match_id}  {info['home']} vs {info['away']}  {time.time() - t0:5.0f}s  "
              f"frames {info['n_frames']}  graded {info['n_graded']}  D {info['d_share_graded']:.0%}  "
              f"poss {info['n_possessions']} (raw {info['n_possessions_raw']}, mean "
              f"{info['mean_possession_s']:.1f}s, <2s: {info['possessions_under_2s']})  "
              f"baseline spellF1 {b['spell_f1']:.3f} macroF1 {b['macro_f1']:.3f} "
              f"pred {b['n_pred_spells']} recD {b['recall_D']:.2f}")

    print(f"\nsummary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
