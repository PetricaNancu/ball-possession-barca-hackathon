"""Automated benchmarking, validation-split switching, rule ablation, and hyperparameter tuning.

Supports multiple validation strategies and hyperparameter/rule sweeps:
  1. --val-split metrica       Train on Sportec (7 matches), validate on Metrica (3 matches)
  2. --val-split metrica-loo   Train on Sportec (7) + 2 Metrica matches, validate on held-out Metrica (3 folds)
  3. --val-split loo           Full 10-match Leave-One-Match-Out CV (Sportec x7 + Metrica x3)
  4. --val-split custom        Custom validation matches via --val-matches (e.g., --val-matches metrica1 J03WPY)
  5. --val-split public-5min   Fast sweep on the 5-minute BAR-ATH public benchmark using trained model

Usage:
    python scripts/benchmark_tune.py --val-split public-5min --update-submission
    python scripts/benchmark_tune.py --val-split metrica --update-submission
    python scripts/benchmark_tune.py --val-split metrica-loo --seeds 0 1 --update-submission
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

import mtah  # noqa: F401
import possession_metrics as pm
from mtah import paths
from mtah.config import load_config
from mtah.decode import argmax_labels, decode, postprocess_stoppages, smart_tagging_prior
from mtah.degrade import NoiseModel, degrade
from mtah.evaluate import score, summarize
from mtah.external import corpus_labels
from mtah.features import LABEL_TO_CLASS, compute_features
from mtah.io import Tracking, load_tracking
from mtah.labels import mask_ungraded, write_submission
from mtah.model import DEFAULT_PARAMS, load, predict_proba, save, train, training_matrix

BENCHMARK_OUT = paths.ROOT / "corpus" / "benchmark_tuning_report.json"


def load_degraded_corpus(noise_model: NoiseModel, seeds: list[int], sandwich: int) -> list[dict]:
    parts = []
    for p in sorted(paths.CACHE_DIR.glob("*_tracking.npz")):
        if "BAR-ATH" in p.name:
            continue
        clean = Tracking.load(p)
        labels = corpus_labels(clean, sandwich)
        y = np.array([LABEL_TO_CLASS.get(l, -1) for l in labels], dtype=np.int8)
        for seed in seeds:
            t0 = time.time()
            d = degrade(clean, noise_model, seed=seed)
            X, names = compute_features(d)
            parts.append({
                "game": clean.game, "source": clean.source, "seed": seed,
                "X": X, "y": y, "labels": labels, "blocks": clean.blocks, "names": names,
            })
            print(f"  loaded {clean.game:<10} ({clean.source:<7}) seed={seed} shape={X.shape} in {time.time() - t0:.1f}s")
    return parts


def build_weighted_train_matrix(parts: list[dict], held_out_games: set[str],
                                metrica_weight: int, stride: int):
    names = parts[0]["names"]
    train_parts = []
    for p in parts:
        if p["game"] in held_out_games:
            continue
        train_parts.append((p["X"], p["y"], names))
        if p["source"] == "metrica" and metrica_weight > 1:
            for _ in range(metrica_weight - 1):
                train_parts.append((p["X"], p["y"], names))
    return training_matrix(train_parts, stride=stride, augment=True)


def run_validation_split(parts: list[dict], val_split: str, custom_val: list[str],
                         n_rounds: int, stride: int, metrica_weight: int) -> list[dict]:
    names = parts[0]["names"]
    all_games = sorted({p["game"] for p in parts})
    primary_seed = parts[0]["seed"]

    if val_split == "metrica":
        folds = [[g for g in all_games if g.startswith("metrica")]]
    elif val_split == "metrica-loo":
        folds = [[g] for g in all_games if g.startswith("metrica")]
    elif val_split == "loo":
        folds = [[g] for g in all_games]
    elif val_split == "custom":
        if not custom_val:
            raise ValueError("--val-matches is required when --val-split=custom")
        folds = [custom_val]
    else:
        return []

    oof = []
    for held_list in folds:
        held_set = set(held_list)
        t0 = time.time()
        X_tr, y_tr = build_weighted_train_matrix(parts, held_set, metrica_weight, stride)
        booster = train(X_tr, y_tr, names, n_rounds=n_rounds)
        for p in parts:
            if p["game"] in held_set and p["seed"] == primary_seed:
                proba = predict_proba(booster, p["X"])
                oof.append({
                    "game": p["game"], "source": p["source"], "proba": proba,
                    "X": p["X"], "names": names, "labels": p["labels"], "blocks": p["blocks"],
                })
                raw = score(p["labels"], argmax_labels(proba, p["blocks"]))
                print(f"  val {p['game']:<10} trained on {len(X_tr):>8} rows ({time.time() - t0:.0f}s) -> "
                      f"argmax spellF1={raw['spell_f1']:.4f} macroF1={raw['macro_f1']:.4f}")
        del X_tr, y_tr, booster
    return oof


def sweep_on_5min(proba_bar: np.ndarray, X_bar: np.ndarray, names: list[str],
                  bar_tracking: Tracking, gt_codes: np.ndarray,
                  grid_team: list[int], grid_dead: list[int],
                  grid_min: list[int], grid_phase: list[float]) -> tuple[dict, list[dict]]:
    table = []
    # 1. Argmax ablation
    raw_arg = argmax_labels(proba_bar, bar_tracking.blocks)
    r_arg = pm.evaluate(gt_codes, pm.label_codes(raw_arg))
    table.append({
        "variant": "argmax_only", "switch_team": 0, "switch_dead": 0, "min_frames": 0,
        "phase_weight": 0.0, "postprocess": False,
        "spell_f1": float(r_arg["spell_f1_mean"]), "macro_f1": float(r_arg["macro_f1"]),
        "accuracy": float(r_arg["accuracy"]),
        "f1_H": float(r_arg["per_class"]["H"]["f1"]),
        "f1_A": float(r_arg["per_class"]["A"]["f1"]),
        "f1_D": float(r_arg["per_class"]["D"]["f1"]),
        "recall_D": float(r_arg["per_class"]["D"]["recall"]),
        "n_pred_spells": int(r_arg["n_pred_spells"]),
    })

    # 2. Grid search over decoder + Smart Tagging phase weight + stoppage rules
    priors = {pw: smart_tagging_prior(bar_tracking, phase_weight=pw) for pw in grid_phase}
    for st_pen, sd_pen, mf, pw in itertools.product(grid_team, grid_dead, grid_min, grid_phase):
        viterbi_pred = decode(proba_bar, bar_tracking.blocks, st_pen, sd_pen, mf, prior=priors[pw])
        for use_post in (False, True):
            pred = (postprocess_stoppages(viterbi_pred, X_bar, names, bar_tracking.blocks, tracking=bar_tracking)
                    if use_post else viterbi_pred)
            res = pm.evaluate(gt_codes, pm.label_codes(pred))
            table.append({
                "variant": "viterbi+rules" if use_post else "viterbi_only",
                "switch_team": st_pen, "switch_dead": sd_pen, "min_frames": mf,
                "phase_weight": pw, "postprocess": use_post,
                "spell_f1": float(res["spell_f1_mean"]), "macro_f1": float(res["macro_f1"]),
                "accuracy": float(res["accuracy"]),
                "f1_H": float(res["per_class"]["H"]["f1"]),
                "f1_A": float(res["per_class"]["A"]["f1"]),
                "f1_D": float(res["per_class"]["D"]["f1"]),
                "recall_D": float(res["per_class"]["D"]["recall"]),
                "n_pred_spells": int(res["n_pred_spells"]),
            })

    # Rank by spell_f1 desc, then macro_f1 desc, then accuracy desc
    table.sort(key=lambda r: (-r["spell_f1"], -r["macro_f1"], -r["accuracy"]))
    return table[0], table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-split", default="public-5min",
                        choices=["public-5min", "metrica", "metrica-loo", "loo", "custom"],
                        help="validation split strategy")
    parser.add_argument("--val-matches", nargs="+", default=[],
                        help="match IDs for --val-split custom")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="degradation seeds per external match")
    parser.add_argument("--n-rounds", type=int, default=300,
                        help="LightGBM boosting rounds")
    parser.add_argument("--stride", type=int, default=2,
                        help="training frame subsampling stride")
    parser.add_argument("--metrica-weight", type=int, default=2,
                        help="sample weight multiplier for Metrica matches")
    parser.add_argument("--retrain-final", action="store_true",
                        help="retrain final LightGBM model on all external matches")
    parser.add_argument("--update-submission", action="store_true",
                        help="save best decoder params and regenerate team submission CSV")
    parser.add_argument("--config", help="optional YAML config override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sandwich = int(cfg["corpus"]["sportec_sandwich_max_frames"])
    noise_model = NoiseModel.load(paths.MODELS_DIR / "noise_model.npz")

    oof_summary = []
    if args.val_split != "public-5min" or args.retrain_final or not (paths.MODELS_DIR / "engine_lgbm.txt").is_file():
        print(f"=== Loading degraded external corpus (seeds={args.seeds}, sandwich={sandwich}) ===")
        parts = load_degraded_corpus(noise_model, args.seeds, sandwich)
        names = parts[0]["names"]

        if args.val_split != "public-5min":
            print(f"\n=== Running external validation split: {args.val_split} ===")
            oof = run_validation_split(
                parts, args.val_split, args.val_matches,
                args.n_rounds, args.stride, args.metrica_weight
            )
            for o in oof:
                dec = decode(o["proba"], o["blocks"], 16, 15, 45)
                sc = score(o["labels"], dec)
                oof_summary.append({"game": o["game"], "source": o["source"], **sc})
                print(f"  {o['game']:<10} ({o['source']:<7}) decoded spellF1={sc['spell_f1']:.4f} "
                      f"macroF1={sc['macro_f1']:.4f} f1_D={sc['f1_D']:.4f} poss={sc['n_pred_spells']}/{sc['n_gt_spells']}")
            if oof_summary:
                mean_sp = np.mean([r["spell_f1"] for r in oof_summary])
                mean_ma = np.mean([r["macro_f1"] for r in oof_summary])
                print(f"  Validation Mean ({args.val_split}): spellF1={mean_sp:.4f} macroF1={mean_ma:.4f}")

        if args.retrain_final or not (paths.MODELS_DIR / "engine_lgbm.txt").is_file():
            print("\n=== Training final LightGBM model on full external corpus ===")
            t0 = time.time()
            X_tr, y_tr = build_weighted_train_matrix(parts, set(), args.metrica_weight, args.stride)
            booster = train(X_tr, y_tr, names, n_rounds=args.n_rounds)
            save(booster, names, paths.MODELS_DIR, args.n_rounds, DEFAULT_PARAMS)
            print(f"  saved model ({len(X_tr)} rows, {args.n_rounds} rounds, {time.time() - t0:.1f}s) -> {paths.MODELS_DIR / 'engine_lgbm.txt'}")

    print("\n=== Benchmarking & Hyperparameter Sweep on BAR-ATH 5-min Public GT ===")
    booster, model_names = load(paths.MODELS_DIR)
    bar = load_tracking(cfg["game"])
    X_bar, feat_names = compute_features(bar)
    if feat_names != model_names:
        raise RuntimeError(
            f"Feature mismatch ({len(feat_names)} vs {len(model_names)} in saved model). "
            "Run with --retrain-final first."
        )
    proba_bar = predict_proba(booster, X_bar)
    gt_path = str(paths.GT_DIR / "BAR-ATH_possession_gt_public.csv")
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))

    grid_team = [8, 10, 12, 14, 15, 16, 18]
    grid_dead = [10, 12, 15, 20, 25]
    grid_min = [0, 25, 45, 50]
    grid_phase = [0.0, 0.05]

    best, table = sweep_on_5min(
        proba_bar, X_bar, feat_names, bar, gt_codes,
        grid_team, grid_dead, grid_min, grid_phase
    )

    print(f"\nTop 10 Configurations on 5-min Public Benchmark:")
    print(f"{'rank':>4}  {'variant':<14} {'st':>3} {'sd':>3} {'mf':>3} {'pw':>4}  "
          f"{'spellF1':>8} {'macroF1':>8} {'acc':>8} {'f1_D':>7} {'rec_D':>7} {'poss':>5}")
    print("-" * 88)
    for rank, row in enumerate(table[:10], start=1):
        print(f"{rank:>4}  {row['variant']:<14} {row['switch_team']:>3} {row['switch_dead']:>3} "
              f"{row['min_frames']:>3} {row['phase_weight']:>4.2f}  "
              f"{row['spell_f1']:>8.4f} {row['macro_f1']:>8.4f} {row['accuracy']:>8.4f} "
              f"{row['f1_D']:>7.4f} {row['recall_D']:>7.4f} {row['n_pred_spells']:>5}")

    # Also show ablation comparison (argmax vs viterbi_only vs viterbi+rules)
    best_arg = next(r for r in table if r["variant"] == "argmax_only")
    best_vit = next(r for r in table if r["variant"] == "viterbi_only")
    best_rul = next(r for r in table if r["variant"] == "viterbi+rules")
    print("\n=== Ablation Summary (5-min Public Benchmark) ===")
    for label, row in [
        ("1. Emission Argmax only (no Viterbi)", best_arg),
        ("2. Emission + Viterbi (no stoppage rules)", best_vit),
        ("3. Full Engine (Viterbi + Smart Tagging + Rules)", best_rul),
    ]:
        print(f"  {label:<48} spellF1={row['spell_f1']:.4f}  macroF1={row['macro_f1']:.4f}  "
              f"acc={row['accuracy']:.4f}  f1_D={row['f1_D']:.4f}  poss={row['n_pred_spells']}")

    report_payload = {
        "val_split": args.val_split,
        "external_validation": oof_summary,
        "best_5min_config": best,
        "ablation": {
            "argmax_only": best_arg,
            "viterbi_only": best_vit,
            "viterbi_plus_rules": best_rul,
        },
        "top_25_configs": table[:25],
    }
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(json.dumps(report_payload, indent=2))
    print(f"\nSaved benchmark report -> {BENCHMARK_OUT}")

    if args.update_submission:
        decoder_cfg = {
            "switch_team": best_rul["switch_team"],
            "switch_dead": best_rul["switch_dead"],
            "min_frames": best_rul["min_frames"],
        }
        (paths.MODELS_DIR / "decoder.json").write_text(json.dumps(decoder_cfg, indent=2))
        print(f"Updated {paths.MODELS_DIR / 'decoder.json'} -> {decoder_cfg}")

        prior = smart_tagging_prior(bar, phase_weight=best_rul["phase_weight"])
        raw_labels = decode(proba_bar, bar.blocks, prior=prior, **decoder_cfg)
        final_labels = postprocess_stoppages(raw_labels, X_bar, feat_names, bar.blocks, tracking=bar)
        final_labels = mask_ungraded(final_labels, game=cfg["game"])

        team_name = cfg["team"]
        out_path = paths.RESULTS_DIR / f"{team_name}.csv"
        write_submission(out_path, final_labels)
        print(f"Wrote submission -> {out_path}")

        sibling_results = paths.ROOT.parent / "more-than-a-hack-2027" / "evaluation" / "results"
        if sibling_results.is_dir():
            sib_path = sibling_results / f"{team_name}.csv"
            write_submission(sib_path, final_labels)
            print(f"Synced submission -> {sib_path}")


if __name__ == "__main__":
    main()
