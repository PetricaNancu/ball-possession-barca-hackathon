"""Automated external-corpus benchmarking, validation-split switching, rule ablation, and decoder tuning.

All hyperparameter tuning and model/rule selection run strictly on the external degraded corpus
(Sportec x7 + Metrica x3) — never on BAR-ATH or the 5-minute public ground truth (RULES.md).

Supported external validation splits:
  1. --val-split loo           Full 10-match Leave-One-Match-Out CV (Sportec x7 + Metrica x3) [default]
  2. --val-split metrica-loo   Train on Sportec (7) + 2 Metrica matches, validate on held-out Metrica (3 folds)
  3. --val-split metrica       Train on Sportec (7 matches), validate on Metrica (3 matches)
  4. --val-split custom        Custom external validation matches via --val-matches (e.g., --val-matches metrica1 J03WPY)

Usage:
    python scripts/benchmark_tune.py --val-split loo --reuse-oof --update-submission
    python scripts/benchmark_tune.py --val-split metrica-loo --update-submission
    python scripts/benchmark_tune.py --val-split metrica --retrain-final --update-submission
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

import mtah  # noqa: F401
from mtah import paths
from mtah.config import load_config
from mtah.decode import argmax_labels, decode, postprocess_stoppages
from mtah.degrade import NoiseModel, degrade
from mtah.evaluate import score
from mtah.external import corpus_labels
from mtah.features import LABEL_TO_CLASS, compute_features
from mtah.io import Tracking
from mtah.model import DEFAULT_PARAMS, predict_proba, save, train, training_matrix

BENCHMARK_OUT = paths.ROOT / "corpus" / "benchmark_tuning_report.json"
OOF_PATH = paths.ROOT / "corpus" / "oof_proba.npz"


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
        invalid = [g for g in custom_val if "BAR-ATH" in g or g not in all_games]
        if invalid:
            raise ValueError(f"Invalid or disallowed validation matches: {invalid}; valid external matches: {all_games}")
        folds = [custom_val]
    else:
        raise ValueError(f"Unsupported validation split: {val_split}")

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
                      f"argmax spellF1={raw['spell_f1']:.4f} macroF1={raw['macro_f1']:.4f}", flush=True)
        del X_tr, y_tr, booster
    return oof


def load_cached_oof_for_split(parts: list[dict], val_split: str, custom_val: list[str]) -> list[dict]:
    if not OOF_PATH.is_file():
        raise FileNotFoundError(f"No cached OOF file at {OOF_PATH}; run without --reuse-oof first.")
    z = np.load(OOF_PATH)
    primary_seed = parts[0]["seed"]
    by_game = {p["game"]: p for p in parts if p["seed"] == primary_seed}
    cached_games = sorted({k.split("__")[0] for k in z.files})
    if val_split in ("metrica", "metrica-loo"):
        wanted = [g for g in cached_games if g.startswith("metrica")]
    elif val_split == "loo":
        wanted = cached_games
    elif val_split == "custom":
        wanted = [g for g in custom_val if g in cached_games and "BAR-ATH" not in g]
    else:
        raise ValueError(f"Unsupported validation split: {val_split}")

    oof = []
    for g in wanted:
        p = by_game[g]
        oof.append({
            "game": g,
            "source": str(z[f"{g}__source"]),
            "proba": z[f"{g}__proba"].astype(np.float32),
            "X": p["X"],
            "names": p["names"],
            "labels": z[f"{g}__labels"],
            "blocks": z[f"{g}__blocks"],
        })
    return oof


def _agg_external(rows: list[dict], sources: list[str], metrica_weight: float) -> dict:
    weights = np.array([metrica_weight if s == "metrica" else 1.0 for s in sources], dtype=np.float64)
    weights /= weights.sum()
    w_mean = lambda key: float(np.sum([r[key] * w for r, w in zip(rows, weights)]))
    u_mean = lambda key: float(np.mean([r[key] for r in rows]))
    out = {
        "spell_f1": w_mean("spell_f1"),
        "macro_f1": w_mean("macro_f1"),
        "f1_H": w_mean("f1_H"),
        "f1_A": w_mean("f1_A"),
        "f1_D": w_mean("f1_D"),
        "recall_D": w_mean("recall_D"),
        "mean_pred_spells": u_mean("n_pred_spells"),
        "corpus_spell_f1": u_mean("spell_f1"),
        "corpus_macro_f1": u_mean("macro_f1"),
    }
    m_rows = [r for r, s in zip(rows, sources) if s == "metrica"]
    s_rows = [r for r, s in zip(rows, sources) if s == "sportec"]
    if m_rows:
        out["metrica_spell_f1"] = float(np.mean([r["spell_f1"] for r in m_rows]))
        out["metrica_macro_f1"] = float(np.mean([r["macro_f1"] for r in m_rows]))
    if s_rows:
        out["sportec_spell_f1"] = float(np.mean([r["spell_f1"] for r in s_rows]))
        out["sportec_macro_f1"] = float(np.mean([r["macro_f1"] for r in s_rows]))
    return out


def sweep_on_external_oof(oof: list[dict], grid_team: list[int], grid_dead: list[int],
                          grid_min: list[int], metrica_weight: float, top_k: int = 4) -> tuple[dict, list[dict]]:
    sources = [o["source"] for o in oof]
    table = []

    # 1. Argmax ablation on held-out external matches
    arg_rows = [score(o["labels"], argmax_labels(o["proba"], o["blocks"])) for o in oof]
    table.append({
        "variant": "argmax_only", "switch_team": 0, "switch_dead": 0, "min_frames": 0,
        "postprocess": False,
        **_agg_external(arg_rows, sources, metrica_weight),
    })

    # 2. Stage 1: sweep (switch_team, switch_dead) at min_frames=0 for Viterbi-only and Viterbi+rules
    stage1 = []
    for st_pen, sd_pen in itertools.product(grid_team, grid_dead):
        vit_preds = [decode(o["proba"], o["blocks"], st_pen, sd_pen, 0) for o in oof]
        rows_vit = [score(o["labels"], p) for o, p in zip(oof, vit_preds)]
        rec_vit = {
            "variant": "viterbi_only", "switch_team": st_pen, "switch_dead": sd_pen,
            "min_frames": 0, "postprocess": False,
            **_agg_external(rows_vit, sources, metrica_weight),
        }
        stage1.append(rec_vit)

        rows_rul = [
            score(o["labels"], postprocess_stoppages(p, o["X"], o["names"], o["blocks"]))
            for o, p in zip(oof, vit_preds)
        ]
        rec_rul = {
            "variant": "viterbi+rules", "switch_team": st_pen, "switch_dead": sd_pen,
            "min_frames": 0, "postprocess": True,
            **_agg_external(rows_rul, sources, metrica_weight),
        }
        stage1.append(rec_rul)

    table.extend(stage1)

    # 3. Stage 2: sweep min_frames > 0 on top_k candidates of each variant
    top_candidates = []
    for variant in ("viterbi_only", "viterbi+rules"):
        sub = sorted([r for r in stage1 if r["variant"] == variant],
                     key=lambda r: (-r["spell_f1"], -r["macro_f1"]))
        top_candidates.extend(sub[:top_k])

    for base in top_candidates:
        for mf in grid_min:
            if mf == 0:
                continue
            vit_preds = [decode(o["proba"], o["blocks"], base["switch_team"], base["switch_dead"], mf) for o in oof]
            if base["postprocess"]:
                preds = [postprocess_stoppages(p, o["X"], o["names"], o["blocks"]) for o, p in zip(oof, vit_preds)]
            else:
                preds = vit_preds
            rows = [score(o["labels"], p) for o, p in zip(oof, preds)]
            table.append({
                "variant": base["variant"], "switch_team": base["switch_team"],
                "switch_dead": base["switch_dead"], "min_frames": mf,
                "postprocess": base["postprocess"],
                **_agg_external(rows, sources, metrica_weight),
            })

    table.sort(key=lambda r: (-r["spell_f1"], -r["macro_f1"]))
    return table[0], table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-split", default="loo",
                        choices=["loo", "metrica-loo", "metrica", "custom"],
                        help="external validation split strategy (never BAR-ATH)")
    parser.add_argument("--val-matches", nargs="+", default=[],
                        help="external match IDs for --val-split custom")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="degradation seeds per external match")
    parser.add_argument("--n-rounds", type=int, default=300,
                        help="LightGBM boosting rounds")
    parser.add_argument("--stride", type=int, default=2,
                        help="training frame subsampling stride")
    parser.add_argument("--metrica-weight", type=int, default=2,
                        help="provider weight multiplier for Metrica matches")
    parser.add_argument("--reuse-oof", action="store_true",
                        help="reuse cached corpus/oof_proba.npz instead of retraining CV folds")
    parser.add_argument("--retrain-final", action="store_true",
                        help="retrain final LightGBM model on all external matches")
    parser.add_argument("--update-submission", action="store_true",
                        help="save best external-CV decoder params and regenerate submission CSV")
    parser.add_argument("--config", help="optional YAML config override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sandwich = int(cfg["corpus"]["sportec_sandwich_max_frames"])
    noise_model = NoiseModel.load(paths.MODELS_DIR / "noise_model.npz")

    print(f"=== Loading degraded external corpus (seeds={args.seeds}, sandwich={sandwich}) ===")
    parts = load_degraded_corpus(noise_model, args.seeds, sandwich)
    names = parts[0]["names"]

    if args.reuse_oof:
        print(f"\n=== Loading cached out-of-fold predictions for external split: {args.val_split} ===")
        oof = load_cached_oof_for_split(parts, args.val_split, args.val_matches)
    else:
        print(f"\n=== Running external validation split: {args.val_split} ===")
        oof = run_validation_split(
            parts, args.val_split, args.val_matches,
            args.n_rounds, args.stride, args.metrica_weight
        )
        if args.val_split == "loo":
            payload = {}
            for o in oof:
                payload[f"{o['game']}__proba"] = o["proba"].astype(np.float16)
                payload[f"{o['game']}__labels"] = o["labels"]
                payload[f"{o['game']}__blocks"] = o["blocks"]
                payload[f"{o['game']}__source"] = np.array(o["source"])
            np.savez_compressed(OOF_PATH, **payload)
            print(f"  cached 10-fold external OOF probabilities -> {OOF_PATH}")

    print(f"\n=== Hyperparameter & Rule Ablation Sweep on External Split ({args.val_split}, n={len(oof)} matches) ===")
    grid_team = [4, 6, 7, 8, 10, 12, 15, 18, 20, 25]
    grid_dead = [10, 15, 20, 25, 30, 35, 40, 50]
    grid_min = [0, 13, 20, 25, 35, 45]

    best, table = sweep_on_external_oof(
        oof, grid_team, grid_dead, grid_min, metrica_weight=float(args.metrica_weight)
    )

    print(f"\nTop 10 Configurations on External Validation ({args.val_split}):")
    print(f"{'rank':>4}  {'variant':<14} {'st':>3} {'sd':>3} {'mf':>3}  "
          f"{'w_spF1':>8} {'w_maF1':>8} {'met_spF1':>9} {'spo_spF1':>9} {'f1_D':>7} {'poss':>6}")
    print("-" * 92)
    for rank, row in enumerate(table[:10], start=1):
        m_sp = row.get("metrica_spell_f1", float("nan"))
        s_sp = row.get("sportec_spell_f1", float("nan"))
        print(f"{rank:>4}  {row['variant']:<14} {row['switch_team']:>3} {row['switch_dead']:>3} "
              f"{row['min_frames']:>3}  "
              f"{row['spell_f1']:>8.4f} {row['macro_f1']:>8.4f} {m_sp:>9.4f} {s_sp:>9.4f} "
              f"{row['f1_D']:>7.4f} {row['mean_pred_spells']:>6.1f}")

    best_arg = next(r for r in table if r["variant"] == "argmax_only")
    best_vit = next(r for r in table if r["variant"] == "viterbi_only")
    best_rul = next(r for r in table if r["variant"] == "viterbi+rules")
    print(f"\n=== External Validation Ablation Summary ({args.val_split}) ===")
    for label, row in [
        ("1. Emission Argmax only (no Viterbi)", best_arg),
        ("2. Emission + Viterbi (no stoppage rules)", best_vit),
        ("3. Full Engine (Viterbi + Stoppage Rules)", best_rul),
    ]:
        print(f"  {label:<46} w_spellF1={row['spell_f1']:.4f}  w_macroF1={row['macro_f1']:.4f}  "
              f"f1_D={row['f1_D']:.4f}  mean_poss={row['mean_pred_spells']:.1f}")

    oof_per_match = []
    for o in oof:
        vit = decode(o["proba"], o["blocks"], best["switch_team"], best["switch_dead"], best["min_frames"])
        pred = postprocess_stoppages(vit, o["X"], o["names"], o["blocks"]) if best["postprocess"] else vit
        sc = score(o["labels"], pred)
        oof_per_match.append({"game": o["game"], "source": o["source"], **sc})
        print(f"  {o['game']:<10} ({o['source']:<7}) spellF1={sc['spell_f1']:.4f} "
              f"macroF1={sc['macro_f1']:.4f} f1_D={sc['f1_D']:.4f} poss={sc['n_pred_spells']}/{sc['n_gt_spells']}")

    if args.retrain_final or not (paths.MODELS_DIR / "engine_lgbm.txt").is_file():
        print("\n=== Training final LightGBM model on full external corpus ===")
        t0 = time.time()
        X_tr, y_tr = build_weighted_train_matrix(parts, set(), args.metrica_weight, args.stride)
        booster = train(X_tr, y_tr, names, n_rounds=args.n_rounds)
        save(booster, names, paths.MODELS_DIR, args.n_rounds, DEFAULT_PARAMS)
        print(f"  saved model ({len(X_tr)} rows, {args.n_rounds} rounds, {time.time() - t0:.1f}s) -> {paths.MODELS_DIR / 'engine_lgbm.txt'}")

    report_payload = {
        "val_split": args.val_split,
        "metrica_weight": args.metrica_weight,
        "best_external_config": best,
        "external_per_match": oof_per_match,
        "ablation": {
            "argmax_only": best_arg,
            "viterbi_only": best_vit,
            "viterbi_plus_rules": best_rul,
        },
        "top_25_configs": table[:25],
    }
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(json.dumps(report_payload, indent=2))
    print(f"\nSaved external benchmark report -> {BENCHMARK_OUT}")

    if args.update_submission:
        decoder_cfg = {
            "switch_team": int(best["switch_team"]),
            "switch_dead": int(best["switch_dead"]),
            "min_frames": int(best["min_frames"]),
            "postprocess": bool(best["postprocess"]),
        }
        (paths.MODELS_DIR / "decoder.json").write_text(json.dumps(decoder_cfg, indent=2))
        print(f"Updated {paths.MODELS_DIR / 'decoder.json'} (from external {args.val_split}) -> {decoder_cfg}")

        sub_cmd = [sys.executable, str(paths.ROOT / "scripts" / "make_submission.py")]
        if args.config:
            sub_cmd.extend(["--config", args.config])
        subprocess.run(sub_cmd, cwd=str(paths.ROOT), check=True)


if __name__ == "__main__":
    main()
