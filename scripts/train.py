"""Train the possession engine on the degraded external corpus and evaluate it leave-one-match-out.

    python scripts/train.py                 # LOO CV (10 folds), decoder tuning, final model
    python scripts/train.py --quick         # fewer trees / coarser stride, for iteration
    python scripts/train.py --no-cv         # final model only (uses saved decoder params)
    python scripts/train.py --decoder-only  # re-tune the decoder on saved out-of-fold probabilities

Protocol (docs/04-engine.md): for each external match, train the emission model on the other
nine (degraded, team-swap augmented) and predict the held-out match; tune the decoder penalties
on the pooled out-of-fold probabilities with the official metric; report per match and per source
next to the reference baseline on the same degraded matches. Then fit the final model on all ten
and save it with the decoder parameters. BAR-ATH is never read here.
"""

import argparse
import itertools
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from mtah import paths
from mtah.config import load_config
from mtah.decode import argmax_labels, decode
from mtah.degrade import NoiseModel, degrade
from mtah.evaluate import score, summarize
from mtah.external import corpus_labels
from mtah.features import LABEL_TO_CLASS, compute_features
from mtah.io import Tracking
from mtah.model import DEFAULT_PARAMS, predict_proba, save, train, training_matrix

CV_PATH = paths.ROOT / "corpus" / "cv_results.json"
OOF_PATH = paths.ROOT / "corpus" / "oof_proba.npz"
DECODER_PATH = paths.MODELS_DIR / "decoder.json"
GRID_PEN = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40]
GRID_MIN = [0, 13, 25, 38, 50]


def load_corpus(noise_model, seeds, sandwich):
    """[(game, source, X, y, labels, blocks, names)] for every cached external match x seed."""
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
            parts.append({"game": clean.game, "source": clean.source, "seed": seed, "X": X,
                          "y": y, "labels": labels, "blocks": clean.blocks, "names": names})
            print(f"  {clean.game} seed {seed}: features {X.shape} in {time.time() - t0:.1f}s")
    return parts


def tune_decoder(oof, grid_team, grid_dead, grid_min, top_k=3):
    """Grid search on pooled out-of-fold predictions; min_frames is searched for the top_k
    penalty pairs. Returns (best params, table)."""
    table = []
    for st, sd in itertools.product(grid_team, grid_dead):
        rows = [score(o["labels"], decode(o["proba"], o["blocks"], st, sd, 0)) for o in oof]
        table.append({"switch_team": st, "switch_dead": sd, "min_frames": 0, **summarize(rows)})
    top = sorted(table, key=lambda r: -r["spell_f1"])[:top_k]
    for base in top:
        for mf in grid_min:
            if mf == 0:
                continue
            rows = [score(o["labels"], decode(o["proba"], o["blocks"], base["switch_team"],
                                                 base["switch_dead"], mf)) for o in oof]
            table.append({"switch_team": base["switch_team"], "switch_dead": base["switch_dead"],
                          "min_frames": mf, **summarize(rows)})
    best = max(table, key=lambda r: r["spell_f1"])
    return {k: best[k] for k in ("switch_team", "switch_dead", "min_frames")}, table


def save_oof(oof):
    payload = {}
    for o in oof:
        payload[f"{o['game']}__proba"] = o["proba"].astype(np.float16)
        payload[f"{o['game']}__labels"] = o["labels"]
        payload[f"{o['game']}__blocks"] = o["blocks"]
        payload[f"{o['game']}__source"] = np.array(o["source"])
    np.savez_compressed(OOF_PATH, **payload)


def load_oof():
    z = np.load(OOF_PATH)
    games = sorted({k.split("__")[0] for k in z.files})
    return [{"game": g, "source": str(z[f"{g}__source"]), "proba": z[f"{g}__proba"].astype(np.float32),
             "labels": z[f"{g}__labels"], "blocks": z[f"{g}__blocks"]} for g in games]


def report(oof, decoder, baseline_ref):
    print(f"\n{'match':<10}{'src':<8}{'base spF1':>10}{'base maF1':>10}{'argmax spF1':>12}"
          f"{'argmax maF1':>12}{'decoded spF1':>13}{'decoded maF1':>13}{'f1_D':>7}{'pred':>6}{'gt':>5}")
    results = {}
    for o in oof:
        raw = score(o["labels"], argmax_labels(o["proba"], o["blocks"]))
        dec = score(o["labels"], decode(o["proba"], o["blocks"], **decoder))
        b = baseline_ref.get(o["game"], {})
        results[o["game"]] = {"source": o["source"], "argmax": raw, "decoded": dec,
                              "baseline": {k: b.get(k) for k in ("spell_f1", "macro_f1")}}
        print(f"{o['game']:<10}{o['source']:<8}{b.get('spell_f1', float('nan')):>10.3f}"
              f"{b.get('macro_f1', float('nan')):>10.3f}{raw['spell_f1']:>12.3f}{raw['macro_f1']:>12.3f}"
              f"{dec['spell_f1']:>13.3f}{dec['macro_f1']:>13.3f}{dec['f1_D']:>7.3f}"
              f"{dec['n_pred_spells']:>6}{dec['n_gt_spells']:>5}")
    for src in ("sportec", "metrica"):
        rows = [r for r in results.values() if r["source"] == src]
        if rows:
            print(f"{src + ' mean':<18}{np.mean([r['baseline']['spell_f1'] or np.nan for r in rows]):>10.3f}"
                  f"{np.mean([r['baseline']['macro_f1'] or np.nan for r in rows]):>10.3f}"
                  f"{np.mean([r['argmax']['spell_f1'] for r in rows]):>12.3f}"
                  f"{np.mean([r['argmax']['macro_f1'] for r in rows]):>12.3f}"
                  f"{np.mean([r['decoded']['spell_f1'] for r in rows]):>13.3f}"
                  f"{np.mean([r['decoded']['macro_f1'] for r in rows]):>13.3f}")
    allr = list(results.values())
    print(f"{'corpus mean':<18}{'':>20}{np.mean([r['argmax']['spell_f1'] for r in allr]):>12.3f}"
          f"{np.mean([r['argmax']['macro_f1'] for r in allr]):>12.3f}"
          f"{np.mean([r['decoded']['spell_f1'] for r in allr]):>13.3f}"
          f"{np.mean([r['decoded']['macro_f1'] for r in allr]):>13.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sandwich = int(cfg["corpus"]["sportec_sandwich_max_frames"])
    n_rounds = 120 if args.quick else int(cfg["engine"]["n_rounds"])
    stride = 4 if args.quick else int(cfg["engine"]["train_stride"])

    baseline_ref = {}
    nv = paths.ROOT / "corpus" / "noise_validation.json"
    if nv.is_file():
        baseline_ref = json.loads(nv.read_text()).get("degraded", {})

    if args.decoder_only:
        oof = load_oof()
        decoder, table = tune_decoder(oof, GRID_PEN, GRID_PEN, GRID_MIN)
        print(f"best decoder: {decoder}")
        results = report(oof, decoder, baseline_ref)
        cv = json.loads(CV_PATH.read_text()) if CV_PATH.is_file() else {}
        cv.update({"decoder": decoder, "decoder_grid": table, "results": results})
        CV_PATH.write_text(json.dumps(cv, indent=2))
        DECODER_PATH.write_text(json.dumps(decoder, indent=2))
        print(f"\n-> {CV_PATH}, {DECODER_PATH}")
        return

    noise_model = NoiseModel.load(paths.MODELS_DIR / "noise_model.npz")
    print("building degraded corpus features ...")
    parts = load_corpus(noise_model, args.seeds, sandwich)
    names = parts[0]["names"]
    games = sorted({p["game"] for p in parts})

    decoder = None
    if not args.no_cv:
        oof = []
        for held in games:
            t0 = time.time()
            train_parts = [(p["X"], p["y"], names) for p in parts if p["game"] != held]
            X, y = training_matrix(train_parts, stride=stride, augment=True)
            booster = train(X, y, names, n_rounds=n_rounds)
            for p in parts:
                if p["game"] == held and p["seed"] == args.seeds[0]:
                    proba = predict_proba(booster, p["X"])
                    oof.append({"game": held, "source": p["source"], "proba": proba,
                                "labels": p["labels"], "blocks": p["blocks"]})
            raw = score(oof[-1]["labels"], argmax_labels(oof[-1]["proba"], oof[-1]["blocks"]))
            print(f"fold {held:<10} trained on {len(X):>8} rows in {time.time() - t0:5.0f}s  "
                  f"argmax spellF1 {raw['spell_f1']:.3f} macroF1 {raw['macro_f1']:.3f}", flush=True)
            del X, y, booster
        save_oof(oof)

        print("\ntuning decoder on out-of-fold probabilities ...", flush=True)
        grid_pen = GRID_PEN if not args.quick else [4, 8, 15]
        grid_min = GRID_MIN if not args.quick else [0, 25]
        decoder, table = tune_decoder(oof, grid_pen, grid_pen, grid_min)
        print(f"best decoder: {decoder}")
        results = report(oof, decoder, baseline_ref)

        CV_PATH.parent.mkdir(parents=True, exist_ok=True)
        CV_PATH.write_text(json.dumps({"decoder": decoder, "decoder_grid": table,
                                       "n_rounds": n_rounds, "stride": stride,
                                       "seeds": args.seeds, "results": results}, indent=2))
        DECODER_PATH.write_text(json.dumps(decoder, indent=2))
        print(f"\n-> {CV_PATH}, {DECODER_PATH}")

    print("\ntraining final model on all matches ...")
    t0 = time.time()
    X, y = training_matrix([(p["X"], p["y"], names) for p in parts], stride=stride, augment=True)
    booster = train(X, y, names, n_rounds=n_rounds)
    save(booster, names, paths.MODELS_DIR, n_rounds, DEFAULT_PARAMS)
    print(f"final model: {len(X)} rows, {n_rounds} rounds, {time.time() - t0:.0f}s -> "
          f"{paths.MODELS_DIR / 'engine_lgbm.txt'}")
    imp = sorted(zip(names, booster.feature_importance("gain")), key=lambda t: -t[1])
    total = sum(v for _, v in imp) or 1
    print("top features by gain: " + ", ".join(f"{n} {v / total:.1%}" for n, v in imp[:12]))


if __name__ == "__main__":
    main()
