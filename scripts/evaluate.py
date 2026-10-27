"""Score one or more frame,label CSVs against any ground truth with the official metric library.

    python scripts/evaluate.py evaluation/results/our_team.csv                 # public 5 min
    python scripts/evaluate.py pred.csv --gt path/to/external_match_gt.csv
    python scripts/evaluate.py a.csv b.csv --gt ... --bootstrap 1000 --json out.json

For the full ranked board with reports, use the organisers' script unchanged:
    python evaluation/run_evaluation.py --bootstrap 1000
"""

import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401

import mtah  # noqa: F401
import possession_metrics as pm
from mtah import paths

KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "mean_spell_duration_s",
        "gt_mean_spell_duration_s", "spell_bootstrap_low", "spell_bootstrap_high")


def resolve_gt(arg: str) -> str:
    if arg == "public":
        hits = glob.glob(str(paths.ROOT / "evaluation" / "gt" / "*_possession_gt_public.csv"))
        if len(hits) != 1:
            raise SystemExit(f"expected one public GT, found {hits}")
        return hits[0]
    if not os.path.isfile(arg):
        raise SystemExit(f"ground truth not found: {arg}")
    return arg


def score(pred_path: str, gt_codes, bootstrap_n: int) -> dict:
    table = pm.read_label_csv(pred_path)
    reason = pm.validate_submission(table, len(gt_codes))
    if reason:
        return {"submission": os.path.basename(pred_path), "error": reason}
    result = pm.evaluate(gt_codes, pm.label_codes(table), bootstrap_n=bootstrap_n)
    out = {"submission": os.path.basename(pred_path)}
    out.update({k: result.get(k) for k in KEYS})
    out["recall_D"] = result["per_class"]["D"]["recall"]
    out["f1_D"] = result["per_class"]["D"]["f1"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--gt", default="public", help="'public' or a path to a frame,label CSV")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()

    gt_path = resolve_gt(args.gt)
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    rows = [score(p, gt_codes, args.bootstrap) for p in args.submissions]

    print(f"ground truth: {gt_path}")
    print(f"{'submission':<28} {'spellF1':>8} {'macroF1':>8} {'poss':>5} {'gt':>4} "
          f"{'dur':>6} {'gtdur':>6} {'recD':>6}")
    for r in rows:
        if "error" in r:
            print(f"{r['submission']:<28} SKIPPED: {r['error']}")
            continue
        ci = ""
        if r["spell_bootstrap_low"] is not None:
            ci = f"   [{r['spell_bootstrap_low']:.3f}, {r['spell_bootstrap_high']:.3f}]"
        print(f"{r['submission']:<28} {r['spell_f1_mean']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{r['n_pred_spells']:>5} {r['n_gt_spells']:>4} {r['mean_spell_duration_s']:>6.1f} "
              f"{r['gt_mean_spell_duration_s']:>6.1f} {r['recall_D']:>6.3f}{ci}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ground_truth": gt_path, "results": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
