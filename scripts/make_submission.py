"""Build the submission file from the provided data. One command, deterministic.

    python scripts/make_submission.py                      # configs/default.yaml
    python scripts/make_submission.py --config configs/x.yaml --team my_team
    python scripts/make_submission.py --method baseline --window 50

Writes evaluation/results/<team>.csv, then runs the organisers' validation on it and prints what
the file claims (label counts, possessions, mean duration). Frames outside the two graded halves
are written as X.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

import mtah  # noqa: F401
import possession_metrics as pm
from mtah import paths
from mtah.baseline import baseline_labels
from mtah.config import load_config
from mtah.io import load_tracking
from mtah.labels import mask_ungraded, spells, write_submission


def build_labels(config: dict) -> np.ndarray:
    method = config["pipeline"]["method"]
    tracking = load_tracking(game=config["game"])
    if method == "baseline":
        return baseline_labels(tracking, window=int(config["baseline"]["window"]))
    raise ValueError(f"unknown pipeline method {method!r}")


def describe(labels: np.ndarray, fps: float) -> str:
    counts = {k: int((labels == k).sum()) for k in ("H", "A", "D", "X")}
    runs = spells(labels)
    n_alive = counts["H"] + counts["A"]
    mean_s = n_alive / fps / len(runs) if runs else 0.0
    return (f"labels {counts}\n      {len(runs)} possessions, mean {mean_s:.1f} s "
            f"(counted over non-X frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="YAML overriding configs/default.yaml")
    parser.add_argument("--team", help="submission name (default: config)")
    parser.add_argument("--method", help="pipeline method (default: config)")
    parser.add_argument("--window", type=int, help="baseline smoothing half-width")
    parser.add_argument("--out", help="output path (default: evaluation/results/<team>.csv)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.team:
        config["team"] = args.team
    if args.method:
        config["pipeline"]["method"] = args.method
    if args.window is not None:
        config["baseline"]["window"] = args.window

    labels = mask_ungraded(build_labels(config), game=config["game"])
    out = args.out or (paths.RESULTS_DIR / f"{config['team']}.csv")
    write_submission(out, labels)

    table = pm.read_label_csv(str(out))
    reason = pm.validate_submission(table, len(labels))
    status = "OK  " if not reason else f"FAIL  {reason}"
    print(f"{status}  {out}")
    print(f"      method={config['pipeline']['method']}  "
          f"{describe(labels, fps=25.0)}")
    if reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
