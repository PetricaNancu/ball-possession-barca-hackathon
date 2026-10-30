"""Choose the DFL harmonisation threshold by matching Metrica's possession-duration distribution.

    python scripts/analyze_conventions.py

Prints, for a sweep of sandwich thresholds, the possession statistics of the harmonised Sportec
matches next to the Metrica matches, and the Kolmogorov-Smirnov distance between the two
duration distributions. The public BAR-ATH window is shown for orientation only and plays no
part in the choice (RULES.md).
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from mtah import paths
from mtah.harmonise import merge_short_sandwiches
from mtah.io import Tracking
from mtah.labels import read_submission, spells

FPS = 25.0


def durations(labels):
    return np.array([(e - s) / FPS for _, s, e in spells(labels)])


def ks_distance(a, b):
    grid = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def describe(name, d, extra=""):
    q = lambda p: np.percentile(d, p)
    print(f"{name:<26} n {len(d):5d}  mean {d.mean():5.1f}  q25/50/75 {q(25):4.1f}/{q(50):4.1f}/{q(75):4.1f}"
          f"  <1s {np.mean(d < 1):5.1%}  <2s {np.mean(d < 2):5.1%}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thresholds", type=int, nargs="+",
                        default=[0, 5, 10, 13, 15, 20, 25, 30, 38, 50, 75])
    args = parser.parse_args()

    tracks = [Tracking.load(p) for p in sorted(paths.CACHE_DIR.glob("*_tracking.npz"))
              if "BAR-ATH" not in p.name]
    metrica = [t for t in tracks if t.source == "metrica"]
    sportec = [t for t in tracks if t.source == "sportec"]

    ref = np.concatenate([durations(t.labels) for t in metrica])
    describe("Metrica x3 (reference)", ref)
    describe("BAR-ATH public (info only)",
             durations(read_submission(paths.GT_DIR / "BAR-ATH_possession_gt_public.csv")))
    print()

    best = None
    for thr in args.thresholds:
        d = np.concatenate([durations(merge_short_sandwiches(t.labels, thr)) for t in sportec])
        ks = ks_distance(d, ref)
        per_match = np.mean([len(spells(merge_short_sandwiches(t.labels, thr))) for t in sportec])
        describe(f"Sportec x7, T={thr:3d} frames", d, f"  poss/match {per_match:5.0f}  KS {ks:.3f}")
        if best is None or ks < best[1]:
            best = (thr, ks)
    print(f"\nsmallest KS distance at T={best[0]} frames ({best[0] / FPS:.2f} s), KS={best[1]:.3f}")


if __name__ == "__main__":
    main()
