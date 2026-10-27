# Third-party notices

The following files are copied unmodified from the organisers' repository
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027),
released under the MIT Licence (see `LICENSE-metrica-sports.txt`):

- `games.py` — per-match constants, asset paths, graded frame blocks
- `evaluation/possession_metrics.py` — the official metric library (spellF1, macroF1, ...)
- `evaluation/run_evaluation.py` — the official scorer
- `evaluation/validate_submission.py` — the official submission validator
- `evaluation/gt/BAR-ATH_possession_gt_public.csv`, `..._meta.json` — the public 5-minute ground truth
- `evaluation/results/baseline_*.csv` — the four published baselines (derived from the ATD only)
- `starter/baseline.py`, `starter/load_atd.py`, `starter/load_smart_tagging.py`,
  `starter/load_homography.py` — reference loaders and baseline

`mtah/io.py` and `mtah/baseline.py` are re-implementations adapted from those starter files.

The match data itself (`data/`) is **not** covered by that licence and is not in this repository.
