# Third-party notices

Everything under `vendor/mtah2027/` is copied **unmodified** from the organisers' repository
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027),
released under the MIT Licence (`vendor/mtah2027/LICENSE`). The folder keeps the upstream layout so
those scripts run as-is:

- `games.py` — per-match constants, asset paths, graded frame blocks
- `evaluation/possession_metrics.py` — the official metric library (spellF1, macroF1, ...)
- `evaluation/run_evaluation.py` — the official scorer
- `evaluation/validate_submission.py` — the official submission validator
- `evaluation/gt/BAR-ATH_possession_gt_public.csv`, `..._meta.json` — the public 5-minute ground truth
- `evaluation/results/baseline_*.csv` — the four published baselines (derived from the ATD only)
- `starter/baseline.py`, `starter/load_atd.py`, `starter/load_smart_tagging.py`,
  `starter/load_homography.py` — reference loaders and baseline

Everything outside `vendor/` is original work of this repository. `mtah/io.py` and
`mtah/baseline.py` are re-implementations adapted from the starter files above.

The match data itself (`data/`) is **not** covered by that licence and is not in this repository.
