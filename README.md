# mtah-possession — The Digital Farmhouse

Frame-level ball possession for **More than a Hack 2027 — Phase 1** (FC Barcelona vs Athletic
Club) by **The Digital Farmhouse**. For every frame of the tactical video (`0..199318`) the system
writes `H` (Barcelona in possession), `A` (Athletic), `D` (dead ball) or `X` (ungraded), from the
provided automatic tracking and Smart Tagging. Ranked on `spellF1`.

- **Submission CSV**: `results/The Digital Farmhouse.csv`
- **Technical Document**: `results/The Digital Farmhouse.md` (and detailed step-by-step reports in `docs/`)

The approach, in one paragraph: a possession engine trained on **other matches** (open Sportec/DFL
and Metrica sample data, degraded to look like automatic tracking), a bidirectional physics/rule
layer (touches, controlled-possession state machine, one-sided past/future rolling windows,
out-of-bounds latch, collective motion speeds), LightGBM emission probabilities, and offline
Viterbi decoding with Smart Tagging priors — with all model and decoder hyperparameters tuned
exclusively via leave-one-match-out cross-validation on the external corpus. No hand labels and no
human-produced data about the competition match are used anywhere, and the 5-minute public ground
truth is used strictly as a post-hoc correctness smoke test. Full technical documentation:
`results/The Digital Farmhouse.md` and `docs/`.

## Layout

Everything needed to train, validate, and reproduce the submission lives in this repository (nothing is required outside it except the gated match data bundle):

```
mtah/                 the package: io (canonical tracking), external (other-match corpora), harmonise,
                      noise + degrade (simulator), features (rule layer), model (LightGBM), decode (Viterbi),
                      evaluate, labels, baseline, config, paths
scripts/              check_data, build_corpus, analyze_conventions, noise_stats, fit_noise, train,
                      benchmark_tune, make_submission, evaluate
configs/default.yaml  team name, pipeline method, model & decoder parameters
data/                 gated competition match bundle (tactical/ & broadcast/), strictly gitignored
results/              Technical Document (The Digital Farmhouse.md) + generated submission CSV (gitignored)
cache/                parsed tracking (.npz), gitignored
corpus/               external-match ground truths, noise targets, CV results; gitignored except noise_targets.json
models/               committed trained model weights & parameters (see below)
docs/                 one file per step: method, metrics, caveats
plan/                 architecture explainers & design notes
tests/                unit & reference-baseline regression tests
vendor/mtah2027/      unmodified copies from the organisers' repo (MIT, see NOTICE.md):
                      games.py, evaluation/ (scorer, validator, metrics, public GT, baselines), starter/
```

## Trained Model & Weights (`models/`)

All trained weights and calibrated parameters are committed directly in `models/` (and `corpus/noise_targets.json`) so `make submission` reproduces the exact submission CSV on CPU without retraining:

| File | Role |
|---|---|
| `models/engine_lgbm.txt` | Trained 3-class LightGBM emission model weights (`300` boosting rounds, `111` features, trained on degraded Sportec ×7 + Metrica ×3 with H/A swap augmentation) |
| `models/engine_features.json` | Ordered `111`-feature schema and LightGBM training hyperparameters |
| `models/decoder.json` | External-CV tuned Viterbi decoder parameters (`switch_team=8`, `switch_dead=50`, `min_frames=20`, `postprocess=false`) |
| `models/noise_model.npz` | Fitted noise-simulator distributions & hazards (calibrated to label-free `BAR-ATH` ATD statistics) |
| `corpus/noise_targets.json` | Label-free `BAR-ATH` input summary targets consumed by `scripts/fit_noise.py` |

## Setup

Unpack the competition data bundle into `./data/` (so `./data/tactical/` and `./data/broadcast/` exist; `data/` is gitignored), or point `MTAH_DATA_DIR` at an existing data folder:

```bash
make setup                                 # .venv with pinned deps (Python 3.12)
# Either unpack tactical/ and broadcast/ into ./data/ (auto-detected), or set:
export MTAH_DATA_DIR=/path/to/data         # optional override if data lives elsewhere
make check                                 # verifies alignment, builds the cache, prints stats
```

The data bundle is gated and is never committed.

## External training corpus (other matches only)

```bash
make corpus           # downloads the 7 Sportec/DFL open matches + 3 Metrica sample matches via kloppy
                      # (~2 min each), writes cache/*.npz, corpus/gt/*.csv, corpus/summary.json
make noise            # label-free BAR-ATH statistics -> fitted noise simulator -> validation
make train            # leave-one-match-out CV on the degraded corpus, decoder tuning, final model
```

Sportec/DFL open data: Bassek, M. et al. (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, Scientific Data, https://doi.org/10.1038/s41597-025-04505-y.
Per-frame `BallStatus` and `BallPossession` map directly to the challenge's `D` / `H` / `A`.

## Reproduce the submission

```bash
make submission       # -> results/<team>.csv  (team name in configs/default.yaml)
make validate         # official validator
make evaluate         # public 5-minute smoke test, next to the published baseline (not a tuning set)
make board            # official ranked board over results/*.csv
make test
```

`make baseline` reproduces the organisers' `baseline_smooth50` through our loader; the test
suite asserts it is identical, which is how the loader is verified.

## Rules compliance

- Labels are produced by code only. No frame of the competition match is hand-labelled, corrected
  or used for training or tuning — including the public 5 minutes.
- All supervised training data comes from other matches and is listed, with source and licence,
  in the technical document.
- The repository regenerates the submitted CSV byte-for-byte from the data bundle.

## Attribution

Scorer, validator, metric library, public ground truth, baselines and reference loaders are
copied from [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(MIT). See `NOTICE.md`.
