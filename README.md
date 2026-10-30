# mtah-possession

Frame-level ball possession for **More than a Hack 2027 — Phase 1** (FC Barcelona vs Athletic
Club). For every frame of the tactical video the system writes `H` (Barcelona in possession),
`A` (Athletic), `D` (dead ball) or `X` (ungraded), from the provided automatic tracking, Smart
Tagging and homographies. Ranked on `spellF1`.

The approach, in one paragraph: a possession engine trained on **other matches** (open Sportec/DFL
and Metrica sample data, degraded to look like automatic tracking), a physics/rule layer (touches,
ballistic gap filling, dead-ball cues), and offline bidirectional decoding (Viterbi) tuned on the
official metric over held-out external matches. No hand labels and no human-produced data about
the competition match are used anywhere; the public 5-minute ground truth is a correctness smoke
test only. Full plan: `docs/`.

## Layout

Everything outside `vendor/` is ours.

```
mtah/                 the package: io (canonical tracking), external (other-match corpora), labels, baseline, config, paths
scripts/              check_data, build_corpus, make_submission, evaluate
configs/default.yaml  team name, pipeline method, parameters
results/              our submission CSV(s)
cache/                parsed tracking (.npz), gitignored
corpus/               external-match ground truths + baseline reference scores, gitignored (rebuilt by `make corpus`)
models/               trained weights, committed once frozen
tests/
vendor/mtah2027/      unmodified copies from the organisers' repo (MIT, see NOTICE.md):
                      games.py, evaluation/ (scorer, validator, metrics, public GT, baselines), starter/
```

## Setup

```bash
make setup                                 # .venv with pinned deps (Python 3.12)
export MTAH_DATA_DIR=/path/to/data         # folder containing tactical/ and broadcast/
make check                                 # verifies alignment, builds the cache, prints stats
```

If `MTAH_DATA_DIR` is unset, `../more-than-a-hack-2027/data` is used when it exists.
The data bundle is gated and is never committed.

## External training corpus (other matches only)

```bash
make corpus           # downloads the 7 Sportec/DFL open matches via kloppy (~2 min each), writes
                      # cache/<id>_sportec_tracking.npz, corpus/gt/<id>_gt.csv, corpus/summary.json
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
