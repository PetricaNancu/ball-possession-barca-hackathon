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

```
mtah/                 the package: io (canonical tracking), labels, baseline, config, paths
scripts/              check_data, make_submission, evaluate
configs/default.yaml  team name, pipeline method, parameters
evaluation/           organisers' scorer + validator + public GT (vendored, see NOTICE.md)
starter/              organisers' reference loaders and baseline (vendored)
games.py              organisers' per-match constants (vendored)
cache/                parsed tracking (.npz), gitignored
models/               trained weights, committed once frozen
tests/
```

## Setup

```bash
make setup                                 # .venv with pinned deps (Python 3.12)
export MTAH_DATA_DIR=/path/to/data         # folder containing tactical/ and broadcast/
make check                                 # verifies alignment, builds the cache, prints stats
```

If `MTAH_DATA_DIR` is unset, `../more-than-a-hack-2027/data` is used when it exists.
The data bundle is gated and is never committed.

## Reproduce the submission

```bash
make submission       # -> evaluation/results/<team>.csv  (team name in configs/default.yaml)
make validate         # official validator
make evaluate         # public 5-minute smoke test (not a tuning set)
make board            # official ranked board over evaluation/results/*.csv
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
