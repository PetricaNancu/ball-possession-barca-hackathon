# Step 1 — External training corpus

*Status: done (2026-10-30). Code: `mtah/external.py`, `scripts/build_corpus.py`, `make corpus`.*

## Why this step exists

The rules forbid any human label on the competition match — not for the submission, not for
training, not for validation — and the public 5-minute ground truth may only be used to check
that the pipeline is wired correctly. Supervised learning therefore needs labelled **other
matches**, and model selection needs a held-out set of them scored with the official metric.

The ground truth we are scored on is "corrected tracking + a per-frame *ball dead* flag + a
per-frame *owning team* flag". Two public datasets provide exactly that kind of signal, so no
human labelling by us is involved anywhere.

## Sources

| source | matches | what it provides | licence / citation |
|---|---|---|---|
| **Sportec / DFL open data** | 7 Bundesliga matches (`J03WMX J03WN1 J03WPY J03WOH J03WQQ J03WOY J03WR9`) | 25 fps positions of all players and the ball, per-frame `BallStatus` (alive/dead) and `BallPossession` (team), event data | Bassek, M. et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, doi:10.1038/s41597-025-04505-y. Downloaded through `kloppy 3.19.1` (`sportec.load_open_tracking_data`). |
| **Metrica Sports sample data** | 3 anonymised matches (`metrica1`, `metrica2`, `metrica3`) | 25 fps positions (games 1–2 CSV, game 3 in the same FIFA/EPTS format as the competition ATD) + a synchronised event stream (pass, carry, recovery, ball lost, ball out, set piece, shot, foul) | github.com/metrica-sports/sample-data (open, "acknowledge the source"). Tracking through `kloppy` (`metrica.load_open_data`), events read directly. |

Neither dataset contains any information about FC Barcelona vs Athletic Club.

## How it works

1. **One canonical object for every match.** `mtah.io.Tracking` holds `ball (N,3)`,
   `players (N, slots, 3)` in metres on the match's pitch (origin top-left, the ATD convention),
   `team_of_slot`, `fps`, per-half `blocks` and a per-frame `labels` array in `{H, A, D, X}`.
   BAR-ATH is loaded into the same object from the ATD, so everything downstream is match-agnostic.
2. **Timeline layout.** Each external match is laid out as `[period 1][halftime gap][period 2]`,
   the gap being the source's own frame-id gap, labelled `X` and empty of players — the same shape
   as a tactical recording, so `graded_mask()` and `half_of_frame()` behave identically.
3. **Labels — Sportec.** Direct: `BallStatus == dead → D`, else `H`/`A` from `BallPossession`
   (home = first team in the metadata, as in the EPTS convention used by the ATD).
4. **Labels — Metrica.** Derived by rule from the event stream (`labels_from_metrica_events`),
   following the definitions in CHALLENGE.md:
   - `D` from the start of a period until its first `SET PIECE`; after `BALL OUT`, `FAULT RECEIVED`,
     an offside, a goal or a shot going out, until the next restart (a `SET PIECE`, or a restart
     taken directly as a `PASS`, e.g. a goal kick).
   - While alive, the owner is the team of the last possession event (`SET PIECE`, `PASS`,
     `CARRY`, `RECOVERY`, `SHOT`). `BALL LOST` belongs to the team that had the ball, so a loose
     ball stays with that team until the opponent's `RECOVERY` / `PASS` / `CARRY`. `CHALLENGE`
     and `CARD` never flip possession on their own.
5. **Outputs per match**: `cache/<id>_<source>_tracking.npz`; `corpus/gt/<id>_gt.csv` in the
   submission format so the organisers' scorer can score any prediction on it with `--gt`; a line
   in `corpus/summary.json`. As a reference point, the organisers' `baseline_smooth50`
   (nearest player + ±2 s majority vote) is run on each **clean** match and scored with the
   official metric.
6. **Cost**: ~2 min per Sportec match (kloppy's XML parse; the download is 350–420 MB each and is
   cached in `~/kloppy_cache`, 2.7 GB in total), ~35 s per Metrica match. 370 MB of `.npz` caches.

## Results

### Per match

| match | teams | graded frames | D share | possessions | mean / median (s) | < 1 s | ball visible | baseline spellF1 | macroF1 | pred. poss. | D recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `J03WMX` | 1. FC Köln vs FC Bayern München | 145,967 | 42% | 511 | 6.7 / 3.0 | 86 | 100% | 0.307 | 0.415 | 360 | 0.00 |
| `J03WN1` | VfL Bochum 1848 vs Bayer 04 Leverkusen | 141,561 | 52% | 500 | 5.5 / 3.1 | 76 | 100% | 0.284 | 0.349 | 436 | 0.00 |
| `J03WPY` | Fortuna Düsseldorf vs 1. FC Nürnberg | 146,211 | 44% | 421 | 7.7 / 3.8 | 57 | 100% | 0.317 | 0.395 | 375 | 0.00 |
| `J03WOH` | Fortuna Düsseldorf vs SSV Jahn Regensburg | 137,214 | 48% | 402 | 7.2 / 3.9 | 46 | 100% | 0.301 | 0.385 | 383 | 0.00 |
| `J03WQQ` | Fortuna Düsseldorf vs FC St. Pauli | 142,345 | 41% | 401 | 8.4 / 4.7 | 37 | 100% | 0.361 | 0.416 | 406 | 0.00 |
| `J03WOY` | Fortuna Düsseldorf vs F.C. Hansa Rostock | 142,536 | 46% | 456 | 6.7 / 3.6 | 41 | 100% | 0.271 | 0.376 | 372 | 0.00 |
| `J03WR9` | Fortuna Düsseldorf vs 1. FC Kaiserslautern | 146,810 | 42% | 370 | 9.2 / 4.2 | 41 | 100% | 0.315 | 0.410 | 314 | 0.00 |
| `metrica1` | Home vs Away | 145,006 | 39% | 323 | 10.9 / 6.0 | 20 | 61% | 0.451 | 0.794 | 482 | 0.74 |
| `metrica2` | Home vs Away | 141,156 | 34% | 294 | 12.7 / 7.3 | 13 | 59% | 0.445 | 0.800 | 464 | 0.78 |
| `metrica3` | Team A vs Team B | 143,761 | 29% | 343 | 11.8 / 6.6 | 14 | 68% | 0.403 | 0.801 | 559 | 0.74 |

Totals: Sportec 1,002,644 graded frames / 3,061 possessions, baseline mean spellF1 0.308,
macroF1 0.392. Metrica 429,923 graded frames / 960 possessions, baseline mean spellF1 0.433,
macroF1 0.798. For comparison the same baseline scores **0.2433 / 0.6724 on the full BAR-ATH**
and 0.3256 / 0.6207 on the public 5 minutes.

### Convention comparison (the important result)

| | median possession | possessions < 1 s | sub-second `X→Y→X` sandwiches | ball visible when dead / alive |
|---|---|---|---|---|
| BAR-ATH public GT (17 possessions) | 7.6 s | 6% | 1 | — (59% overall, graded) |
| Sportec ×7 | 3.0–4.7 s | 9–17% | 18–55 per match | 100% / 100% |
| Metrica ×3 | 6.0–7.3 s | 4–6% | 6–10 per match | 5–12% / 87–93% |

- **Metrica's event-derived labels reproduce the ground truth's conventions**: similar median
  possession length and similar rarity of sub-second flips. They are the convention reference.
- **DFL's `BallPossession` flips on single touches**: dozens of sub-second "team Y touched it
  between two Y-free stretches of X" per match. CHALLENGE.md says a touch without control is
  *not* a change of possession, so DFL labels need a harmonisation step before training (step 2).
- **Metrica's corrected tracking already drops the ball when it is dead** (only 5–12% visible)
  and keeps it ~90% of the time during play. That is why the baseline scores D recall 0.74–0.78
  and macroF1 0.80 on Metrica while scoring 0 D recall on Sportec (ball always present → the
  baseline never says `D`). BAR-ATH is different again: 59% visible overall with gaps **during
  play**. The noise simulator (step 3) therefore has two regimes.
- Red cards (21 players for most of `J03WN1`) and substitution overlaps (23 players for a few
  frames) occur. Nothing in the pipeline assumes exactly 22 players.

### Validation

- `tests/test_external.py` checks every cached match: labels only in `{H, A, D}` on graded frames
  and `X` elsewhere, 20–23 players per graded frame and none in the gap, positions within the
  pitch (with slack for the ball leaving it), both teams present, 200–700 possessions.
- `corpus/gt/*.csv` are scored by the unmodified organisers' metric library, so every number
  above is in the competition's own units.

## What this step does not do

- It does not touch a single frame of BAR-ATH.
- It does not yet harmonise DFL conventions or simulate tracking noise; those are steps 2 and 3.
- Provider speed fields are kept but not trusted (units differ: Metrica ATD ball speed is
  implausibly low, Sportec's looks like km/h). Features derive speed from positions.

## Reproduce

```bash
make setup
make corpus          # ~20 min on an M1, 2.7 GB download, cached
make test
```
