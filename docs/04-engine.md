# Step 4 — The possession engine: rule layer, features, emission model, decoder

*Status: first version done (2026-10-30). Code: `mtah/features.py`, `mtah/model.py`,
`mtah/decode.py`, `mtah/evaluate.py`, `scripts/train.py` (`make train`), `engine` method in
`scripts/make_submission.py`. Artefacts: `models/engine_lgbm.txt`, `models/engine_features.json`,
`models/decoder.json`, `corpus/cv_results.json`.*

## Architecture

```
Tracking (BAR-ATH or degraded external match)
   │
   ▼  rule layer + features (mtah/features.py)            ~1 s per match, 80 columns
   │
   ▼  emission model: LightGBM 3-class (mtah/model.py)     P(H), P(A), P(D) per frame
   │
   ▼  decoder: Viterbi per half + min-duration (mtah/decode.py)
   │
   ▼  H / A / D on graded frames, X elsewhere
```

Every stage is identical for the competition match and for the training matches; nothing in it
is specific to BAR-ATH, and nothing in it has seen a BAR-ATH label.

## 1. Rule layer (`mtah/features.py`)

Generic football physics, offline and **bidirectional**: each frame may use what happens after
it, because the file is complete when we label it.

- **Ball cleanup** — 5-frame moving average inside each visible run; gaps ≤ 2 s linearly
  interpolated and flagged (`ball_interp`); speed, acceleration and heading change from central
  differences of positions (provider speed fields are never used).
- **Touches** — acceleration spike (> 12 m/s²) or sharp turn (> 0.6 rad while > 1.5 m/s),
  attributed to the nearest player within 2.5 m; a touch is *contested* when an opponent is also
  within 2.5 m. Gives `last_touch_sgn`, `next_touch_sgn` (the future!), time since / until a
  touch, touch rate.
- **Controlled possession state machine** — the CHALLENGE.md rules in code: a touch by team T
  flips possession to T only if T also makes the next touch, or the ball stays within 2 m of a T
  player for 0.4 s (carry / controlled reception); a lone touch (deflection, failed clearance)
  does not. Reset at each half. Exposed as `control_sgn`, `control_next_sgn` (team of the next
  controlled touch) and time since / until the last change of control.
- **Ball–player geometry** — nearest and second-nearest distance per team, players within 3 m
  and 8 m per team, signed margin (`d_margin_sgn` = d_away − d_home), distance to sideline /
  goal line (negative when out), out-of-bounds flag, stationary duration.
- **Gap context** — frames since the ball was last observed and until it is next observed, the
  team margin and ball speed at those two sightings (so a frame inside a 10-second gap knows who
  had the ball on both sides of it).
- **Collective motion** (works with no ball at all) — number of tracked objects, collective
  speed, fraction of players walking, slow-phase flag, per-team spread, centroid distance, ball
  position relative to the two centroids.
- **Block context** — frames since the start of the half and whether the ball has moved yet (the
  pre-kick-off stretch of a half is dead by construction).
- **Temporal context** — rolling means over ±0.5 s, ±2 s and ±6 s of the nine most informative
  signals.

How informative the rule layer is on its own (team accuracy on alive frames; the public window
is a sanity check only, the degraded match is the training regime):

| signal | BAR-ATH public 5 min | degraded `J03WPY` |
|---|---|---|
| nearest player (= the organisers' baseline) | 0.777 | 0.749 |
| last touch | 0.827 | 0.797 |
| controlled possession (state machine) | **0.868** | **0.801** |
| team of the next controlled touch | 0.847 | 0.801 |
| rolling margin ±2 s | 0.833 | — |

## 2. Emission model (`mtah/model.py`)

LightGBM, 3 classes, 300 rounds, learning rate 0.05, 63 leaves, ≥ 400 samples per leaf, feature
and bagging fraction 0.8, deterministic. Trained on every second graded frame of the degraded
external matches (frames are highly correlated; the stride halves memory at no measurable cost)
**plus the same frames with the two teams swapped** (`swap_teams` / `swap_labels`), which removes
any home/away bias and doubles the data. NaN is a legitimate input value (ball unknown).

## 3. Decoder (`mtah/decode.py`)

Viterbi over {H, A, D} inside each half: staying is free, switching between the teams costs
`switch_team`, entering or leaving a dead ball costs `switch_dead` (log-probability units); then
spells shorter than `min_frames` are merged into the longer neighbour. The three numbers are
**tuned directly on `spellF1`** over the pooled out-of-fold predictions of the external matches —
the ranking metric, not a per-frame proxy. The decoder always commits on graded frames: the
scorer forward-fills `X` before matching possessions, so abstaining there can only cost macroF1.

## 4. Evaluation protocol

Leave-one-match-out over the 10 external matches: train on 9 (degraded, augmented), predict the
10th (degraded with the same noise model), score with the organisers' metric library against the
harmonised labels. The decoder is tuned on the pooled out-of-fold probabilities, so its numbers
are honest for the matches shown. The reference point on each row is the organisers' baseline
run on the *same degraded match* (docs/03).

## 5. Results (leave-one-match-out, degraded corpus, seed 0; 300 rounds, stride 2)

Decoder chosen on the pooled out-of-fold probabilities: `switch_team = 15`, `switch_dead = 40`,
`min_frames = 0` (the landscape is flat above `switch_team ≈ 10`: every configuration in the
top 20 of the 112-point grid is within 0.003 spellF1; `min_frames` only hurts beyond 25 frames).

| held-out match | source | baseline spellF1 / macroF1 | argmax spellF1 / macroF1 | **engine spellF1 / macroF1** | F1(D) | predicted / true possessions |
|---|---|---|---|---|---|---|
| `J03WMX` | Sportec | 0.275 / 0.639 | 0.338 / 0.889 | **0.612 / 0.894** | 0.958 | 245 / 341 |
| `J03WN1` | Sportec | 0.260 / 0.597 | 0.278 / 0.853 | **0.578 / 0.864** | 0.973 | 272 / 358 |
| `J03WOH` | Sportec | 0.268 / 0.627 | 0.340 / 0.885 | **0.603 / 0.883** | 0.964 | 237 / 306 |
| `J03WOY` | Sportec | 0.242 / 0.578 | 0.334 / 0.879 | **0.634 / 0.884** | 0.965 | 255 / 336 |
| `J03WPY` | Sportec | 0.257 / 0.580 | 0.325 / 0.882 | **0.605 / 0.885** | 0.965 | 235 / 307 |
| `J03WQQ` | Sportec | 0.235 / 0.506 | 0.303 / 0.869 | **0.536 / 0.871** | 0.945 | 235 / 315 |
| `J03WR9` | Sportec | 0.275 / 0.614 | 0.307 / 0.881 | **0.593 / 0.882** | 0.954 | 221 / 296 |
| `metrica1` | Metrica | 0.378 / 0.747 | 0.448 / 0.883 | **0.671 / 0.885** | 0.916 | 246 / 323 |
| `metrica2` | Metrica | 0.386 / 0.766 | 0.434 / 0.865 | **0.685 / 0.866** | 0.867 | 237 / 294 |
| `metrica3` | Metrica | 0.282 / 0.720 | 0.359 / 0.873 | **0.629 / 0.872** | 0.881 | 255 / 343 |
| **Sportec mean** | | 0.259 / 0.592 | 0.318 / 0.877 | **0.594 / 0.880** | | |
| **Metrica mean** | | 0.349 / 0.744 | 0.414 / 0.874 | **0.662 / 0.875** | | |
| **corpus mean** | | 0.286 / 0.637 | 0.347 / 0.876 | **0.615 / 0.879** | | |

Reading the table:

- **2.2–2.4× the baseline on the ranking metric** on matches made to look like BAR-ATH, where
  the baseline itself scores what it scores on the real BAR-ATH (0.26 vs 0.24).
- **The decoder is where spellF1 is made.** The per-frame argmax already has macroF1 0.88 but
  spellF1 0.35 — it flickers. Viterbi leaves macroF1 untouched and nearly doubles spellF1. This
  is the disagreement EVALUATION.md describes, observed on our own system.
- **Dead ball is essentially solved** (F1(D) 0.87–0.97 vs 0 for the baseline on Sportec).
- **The remaining error is missed short possessions**: we predict 221–272 possessions where the
  truth has 296–358. The decoder penalties that maximise spellF1 merge some genuine short
  possessions into their neighbours. Lowering the penalties recovers them but creates more
  spurious splits; the trade-off is flat at this level of per-frame evidence, so the next gain
  has to come from sharper features around transitions (receiver prediction, interception
  geometry), not from the decoder.
- Top features by gain: `ball_speed` 19 %, `nearest_team_sgn_rm50` 14 %, `next_seen_speed` 7 %,
  `last_seen_speed` 6 %, `collective_speed_rm150` 5 %, `collective_speed_rm50` 4 %,
  `spread_y_H/A` 4 % each, `next_seen_margin_sgn` 3 %, `last_seen_margin_sgn` 2.5 %,
  `last_touch_sgn` 2.4 %, `control_sgn_rm50` 2.3 %. The bidirectional gap-context features
  (`next_seen_*`) rank third and fourth: knowing where the ball reappears is worth a lot.

### Smoke test on BAR-ATH (public 5 minutes — correctness only, not a tuning set)

| | spellF1 | macroF1 | possessions | mean length |
|---|---|---|---|---|
| engine (`results/our_team.csv`) | 0.525 [0.34, 0.70] | 0.700 | 15 | 19.0 s |
| `baseline_smooth50` | 0.326 [0.21, 0.45] | 0.621 | 26 | 10.8 s |
| ground truth | | | 17 | 14.2 s |

Right alignment, right label semantics, possession count in the right range (15 vs 17; again
slightly too few). The bootstrap intervals are wide because 17 possessions is a tiny sample —
the number is recorded, not optimised. Full-match file: 278 possessions, mean 14.1 s, H 58,945 /
A 38,816 / D 61,096 frames.

## Caveats

- One degraded copy per match; several seeds per match would be the cheapest robustness gain.
- Metrica scores higher than Sportec (its labels follow the GT's conventions exactly and its ball
  vanishes in stoppages); BAR-ATH probably sits between the two sources.
- Smart Tagging is not used yet (external matches have none to learn from); label-free
  adaptation on BAR-ATH is not used yet.
- The metric's own noise band on a full match is ≈ 0.04; per-match spellF1 here ranges 0.54–0.69.

## Reproduce

```bash
make train            # ~30 min on an M1: features, 10 folds, decoder grid, final model
make submission       # results/<team>.csv with method: engine
make validate && make evaluate
```
