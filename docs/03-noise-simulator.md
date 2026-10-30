# Step 3 — Noise simulator (clean corpus → BAR-ATH-like tracking)

*Status: done (2026-10-30). Code: `mtah/noise.py`, `mtah/degrade.py`, `scripts/noise_stats.py`,
`scripts/fit_noise.py`; artefacts `corpus/noise_targets.{json,npz}`, `models/noise_model.npz`,
`corpus/noise_validation.json`.*

## Why

The external corpus is corrected tracking: the ball is present on 100 % of Sportec frames and
every player is tracked without interruption. BAR-ATH is automatic tracking: the ball is missing
on 41 % of graded frames, 46 anonymous slots cover 22 players with 11.5 track breaks per minute,
and the trajectories are heavily filtered. A model trained on the clean corpus would learn, for
instance, that a missing ball never happens. The simulator closes that gap, and it is the **one
assumption** on which training on other matches rests — so it is fitted only to statistics that
need no label, and it is validated against a number the organisers published.

## What BAR-ATH looks like (label-free measurements, graded frames)

`scripts/noise_stats.py` measures the inputs only — positions and presence — never a label.

| statistic | BAR-ATH | Sportec `J03WPY` (clean) | Metrica 3 (clean) |
|---|---|---|---|
| ball visible | 59.3 % | 100 % | 67.8 % |
| ball gaps (count) / median length / p90 | 150 / 268 fr (10.7 s) / 845 fr | 0 | 160 / 167 / 920 |
| share of missing frames in gaps > 1 s | 99.9 % | — | 99.7 % |
| P(ball lost on next frame \| visible), by ball speed 0–1 / 3–6 / 15+ m/s | 0.0057 / 0.0016 / 0.0001 | 0 | 0.0004 / 0.0017 / 0.0037 |
| P(ball lost on next frame), by players within 3 m: 0 / 3+ | 0.0013 / 0.0041 | 0 | 0.0023 / 0.0055 |
| frames in "slow" phases (collective player speed < 2 m/s for ≥ 2 s) | 66.9 % | 61.2 % | 57.9 % |
| P(ball missing \| slow) / P(ball missing \| moving) | **0.552 / 0.112** | 0 / 0 | 0.527 / 0.040 |
| share of missing frames that fall in slow phases | **90.9 %** | — | 94.8 % |
| ball out of bounds when visible | 4.2 % | 26.8 % | 0.9 % |
| ball jitter (std of 2nd difference, m) | 0.015 | 0.481 | 0.205 |
| tracked objects per frame (mean) / frames > 22 / frames < 20 | 20.49 / 2.4 % / 14.9 % | 22 / 0 / 0 | 22 / 0 / 0 |
| slot lifetime median / p90 (frames) | 888 / 7,116 | whole half | whole match |
| track breaks per graded minute | 11.5 | 0.5 | 0.4 |
| player jitter (median over slots, m) | 0.0022 | 0.0705 | 0.0429 |

Three facts shaped the design:

1. **The ball is not lost in short flickers but in long stretches** (median gap ≈ 11 s), and
   91 % of the missing frames occur while the players are walking. The tracker loses a *slow* ball
   among people — stoppages and slow phases — and almost never a fast one (hazard at 15+ m/s is
   0.0001 per frame).
2. **Metrica's corrected tracking already drops the ball in most stoppages** (visible 5–12 %
   when dead), Sportec never does. The simulator must therefore *top up* each source to the same
   target rather than apply a fixed amount of dropout.
3. **The ATD is far smoother than the corrected sources** (second differences 30× smaller). We do
   not add jitter; we smooth.

## How it works (`mtah/degrade.py`)

All calibrations are bisections on the realised statistic of the match being degraded, so a
single model serves both sources and any match. Everything is seeded; `degrade(tracking, model,
seed)` is deterministic and never touches `labels` or `blocks`.

1. **Smoothing** — centred moving average over each visible run (no bleeding across gaps), window
   fitted to the jitter target but **capped at 15 frames**: matching the ATD's filtering exactly
   needs ~41 frames, which lags fast trajectories and costs the reference baseline ≈ 0.03 spellF1
   (measured in the sweep below). 15 frames reproduces the jitter of visible *play* in BAR-ATH
   (0.034 m in the public window; the 0.015 m full-match figure is dominated by stationary balls).
2. **Out-of-bounds excursions** — whole excursions are dropped until the visible out-of-bounds
   share matches 4.2 %.
3. **Ball dropout by regime** — frames are classified label-free as *slow* or *moving*
   (`noise.slow_phases`). Gaps start with a per-frame hazard proportional to BAR-ATH's measured
   gap-start rate in each regime (slow ≫ moving); gap lengths are drawn from BAR-ATH's empirical
   regime-conditional gap-length distribution (so gaps are long, as in the ATD); one multiplier
   on the hazard is bisected so the overall visible share reaches 59.3 %.
4. **Player fragmentation** — each player's track is cut into fragments whose lengths follow
   BAR-ATH's slot-lifetime distribution, separated by absences drawn from its hole distribution
   (truncated at 30 s) and scaled so the mean tracked-object count reaches 20.49 (scaled down
   when the source itself has fewer than 22 players, e.g. after a red card). Every fragment is
   re-assigned to a free anonymous slot of its team — 24 home, 22 away, as in the ATD — so
   identities disappear exactly as they do in the ATD.
5. **Ghost duplicates** — on frames that already show everyone, a second copy of a present
   player offset by N(0, 0.7 m) for 60 frames; their number is bisected so 2.4 % of frames show
   more than 22 objects.

## Validation

Two tests, both run by `scripts/fit_noise.py` (≈ 1 min for the 10 matches):

**A. Do the degraded matches reproduce BAR-ATH's input statistics?**

| statistic | BAR-ATH target | degraded Sportec (mean of 7) | degraded Metrica (mean of 3) |
|---|---|---|---|
| ball visible | 0.593 | 0.589 | 0.588 |
| P(missing \| slow) / P(missing \| moving) | 0.552 / 0.112 | 0.508 / 0.249 | 0.582 / 0.107 |
| gap median / p90 (frames) | 268 / 845 | 330 / 1,064 | 179 / 923 |
| gaps per hour | 85 | 76 | 108 |
| out of bounds when visible | 0.042 | 0.056 | 0.012 |
| objects per frame / > 22 / < 20 | 20.49 / 0.024 / 0.149 | 20.34 / 0.022 / 0.238 | 20.53 / 0.024 / 0.187 |
| slot lifetime median (frames) / breaks per min | 888 / 11.5 | 783 / 12.9 | 814 / 12.9 |
| ball jitter / player jitter (m) | 0.015 / 0.0022 | 0.041 / 0.003 | 0.026 / 0.008 |

Visibility, object counts, fragmentation and ghost rate are on target. Known residuals: Sportec
keeps too much dropout in moving phases (0.25 vs 0.11) because its stoppages are shorter than the
BAR-ATH gap lengths drawn there, so the overflow lands in play; the jitter is not fully matched
by design (see smoothing); Metrica has too few out-of-bounds frames because its corrected
tracking rarely records them.

**B. Does the organisers' baseline score on the degraded corpus what it scores on BAR-ATH?**
The published full-match numbers are spellF1 **0.2433**, macroF1 **0.6724**
(`baseline_smooth50`). On the clean corpus the same algorithm scored 0.341 / 0.39 (Sportec, no
`D` ever predicted) and 0.43 / 0.80 (Metrica).

| degraded match | spellF1 | macroF1 | D recall |
|---|---|---|---|
| `J03WMX` | 0.275 | 0.639 | 0.58 |
| `J03WN1` | 0.260 | 0.597 | 0.55 |
| `J03WOH` | 0.268 | 0.627 | 0.56 |
| `J03WOY` | 0.242 | 0.578 | 0.51 |
| `J03WPY` | 0.257 | 0.580 | 0.50 |
| `J03WQQ` | 0.235 | 0.506 | 0.41 |
| `J03WR9` | 0.275 | 0.614 | 0.57 |
| `metrica1` | 0.378 | 0.747 | 0.75 |
| `metrica2` | 0.386 | 0.766 | 0.78 |
| `metrica3` | 0.282 | 0.720 | 0.74 |
| **Sportec mean** | **0.259** | **0.592** | 0.53 |
| **Metrica mean** | **0.349** | **0.744** | 0.76 |
| **corpus mean** | **0.286** | **0.637** | — |
| **BAR-ATH, published** | **0.243** | **0.672** | — |

The degraded Sportec matches now score within the metric's own noise band (≈ 0.04) of the
published BAR-ATH spellF1; Metrica remains easier (its ball is absent in stoppages, which the
baseline reads as `D` for free, and its possession labels follow the GT's conventions exactly).
The corpus as a whole brackets BAR-ATH on both metrics. This is the gate the plan set before any
training; it passes, with the caveats below written down rather than tuned away.

Smoothing-window sweep (reference baseline on degraded matches, seed 0):

| window (frames) | `J03WPY` spellF1 / macroF1 | `J03WMX` | `metrica1` | resulting ball jitter |
|---|---|---|---|---|
| 1 (none) | 0.265 / 0.588 | 0.295 / 0.658 | 0.380 / 0.748 | 0.33–0.47 |
| 9 | 0.258 / 0.578 | 0.276 / 0.651 | 0.382 / 0.748 | 0.05–0.06 |
| **15** | 0.257 / 0.580 | 0.275 / 0.639 | 0.378 / 0.747 | 0.03–0.04 |
| 41 | 0.234 / 0.559 | 0.237 / 0.628 | 0.346 / 0.729 | 0.012–0.016 |

## Caveats carried forward

- **Dead-ball visibility is unobservable without labels.** The public window (used for
  understanding only, not fitting) shows the ATD keeps a visible ball through some stoppages and
  loses it entirely in others; the regime proxy cannot tell those apart, so the `D` signature of
  degraded Sportec (ball missing in ≈ 50 % of dead frames) may not be BAR-ATH's. The model stage
  must therefore not lean on "ball missing ⇒ dead" alone; collective-motion and Smart Tagging cues
  carry that decision.
- The public window's D share (20 %) is not representative of a full match (Sportec 41–52 %,
  Metrica 29–39 %); per-class comparisons with it are indicative only.
- One seed per match was used here; training will use several seeds per match.

## Rules compliance

Fitted to label-free BAR-ATH input statistics only (`corpus/noise_targets.json` lists them);
validated against numbers the organisers published; the public 5-minute labels were read only to
*describe* the two failure modes above, and no constant in the simulator comes from them.

## Reproduce

```bash
.venv/bin/python scripts/noise_stats.py --json corpus/noise_targets.json   # ~8 s
.venv/bin/python scripts/fit_noise.py                                        # ~1 min
make test
```
