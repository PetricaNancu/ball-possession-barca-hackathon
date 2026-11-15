# Step 5 — External Validation, Rule Compliance & Improvements (The Digital Farmhouse)

This document explains how we validate and tune our possession engine **strictly on the 10 external degraded matches** (`Sportec x7 + Metrica x3`, 1.43M graded frames) in full compliance with `RULES.md` and `EVALUATION.md`, what improved in v2, why tuning on external CV outperformed 5-minute overfitting, and how to run the external benchmarking and smoke-test scripts.

---

## 1. Strict Separation: External Tuning vs. 5-Minute Smoke Test (`RULES.md` Rule 2)

Per `RULES.md` and `EVALUATION.md`, the 5-minute `BAR-ATH` public ground truth (`17` possessions) is **never** used for training, hyperparameter sweeps, or rule selection:
- **All model training, decoder sweeps, and rule ablations** run exclusively on the **10 external matches** (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`, `metrica1`, `metrica2`, `metrica3`) via 10-fold Leave-One-Match-Out CV (`scripts/train.py` and `scripts/benchmark_tune.py --val-split loo`).
- Because Metrica sample matches come from the exact same provider and share the same possession/stoppage conventions as `BAR-ATH`, external training and out-of-fold validation weight Metrica matches $2\times$ (`metrica_weight: 2`).

---

## 2. External 10-Fold Leave-One-Match-Out CV Results

Selected decoder configuration on external OOF: `{"switch_team": 8, "switch_dead": 50, "min_frames": 20, "postprocess": false}`.

| Validation Subset | Baseline `spellF1` / `macroF1` | Emission Argmax `spellF1` / `macroF1` | **Decoded Engine `spellF1` / `macroF1`** | `F1(D)` | Mean Pred / GT Possessions |
|---|---:|---:|---:|---:|---:|
| **Metrica ×3 Mean** (`metrica1..3`, v1 was `0.662`) | 0.349 / 0.744 | 0.407 / 0.876 | **0.695 / 0.877** | **0.887** | 256 / 320 |
| **Sportec ×7 Mean** (`J03*`) | 0.259 / 0.592 | 0.310 / 0.870 | **0.581 / 0.875** | **0.952** | 258 / 323 |
| **Unweighted 10-Match Mean** | 0.286 / 0.637 | 0.339 / 0.872 | **0.615 / 0.876** | **0.933** | 257 / 322 |
| **Provider-Weighted Mean ($2\times$ Metrica)** | 0.300 / 0.662 | 0.355 / 0.873 | **0.634 / 0.876** | **0.922** | 257 / 322 |

### External Ablation (Why External CV Beat Heuristic Post-Filtering)
| Pipeline Variant (10 External Matches OOF) | Weighted `spellF1` | Weighted `macroF1` | `F1(D)` | Mean Possessions |
|---|---:|---:|---:|---:|
| 1. Emission Argmax only (no Viterbi) | 0.3545 | 0.8731 | 0.9204 | 784.0 |
| **2. Emission + Viterbi (`st=8, sd=50, mf=20`)** | **0.6338** | **0.8760** | **0.9221** | **257.3** |
| 3. Emission + Viterbi + Stoppage Post-Filter | 0.6096 | 0.8587 | 0.9012 | 231.7 |

Evaluating across 10 full 90-minute matches revealed that hard stoppage post-processing rules over-merge possessions on full matches (`231.7` mean possessions, `-0.024` `spellF1`), whereas LightGBM's new directional and stoppage features (`touch_gap_len`, `collective_speed_fwd100`, `oob_latched`) decoded cleanly with Viterbi lift Metrica `spellF1` from `0.662` (v1) to **`0.695`** (`0.704`, `0.727`, `0.653`).

---

## 3. Post-Hoc Smoke Test on `BAR-ATH` Public 5 Minutes (Read-Only Check)

Scored once via `make evaluate` solely to confirm alignment and label semantics:

| Submission | `spellF1` (Ranking) | `macroF1` | Frame Accuracy | `F1(H)` | `F1(A)` | `F1(D)` | Possessions (GT=17) | Mean Dur (GT=14.2s) | Alive Team Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **The Digital Farmhouse** | **0.4941** | **0.8254** | **84.87%** | **0.8688** | **0.8694** | **0.7380** | **17** | **15.6 s** | **91.29%** |
| `baseline_smooth125` | 0.4242 | 0.5812 | 72.75% | 0.7864 | 0.8108 | 0.1464 | 16 | 18.5 s | 88.54% |
| `baseline_smooth50` (Published Floor) | 0.3256 | 0.6207 | 71.32% | 0.7781 | 0.7615 | 0.3225 | 26 | 10.8 s | 83.49% |
| `baseline_smooth25` | 0.2868 | 0.6200 | 69.12% | 0.7620 | 0.7123 | 0.3857 | 36 | 7.6 s | 79.09% |
| `baseline_raw` | 0.0378 | 0.5972 | 65.12% | 0.7290 | 0.6430 | 0.4197 | 131 | 2.0 s | 72.79% |

Full-match `The Digital Farmhouse.csv` summary: **`265` possessions**, mean **`12.9 s`**, `H: 52,536`, `A: 33,019`, `D: 73,302`, `X: 40,462` (`61.4%` Barcelona / `38.6%` Athletic in-play possession share).

---

## 4. Technical Summary of v2 Improvements

1. **Fixed Provider Label-Leakage in Sportec Dead-Ball Simulation (`mtah/degrade.py`)**:
   - `_unfreeze_provider_holds()` replaces DFL's constant-coordinate dead-ball holds (`dx == dy == 0`) with smooth low-speed drift so LightGBM learns genuine stoppage physics rather than `ball_speed == 0`.
2. **Improved Bidirectional Controlled-Touch & Directional Features (`mtah/features.py`)**:
   - Fixed `control_next_sgn` backward fill (`ctrl_v2_bfill_sgn`), excluded linearly interpolated frames from false acceleration spikes, and added one-sided past/future windows (`_bwd35`, `_bwd100`, `_fwd35`, `_fwd100`), `touch_gap_len` (`16.7%` gain), and `oob_latched`.
3. **Algorithmic Smart Tagging Kickoff Priors (`mtah/decode.py`)**:
   - `detect_kickoff_frame()` locates the kickoff strike algorithmically inside each Smart Tagging `Kick Off (Start)` / `Kick Off (After Goal)` window from ball stationarity and departure near the centre spot (no match-specific frame offsets), enforcing `D` prior to kickoff in both halves.

---

## 5. Commands (External Tuning & Smoke Test)

```bash
# 1. Full 10-fold external LOO CV + decoder tuning + final model training
make train

# 2. Fast external decoder/ablation sweep reusing cached 10-fold external OOF predictions
.venv/bin/python scripts/benchmark_tune.py --val-split loo --reuse-oof --update-submission

# 3. Alternative external splits (3-fold Metrica LOO, or Sportec->Metrica cross-dataset)
.venv/bin/python scripts/benchmark_tune.py --val-split metrica-loo --update-submission
.venv/bin/python scripts/benchmark_tune.py --val-split metrica --retrain-final --update-submission

# 4. Validate submission and run read-only 5-minute smoke test
make validate
make evaluate
```
