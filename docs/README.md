# Documentation

One file per step of the pipeline, written as the step is finished: how it works, what it was
measured against, and the numbers. These files are the raw material for the technical document
submitted with the possession file.

| step | file | status |
|---|---|---|
| 0 | Scaffolding, canonical loader, vendored scorer, baseline reproduced bit-for-bit | done (README) |
| 1 | [External training corpus](01-external-corpus.md) — Sportec ×7, Metrica ×3, labels, conventions | done |
| 2 | [Label harmonisation](02-label-harmonisation.md) — DFL touches → Metrica/GT conventions, `T = 50` frames | done |
| 3 | [Noise simulator](03-noise-simulator.md) — clean corpus → BAR-ATH-like tracking, validated against the published baseline | done |
| 4 | [The engine](04-engine.md) — rule layer, features, LightGBM, Viterbi; LOO spellF1 0.615 vs baseline 0.286 | done |
| 5 | [Evaluation & improvements](05-evaluation-and-improvements.md) — v2 rules, Smart Tagging priors, 5-min benchmark (`spellF1 = 0.6571`, `macroF1 = 0.9269`, `acc = 92.49%`) | done |
| 6 | [Technical Document](../results/The%20Digital%20Farmhouse.md) — official deliverable for Phase 1 submission | done |

Metric vocabulary (from the organisers' EVALUATION.md): **spellF1** = possession-level F1 over
temporal IoU 0.3–0.7, the ranking metric; **macroF1** = per-frame mean F1 over H/A/D, the
tie-break. Noise band ≈ 0.04 spellF1 on a full match.
