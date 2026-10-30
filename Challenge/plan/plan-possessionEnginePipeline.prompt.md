# Plan: Possession Engine + Video Ball Recovery + Broadcast Fusion

## What the challenge actually is

**It is not "fix a generated file".** Nothing is pre-labelled. You receive raw, uncorrected automatic tracking (player + ball positions per frame, no identities), automatic tactical tags, camera homographies and two videos — and you must **build an algorithm that outputs, for each of the 199,319 frames of the *tactical* video, one of `H` (Barça has the ball), `A` (Athletic has it), `D` (ball out of play) or `X` (abstain)**. Only frames 11185–90651 and 110808–190197 are scored (the two halves); warm-up, halftime and post-match are ignored — write `X` there.

Key facts that shape every design decision:
- **Ranked on `spellF1`**: each *possession* (maximal run of `H` or `A`) is matched 1:1 against the truth by temporal overlap (IoU 0.3–0.7). Per-frame accuracy is almost irrelevant; what matters is **the right number of possessions with the right boundaries**. Being 1 s late on a boundary costs ~35%.
- **Ground truth = corrected tracking**: `D` where the ball is dead, else the team "owning" the ball. Loose balls stay with the last team; deflections do not change possession; possessions can be as short as 19 frames (0.76 s).
- **Ball is missing on 41% of graded frames** — the central difficulty.
- The baseline (nearest player + majority vote, spellF1 0.2433) fails in two ways visible in `evaluation/reports/baseline_smooth50_transitions.csv`: it **flickers `H`↔`A`** on duels/passes, and its **`D` recall is 0.21** — the tracker keeps following the ball while it is dead, so the baseline calls it a possession. Dead-ball detection and transition precision are the two biggest levers.
- **Broadcast video is optional**, on its own timeline with cuts/replays, with no mapping provided. The videos are *not* required to score — many teams will only use the tracking.
- **Rules tightened after launch (commits `c073f29`, `f8f2629`)**: no hand labels on *this* game for any purpose — not the submission, not training, not validation, not "transitions only"; the public 5 minutes are **not training data and not a tuning set**; no human-produced data about this game from any source (Opta/StatsBomb/Wyscout events, corrected tracking); no manual corrections of the provided tracking. **Allowed**: hand labels and datasets about **other games** in any amount (accessible + documented), unsupervised/self-supervised learning and **algorithmic pseudo-labels** on this game, and watching the video/data to understand and debug — but not turning what you saw into labels, corrections, or rules/constants tied to specific frames. The technical document must list every labelled dataset used (source, who labelled, how much).
- Deliverables: `<team>.csv` + public repo that regenerates it byte-exactly + a few-page technical doc. Deadline **Nov 17**.

---

## Approach

Build a tiered pipeline where Tier 1 (tracking + tagging → hybrid rules/model → HMM) is always shippable, and Tiers 2–3 only *improve the ball track* fed into it. All supervised training and all model selection happen on an **external corpus of other matches** (Sportec/DFL ×7 + Metrica sample ×3) degraded to look like our automatic tracking; the public 5 min is a correctness smoke test only.

## Steps

### Phase 0 — Setup & scaffolding (day 1, all parallel)
1. Create the public submission repo (data gitignored): `src/mtah/{io,features,touches,deadball,model,decode,video_ball,broadcast_align}.py`, `scripts/make_submission.py` (single command), `configs/`, `models/`, `docs/`, pinned `requirements.txt`. Copy the loader patterns from `starter/load_atd.py`, `starter/load_smart_tagging.py`, `starter/load_homography.py`; reuse `games.py` for paths/frame blocks.
2. Build `load_tactical()` → numpy arrays: ball `(N,3)`, players `(N,46,3)` in metres (105.3×68), team-per-slot from ATD XML (first team = home, as in `team_of_each_track` in `starter/baseline.py`). Cache as `.npz`.
3. Build `evaluate.py` wrapper around `evaluation/run_evaluation.py` + `--bootstrap 1000` that can score against *any* `frame,label` CSV (the `--gt` path option) — needed to score the external matches with the official metric.

### Phase 1 — Understand the conventions & build an external training/validation corpus (parallel with Phase 2)
4. Overlay ATD on the tactical video for the public window (homography `project()`) and watch it next to the public GT to learn the *generic* conventions: when `D` starts (ball crossing the line vs. whistle), when `D` ends (restart touch), how interceptions/deflections/aerials are treated. Output is a written list of general rules — **no thresholds or constants tied to specific frames of this game** (RULES.md). Cross-read Metrica's event-definition PDF in `metrica-sports/sample-data/documentation`: same provider, same conventions as the corrected tracking the GT comes from.
5. Build the **external labelled corpus** (hand labelling on this game is banned):
   - **Sportec/DFL open data** — 7 Bundesliga matches, 25 fps tracking with per-frame `ball_state` (alive/dead) and `ball_owning_team`: `kloppy.sportec.load_open_tracking_data(match_id)` / `load_open_event_data(match_id)` (ids `J03WMX`, `J03WN1`, `J03WPY`, `J03WOH`, `J03WQQ`, `J03WOY`, `J03WR9`). Maps **directly** to `H/A/D` per frame.
   - **Metrica sample data** — 3 anonymised matches (`github.com/metrica-sports/sample-data`); game 3 is in the *same* EPTS XML/TXT format as our ATD, tracking synchronised to events (pass, reception, ball out, set piece, recovery…) → derive `H/A/D` per frame **algorithmically** from the events.
   Convert each to (a) the challenge `frame,label` CSV and (b) the same numpy arrays `load_tactical()` produces, so the pipeline and the official scorer run unchanged. Split **by match** (e.g. 7 train / 3 held-out); never tune on held-out.
6. **Noise simulator** (`src/mtah/degrade.py`) + data audit: the external data is clean/corrected, ours is automatic. Measure BAR-ATH's *unlabelled* statistics — ball gap-length distribution and its dependence on ball speed/crowding, track fragmentation and slot re-assignment, jitter, out-of-bounds frames, per-half attack direction (kick-off: team with mean x<0.5 attacks right), Smart Tagging set-piece timing, broadcast XML frame rate — and fit a degradation model that drops identities (→ anonymous slots), fragments tracks and removes the ball with matching statistics. Fitting to unlabelled statistics is algorithmic and allowed. Then run `starter/baseline.py`'s algorithm on the degraded external matches: if it lands near spellF1 0.24 there, the external benchmark is a credible proxy for the private score.

### Phase 2 — Tier 1 possession engine (core; always shippable)
7. *Ball cleanup*: interpolate NaN gaps ≤ ~0.5 s; Savitzky–Golay smooth; compute speed/acceleration/heading; flag out-of-bounds.
8. *Touch detection*: acceleration spikes + heading changes → touch events; attribute each touch to the nearest player (≤ ~2 m) → team, with ball speed before/after and distance to 2nd-nearest opponent as "contest" features.
9. *Dead-ball detector*: candidate `D` from (a) ball out of bounds, (b) ball stationary + low collective player speed, (c) Smart Tagging `SET PIECES` / `GOALS` instances (restart ends `D`; goal starts `D` until `Kick Off (After Goal)`), (d) the pre-kick-off block at the start of each half, detected algorithmically (Smart Tagging `Kick Off (Start)` instance + both teams inside their own halves with the ball at the centre spot); the organiser-published kick-off frames in `games.py` are provided data and may serve as a cross-check. Train the alive/dead classifier on collective-motion features (mean speed, spread, convex hull, fraction stationary) using **external matches' `ball_state`**, never the public GT.
10. *Possession state machine*: team = team of last **controlled** touch; controlled = two consecutive same-team touches, or carry ≥ ~0.4 s, or ball speed drops after touch; deflections (speed kept, small heading change) do not flip. Loose/contested → keep last team. Calibrate thresholds on the external corpus; step 4 only informs which rules exist.
11. *Per-frame feature vector* (`features.py`): min distance ball→H / ball→A, ball speed, time since last touch, last-touch team, touch sequence counts, ball-missing flag + gap length, team centroids/spread/mean speed, pitch-control proxy, state-machine output, Smart Tagging phase code + team one-hot, frames to/from nearest set piece.
12. *Emission model*: LightGBM 3-class {H,A,D} trained on the **degraded external corpus** with leave-matches-out CV; output probabilities per frame. Never trained on BAR-ATH labels. Optional and allowed on BAR-ATH: label-free adaptation — HMM/EM re-estimation on the emission probabilities, or self-training on high-confidence *algorithmic* pseudo-labels. (*depends on 5, 6, 11*)
13. *Decoder*: 3-state HMM / Viterbi per half with tuned switch penalties (H↔A, H/A↔D), then a post-filter merging spells shorter than a minimum duration. Tune penalties on held-out external matches' spellF1; sanity-check BAR-ATH's possession count / mean duration against the *external* matches' distributions (not against the public window's 17 / 14.2 s — that would be tuning on the public GT). `X` only on ungraded frames (on graded frames the HMM already provides continuity; `X` would only cost macroF1).
14. Write `make_submission.py` → `evaluation/results/<team>.csv`; run `evaluation/validate_submission.py`. **Checkpoint: first valid submission that beats the baseline on held-out external matches, plus one smoke-test run on the public window to confirm alignment and label semantics (record the number, do not select on it).**

### Phase 3 — Tier 2: recover the ball from the tactical video (parallel with Phase 2 steps 8–13; CV member)
15. Auto-label: for frames where ATD ball exists, project to pixels via tactical homography; keep only low-speed / near-player frames (ball on ground, projection reliable). Spot-check crops only to design the *automatic* filter — never drop or fix individual samples by hand (that is hand correction). Optionally pre-train on public ball-detection data from other games (e.g. SoccerNet).
16. Fine-tune a small detector (YOLOv8/11-n/s, 1280 input or tiled crops) on ~5–10k auto-labelled frames; hold out frames to measure recall/precision against ATD.
17. Inference on the ~86k missing-ball frames; link detections into tracklets (Kalman), reject isolated/false positives; inverse-homography to pitch coords; mark source + confidence. Feed as the ball input to Phase 2 with source flags as features; measure gain with the Phase 5 protocol. *Fixed seeds, deterministic inference flags; quantise coordinates to 1e-3 so tiny float noise cannot change the final CSV.*

### Phase 4 — Tier 3: broadcast ↔ tactical alignment (parallel with Phases 2–3; separate member)
18. Segment the broadcast into shots: homography `valid` flips + jumps in matrix / player positions (optionally frame-difference shot detection on the video).
19. For each valid shot, find a constant offset into tactical time: coarse search via cross-correlation of team-centroid time series, refine via Chamfer distance between per-team player point sets. Enforce monotonic, near-constant offset within each half; reject shots with poor score or inconsistent offset (replays, slow-mo). No manual offset fixes — a rejected shot stays rejected.
20. Validate alignment on frames where both balls exist (median ball disagreement < ~1–2 m). Fuse: broadcast ball where tactical ball is missing/implausible. Feed into Phase 2.

### Phase 5 — Fusion, ablation, freeze (depends on 14; 17 and 20 optional)
21. Ablations on held-out external matches with bootstrap CIs: tracking-only vs +video ball vs +broadcast ball; keep each addition only if it clears the ~0.04 noise band. On BAR-ATH itself use only label-free proxies: ball coverage on graded frames, and agreement between ATD / video / broadcast balls on frames where more than one source exists. (Video and broadcast ball recovery can only be *scored* on BAR-ATH indirectly — so the external ablation uses the degraded corpus with simulated "recovered" ball coverage matching what Tiers 2–3 achieve.)
22. Freeze config; regenerate from a **fresh clone on a second machine**; `diff` the CSVs — must be identical. Validate the exact file to be uploaded.

### Phase 6 — Deliverables
23. Technical doc (`docs/<team>.md` → PDF): approach & why, assets used/dropped, **training data (every external dataset: source, who labelled it, how much — none from BAR-ATH)**, possession rules implemented, handling of missing ball, any LLM components and how their outputs are made reproducible, limitations, reproduction command. README with install + `make submission`.
24. Submit `<team>.csv`, repo URL, `<team>.pdf` via the form before Nov 17.

## Relevant files
- `starter/baseline.py` — `nearest_player_labels`, `smooth`, `write_submission`, `team_of_each_track`: reuse as reference for ATD parsing, team mapping and CSV writing.
- `starter/load_homography.py` — `load`, `project` (and `np.linalg.inv(H)` for unprojection) for auto-labels and video→pitch.
- `starter/load_smart_tagging.py` — intervals → video frames (`round(sec*25)`), case-insensitive tag matching.
- `games.py` — `asset_paths`, `half_video_blocks`, pitch 105.3×68, kickoff frames.
- `evaluation/possession_metrics.py` — the spec; reuse its spell/transition functions in the internal evaluator.
- `evaluation/run_evaluation.py`, `evaluation/validate_submission.py` — scoring (`--gt <path>`, `--bootstrap`) and the pre-submit check.
- `evaluation/gt/BAR-ATH_possession_gt_public.csv` — correctness smoke test only (not training, not tuning); `evaluation/reports/baseline_smooth50_transitions.csv` — failure-mode catalogue.
- External corpus: `kloppy.sportec.load_open_tracking_data` / `load_open_event_data` (7 DFL matches, per-frame `ball_state` + `ball_owning_team`); `github.com/metrica-sports/sample-data` (3 matches, game 3 in EPTS format, `documentation/` has the event definitions).

## Verification
1. `python evaluation/validate_submission.py evaluation/results/<team>.csv` passes (199,319 rows, frames 0–199318, labels H/A/D/X).
2. `python evaluation/run_evaluation.py --bootstrap 1000` on public, run once per milestone as a smoke test: labels mean what we think, 2nd-half frames before kick-off come out `D`, D recall ≫ 0.21, no absurd possession count. Not used to choose between versions.
3. Same scorer with `--gt <external_match.csv>` on held-out external matches: spellF1 beats the baseline algorithm run on the same degraded matches, beyond the bootstrap interval; tracked per ablation.
4. Alignment sanity: Smart Tagging kick-off instances contain frames 11185 and 113742; 2nd-half frames 110808–113741 come out `D`.
5. Detector: recall/precision vs held-out ATD ball frames; fused ball coverage on graded frames reported (target well above 59%).
6. Broadcast: per-shot offset plot monotonic within halves; ball agreement on overlap < 2 m median.
7. Reproducibility: fresh clone + `make submission` on two machines → `diff` identical.

## Decisions
- Tactical ATD + Smart Tagging is the primary input; video and broadcast only *augment the ball track*, so Tier 1 is always a complete submission.
- Hybrid: rule layer (touches, dead-ball, state machine) produces features and priors; LightGBM emissions; HMM/Viterbi + min-duration post-filter decode.
- No hand labels and no human-produced data about BAR-ATH anywhere in the pipeline. All supervised learning on external matches (Sportec ×7, Metrica ×3) with simulated tracking noise; only unsupervised / algorithmic-pseudo-label methods touch BAR-ATH.
- No `X` on graded frames.
- Model selection never uses the public 5 minutes; it is a smoke test run at milestones only.
- Any LLM that sees match data runs locally (open weights); hosted LLMs are used only on code and prose that contain no data.

## Where LLMs fit (cost-bounded, never frame-by-frame)

Constraints first:
- The T&C forbid sending **any** data, video or extract to a third-party hosted LLM. So **cloud LLMs (ChatGPT / Claude / Gemini / Copilot) only for work that contains no match data** — code, docs, rule design, metric analysis. **Anything that reads match data must be a local open-weight model** (e.g. Qwen2.5 / Llama 3.x 7–14B via llama.cpp or Ollama on the M1 or your own GPU; Qwen2-VL for images).
- Byte-exact reproducibility: every in-pipeline LLM call runs at temperature 0 with a fixed seed, and its outputs are **cached as a JSON artefact in the repo** together with the script that regenerates it. Ask the organisers (GitHub issue) whether cached LLM outputs satisfy rule 3.
- LLM use is not scored (OVERVIEW.md). Keep a component only if the external-corpus ablation shows a gain.

Ranked by expected value per unit cost:
1. **Development copilot (cloud OK, no data)** — write the rule engine, degradation model, HMM, tests, README and technical doc; have it read `possession_metrics.py` and the rules to enumerate what the metric rewards and penalises. Highest and most certain value.
2. **LLM-guided rule / feature search (cloud OK if prompts contain only code and metric numbers from *external* games)** — the LLM proposes candidate possession rules and features as Python functions from the textual definitions in CHALLENGE.md; an automatic loop scores each candidate on the external corpus and feeds the score back (FunSearch-style). A few dozen iterations, cheap, and the labels never come from a person. Keep BAR-ATH out of the loop entirely.
3. **Local LLM adjudication of ambiguous windows only (local, data-touching)** — compress tracking into a symbolic event log (~3–5k touch events per match: time, team, speed before/after, nearest-opponent distance, zone, Smart Tagging phase). Only windows where the rule engine and the GBM disagree, or confidence is low (expect a few hundred per match), are sent as short text prompts: "deflection, interception, or loose ball retained?" The answer enters the HMM as one more emission term. Runs on an M1 in well under an hour; gain uncertain — gate it behind the ablation.
4. **Local VLM on sparse keyframes (local, data-touching)** — at candidate dead-ball boundaries (~200–500 per match) show 1–3 frames to a local VLM: "ball out / referee signal / throw-in?" Tactical wide shots make the ball tiny, so accuracy will be modest; broadcast close-ups are better but need the Phase 4 alignment. Lowest priority; only after Phases 2–4.

Not worth doing: per-frame LLM calls (199k × anything is slow and non-reproducible), or sending any clip / CSV / extract to a hosted model.

## Further considerations
1. **Cloud compute & data terms** — the FAQ forbids sending data to third-party hosted LLMs, and the data T&C you signed likely restricts uploading to third parties generally. Recommendation: run video work on the M1 or a VM you control; only use Colab if the signed terms clearly permit it. Ask the organisers via a GitHub issue if unsure.
2. **Byte-exact reproduction with a GPU detector** — GPU inference can be nondeterministic. Recommendation: deterministic flags + coordinate quantisation + publish trained weights in the repo; ask the organisers whether publishing weights trained on match frames is acceptable (it is not frame-level tracking data, but confirm).
3. **Scope guard** — Tiers 2–3 can consume the whole three weeks. Recommendation: hard checkpoint around Nov 8 — if video/broadcast gains are not above the noise band on the external corpus, drop them and polish Tier 1 + documentation.
4. **Domain gap of the external corpus** — Bundesliga/Metrica sample matches are clean and from other competitions; the degradation model (step 6) is what makes them a valid proxy. Validate the proxy by checking that the repo baseline scores similarly on degraded external matches and on the full BAR-ATH (0.24). If it does not, the noise model is wrong before anything else is.
5. **Public-window discipline** — you will inevitably see the public score. Write the protocol down in the repo (one run per milestone, never a selection criterion) so the technical document can state it honestly.
