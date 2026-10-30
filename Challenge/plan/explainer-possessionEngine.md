# Explainer — what we are building, how it works, and what each term means

Companion to `plan-possessionEngine-v2.prompt.md`. Written for a team member joining cold.

---

## 1. The idea in bullets

**What the system does**
- Reads one football match's automatic tracking (22 players + ball, 25 times per second, 199,319 frames) and writes one letter per frame: `H` (Barcelona has the ball), `A` (Athletic has it), `D` (ball out of play), `X` (frame not graded).
- It is scored on how many *whole possessions* it gets right (`spellF1`), not on how many frames. A possession = an unbroken run of `H` or of `A`. So the job is really: **find every moment the ball changes team and every moment it goes dead, to within a few frames**.

**How it works, end to end**
1. **Load** the tactical tracking (ATD), the Smart Tagging and the homography for BAR-ATH.
2. **Clean the ball track**: smooth it, flag impossible jumps, and fill short gaps when the ball's path before and after the gap is consistent (ballistic gap fill).
3. **Optionally add more ball observations** from the tactical video (our own detector) and from the broadcast tracking (after aligning its clock) — "ball fusion".
4. **Rule layer** turns raw positions into football events: touches (who touched the ball, when), out-of-bounds, stationary ball, set pieces, a simple "last controlled touch" possession state.
5. **Features**: for every frame, a vector of numbers describing the situation (distance from ball to nearest Barça / Athletic player, ball speed, time since last touch, who the pass seems to be going to, team shapes, Smart Tagging phase, …).
6. **Emission model** (a trained classifier) maps each frame's features to three probabilities: P(H), P(A), P(D).
7. **Decoder (Viterbi / HMM)** turns 199k noisy per-frame probabilities into a small number of clean, long segments by penalising label changes.
8. **Label-free adaptation** nudges the model toward BAR-ATH's own statistics without any human labels (allowed by the rules).
9. **Write** `frame,label` CSV and validate it with the organisers' script.

**Where the training data comes from (the crucial rule)**
- We may **not** label a single frame of BAR-ATH by hand, nor train or tune on the public 5-minute ground truth.
- We **may** use labelled data from **other matches**. Two public sources give exactly our labels (alive/dead + which team owns the ball) per frame: **Sportec/DFL (7 Bundesliga matches)** and **Metrica sample data (3 matches)**.
- Those datasets are *corrected* (clean). Ours is *automatic* (ball missing 41% of frames, tracks that break and swap). So we run them through a **noise simulator** that makes them look like our data before training. That simulator is the one big assumption of the project, and we test it.

**What you need to have / find to start**
- [ ] The BAR-ATH bundle in `data/` (already present: tactical + broadcast ATD, homographies, Smart Tagging, both videos).
- [ ] Python ≥ 3.9 venv with `numpy pandas kloppy lightgbm scikit-learn scipy matplotlib`; later `torch` (BiGRU), `ultralytics` + `opencv-python` (video ball detector), `optuna` (decoder tuning).
- [ ] The repo's scorer working: `python evaluation/run_evaluation.py` prints the four baselines.
- [ ] External data downloaded by script: Sportec via `kloppy.sportec.load_open_tracking_data(...)`; Metrica via `git clone https://github.com/metrica-sports/sample-data`.
- [ ] A public GitHub repo for the submission (data gitignored), a team name, a Google account for the form.
- [ ] Compute: the M1 is enough for everything except training the video ball detector; for that, a GPU you control (check the data terms before using Colab — the data may not be uploaded to third parties).
- [ ] Decisions logged: public-window protocol (smoke test only), which external datasets and licences.

---

## 2. Glossary, in pipeline order

### Tactical video vs broadcast video
- **Tactical**: a fixed wide camera high in the stand showing the whole pitch all the time. 199,319 frames at 25 fps. **This is the reference clock** — row *i* of our CSV is frame *i* of this video.
- **Broadcast**: the TV feed. Cuts, close-ups, replays, its own frame count (164,488). Not synchronised to tactical; if we want to use it we must work out, shot by shot, which tactical frame each broadcast frame corresponds to.

### ATD — Automatic Tracking Data
Metrica's computer-vision output: for every video frame, the (x, y) position and speed of every object it could see, **including the ball**, with no human correction. It comes as a metadata XML (teams, 46 "track slots", 25 fps, pitch 105×68) and a text file with one line per frame:

```
<frame_id>:<x,y,speed>;<x,y,speed>;…(46 slots)…:<ball x,y,speed>
1713242:0.512,0.488,1.3;NaN;0.300,0.610,4.2;…:0.498,0.501,0.0
```

- Coordinates are **normalised 0–1** (multiply by 105.3 and 68.0 to get metres). `NaN` = not seen.
- **Track slots are not players.** There are 46 slots for 22 people; a slot is a fragment of a trajectory that belongs to one team. When the tracker loses someone and finds them again, they may come back in a different slot. We therefore never reason "player 7 did X", only "a Barcelona player did X".
- `video_frame = frame_id − 1`. The ball is present on 59% of graded frames.

### Homography
A 3×3 matrix **H** per video frame that converts a pitch position into the pixel where it appears in that frame. If the pitch position is (x, y) in 0–1 coordinates:

$$
\begin{pmatrix}u\\v\\w\end{pmatrix} = H \begin{pmatrix}x\\y\\1\end{pmatrix},\qquad
\text{pixel} = \left(\frac{u}{w}, \frac{v}{w}\right)
$$

The inverse matrix does the reverse: pixel → pitch position. We use it to (a) draw the tracking on the video to understand it, (b) create automatic training labels for a ball detector (we know where the ball is on frames where ATD saw it → crop the video there), and (c) convert our own detections back to pitch coordinates. The CSV has `valid` (does a matrix exist?) and `reprojection_error` (how well it fits; lower is better). Tactical is valid on every frame; broadcast on 65.8%.

### Smart Tagging
Metrica's **automatic tactical annotation** of the tactical video, as a SportsCode XML of 638 "instances". Each instance is a time interval with a code and tags:

```xml
<instance>
  <start>443.88</start> <end>448.88</end>          <!-- seconds from video start -->
  <code>SET PIECES</code>
  <label><group>Team</group><text>N/A</text></label>
  <label><group>Type</group><text>Kick Off (Start)</text></label>
  <label><group>Half</group><text>1st Half</text></label>
</instance>
```

Codes: `SET PIECES` (throw-in, corner, goal kick, free kick, kick-off), `GOALS`, and phases like `BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `DEFENDING IN MIDDLE THIRD`, `BALL IN THE BOX`… each with a `Team` tag (Barcelona / Atletic Club / N/A). `video_frame = round(seconds × 25)`.

How we use it: a set piece is a **restart**, so it marks the end of a dead-ball stretch and the team that restarts; a goal starts a dead-ball stretch; a phase tagged "Barcelona" is a soft hint that Barcelona is in possession during it. It is inferred automatically, incomplete (only 59 set pieces in a match that has far more stoppages) and sometimes wrong, so it is evidence, never the answer.

### Ball cleanup + ballistic gap fill
Goal: a ball track that is as continuous and physically sensible as possible before anything else looks at it.

1. **Outlier removal** — a ball cannot jump 30 m between two frames (that is 750 m/s). Flag positions that imply impossible speed or that disagree with both neighbours.
2. **Smoothing** — Savitzky–Golay filter (a local polynomial fit) on x and y to remove pixel jitter, then compute speed, acceleration and heading per frame.
3. **Out-of-bounds flag** — position outside the 105.3 × 68 rectangle (with a small margin), a strong dead-ball cue.
4. **Ballistic gap fill** — for every run of `NaN` frames, look at the ball just before and just after:
   - *Example*: last seen at frame 1000 at (50 m, 30 m) moving 15 m/s toward +x; next seen at frame 1040 at (73 m, 31 m). The gap is 1.6 s and the displacement 23 m → 14.4 m/s along the same heading. That is one consistent flight → interpolate the 39 missing positions along the line (speed allowed to decay a little).
   - *Counter-example*: next seen at (20 m, 60 m): a 3.3 m/s "flight" in the opposite direction is not one flight — something happened in between (a touch, a header, a deflection). Do **not** fill; keep the endpoints and record "gap of 40 frames, inconsistent", which is itself a feature (something happened here).
   - "Ballistic" is used loosely: we work in the 2D ground plane, so the test is straight-line, roughly constant-or-decaying speed, not a 3D parabola.
5. Every filled frame is marked `source = interpolated` so later stages can trust it less than a real observation.

### Ball fusion (Tiers 2 and 3)
When we have more than one opinion about where the ball is, pick one per frame:

| priority | source | when used |
|---|---|---|
| 1 | tactical ATD ball | present and passes the plausibility checks |
| 2 | our video detector (Tier 2) | ATD missing; detection confidence high and consistent with neighbouring frames |
| 3 | broadcast ball mapped to tactical time (Tier 3) | both above missing; alignment confident; homography valid |
| 4 | ballistic interpolation | endpoints consistent |
| 5 | `NaN` | nothing trustworthy |

Plausibility = speed ≤ ~40 m/s, no jump against both neighbours, inside the pitch (or just outside). The chosen **source and a confidence number become features**, so the classifier can learn to weigh a detector ball differently from an ATD ball.

### Sportec / DFL ×7 (the main external training set)
Seven full Bundesliga matches released openly by the DFL and Sportec Solutions with a 2025 *Scientific Data* paper (Bassek et al.). For each match: positional data at 25 Hz for all players and the ball **with per-frame `ball_state` (alive/dead) and `ball_owning_team`**, plus event data (passes, shots, set pieces with type). Those two flags are exactly our `H / A / D`. Loaded in two lines:

```python
from kloppy import sportec
tracking = sportec.load_open_tracking_data(match_id="J03WMX")   # also J03WN1 J03WPY J03WOH J03WQQ J03WOY J03WR9
events   = sportec.load_open_event_data(match_id="J03WMX")
```

Note the licence on the figshare page and cite the paper in the technical document.

### Metrica sample data ×3 (same provider as our match)
`https://github.com/metrica-sports/sample-data` — three anonymised matches from Metrica's own corrected ("Elite") pipeline.
- Games 1–2: tracking as CSV (one file per team) + events CSV.
- Game 3: tracking in the **same EPTS XML/TXT format as our ATD** + events JSON. `kloppy.metrica.load_tracking_epts(...)` and `load_event(...)` read them.
- Events include `SET PIECE`, `PASS`, `RECOVERY`, `BALL LOST`, `BALL OUT`, `SHOT`, `CHALLENGE`, `FAULT RECEIVED`, … with the `documentation/` folder defining each. We derive `H / A / D` per frame **by code**: possession = team of the current event sequence; `D` from a `BALL OUT` / foul / goal until the next `SET PIECE`.
- Why they matter even though there are only three: the ground truth we are scored on was produced by Metrica's corrected tracking, so these matches carry the *same conventions* (when does a loose ball change hands, when does dead start/end).

### Finding more sample matches (optional)
What a dataset must have to be useful: player **and ball** positions at ≥ 10 Hz, and either per-frame possession / ball-status flags or event data synchronised to the tracking. Where to look:
- **Hugging Face** — search `soccer tracking`, `football tracking data`, `SoccerNet` (SoccerNet hosts video/tracking datasets there; most have no possession labels, useful mainly for ball-detector pre-training).
- **Kaggle** — search `football tracking data`, `soccer positional data`, `DFL Bundesliga` (the "DFL Bundesliga Data Shootout" is video + events, no tracking — good for detector pre-training only).
- **GitHub** — `SkillCorner/opendata` (9 matches of *automatic broadcast* tracking: noisier and closer to ours, but partial pitch coverage; check whether its possession fields are usable), `Friends-of-Tracking-Data-FoTD`.
- Rules reminder: anything about **this** match (any LaLiga event feed, any corrected possession data for BAR-ATH) is banned; only other games. Record source, licence, who labelled it and how much — the technical document must list it.

### Noise simulator (`degrade.py`)
**Problem**: external data is clean; ours is automatic. A model trained on clean data learns "the ball is always there", then meets a match where it is missing 41% of the time.
**Solution**: a function that takes a clean match and returns a version that *statistically* looks like BAR-ATH:
1. **Measure BAR-ATH without labels** (allowed — these are statistics of the input, not answers): how often and for how long the ball disappears, and how that depends on ball speed, how crowded the area is, and distance to the touchline; how long a track slot lives before it breaks; how often two slots cover one person; how jittery positions are (from frame-to-frame second differences).
2. **Fit simple models to those numbers**: a hazard "probability the ball drops out this frame given speed and crowding", a gap-length distribution, a track-break hazard, a jitter standard deviation.
3. **Apply to the clean match**: remove player identities (assign to anonymous slots that break and re-assign exactly like ours), delete ball observations according to the hazard, add jitter and occasional duplicates. Use several random seeds per match to get 3–5 different degraded versions.
4. **Validate** by running the organisers' own baseline (`baseline_smooth50`) on the degraded matches: it should score about what it scores on the full BAR-ATH (spellF1 ≈ 0.24, macroF1 ≈ 0.67), and the degraded ball-visible fraction should be ≈ 59%, mean tracked objects ≈ 20.5. If not, the simulator is wrong and must be fixed before anything is trained on it.

### Rule layer: touches, dead-ball cues, state machine
- **Touch** = a sharp change in the ball's speed or direction. Attribute it to the nearest player within ~2 m → a team. Also record how contested it was (distance to the nearest opponent).
- **Receiver prediction**: when the ball disappears mid-flight, extend its last velocity forward; the players lying on that line over the next second or two, and whoever is nearest where it reappears, tell us which team probably received it. Because we process offline, we can use the *future* frame where it reappears.
- **Dead-ball cues**: out of bounds; ball not moving while everyone slows down; both teams drifting back into shape; number of tracked objects changing (substitution, injury); a set-piece tag; the pre-kick-off minutes of each half.
- **State machine**: possession belongs to the team of the last *controlled* touch; "controlled" = two consecutive touches by the same team, or carrying the ball for ≥ ~0.4 s, or the ball slowing right after the touch. A deflection (ball keeps its speed, barely changes direction) does not change possession. A loose ball stays with the last team. All thresholds are fitted on the external matches, never on BAR-ATH.

### Features, training, inference (inputs and outputs)
- **Unit of training**: one frame. **Input**: a feature vector of ~50–100 numbers for that frame, built from a window around it (e.g. ±2 s) so it carries context. **Output**: the true label `H / A / D` from the external match.
- **Orientation**: features are expressed relative to the attacking direction ("toward own goal / toward opponent goal") so Bundesliga and LaLiga matches look alike; and every training match is also fed with H and A swapped, which removes home/away bias and doubles the data.
- **Models**: (a) **LightGBM** — gradient-boosted decision trees, fast, strong on tabular features, deterministic on CPU; (b) a small **BiGRU** — a recurrent neural network that reads the sequence forwards *and* backwards, so each frame's output can depend on what happens later. Both output P(H), P(A), P(D) per frame. If averaging them helps on held-out matches, we average.
- **Validation**: leave-one-match-out — train on all external matches but one, score on the one left out with the official scorer, repeat. The mean held-out `spellF1` is what we compare variants on.
- **Inference on BAR-ATH**: compute the same features from the (cleaned, fused) BAR-ATH tracking → probabilities per frame → decoder → `frame,label` CSV. One command: `make submission`.

### HMM / Viterbi (the decoder)
Per-frame probabilities flicker: a crowded tackle gives `H A H A H` across a few frames, and every flicker would count as several fake possessions. A **Hidden Markov Model** says: there is a hidden true state per frame (H, A or D); the state tends to **stay the same** from one frame to the next (high self-transition probability, small probability of switching); and each frame's features are noisy evidence about the state (the classifier's probabilities).

**Viterbi** is the algorithm that finds the single most probable *whole sequence* of states given all 199k frames of evidence at once. In practice it maximises

$$
\sum_t \log P(\text{label}_t \mid \text{features}_t)\;-\;\sum_t \text{penalty}(\text{label}_{t-1} \rightarrow \text{label}_t)
$$

so a switch is only made when the evidence after it outweighs the penalty. Tiny flickers are not worth the penalty and disappear; real changes with sustained evidence survive. We use **different penalties for H↔A and for H/A↔D**, add small bonuses from Smart Tagging (e.g. toward `D` just before a set piece), and tune the penalties directly on held-out external `spellF1`. A final **minimum-duration filter** merges any segment shorter than a few frames into its neighbours. Optional upgrade: a semi-Markov variant that models how long possessions typically last.

### Label-free adaptation on BAR-ATH
Three permitted ways to adjust to our match without any human label:
- **EM re-estimation**: let the HMM re-estimate its switch rates from BAR-ATH's own decoded output (how often possession actually changes in this match).
- **Self-training**: decode BAR-ATH, keep the long, high-confidence segments as *algorithmic* pseudo-labels, retrain once including them.
- **Smart Tagging weak labels**: inside a phase tagged "Barcelona", far from its edges, frames are very probably `H`; combine such noisy votes with a label model and use them the same way.
Each is first tested on a held-out external match (adapt to it label-free, check the score goes up) before it touches BAR-ATH.

### Tier 2 — ball re-detection from the tactical video
Train a small object detector (YOLO) to find the ball in the tactical frames where ATD lost it. Training labels come **for free and automatically**: on frames where ATD *did* see the ball, project its position into the image with the homography and crop a box there. Run the detector on the missing frames, link detections over time (Kalman filter) to reject one-off false positives, map back to pitch coordinates with the inverse homography, and hand the result to ball fusion. Done only if a first look shows the ball is actually visible on a worthwhile share of the missing frames (not hidden inside a crowd).

### Tier 3 — broadcast alignment
The broadcast has its own clock. For each continuous camera shot (detected from where its homography is valid and stable), find the constant offset that best matches its player positions to the tactical ones (cross-correlate team centroids, then refine by comparing the two point clouds). Offsets must grow monotonically through a half; shots that do not fit (replays, slow motion) are discarded automatically. Where aligned, the broadcast ball becomes a third ball source. Highest effort, most uncertain gain — do last, and only if the complementary coverage is meaningful.

### spellF1 (what we are ranked on) — one paragraph
Take all true possessions (runs of `H` or `A`) and all predicted ones; match each predicted possession to at most one true possession if their time overlap (IoU) exceeds a threshold; compute F1; average over thresholds 0.3–0.7. A constant guess scores 0; perfect scores 1; a boundary 1 s late on every possession scores ≈ 0.65. Too many short predicted possessions (flicker) or merged possessions (missed changes) both hurt. `macroF1` (per-frame, average over the three classes) is only a tie-break.

### Why LLMs are tools here, not the model
Hosted LLMs must never see the match data (terms and conditions), and the submission must be byte-exactly reproducible. So we use them to write code, to propose candidate possession rules that an automatic loop scores on the external matches, and to analyse errors on *public* external data. A local open-weight model may optionally adjudicate a few hundred ambiguous windows, with outputs cached in the repo — kept only if it measurably helps.
