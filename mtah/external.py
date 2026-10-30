"""External (other-match) corpora -> the canonical `Tracking` object, with per-frame labels.

Sportec / DFL open data (Bassek et al. 2025, 7 Bundesliga matches): 25 fps positional data for
all players and the ball, with per-frame BallStatus (alive/dead) and BallPossession (team). Those
two flags are exactly the challenge's H / A / D, so the label needs no derivation.

The two periods are laid out on one video-like timeline: [period 1][halftime gap][period 2], the
gap being the real frame-id gap in the source (all-NaN, labelled X), so the result looks like a
tactical recording and `graded_mask()` / `half_of_frame()` behave as they do for BAR-ATH.

Coordinates are requested from kloppy in Metrica's normalised system (origin top-left, 0-1) and
scaled by the match's pitch, which is the same convention `load_tracking` applies to the ATD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths
from .io import AWAY, HOME, Tracking

SPORTEC_MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
CORPUS_GT_DIR = paths.ROOT / "corpus" / "gt"


def tracking_from_kloppy(dataset, game: str, source: str, require_labels: bool = True) -> Tracking:
    """Any kloppy TrackingDataset (normalised Metrica coordinates) -> Tracking with labels.

    Slots are real players here (one per player who appears), home first. Labels: D where the
    ball is dead, else H/A from ball_owning_team; X on the synthetic halftime gap. Speed columns
    are the provider's field as-is (units differ between providers; features derive their own).
    With require_labels=False (no per-frame flags in the source) labels are left None and
    meta["frame_ids"] carries the record -> row mapping so a caller can derive them.
    """
    md = dataset.metadata
    pitch = (float(md.pitch_dimensions.pitch_length), float(md.pitch_dimensions.pitch_width))
    fps = float(md.frame_rate)
    home, away = md.teams[0], md.teams[1]
    if str(home.ground) != "home":
        raise ValueError(f"{game}: first team is {home.ground}, expected home")

    df = dataset.to_df()
    player_ids, team_of_slot = [], []
    for team, code in ((home, HOME), (away, AWAY)):
        for player in team.players:
            if f"{player.player_id}_x" in df.columns:
                player_ids.append(player.player_id)
                team_of_slot.append(code)
    team_of_slot = np.asarray(team_of_slot, dtype=np.int8)

    period_ids = df["period_id"].to_numpy(dtype=np.int64)
    frame_ids = df["frame_id"].to_numpy(dtype=np.int64)
    periods = sorted(set(period_ids.tolist()))
    if periods != [1, 2]:
        raise ValueError(f"{game}: expected periods [1, 2], got {periods}")

    # Timeline: period 1, then the source's own frame-id gap, then period 2.
    first = {p: int(frame_ids[period_ids == p].min()) for p in periods}
    last = {p: int(frame_ids[period_ids == p].max()) for p in periods}
    for p in periods:
        ids = frame_ids[period_ids == p]
        if not np.array_equal(ids, np.arange(first[p], last[p] + 1)):
            raise ValueError(f"{game}: period {p} frame ids are not contiguous")
    n1 = last[1] - first[1] + 1
    gap = first[2] - last[1] - 1
    n2 = last[2] - first[2] + 1
    n = n1 + gap + n2
    row = np.where(period_ids == 1, frame_ids - first[1], n1 + gap + frame_ids - first[2])

    def col(name, scale=1.0):
        return df[name].to_numpy(dtype=np.float64) * scale if name in df.columns else np.nan

    ball = np.full((n, 3), np.nan)
    ball[row, 0] = col("ball_x", pitch[0])
    ball[row, 1] = col("ball_y", pitch[1])
    ball[row, 2] = col("ball_speed")

    players = np.full((n, len(player_ids), 3), np.nan)
    for s, pid in enumerate(player_ids):
        players[row, s, 0] = col(f"{pid}_x", pitch[0])
        players[row, s, 1] = col(f"{pid}_y", pitch[1])
        players[row, s, 2] = col(f"{pid}_s")

    labels = None
    meta = {"home_name": home.name, "away_name": away.name, "provider": str(md.provider),
            "period_frame_ids": {str(p): [first[p], last[p]] for p in periods},
            "halftime_gap_frames": int(gap)}
    if require_labels:
        labels = np.full(n, "X", dtype="<U1")
        state = df["ball_state"].astype(str).str.lower().to_numpy()
        owner = df["ball_owning_team_id"].astype(str).to_numpy()
        frame_labels = np.where(state == "dead", "D",
                                np.where(owner == str(home.team_id), "H",
                                         np.where(owner == str(away.team_id), "A", "X")))
        labels[row] = frame_labels
    else:
        meta["frame_ids"] = {"frame_id": frame_ids.tolist(), "period_id": period_ids.tolist(),
                             "row": row.tolist()}

    blocks = np.array([[1, 0, n1 - 1], [2, n1 + gap, n - 1]], dtype=np.int64)
    return Tracking(
        ball=ball, players=players, team_of_slot=team_of_slot, fps=fps, pitch=pitch,
        game=game, source=source, labels=labels, blocks=blocks, meta=meta,
    )


def load_sportec(match_id: str, use_cache: bool = True) -> Tracking:
    """One Sportec open match as Tracking (downloaded through kloppy, cached in ~/kloppy_cache)."""
    cached = paths.CACHE_DIR / f"{match_id}_sportec_tracking.npz"
    if use_cache and cached.is_file():
        return Tracking.load(cached)
    from kloppy import sportec
    dataset = sportec.load_open_tracking_data(match_id=match_id, coordinates="metrica")
    tracking = tracking_from_kloppy(dataset, game=match_id, source="sportec")
    if use_cache:
        tracking.save(cached)
    return tracking


# --------------------------------------------------------------------------------------------
# Metrica sample data: tracking from kloppy, labels derived from the event stream
# --------------------------------------------------------------------------------------------

METRICA_MATCHES = ("metrica1", "metrica2", "metrica3")
_METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
_METRICA_EVENTS = {
    1: f"{_METRICA_RAW}/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    2: f"{_METRICA_RAW}/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    3: f"{_METRICA_RAW}/Sample_Game_3/Sample_Game_3_events.json",
}

# Event types that establish (or confirm) which team has the ball. BALL LOST belongs to the team
# that had it, so it never flips possession; CHALLENGE/CARD never do either -- the flip is
# materialised by the opponent's RECOVERY / PASS / CARRY that follows.
_POSSESSION_EVENTS = {"SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST"}


def _metrica_cache_file(url: str) -> Path:
    import urllib.request
    folder = Path.home() / "kloppy_cache" / "metrica"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / url.rsplit("/", 1)[1]
    if not dst.is_file():
        urllib.request.urlretrieve(url, dst)
    return dst


def metrica_events(match_id: int, home_team_id: str) -> "pd.DataFrame":
    """Metrica sample events as one table: type, subtypes (list), side (H/A), period, frames."""
    import json
    import pandas as pd

    path = _metrica_cache_file(_METRICA_EVENTS[match_id])
    rows = []
    if match_id == 3:
        for e in json.load(open(path))["data"]:
            st = e.get("subtypes") or []
            st = [st["name"]] if isinstance(st, dict) else [s["name"] for s in st]
            rows.append({
                "type": e["type"]["name"].upper(), "subtypes": [s.upper() for s in st],
                "side": "H" if e["team"]["id"] == home_team_id else "A",
                "period": int(e["period"]),
                "start_frame": int(e["start"]["frame"]), "end_frame": int(e["end"]["frame"]),
            })
    else:
        raw = pd.read_csv(path)
        for _, e in raw.iterrows():
            st = e["Subtype"] if isinstance(e["Subtype"], str) else ""
            rows.append({
                "type": str(e["Type"]).upper(),
                "subtypes": [s.strip().upper() for s in st.split("-") if s.strip()],
                "side": "H" if str(e["Team"]).strip().lower() == "home" else "A",
                "period": int(e["Period"]),
                "start_frame": int(e["Start Frame"]), "end_frame": int(e["End Frame"]),
            })
    return pd.DataFrame(rows).sort_values(["start_frame", "end_frame"], kind="stable")


def labels_from_metrica_events(events, frame_ids: np.ndarray, period_ids: np.ndarray) -> np.ndarray:
    """Per-frame H/A/D from Metrica's event stream, following CHALLENGE.md's conventions.

    - D from the start of each period until its first SET PIECE, and after a BALL OUT, a FAULT
      RECEIVED, an offside, a goal or a shot going out, until the next SET PIECE (or a restart
      taken directly as a PASS, e.g. a goal kick).
    - While alive, the owning team is that of the last possession event (SET PIECE, PASS, CARRY,
      RECOVERY, SHOT). BALL LOST does not flip: a loose ball stays with the team that lost it until
      the opponent's RECOVERY / PASS / CARRY. CHALLENGE and CARD are ignored.
    """
    labels = np.full(len(frame_ids), "D", dtype="<U1")
    order = np.argsort(frame_ids, kind="stable")
    sorted_ids = frame_ids[order]

    def row_of(frame):
        return int(np.searchsorted(sorted_ids, frame))

    for period in np.unique(period_ids):
        ev = events[events["period"] == period]
        in_period = order[period_ids[order] == period]
        if not len(in_period) or not len(ev):
            continue
        state, since = "D", int(in_period[0])
        timeline = []  # (row, label) change points
        for e in ev.itertuples(index=False):
            subs = set(e.subtypes)
            kind = e.type
            if kind == "SET PIECE":
                new = e.side
                at = e.start_frame
            elif kind in ("PASS", "CARRY", "RECOVERY"):
                new = e.side
                at = e.start_frame
            elif kind == "SHOT":
                new = e.side
                at = e.start_frame
                timeline.append((row_of(at), new))
                if "GOAL" in subs or "OUT" in subs or "OFF TARGET" in subs:
                    timeline.append((row_of(e.end_frame), "D"))
                continue
            elif kind == "BALL LOST":
                if "OFFSIDE" in subs:
                    timeline.append((row_of(e.end_frame), "D"))
                continue
            elif kind in ("BALL OUT", "FAULT RECEIVED"):
                new, at = "D", e.start_frame
            else:
                continue
            timeline.append((row_of(at), new))

        timeline.sort(key=lambda t: t[0])
        rows = in_period
        for (r0, lab), (r1, _) in zip(timeline, timeline[1:] + [(int(rows[-1]) + 1, None)]):
            lo, hi = max(r0, int(rows[0])), min(r1, int(rows[-1]) + 1)
            if hi > lo:
                labels[order[lo:hi]] = lab
    return labels


def load_metrica_sample(match_id: int, use_cache: bool = True) -> Tracking:
    """Metrica sample game 1, 2 or 3 as Tracking, labels derived from its events."""
    cached = paths.CACHE_DIR / f"metrica{match_id}_metrica_tracking.npz"
    if use_cache and cached.is_file():
        return Tracking.load(cached)
    from kloppy import metrica
    dataset = metrica.load_open_data(match_id=match_id, coordinates="metrica")
    tracking = tracking_from_kloppy(dataset, game=f"metrica{match_id}", source="metrica",
                                    require_labels=False)
    events = metrica_events(match_id, home_team_id=dataset.metadata.teams[0].team_id)

    df_ids = tracking.meta["frame_ids"]
    frame_ids = np.asarray(df_ids["frame_id"], dtype=np.int64)
    period_ids = np.asarray(df_ids["period_id"], dtype=np.int64)
    rows = np.asarray(df_ids["row"], dtype=np.int64)
    per_record = labels_from_metrica_events(events, frame_ids, period_ids)
    labels = np.full(tracking.n_frames, "X", dtype="<U1")
    labels[rows] = per_record
    tracking.labels = labels
    tracking.meta.pop("frame_ids")
    tracking.meta["n_events"] = int(len(events))
    if use_cache:
        tracking.save(cached)
    return tracking


def write_corpus_gt(tracking: Tracking, out_dir: Path = CORPUS_GT_DIR) -> Path:
    """frame,label CSV for the official scorer (`--gt <path>`), X outside the graded blocks."""
    from .labels import write_submission
    return write_submission(Path(out_dir) / f"{tracking.game}_gt.csv", tracking.labels)


def load_external(match_id: str, use_cache: bool = True) -> Tracking:
    """Dispatch on the id: 'J03WPY' (Sportec) or 'metrica3' (Metrica sample)."""
    if match_id.startswith("metrica"):
        return load_metrica_sample(int(match_id[len("metrica"):]), use_cache=use_cache)
    return load_sportec(match_id, use_cache=use_cache)
