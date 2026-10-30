"""Load the provided assets into one canonical, match-agnostic representation.

Everything downstream (rules, features, models, decoder) consumes a `Tracking` object: ball and
player positions in metres on a (length, width) pitch, a team per track slot, and -- for external
training matches -- a per-frame label. External loaders (Sportec, Metrica sample data) produce the
same object, so the pipeline never knows which match it is looking at.

Raw ATD line format (Metrica FIFA/EPTS export), one line per video frame:

    <frame_id>:<x,y,speed>;<x,y,speed>;...(one group per slot)...:<ball x,y,speed>

Missing observations are NaN. frame_id counts from 1; submissions count from 0.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import paths  # noqa: F401  sets MTAH_DATA_DIR before games is imported
import games

HOME, AWAY = 0, 1
TEAM_LETTER = {HOME: "H", AWAY: "A"}

_CHUNK_LINES = 20000


@dataclass
class Tracking:
    """One match (or one recording of it) in metres, one record per video frame."""

    ball: np.ndarray            # (N, 3) x_m, y_m, speed_mps; NaN where not observed
    players: np.ndarray         # (N, S, 3) x_m, y_m, speed_mps per track slot
    team_of_slot: np.ndarray    # (S,) HOME or AWAY
    fps: float
    pitch: tuple                # (length_m, width_m)
    game: str
    source: str
    labels: Optional[np.ndarray] = None   # (N,) '<U1' in {H, A, D, X}; external matches only
    blocks: Optional[np.ndarray] = None   # (K, 3) int64 rows (half, first_frame, last_frame)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.players.shape[1])

    @property
    def ball_present(self) -> np.ndarray:
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_present(self) -> np.ndarray:
        return ~np.isnan(self.players[:, :, 0])

    def graded_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.blocks:
            mask[first:last + 1] = True
        return mask

    def half_of_frame(self) -> np.ndarray:
        """(N,) int8: half number inside a graded block, 0 elsewhere."""
        half = np.zeros(self.n_frames, dtype=np.int8)
        for h, first, last in self.blocks:
            half[first:last + 1] = h
        return half

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ball": self.ball, "players": self.players, "team_of_slot": self.team_of_slot,
            "fps": np.float64(self.fps), "pitch": np.asarray(self.pitch, dtype=np.float64),
            "game": np.array(self.game), "source": np.array(self.source),
            "blocks": np.asarray(self.blocks, dtype=np.int64),
            "meta_json": np.array(json.dumps(self.meta)),
        }
        if self.labels is not None:
            payload["labels"] = self.labels
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> "Tracking":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                ball=z["ball"], players=z["players"], team_of_slot=z["team_of_slot"],
                fps=float(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                game=str(z["game"]), source=str(z["source"]),
                labels=z["labels"] if "labels" in z.files else None,
                blocks=z["blocks"] if "blocks" in z.files else None,
                meta=json.loads(str(z["meta_json"])) if "meta_json" in z.files else {},
            )


# --------------------------------------------------------------------------------------------
# ATD metadata
# --------------------------------------------------------------------------------------------

def read_atd_meta(meta_path) -> dict:
    """Teams and slot->team mapping from the FIFA/EPTS metadata XML.

    The first team listed is the home side (EPTS convention; Barcelona for BAR-ATH).
    """
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    home_id = teams[0][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    team_of_slot = np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)
    frame_rate = root.findtext(".//FrameRate")
    return {
        "home_name": teams[0][1], "away_name": teams[1][1],
        "team_of_slot": team_of_slot, "n_slots": len(slots),
        "fps": float(frame_rate) if frame_rate else None,
    }


# --------------------------------------------------------------------------------------------
# Raw ATD parsing
# --------------------------------------------------------------------------------------------

def _parse_chunk(lines, n_fields):
    text = "".join(lines).replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, na_values=["NaN"])
    if table.shape[1] != n_fields:
        raise ValueError(f"ATD line has {table.shape[1]} fields, expected {n_fields}")
    return table.to_numpy()


def parse_atd_raw(raw_path, n_slots: int):
    """(frame_ids, players_norm (N,S,3), ball_norm (N,3)) in the file's 0-1 coordinates."""
    n_fields = 1 + 3 * n_slots + 3
    expected_commas = n_fields - 1
    blocks, buffer = [], []
    with open(raw_path) as fh:
        for line in fh:
            if line.count(",") + line.count(";") + line.count(":") != expected_commas:
                frame = line.split(":", 1)[0]
                raise ValueError(f"ATD frame {frame}: unexpected field count")
            buffer.append(line)
            if len(buffer) >= _CHUNK_LINES:
                blocks.append(_parse_chunk(buffer, n_fields))
                buffer = []
    if buffer:
        blocks.append(_parse_chunk(buffer, n_fields))
    values = np.concatenate(blocks, axis=0)
    frame_ids = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:]
    return frame_ids, players, ball


def _to_metres(xyz: np.ndarray, pitch) -> np.ndarray:
    out = xyz.copy()
    out[..., 0] *= pitch[0]
    out[..., 1] *= pitch[1]
    return out


# --------------------------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------------------------

def cache_path(game: str, source: str) -> Path:
    return paths.CACHE_DIR / f"{game}_{source}_tracking.npz"


def load_tracking(game: str = games.GAME, source: str = games.DEFAULT_SOURCE,
                  use_cache: bool = True) -> Tracking:
    """The provided ATD for one recording, in metres, one record per video frame (from 0).

    Verifies that frame ids are exactly 1..N for that recording's video length, so the result
    can be indexed by submission row directly.
    """
    cached = cache_path(game, source)
    if use_cache and cached.is_file():
        tracking = Tracking.load(cached)
        if tracking.blocks is not None:
            return tracking
        # cache written before `blocks` existed: rebuild

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, None, source)
    meta = read_atd_meta(meta_path)
    frame_ids, players_norm, ball_norm = parse_atd_raw(raw_path, meta["n_slots"])

    n_expected = games.source_frames(source, game)
    if len(frame_ids) != n_expected or not np.array_equal(
            frame_ids, np.arange(1, n_expected + 1)):
        raise ValueError(
            f"{raw_path}: expected frame ids 1..{n_expected}, got {len(frame_ids)} records "
            f"spanning {frame_ids.min()}..{frame_ids.max()}")

    pitch = tuple(games.GAMES[game]["pitch"])
    tracking = Tracking(
        ball=_to_metres(ball_norm, pitch), players=_to_metres(players_norm, pitch),
        team_of_slot=meta["team_of_slot"], fps=float(games.GAMES[game]["fps"]),
        pitch=pitch, game=game, source=source,
        blocks=np.asarray(games.half_video_blocks(game), dtype=np.int64),
        meta={"home_name": meta["home_name"], "away_name": meta["away_name"]},
    )
    if use_cache:
        tracking.save(cached)
    return tracking


def load_smart_tagging(game: str = games.GAME) -> pd.DataFrame:
    """Smart Tagging instances as a table, one row per instance, intervals in video frames.

    Columns: code, start_s, end_s, start_frame, end_frame (half-open), one column per tag group
    (Team, Half, Type, Side, ...) and `team_side` in {H, A, None} mapped from the ATD team names.
    """
    path = games.require(["smart_tagging"], game)[0]
    fps = games.GAMES[game]["fps"]
    rows = []
    for inst in ET.parse(path).iter("instance"):
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        row = {
            "code": (inst.findtext("code") or "").strip(),
            "start_s": start_s, "end_s": end_s,
            "start_frame": int(round(start_s * fps)), "end_frame": int(round(end_s * fps)),
        }
        for label in inst.findall("label"):
            row[(label.findtext("group") or "").strip()] = (label.findtext("text") or "").strip()
        rows.append(row)
    table = pd.DataFrame(rows)

    meta = read_atd_meta(games.require(["atd_meta"], game)[0])
    names = {meta["home_name"].lower(): "H", meta["away_name"].lower(): "A"}
    if "Team" in table.columns:
        table["team_side"] = table["Team"].str.lower().map(names)
    else:
        table["team_side"] = None
    return table


def load_homography(game: str = games.GAME, source: str = games.DEFAULT_SOURCE):
    """(matrices (N,3,3), reprojection_error (N,), valid (N,) bool), indexed by video frame.

    Matrices map normalised pitch coordinates (0-1) to pixels; NaN where valid is False.
    """
    path = games.require(["homography"], game, None, source)[0]
    table = pd.read_csv(path)
    cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    matrices = table[cols].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return matrices, table["reprojection_error"].to_numpy(), table["valid"].to_numpy().astype(bool)


def project(matrix: np.ndarray, x, y):
    """Normalised pitch (0-1) -> pixels. `unproject` is the same with np.linalg.inv(matrix)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w)


def unproject(matrix: np.ndarray, px, py):
    return project(np.linalg.inv(matrix), px, py)
