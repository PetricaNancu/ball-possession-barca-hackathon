"""Labels, graded frames and the submission file format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths  # noqa: F401
import games

LABELS = ("H", "A", "D", "X")


def graded_blocks(game: str = games.GAME):
    """[(half, first_video_frame, last_video_frame)] -- the frames that are scored."""
    return games.half_video_blocks(game)


def graded_mask(n_frames: int, game: str = games.GAME) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in graded_blocks(game):
        mask[first:last + 1] = True
    return mask


def half_of_frame(n_frames: int, game: str = games.GAME) -> np.ndarray:
    """(N,) int8: 1 or 2 inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in graded_blocks(game):
        half[first:last + 1] = h
    return half


def mask_ungraded(labels: np.ndarray, game: str = games.GAME) -> np.ndarray:
    """Write X on every frame the ground truth never scores. Costs nothing, avoids noise."""
    out = np.asarray(labels, dtype="<U1").copy()
    out[~graded_mask(len(out), game)] = "X"
    return out


def mask_by_blocks(labels: np.ndarray, blocks) -> np.ndarray:
    """Same as mask_ungraded, for a match whose graded blocks are given explicitly."""
    out = np.asarray(labels, dtype="<U1").copy()
    keep = np.zeros(len(out), dtype=bool)
    for _, first, last in blocks:
        keep[first:last + 1] = True
    out[~keep] = "X"
    return out


def write_submission(path: Path, labels: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype="<U1")
    bad = set(np.unique(labels)) - set(LABELS)
    if bad:
        raise ValueError(f"invalid labels {sorted(bad)}")
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(
        path, index=False, lineterminator="\n")
    return path


def read_submission(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    return table["label"].astype(str).str.strip().str.upper().to_numpy(dtype="<U1")


def spells(labels: np.ndarray):
    """[(label, start, end_exclusive)] maximal runs of H or A (dead ball and X break runs)."""
    labels = np.asarray(labels, dtype="<U1")
    out, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else None
        if start is not None and (cur != labels[start]):
            if labels[start] in ("H", "A"):
                out.append((labels[start], start, i))
            start = None
        if start is None and cur in ("H", "A"):
            start = i
    return out
