"""Emission model: per-frame P(H), P(A), P(D) from the feature vector (LightGBM, 3 classes).

Trained only on degraded external matches. Every training match is also fed with the two teams
swapped (features via `swap_teams`, labels via `swap_labels`), which removes any home/away bias
and doubles the data. Deterministic given the same data and parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import swap_labels, swap_teams

DEFAULT_PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "deterministic": True,
    "force_row_wise": True, "num_threads": 8, "verbose": -1, "seed": 7,
}


def training_matrix(parts, stride: int = 2, augment: bool = True):
    """Stack (X, y) of graded, labelled frames from several matches, subsampled by `stride`,
    optionally with the team-swapped copy appended."""
    xs, ys = [], []
    for X, y, names in parts:
        keep = np.flatnonzero(y >= 0)[::stride]
        xs.append(X[keep])
        ys.append(y[keep])
        if augment:
            xs.append(swap_teams(X[keep], names))
            ys.append(swap_labels(y[keep]))
    return np.concatenate(xs), np.concatenate(ys)


def train(X: np.ndarray, y: np.ndarray, names: list[str], n_rounds: int = 300, params=None):
    import lightgbm as lgb
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    dataset = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True)
    return lgb.train(p, dataset, num_boost_round=n_rounds)


def predict_proba(booster, X: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities; NaN rows (ungraded) come back as NaN."""
    out = np.full((len(X), 3), np.nan, dtype=np.float32)
    ok = ~np.all(np.isnan(X), axis=1)
    if ok.any():
        out[ok] = booster.predict(X[ok], num_threads=DEFAULT_PARAMS["num_threads"])
    return out


def save(booster, names: list[str], folder: Path, n_rounds: int, params: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(folder / "engine_lgbm.txt"))
    (folder / "engine_features.json").write_text(json.dumps(
        {"features": names, "n_rounds": n_rounds, "params": params}, indent=2))


def load(folder: Path):
    import lightgbm as lgb
    folder = Path(folder)
    booster = lgb.Booster(model_file=str(folder / "engine_lgbm.txt"))
    meta = json.loads((folder / "engine_features.json").read_text())
    return booster, meta["features"]
