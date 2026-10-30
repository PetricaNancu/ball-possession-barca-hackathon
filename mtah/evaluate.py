"""In-memory scoring with the organisers' metric library (no CSV round trip)."""

from __future__ import annotations

import numpy as np

import possession_metrics as pm  # vendored, on sys.path via mtah.paths


def score(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Official metrics for one match. Both arrays are full-length label strings (X allowed)."""
    gt = pm.label_codes(np.asarray(gt_labels, dtype=object))
    pred = pm.label_codes(np.asarray(pred_labels, dtype=object))
    r = pm.evaluate(gt, pred)
    return {
        "spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
        "f1_H": float(r["per_class"]["H"]["f1"]), "f1_A": float(r["per_class"]["A"]["f1"]),
        "f1_D": float(r["per_class"]["D"]["f1"]), "recall_D": float(r["per_class"]["D"]["recall"]),
        "precision_D": float(r["per_class"]["D"]["precision"]),
        "n_pred_spells": int(r["n_pred_spells"]), "n_gt_spells": int(r["n_gt_spells"]),
        "mean_spell_s": float(r["mean_spell_duration_s"]),
        "gt_mean_spell_s": float(r["gt_mean_spell_duration_s"]),
    }


def summarize(rows: list[dict], keys=("spell_f1", "macro_f1", "f1_D", "recall_D")) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
