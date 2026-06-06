from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "sen": float(sen),
        "spe": float(spe),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if prob is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, prob[:, 1]))
        except ValueError:
            out["auc"] = float("nan")
    return out


def summarize_metrics(fold_metrics):
    keys = sorted(fold_metrics[0].keys())
    return {key: (float(np.nanmean([m[key] for m in fold_metrics])), float(np.nanstd([m[key] for m in fold_metrics]))) for key in keys}
