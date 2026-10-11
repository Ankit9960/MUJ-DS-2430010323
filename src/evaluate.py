"""Evaluation metrics, reported the way the ECG literature reports them.

AAMI EC57 scores each arrhythmia class with sensitivity (Se) and positive
predictivity (P+), not with overall accuracy, because overall accuracy on a
90%-normal dataset is close to meaningless. Both are computed here so the
results table can be placed directly alongside published work.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from src import config as C


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int = 5) -> dict:
    """One-vs-rest Se / P+ / F1 / specificity for every AAMI class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    out = {}
    for i, cls in enumerate(C.AAMI_CLASSES[:n_classes]):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        se = tp / (tp + fn) if (tp + fn) else float("nan")     # recall
        pp = tp / (tp + fp) if (tp + fp) else float("nan")     # precision
        sp = tn / (tn + fp) if (tn + fp) else float("nan")
        f1 = (2 * se * pp / (se + pp)) if (se and pp and not np.isnan(se * pp)) else 0.0
        out[cls] = dict(support=int(tp + fn), tp=tp, fp=fp, fn=fn,
                        sensitivity=se, positive_predictivity=pp,
                        specificity=sp, f1=f1)
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> dict:
    """Full metric bundle for one run."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    present = np.unique(y_true)
    per_class = per_class_metrics(y_true, y_pred, n_classes)

    # Macro-F1 over classes that actually occur in the reference labels;
    # averaging over absent classes would silently drag the score down.
    macro_f1 = float(f1_score(y_true, y_pred, labels=present,
                              average="macro", zero_division=0))

    return dict(
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=macro_f1,
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        # The headline clinical numbers, pulled out for the comparison table.
        S_sensitivity=per_class["S"]["sensitivity"],
        S_positive_predictivity=per_class["S"]["positive_predictivity"],
        V_sensitivity=per_class["V"]["sensitivity"],
        V_positive_predictivity=per_class["V"]["positive_predictivity"],
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        n_test_beats=int(len(y_true)),
    )


def aggregate_seeds(runs: list[dict]) -> dict:
    """Mean and standard deviation across seeds for the scalar metrics."""
    scalar_keys = ["accuracy", "macro_f1", "kappa",
                   "S_sensitivity", "S_positive_predictivity",
                   "V_sensitivity", "V_positive_predictivity"]
    agg = {}
    for k in scalar_keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))

    per_cls = {}
    for cls in C.AAMI_CLASSES:
        for metric in ("sensitivity", "positive_predictivity", "f1"):
            vals = np.array([r["per_class"][cls][metric] for r in runs], dtype=float)
            per_cls.setdefault(cls, {})[f"{metric}_mean"] = float(np.nanmean(vals))
            per_cls.setdefault(cls, {})[f"{metric}_std"] = float(np.nanstd(vals))
    agg["per_class"] = per_cls

    cms = np.array([r["confusion_matrix"] for r in runs])
    agg["confusion_matrix_sum"] = cms.sum(axis=0).tolist()
    agg["confusion_matrix_mean"] = cms.mean(axis=0).round(1).tolist()
    agg["n_seeds"] = len(runs)
    return agg


def format_confusion_matrix(cm, normalise: bool = False) -> str:
    """Plain-text confusion matrix for the console and the markdown report."""
    cm = np.asarray(cm, dtype=float)
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100

    header = "true\\pred " + "".join(f"{c:>10}" for c in C.AAMI_CLASSES)
    lines = [header, "-" * len(header)]
    for i, cls in enumerate(C.AAMI_CLASSES):
        cells = "".join(f"{v:>10.1f}" if normalise else f"{int(v):>10d}"
                        for v in cm[i])
        lines.append(f"{cls:<10}" + cells)
    return "\n".join(lines)
