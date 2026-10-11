"""The two evaluation protocols this project contrasts.

inter-patient : de Chazal DS1/DS2. Train and test patients are disjoint.
intra-patient : all beats pooled, then split at random. Beats from the same
                patient appear on both sides -- this is the leakage condition.

Keeping both behind one interface makes the comparison a single flag, which is
exactly what the ablation needs.
"""
from __future__ import annotations

import numpy as np

from src import config as C
from src.preprocessing import load_records


def _subset(data: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in data.items()}


def inter_patient_split() -> tuple[dict, dict, dict]:
    """DS1 (minus a held-out validation group) / DS1_VAL / DS2.

    Validation records are themselves disjoint from both the training records
    and DS2, so early stopping never touches a patient it will be scored on.
    """
    train = load_records(C.DS1_TRAIN)
    val = load_records(C.DS1_VAL)
    test = load_records(C.DS2)
    return train, val, test


def intra_patient_split(seed: int = 0,
                        val_frac: float = 0.10,
                        test_frac: float = 0.20) -> tuple[dict, dict, dict]:
    """Pool every record, then split beats at random with stratification.

    This deliberately reproduces the flawed protocol used by a large share of
    the published literature. Because consecutive beats from one patient share
    morphology, baseline wander and electrode placement, a random split lets the
    model memorise the patient rather than learn the arrhythmia.
    """
    data = load_records(C.ALL_RECORDS)
    labels = data["labels"]
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])
        train_idx.append(idx[n_test + n_val:])

    cat = lambda parts: np.sort(np.concatenate(parts))
    return (_subset(data, cat(train_idx)),
            _subset(data, cat(val_idx)),
            _subset(data, cat(test_idx)))


def get_split(protocol: str, seed: int = 0) -> tuple[dict, dict, dict]:
    if protocol == "inter":
        return inter_patient_split()
    if protocol == "intra":
        return intra_patient_split(seed=seed)
    raise ValueError(f"Unknown protocol {protocol!r}; expected 'inter' or 'intra'.")


def patient_overlap(train: dict, test: dict) -> dict:
    """Quantify leakage. Reported in the results so the claim is evidenced,
    not asserted: under 'inter' this must be zero by construction."""
    tr = set(np.unique(train["record"]).tolist())
    te = set(np.unique(test["record"]).tolist())
    shared = sorted(tr & te)
    n_shared_beats = int(np.isin(test["record"], list(shared)).sum()) if shared else 0
    return dict(
        n_train_patients=len(tr),
        n_test_patients=len(te),
        n_shared_patients=len(shared),
        shared_patients=shared,
        test_beats_from_seen_patients=n_shared_beats,
        pct_test_beats_from_seen_patients=round(
            100.0 * n_shared_beats / max(len(test["labels"]), 1), 2),
    )


def class_counts(data: dict) -> dict:
    counts = np.bincount(data["labels"], minlength=len(C.AAMI_CLASSES))
    return {c: int(n) for c, n in zip(C.AAMI_CLASSES, counts)}
