"""Synthetic ECG generator -- used ONLY for the smoke test.

Its purpose is to prove the pipeline runs end to end (shapes, losses, metrics,
plots) in under a minute, without waiting on a PhysioNet download. It carries a
deliberate patient effect so that the intra/inter contrast is reproduced
qualitatively, which makes it a useful sanity check for the leakage experiment.

Numbers produced from synthetic data are NOT results and must never appear in
the report. `scripts/run_experiment.py --smoke-test` stamps every output it
writes with `synthetic=True` for this reason.
"""
from __future__ import annotations

import numpy as np

from src import config as C


def _wave(t: np.ndarray, centre: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)


def _beat_template(kind: str, n: int, rng: np.random.Generator,
                   patient_shift: float, patient_gain: float) -> np.ndarray:
    """Crude P-QRS-T morphology, distorted per AAMI class."""
    t = np.linspace(-0.5, 0.5, n)
    p_amp, qrs_w, qrs_amp, t_amp = 0.12, 0.022, 1.0, 0.22
    p_centre = -0.20 + patient_shift

    if kind == "S":          # premature atrial: P buried/early, QRS normal
        p_amp, p_centre = 0.05, -0.13 + patient_shift
    elif kind == "V":        # ventricular: wide bizarre QRS, no P, opposite T
        p_amp, qrs_w, qrs_amp, t_amp = 0.0, 0.055, 1.25, -0.35
    elif kind == "F":        # fusion: halfway between N and V
        p_amp, qrs_w, qrs_amp, t_amp = 0.06, 0.038, 1.1, -0.10
    elif kind == "Q":        # paced / unknown: narrow spike + blunt T
        p_amp, qrs_w, qrs_amp, t_amp = 0.0, 0.012, 1.4, 0.05

    sig = (_wave(t, p_centre, 0.035, p_amp)
           + _wave(t, 0.0 + patient_shift * 0.2, qrs_w, qrs_amp)
           - _wave(t, 0.035 + patient_shift * 0.2, 0.02, 0.28 * qrs_amp)
           + _wave(t, 0.22 + patient_shift, 0.06, t_amp))
    sig = sig * patient_gain
    sig += 0.05 * rng.standard_normal(n)                     # sensor noise
    sig += 0.08 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))  # wander
    return sig


def make_synthetic_records(n_records: int = 12,
                           beats_per_record: int = 260,
                           seed: int = 0) -> dict:
    """Return a dict with the same keys and shapes as a real cached record set."""
    rng = np.random.default_rng(seed)
    # Realistically skewed prior: N dominates, F and Q are rare.
    probs = np.array([0.83, 0.06, 0.08, 0.015, 0.015])

    beats, ctxs, rrs, labels, records = [], [], [], [], []

    for r in range(n_records):
        # Patient-level nuisance factors -- the thing a leaky split lets a model
        # memorise instead of learning genuine arrhythmia morphology.
        shift = rng.uniform(-0.05, 0.05)
        gain = rng.uniform(0.75, 1.30)
        base_rr = rng.uniform(0.7, 1.0)

        cls_idx = rng.choice(len(C.AAMI_CLASSES), size=beats_per_record, p=probs)
        peaks = np.cumsum(rng.normal(base_rr, 0.05, beats_per_record) * C.FS).astype(int)

        for k in cls_idx:
            kind = C.AAMI_CLASSES[k]
            beat = _beat_template(kind, C.BEAT_LEN, rng, shift, gain)
            ctx = _beat_template(kind, C.CTX_LEN, rng, shift, gain)
            beats.append(np.stack([beat, 0.6 * beat + 0.04 * rng.standard_normal(C.BEAT_LEN)], 1))
            ctxs.append(np.stack([ctx, 0.6 * ctx + 0.04 * rng.standard_normal(C.CTX_LEN)], 1))
            labels.append(k)
            records.append(100 + r)

        from src.preprocessing import rr_features
        rrs.append(rr_features(peaks))

    return dict(
        beats=np.asarray(beats, np.float32),
        context=np.asarray(ctxs, np.float32),
        rr=np.concatenate(rrs, axis=0).astype(np.float32),
        labels=np.asarray(labels, np.int64),
        record=np.asarray(records, np.int64),
    )


def synthetic_split(protocol: str, seed: int = 0):
    """Mimic get_split() on synthetic data, honouring the patient boundary."""
    data = make_synthetic_records(seed=seed)
    recs = np.unique(data["record"])
    sub = lambda m: {k: v[m] for k, v in data.items()}

    if protocol == "inter":
        tr_r, va_r, te_r = recs[:7], recs[7:9], recs[9:]
        return (sub(np.isin(data["record"], tr_r)),
                sub(np.isin(data["record"], va_r)),
                sub(np.isin(data["record"], te_r)))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["labels"]))
    n = len(idx)
    te, va = idx[:int(0.2 * n)], idx[int(0.2 * n):int(0.3 * n)]
    tr = idx[int(0.3 * n):]
    return sub(tr), sub(va), sub(te)
