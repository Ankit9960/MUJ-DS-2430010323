"""MIT-BIH preprocessing: filtering, beat segmentation, AAMI labelling, RR features.

Produces one .npz cache per record so that the expensive wfdb read and filtering
happen exactly once. Everything downstream reads the cache.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from src import config as C


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def bandpass_filter(sig: np.ndarray, fs: int = C.FS) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Operates on (n_samples, n_channels)."""
    nyq = 0.5 * fs
    low, high = C.BANDPASS[0] / nyq, C.BANDPASS[1] / nyq
    b, a = butter(C.FILTER_ORDER, [low, high], btype="band")
    # filtfilt along the time axis; padlen guards against very short signals.
    padlen = min(3 * max(len(a), len(b)), sig.shape[0] - 1)
    return filtfilt(b, a, sig, axis=0, padlen=padlen)


def zscore(sig: np.ndarray) -> np.ndarray:
    """Per-channel z-score. Normalising per record removes inter-patient gain
    differences that would otherwise act as a patient fingerprint."""
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (sig - mu) / sd


# --------------------------------------------------------------------------
# RR interval features
# --------------------------------------------------------------------------
def rr_features(r_peaks: np.ndarray, fs: int = C.FS) -> np.ndarray:
    """Four RR-interval features per beat, in seconds / ratios.

    Returns (n_beats, 4): [pre_RR, post_RR, local_RR, pre_RR / local_RR].
    local_RR is a causal 10-beat moving average, which is what a clinician
    effectively uses to judge whether a beat arrived 'early'.
    """
    n = len(r_peaks)
    if n == 0:
        return np.zeros((0, C.N_RR_FEATURES), dtype=np.float32)

    rr = np.diff(r_peaks) / fs                      # (n-1,)
    pre = np.concatenate([[rr[0] if n > 1 else 0.8], rr])          # (n,)
    post = np.concatenate([rr, [rr[-1] if n > 1 else 0.8]])        # (n,)

    local = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 10)
        window = pre[lo:i + 1]
        local[i] = window.mean() if len(window) else pre[i]

    ratio = np.divide(pre, local, out=np.ones_like(pre), where=local > 1e-6)
    feats = np.stack([pre, post, local, ratio], axis=1)
    return np.clip(feats, 0.0, 4.0).astype(np.float32)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def segment_record(signal: np.ndarray,
                   r_peaks: np.ndarray,
                   symbols: list[str]) -> dict:
    """Cut beat windows and rhythm-context windows around each annotated R peak.

    Parameters
    ----------
    signal : (n_samples, 2) already filtered and z-scored
    r_peaks : sample index of each annotation
    symbols : annotation symbol for each r_peak

    Returns dict of arrays, keeping only beats that (a) map to an AAMI class and
    (b) have both windows fully inside the recording.
    """
    n_samples = signal.shape[0]
    beats, ctxs, labels, kept_peaks = [], [], [], []

    for peak, sym in zip(r_peaks, symbols):
        if sym in C.NON_BEAT_SYMBOLS:
            continue
        aami = C.SYMBOL_TO_AAMI.get(sym)
        if aami is None:
            continue  # unmapped symbol -> drop rather than silently mislabel

        b0, b1 = peak - C.BEAT_PRE, peak + C.BEAT_POST
        c0, c1 = peak - C.CTX_PRE, peak + C.CTX_POST
        if b0 < 0 or b1 > n_samples or c0 < 0 or c1 > n_samples:
            continue  # edge beat: skip instead of zero-padding

        beats.append(signal[b0:b1, :])
        ctxs.append(signal[c0:c1:C.CTX_DECIM, :])
        labels.append(C.CLASS_TO_IDX[aami])
        kept_peaks.append(peak)

    if not beats:
        return dict(
            beats=np.zeros((0, C.BEAT_LEN, 2), np.float32),
            context=np.zeros((0, C.CTX_LEN, 2), np.float32),
            rr=np.zeros((0, C.N_RR_FEATURES), np.float32),
            labels=np.zeros((0,), np.int64),
        )

    kept_peaks = np.asarray(kept_peaks)
    return dict(
        beats=np.asarray(beats, dtype=np.float32),        # (n, 360, 2)
        context=np.asarray(ctxs, dtype=np.float32),       # (n, 360, 2)
        rr=rr_features(kept_peaks),                       # (n, 4)
        labels=np.asarray(labels, dtype=np.int64),        # (n,)
    )


def process_record(record_id: int, raw_dir=None) -> dict:
    """Read one MIT-BIH record with wfdb and return its segmented beats."""
    import wfdb  # imported lazily so the rest of the pipeline is testable without it

    raw_dir = raw_dir or C.DATA_RAW
    path = str(raw_dir / str(record_id))

    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    sig = np.asarray(rec.p_signal, dtype=np.float64)  # (n_samples, 2)
    if sig.shape[1] < 2:
        sig = np.repeat(sig[:, :1], 2, axis=1)

    # Put the limb lead (MLII where available) in channel 0 for consistency.
    names = [str(s).strip() for s in (rec.sig_name or [])]
    if "MLII" in names:
        idx = names.index("MLII")
        order = [idx] + [i for i in range(sig.shape[1]) if i != idx]
        sig = sig[:, order]
    elif record_id in C.REVERSED_LEAD_RECORDS:
        sig = sig[:, ::-1]

    sig = zscore(bandpass_filter(sig[:, :2]))
    out = segment_record(sig, np.asarray(ann.sample), list(ann.symbol))
    out["record"] = np.full(len(out["labels"]), record_id, dtype=np.int64)
    return out


def cache_path(record_id: int) -> "C.Path":
    return C.DATA_PROC / f"{record_id}.npz"


def build_cache(record_ids=None, force: bool = False, verbose: bool = True) -> None:
    """Segment every record once and store it as .npz."""
    record_ids = record_ids or C.ALL_RECORDS
    for rid in record_ids:
        out_file = cache_path(rid)
        if out_file.exists() and not force:
            if verbose:
                print(f"  [skip] {rid} (cached)")
            continue
        data = process_record(rid)
        np.savez_compressed(out_file, **data)
        if verbose:
            counts = np.bincount(data["labels"], minlength=len(C.AAMI_CLASSES))
            dist = " ".join(f"{c}={n}" for c, n in zip(C.AAMI_CLASSES, counts))
            print(f"  [ok]   {rid}: {len(data['labels']):>6} beats | {dist}")


def load_records(record_ids) -> dict:
    """Concatenate cached beats for a list of records."""
    parts = {k: [] for k in ("beats", "context", "rr", "labels", "record")}
    for rid in record_ids:
        f = cache_path(rid)
        if not f.exists():
            raise FileNotFoundError(
                f"No cache for record {rid}. Run `python scripts/prepare_data.py` first."
            )
        with np.load(f) as z:
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
