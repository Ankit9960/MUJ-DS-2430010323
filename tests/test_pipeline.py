"""Tests for every component that does not require PyTorch.

Run with:  python -m pytest tests/ -v      (or)   python tests/test_pipeline.py

These cover the parts where a silent bug would corrupt the results without
crashing: label mapping, window alignment, split integrity and the metric
definitions. The torch model/training loop is exercised separately by
`python scripts/run_experiment.py --smoke-test`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config as C
from src.preprocessing import bandpass_filter, zscore, rr_features, segment_record
from src.synthetic import make_synthetic_records, synthetic_split
from src.evaluate import evaluate, per_class_metrics, aggregate_seeds, format_confusion_matrix


# --------------------------------------------------------------------------
def test_config_integrity():
    assert len(C.DS1) == 22 and len(C.DS2) == 22
    assert not set(C.DS1) & set(C.DS2), "DS1 and DS2 must be disjoint patients"
    assert not set(C.ALL_RECORDS) & set(C.PACED_RECORDS), "paced records must be excluded"
    assert set(C.DS1_TRAIN) | set(C.DS1_VAL) == set(C.DS1)
    assert not set(C.DS1_TRAIN) & set(C.DS1_VAL)
    assert not set(C.DS1_VAL) & set(C.DS2), "validation must not touch test patients"
    assert C.BEAT_LEN == 360 and C.CTX_LEN == 360
    # Every mapped symbol lands in a real AAMI class.
    assert set(C.SYMBOL_TO_AAMI.values()) <= set(C.AAMI_CLASSES)
    print("  [ok] config integrity: DS1/DS2 disjoint, val disjoint from both")


def test_bandpass_removes_baseline_wander():
    t = np.linspace(0, 10, 3600)
    wander = 3.0 * np.sin(2 * np.pi * 0.05 * t)          # 0.05 Hz, below the passband
    beatish = np.sin(2 * np.pi * 10 * t)                  # 10 Hz, inside the passband
    sig = (wander + beatish)[:, None]
    out = bandpass_filter(sig)
    # Wander should be strongly attenuated while the 10 Hz content survives.
    assert out.std() < sig.std(), "filter did not reduce total variance"
    assert np.corrcoef(out[:, 0], beatish)[0, 1] > 0.9, "in-band signal was destroyed"
    print(f"  [ok] bandpass: corr with in-band signal "
          f"{np.corrcoef(out[:,0], beatish)[0,1]:.3f}")


def test_zscore():
    x = np.random.default_rng(0).normal(5, 3, (1000, 2))
    z = zscore(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1, atol=1e-9)
    # Constant channel must not divide by zero.
    assert np.isfinite(zscore(np.ones((100, 1)))).all()
    print("  [ok] zscore: mean 0, std 1, constant-channel safe")


def test_rr_features():
    peaks = np.arange(0, 20) * 360          # exactly 1.0 s apart at 360 Hz
    f = rr_features(peaks)
    assert f.shape == (20, C.N_RR_FEATURES)
    assert np.allclose(f[5:, 0], 1.0, atol=1e-6), "pre-RR should be 1.0 s"
    assert np.allclose(f[5:15, 3], 1.0, atol=1e-6), "pre/local ratio should be 1.0"
    # A premature beat must show a pre-RR ratio well below 1.
    peaks2 = np.array([0, 360, 720, 900, 1260, 1620])
    f2 = rr_features(peaks2)
    assert f2[3, 3] < 0.85, f"premature beat ratio not detected: {f2[3,3]:.3f}"
    assert rr_features(np.array([])).shape == (0, C.N_RR_FEATURES)
    print(f"  [ok] rr_features: premature-beat ratio {f2[3,3]:.3f} (< 0.85)")


def test_segment_record_alignment_and_labels():
    n = 40_000
    sig = np.zeros((n, 2))
    peaks = np.arange(2000, n - 2000, 700)
    # Put a unique spike at each R peak so window centring is verifiable.
    for i, p in enumerate(peaks):
        sig[p, :] = i + 1

    symbols = []
    for i in range(len(peaks)):
        symbols.append(["N", "V", "A", "F", "/"][i % 5])

    out = segment_record(sig, peaks, symbols)
    assert out["beats"].shape[1:] == (C.BEAT_LEN, 2)
    assert out["context"].shape[1:] == (C.CTX_LEN, 2)
    assert len(out["labels"]) == len(out["beats"]) == len(out["rr"])
    # The R peak must sit exactly at index BEAT_PRE of the beat window.
    assert out["beats"][0, C.BEAT_PRE, 0] != 0, "R peak is not at the window centre"
    assert out["beats"][0, C.BEAT_PRE - 1, 0] == 0
    # Label mapping: N->N, V->V, A->S, F->F, /->Q
    expected = [C.CLASS_TO_IDX[c] for c in ["N", "V", "S", "F", "Q"]]
    assert list(out["labels"][:5]) == expected, out["labels"][:5]
    print(f"  [ok] segmentation: {len(out['labels'])} beats, R peak centred, "
          f"symbols mapped to AAMI")


def test_segment_drops_non_beats_and_edges():
    n = 5000
    sig = np.random.default_rng(1).normal(size=(n, 2))
    # First peak is too close to the start; '+' is a rhythm marker, not a beat.
    peaks = np.array([10, 2000, 2700, 4995])
    symbols = ["N", "+", "N", "N"]
    out = segment_record(sig, peaks, symbols)
    assert len(out["labels"]) == 1, f"expected 1 usable beat, got {len(out['labels'])}"
    print("  [ok] segmentation drops edge beats and non-beat annotations")


def test_synthetic_split_respects_patients():
    tr, va, te = synthetic_split("inter", seed=0)
    shared = set(np.unique(tr["record"])) & set(np.unique(te["record"]))
    assert not shared, f"inter-patient split leaked patients: {shared}"

    tr2, va2, te2 = synthetic_split("intra", seed=0)
    shared2 = set(np.unique(tr2["record"])) & set(np.unique(te2["record"]))
    assert shared2, "intra-patient split should share patients (that is the point)"
    print(f"  [ok] splits: inter shares 0 patients, intra shares {len(shared2)}")


def test_synthetic_shapes():
    d = make_synthetic_records(n_records=4, beats_per_record=50, seed=0)
    n = len(d["labels"])
    assert d["beats"].shape == (n, C.BEAT_LEN, 2)
    assert d["context"].shape == (n, C.CTX_LEN, 2)
    assert d["rr"].shape == (n, C.N_RR_FEATURES)
    assert d["record"].shape == (n,)
    print(f"  [ok] synthetic generator: {n} beats with consistent shapes")


def test_metrics_against_hand_worked_example():
    # 2 classes used, hand-computable.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 2, 2, 0])
    m = per_class_metrics(y_true, y_pred)
    # Class 0: tp=3, fn=1, fp=2 -> Se 3/4, P+ 3/5
    assert abs(m["N"]["sensitivity"] - 0.75) < 1e-9
    assert abs(m["N"]["positive_predictivity"] - 0.6) < 1e-9
    # Class 1 (S): tp=2, fn=1, fp=1 -> Se 2/3, P+ 2/3
    assert abs(m["S"]["sensitivity"] - 2 / 3) < 1e-9
    assert abs(m["S"]["positive_predictivity"] - 2 / 3) < 1e-9
    # Class 2 (V): tp=2, fn=1, fp=0 -> Se 2/3, P+ 1.0
    assert abs(m["V"]["positive_predictivity"] - 1.0) < 1e-9
    res = evaluate(y_true, y_pred)
    assert abs(res["accuracy"] - 0.7) < 1e-9
    assert np.array(res["confusion_matrix"]).shape == (5, 5)
    print(f"  [ok] metrics match hand-worked values (acc {res['accuracy']:.2f})")


def test_metrics_absent_class_does_not_crash():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    res = evaluate(y_true, y_pred)          # classes F and Q never occur
    assert abs(res["accuracy"] - 1.0) < 1e-9
    assert abs(res["macro_f1"] - 1.0) < 1e-9, res["macro_f1"]
    assert np.isnan(res["per_class"]["F"]["sensitivity"])
    print("  [ok] absent classes handled without crashing or skewing macro F1")


def test_aggregate_seeds():
    rng = np.random.default_rng(0)
    runs = [evaluate(rng.integers(0, 5, 400), rng.integers(0, 5, 400)) for _ in range(3)]
    agg = aggregate_seeds(runs)
    assert agg["n_seeds"] == 3
    assert "accuracy_mean" in agg and "accuracy_std" in agg
    assert np.array(agg["confusion_matrix_sum"]).shape == (5, 5)
    assert np.array(agg["confusion_matrix_sum"]).sum() == 1200
    print("  [ok] seed aggregation: mean/std and summed confusion matrix")


def test_format_confusion_matrix():
    cm = np.arange(25).reshape(5, 5)
    txt = format_confusion_matrix(cm)
    assert "true\\pred" in txt and all(c in txt for c in C.AAMI_CLASSES)
    pct = format_confusion_matrix(cm, normalise=True)
    assert "%" not in pct  # values only; header supplies the unit
    print("  [ok] confusion-matrix formatting")


def test_sota_table():
    from src import sota
    md = sota.markdown_table()
    assert md.count("\n") >= len(sota.SOTA_INTER_PATIENT) + 1
    assert "de Chazal" in md and "86.2" in md
    assert sota.PROTOCOL_EVIDENCE["delta"] > 10
    print("  [ok] SOTA comparison table renders")


def test_figures_render():
    """Every figure must render without error on representative data."""
    from src import plots
    tmp = C.FIGURES / "_selftest"
    rng = np.random.default_rng(0)

    cm = np.array([[44000, 300, 250, 30, 5],
                   [900, 1100, 120, 10, 2],
                   [400, 90, 5800, 120, 8],
                   [300, 20, 180, 250, 3],
                   [60, 5, 20, 4, 40]])
    plots.plot_confusion_matrix(cm, "self-test", tmp / "cm.png")

    mk = lambda a, s, v: dict(accuracy_mean=a, accuracy_std=0.004,
                              macro_f1_mean=a - 0.15, macro_f1_std=0.01,
                              S_sensitivity_mean=s, S_sensitivity_std=0.02,
                              V_sensitivity_mean=v, V_sensitivity_std=0.01)
    plots.plot_leakage_comparison(mk(0.988, 0.94, 0.98), mk(0.892, 0.51, 0.91),
                                  tmp / "leak.png")

    rows = [dict(label=f"variant {i}", is_base=(i == 0),
                 macro_f1_mean=0.7 - 0.04 * i, macro_f1_std=0.01,
                 S_sensitivity_mean=0.5 - 0.05 * i, S_sensitivity_std=0.02)
            for i in range(6)]
    plots.plot_ablation(rows, tmp / "abl.png")
    plots.plot_ablation(rows, tmp / "abl_s.png", "S_sensitivity_mean", "S Se")

    per = {c: dict(f1_mean=float(rng.uniform(0.3, 0.99)), f1_std=0.01)
           for c in C.AAMI_CLASSES}
    plots.plot_per_class_f1({"baseline": dict(per_class=per),
                             "proposed": dict(per_class=per)}, tmp / "f1.png")

    hist = [dict(epoch=i + 1, train_loss=1 / (i + 1),
                 val_macro_f1=0.4 + 0.03 * i, val_accuracy=0.8 + 0.01 * i)
            for i in range(12)]
    plots.plot_training_curves(hist, tmp / "curve.png")
    plots.plot_class_distribution({"N": 90000, "S": 2780, "V": 7000, "F": 800, "Q": 15},
                                  tmp / "dist.png")

    made = sorted(p.name for p in tmp.glob("*.png"))
    assert len(made) == 7, made
    print(f"  [ok] figures render: {', '.join(made)}")


# --------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("=" * 74)
    print(f"Running {len(tests)} tests (torch-free components)")
    print("=" * 74)
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"  [FAIL] {t.__name__}: {e}")
    print("=" * 74)
    if failed:
        print(f"{len(failed)} of {len(tests)} FAILED")
        for name, e in failed:
            print(f"  - {name}: {e}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
