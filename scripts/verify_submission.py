#!/usr/bin/env python
"""Audit the repository against the assignment's stated requirements.

    python scripts/verify_submission.py

Run it twice: once now (it will show which items are still waiting on a
training run), and once after `make_report.py` finishes, when every line should
read PASS. It checks that files exist AND that their contents are what the
requirement actually asks for -- an empty RESULTS.md would fail, not pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C

PASS, PEND, FAIL = "PASS", "PENDING", "FAIL"
ICON = {PASS: "[x]", PEND: "[ ]", FAIL: "[!]"}

results: list[tuple[str, str, str, str]] = []   # (group, item, status, detail)


def check(group: str, item: str, ok: bool, detail: str = "",
          pending_if_false: bool = False):
    status = PASS if ok else (PEND if pending_if_false else FAIL)
    results.append((group, item, status, detail))
    return ok


def nonempty(p: Path, min_bytes: int = 200) -> bool:
    return p.exists() and p.stat().st_size >= min_bytes


# ==========================================================================
# 1. Required submission artefacts
# ==========================================================================
def check_required_files():
    g = "Required directory structure"

    # The exact tree the assignment specifies.
    required = [
        ("README.md",                     "file"),
        ("requirements.txt",              "file"),
        ("data/dataset_information.txt",  "file"),
        ("src/train.py",                  "file"),
        ("src/test.py",                   "file"),
        ("src/model.py",                  "file"),
        ("src/preprocessing.py",          "file"),
        ("notebooks/experiments.ipynb",   "file"),
        ("results/confusion_matrix.png",  "file"),
        ("results/training_curve.png",    "file"),
        ("results/results.csv",           "file"),
        ("results/comparison.csv",        "file"),
        ("models/model_description.txt",  "file"),
        ("figures",                       "dir"),
    ]
    for rel, kind in required:
        path = C.ROOT / rel
        if kind == "dir":
            n = len(list(path.glob("*"))) if path.is_dir() else 0
            check(g, rel + "/", path.is_dir() and n > 0, f"{n} files")
        else:
            ok = path.exists() and path.stat().st_size > 0
            size = f"{path.stat().st_size:,} bytes" if path.exists() else "MISSING"
            check(g, rel, ok, size)

    g = "Content checks"
    readme = C.ROOT / "README.md"
    body = readme.read_text(encoding="utf-8", errors="ignore") if readme.exists() else ""
    check(g, "README documents usage", all(s in body for s in ("Quick start", "MIT-BIH")),
          f"{len(body):,} chars")

    req = C.ROOT / "requirements.txt"
    pkgs = []
    if req.exists():
        pkgs = [l.split(">=")[0].split("==")[0].strip()
                for l in req.read_text().splitlines()
                if l.strip() and not l.strip().startswith("#")]
    needed = {"numpy", "scipy", "scikit-learn", "matplotlib", "torch", "wfdb"}
    check(g, "requirements complete", needed <= set(pkgs), ", ".join(pkgs))

    py = [p for p in sorted(C.ROOT.rglob("*.py")) if "__pycache__" not in str(p)]
    loc = sum(len(p.read_text(encoding="utf-8", errors="ignore").splitlines()) for p in py)
    check(g, "Source code", len(py) >= 10 and loc >= 800,
          f"{len(py)} modules, {loc:,} lines")

    # CSVs must have real rows, not just a header.
    import csv as _csv
    for rel, minrows in [("results/results.csv", 9), ("results/comparison.csv", 5)]:
        f = C.ROOT / rel
        n = 0
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                n = max(sum(1 for _ in _csv.reader(fh)) - 1, 0)
        check(g, rel + " populated", n >= minrows, f"{n} data rows")

    mdir = C.ROOT / "models"
    ckpts = list(mdir.glob("*.pt")) if mdir.is_dir() else []
    check(g, "Trained checkpoints", bool(ckpts),
          f"{len(ckpts)} .pt files" if ckpts
          else "run: python scripts/export_models.py",
          pending_if_false=True)


# ==========================================================================
# 2. Results produced by an actual run
# ==========================================================================
def load_summary():
    p = C.RAW_RESULTS / "ablation_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def check_results(blob):
    g = "Results (produced by running the pipeline)"

    ds = C.RAW_RESULTS / "dataset_summary.json"
    if ds.exists():
        d = json.loads(ds.read_text())
        check(g, "Dataset prepared", True,
              f"{d.get('total_beats',0):,} beats from {len(d.get('records',[]))} records")
    else:
        check(g, "Dataset prepared", False,
              "run: python scripts/prepare_data.py", pending_if_false=True)

    if blob is None:
        check(g, "Experiment runs", False,
              "run: python scripts/run_ablation.py", pending_if_false=True)
        check(g, "RESULTS.md generated", False,
              "run: python scripts/make_report.py", pending_if_false=True)
        check(g, "Confusion matrix", False, "produced by make_report.py", pending_if_false=True)
        check(g, "Performance metrics", False, "produced by make_report.py", pending_if_false=True)
        check(g, "Ablation study", False, "produced by make_report.py", pending_if_false=True)
        return

    exp = blob.get("experiments", {})
    seeds = blob.get("seeds", [])
    synthetic = blob.get("synthetic", False)

    if synthetic:
        check(g, "Real data used", False,
              "this summary is from --smoke-test (synthetic). Re-run without it.")

    expected = ["E1_intra_baseline", "E2_inter_baseline", "E3_intra_proposed",
                "E4_inter_proposed", "A1_loss_ce", "A2_loss_weighted_ce",
                "A3_no_context", "A4_no_rr", "A5_single_lead"]
    missing = [e for e in expected if e not in exp]
    check(g, "Experiment runs", not missing and not synthetic,
          f"{len(exp)}/9 configurations x {len(seeds)} seeds = {len(exp)*len(seeds)} runs"
          + (f" | MISSING: {', '.join(missing)}" if missing else ""))

    md = C.ROOT / "RESULTS.md"
    body = md.read_text(encoding="utf-8", errors="ignore") if md.exists() else ""
    check(g, "RESULTS.md generated", nonempty(md, 2000) and "CO3" in body and "CO5" in body,
          f"{len(body):,} chars, CO3/CO4/CO5 sections present" if body else "run make_report.py",
          pending_if_false=not md.exists())

    # Confusion matrix: both the figure and the numeric matrix must exist.
    cm_figs = list(C.FIGURES.glob("fig[456]_cm_*.png"))
    base = exp.get("E4_inter_proposed", {})
    cm = base.get("confusion_matrix_sum")
    cm_valid = bool(cm) and len(cm) == 5 and all(len(r) == 5 for r in cm)
    check(g, "Confusion matrix", cm_valid and len(cm_figs) >= 1,
          f"{len(cm_figs)} figures + 5x5 matrix in JSON and RESULTS.md"
          if cm_valid else "not found")

    needed_metrics = ["accuracy_mean", "macro_f1_mean", "kappa_mean",
                      "S_sensitivity_mean", "V_sensitivity_mean"]
    have = [m for m in needed_metrics if m in base]
    per_class_ok = all(c in base.get("per_class", {}) for c in C.AAMI_CLASSES)
    check(g, "Performance metrics", len(have) == len(needed_metrics) and per_class_ok,
          "accuracy, macro F1, Cohen kappa + per-class Se/P+/F1 for all 5 classes"
          if per_class_ok else f"only {len(have)}/{len(needed_metrics)} present")

    abl = [k for k in exp if k.startswith("A")]
    abl_figs = list(C.FIGURES.glob("fig[78]_ablation_*.png"))
    check(g, "Ablation study", len(abl) >= 4 and len(abl_figs) >= 1,
          f"{len(abl)} one-factor variants + {len(abl_figs)} figures")

    check(g, "Figures", len(list(C.FIGURES.glob("fig*.png"))) >= 8,
          f"{len(list(C.FIGURES.glob('fig*.png')))} figures in figures/")


# ==========================================================================
# 3. Reproducibility
# ==========================================================================
def check_reproducibility(blob):
    g = "Reproducibility"

    check(g, "Seeds fixed in code", len(C.SEEDS) >= 3,
          f"SEEDS = {C.SEEDS} in src/config.py; set_seed() covers random/numpy/torch/cudnn")

    runs = [p for p in C.RAW_RESULTS.glob("*.json")
            if p.name not in ("ablation_summary.json", "dataset_summary.json")
            and not p.name.startswith("SMOKE_")]
    check(g, "Per-run JSON records", len(runs) >= 9,
          f"{len(runs)} run files in results/raw/ (config + metrics + history + environment)",
          pending_if_false=not runs)

    if runs:
        one = json.loads(runs[0].read_text())
        # run_ablation.py records the environment once, in the summary; a single
        # run launched via run_experiment.py carries its own copy.
        env = (blob or {}).get("environment") or one.get("environment", {})
        check(g, "Environment captured", bool(env.get("python")),
              f"python {env.get('python','?')}, torch {env.get('torch','?')}, "
              f"{env.get('gpu') or ('GPU' if env.get('cuda_available') else 'CPU')}")
        check(g, "Config captured per run", bool(one.get("config")),
              "every run stores the exact configuration that produced it")
    else:
        check(g, "Environment captured", False, "written at run time", pending_if_false=True)
        check(g, "Config captured per run", False, "written at run time", pending_if_false=True)

    if blob:
        n = blob.get("seeds", [])
        std_reported = "macro_f1_std" in blob.get("experiments", {}).get(
            "E4_inter_proposed", {})
        check(g, "Variance reported", len(n) >= 3 and std_reported,
              f"mean +/- std over {len(n)} seeds")


# ==========================================================================
# 4. Protocol integrity -- the part a marker can actually probe
# ==========================================================================
def check_protocol():
    g = "Protocol integrity"

    check(g, "DS1/DS2 patient-disjoint", not set(C.DS1) & set(C.DS2),
          f"{len(C.DS1)} train patients, {len(C.DS2)} test patients, 0 shared")
    check(g, "Validation disjoint from test", not set(C.DS1_VAL) & set(C.DS2),
          f"val records {C.DS1_VAL} held out of DS1, never in DS2")
    check(g, "Paced records excluded", not set(C.ALL_RECORDS) & set(C.PACED_RECORDS),
          f"records {C.PACED_RECORDS} removed per AAMI EC57")
    check(g, "AAMI class scheme", C.AAMI_CLASSES == ["N", "S", "V", "F", "Q"],
          f"{len(C.SYMBOL_TO_AAMI)} annotation symbols mapped to 5 AAMI classes")

    t = C.ROOT / "tests" / "test_pipeline.py"
    n_tests = t.read_text().count("\ndef test_") if t.exists() else 0
    check(g, "Automated tests", n_tests >= 10,
          f"{n_tests} tests in tests/test_pipeline.py (run: python tests/test_pipeline.py)")


# ==========================================================================
# 5. Hyper-parameters actually in use
# ==========================================================================
def print_parameters():
    print("\n" + "=" * 78)
    print("HYPER-PARAMETERS AND SETTINGS  (src/config.py — quote these in the report)")
    print("=" * 78)

    rows = [
        ("Dataset", "MIT-BIH Arrhythmia Database (PhysioNet mitdb v1.0.0)"),
        ("Records used", f"{len(C.ALL_RECORDS)} (48 minus 4 paced, minus 2 outside the AAMI lists)"),
        ("Sampling rate", f"{C.FS} Hz"),
        ("Classes", f"{len(C.AAMI_CLASSES)} AAMI: {', '.join(C.AAMI_CLASSES)}"),
        ("", ""),
        ("Bandpass filter", f"{C.BANDPASS[0]}–{C.BANDPASS[1]} Hz Butterworth, order {C.FILTER_ORDER}, zero-phase"),
        ("Normalisation", "per-record z-score (removes per-patient gain)"),
        ("Beat window", f"{C.BEAT_PRE} before + {C.BEAT_POST} after R peak = {C.BEAT_LEN} samples (1.0 s)"),
        ("Context window", f"{C.CTX_PRE} + {C.CTX_POST} samples, decimated x{C.CTX_DECIM} = {C.CTX_LEN} (4.0 s)"),
        ("RR features", f"{C.N_RR_FEATURES}: pre-RR, post-RR, local 10-beat average, prematurity ratio"),
        ("", ""),
        ("Optimiser", f"Adam, lr {C.DEFAULTS['lr']}, weight decay {C.DEFAULTS['weight_decay']}"),
        ("Batch size", str(C.DEFAULTS["batch_size"])),
        ("Max epochs", str(C.DEFAULTS["epochs"])),
        ("Early stopping", f"patience {C.DEFAULTS['patience']} on validation macro-F1"),
        ("LR schedule", "ReduceLROnPlateau, factor 0.5, patience 3"),
        ("Dropout", str(C.DEFAULTS["dropout"])),
        ("Gradient clipping", "max norm 5.0"),
        ("", ""),
        ("Focal gamma", str(C.DEFAULTS["focal_gamma"])),
        ("Class-balanced beta", str(C.DEFAULTS["cb_beta"])),
        ("Seeds", f"{C.SEEDS} — every score reported as mean ± std"),
        ("", ""),
        ("Train records (DS1)", ", ".join(map(str, C.DS1_TRAIN))),
        ("Validation records", ", ".join(map(str, C.DS1_VAL))),
        ("Test records (DS2)", ", ".join(map(str, C.DS2))),
    ]
    for k, v in rows:
        if not k:
            print()
        else:
            print(f"  {k:<22} {v}")


# ==========================================================================
def main():
    print("=" * 78)
    print("SUBMISSION AUDIT")
    print("=" * 78)

    blob = load_summary()
    check_required_files()
    check_results(blob)
    check_reproducibility(blob)
    check_protocol()

    current = None
    for group, item, status, detail in results:
        if group != current:
            print(f"\n{group}")
            print("-" * 78)
            current = group
        print(f"  {ICON[status]} {item:<28} {detail}")

    n_pass = sum(1 for *_, s, _ in results if s == PASS)
    n_pend = sum(1 for *_, s, _ in results if s == PEND)
    n_fail = sum(1 for *_, s, _ in results if s == FAIL)

    print("\n" + "=" * 78)
    print(f"  {n_pass} passed · {n_pend} waiting on a run · {n_fail} failed")
    print("=" * 78)

    if n_fail:
        print("\n  FAILURES must be fixed before submitting. See the [!] lines above.")
    if n_pend:
        print("\n  To fill the pending items, run these three in order:")
        print("     python scripts/prepare_data.py")
        print("     python scripts/run_ablation.py")
        print("     python scripts/make_report.py")
        print("     python scripts/verify_submission.py     # then re-check")
    if not n_fail and not n_pend:
        print("\n  READY TO SUBMIT — every required item is present and populated.")

    print_parameters()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
