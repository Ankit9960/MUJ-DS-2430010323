#!/usr/bin/env python
"""Step 1: download MIT-BIH and build the beat cache.

    python scripts/prepare_data.py            # download + segment everything
    python scripts/prepare_data.py --force    # re-segment from scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config as C
from src.download import download
from src.preprocessing import build_cache, load_records
from src.splits import class_counts, patient_overlap, get_split
from src.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rebuild the beat cache even if it exists")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("STEP 1/3  Preparing the MIT-BIH Arrhythmia Database")
    print("=" * 74)
    print(f"Records: {len(C.ALL_RECORDS)} "
          f"(48 total minus the 4 paced records {C.PACED_RECORDS}, "
          f"minus 2 not in the AAMI lists)")

    if not args.skip_download:
        print("\n-- Downloading from PhysioNet --")
        download(C.ALL_RECORDS)

    print("\n-- Segmenting beats --")
    build_cache(C.ALL_RECORDS, force=args.force)

    print("\n-- Dataset summary --")
    everything = load_records(C.ALL_RECORDS)
    counts = class_counts(everything)
    total = sum(counts.values())
    print(f"Total beats: {total:,}")
    for cls, n in counts.items():
        print(f"  {cls}: {n:>7,}  ({100 * n / total:5.2f}%)")
    print(f"Imbalance ratio (largest:smallest) = "
          f"{max(counts.values()) / max(min(counts.values()), 1):.0f} : 1")

    print("\n-- Split integrity check --")
    for protocol in ("inter", "intra"):
        tr, va, te = get_split(protocol, seed=0)
        ov = patient_overlap(tr, te)
        print(f"\n[{protocol}-patient]")
        print(f"  train {len(tr['labels']):>7,} beats / {ov['n_train_patients']} patients")
        print(f"  val   {len(va['labels']):>7,} beats")
        print(f"  test  {len(te['labels']):>7,} beats / {ov['n_test_patients']} patients")
        print(f"  patients in BOTH train and test: {ov['n_shared_patients']}")
        print(f"  test beats from a patient the model trained on: "
              f"{ov['test_beats_from_seen_patients']:,} "
              f"({ov['pct_test_beats_from_seen_patients']}%)")

    # Figure 1 for the report.
    from src.plots import plot_class_distribution
    fig = plot_class_distribution(counts, C.FIGURES / "fig1_class_distribution.png")
    print(f"\nWrote {fig}")

    save_json(dict(total_beats=total, class_counts=counts,
                   records=C.ALL_RECORDS, ds1=C.DS1, ds2=C.DS2),
              C.RAW_RESULTS / "dataset_summary.json")
    print(f"Wrote {C.RAW_RESULTS / 'dataset_summary.json'}")
    print("\nData ready. Next:  python scripts/run_ablation.py")


if __name__ == "__main__":
    main()
