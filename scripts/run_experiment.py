#!/usr/bin/env python
"""Run a single configuration.

    # 60-second end-to-end check on synthetic data (no download needed)
    python scripts/run_experiment.py --smoke-test

    # one real run
    python scripts/run_experiment.py --protocol inter --arch multiscale --loss focal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.evaluate import format_confusion_matrix
from src.train import ExperimentConfig, train_one
from src.utils import save_json, environment_report


def load_split(protocol: str, seed: int, synthetic: bool):
    if synthetic:
        from src.synthetic import synthetic_split
        return synthetic_split(protocol, seed)
    from src.splits import get_split
    return get_split(protocol, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="inter", choices=["inter", "intra"])
    ap.add_argument("--arch", default="multiscale", choices=["baseline", "multiscale"])
    ap.add_argument("--loss", default="focal", choices=["ce", "weighted_ce", "focal"])
    ap.add_argument("--leads", default="dual", choices=["single", "dual"])
    ap.add_argument("--no-rr", action="store_true", help="drop the RR-interval features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.DEFAULTS["epochs"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny synthetic run that verifies the pipeline end to end")
    args = ap.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 4)

    cfg = ExperimentConfig(
        name=args.name or f"{args.protocol}_{args.arch}_{args.loss}_{args.leads}",
        protocol=args.protocol, arch=args.arch, loss=args.loss, leads=args.leads,
        use_rr=not args.no_rr, seed=args.seed, epochs=args.epochs,
        synthetic=args.smoke_test,
    )

    print("=" * 74)
    print(f"RUN: {cfg.name}  (seed {cfg.seed})")
    if cfg.synthetic:
        print("*** SMOKE TEST — SYNTHETIC DATA. These numbers are not results. ***")
    print("=" * 74)
    for k, v in cfg.to_dict().items():
        if k not in ("tags", "notes"):
            print(f"  {k:<16} {v}")

    print("\nLoading data...")
    train, val, test = load_split(cfg.protocol, cfg.seed, cfg.synthetic)
    print(f"  train {len(train['labels']):,} | val {len(val['labels']):,} "
          f"| test {len(test['labels']):,} beats")

    print("\nTraining...")
    metrics = train_one(train, val, test, cfg)
    metrics["environment"] = environment_report()

    print("\n" + "-" * 74)
    print(f"RESULTS — {cfg.name}")
    print("-" * 74)
    print(f"  accuracy   {metrics['accuracy']:.4f}")
    print(f"  macro F1   {metrics['macro_f1']:.4f}")
    print(f"  kappa      {metrics['kappa']:.4f}")
    print(f"  parameters {metrics['n_parameters']:,}   "
          f"time {metrics['train_seconds']}s   epochs {metrics['epochs_run']}")
    print(f"\n  {'class':<6}{'Se':>9}{'P+':>9}{'F1':>9}{'support':>10}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<6}{m['sensitivity']:>9.4f}{m['positive_predictivity']:>9.4f}"
              f"{m['f1']:>9.4f}{m['support']:>10,}")

    print("\n  Confusion matrix (counts)")
    print("  " + format_confusion_matrix(metrics["confusion_matrix"]).replace("\n", "\n  "))
    print("\n  Confusion matrix (row-normalised %)")
    print("  " + format_confusion_matrix(metrics["confusion_matrix"], True).replace("\n", "\n  "))

    tag = "SMOKE_" if cfg.synthetic else ""
    out = C.RAW_RESULTS / f"{tag}{cfg.name}_seed{cfg.seed}.json"
    save_json(metrics, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
