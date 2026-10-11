#!/usr/bin/env python
"""Evaluate a trained checkpoint on the held-out test set.

    python src/test.py                                    # proposed model, DS2
    python src/test.py --checkpoint models/E2_inter_baseline_seed0.pt
    python src/test.py --protocol intra                   # the leaky split

Loads weights produced by scripts/export_models.py, runs inference over the
test set, and reports accuracy, macro F1, Cohen's kappa, per-class sensitivity
and positive predictivity, and the confusion matrix. Optionally writes the
confusion-matrix figure.

This is inference only -- nothing here trains. Use src/train.py for that.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src import config as C
from src.evaluate import evaluate, format_confusion_matrix
from src.model import build_model, count_parameters
from src.splits import get_split, patient_overlap
from src.train import ExperimentConfig, make_loaders, predict
from src.utils import get_device


DEFAULT_CKPT = C.MODELS_DIR / "E4_inter_proposed_seed0.pt"


def load_checkpoint(path: Path):
    """Rebuild the model from the checkpoint's own metadata and load weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["n_classes"],
                        ckpt["dropout"], use_rr=ckpt["use_rr"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="path to a .pt file from scripts/export_models.py")
    ap.add_argument("--protocol", default=None, choices=["inter", "intra"],
                    help="override the protocol stored in the checkpoint")
    ap.add_argument("--save-figure", action="store_true",
                    help="write results/confusion_matrix.png")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\n"
                 f"Create one with:  python scripts/export_models.py")

    model, ckpt = load_checkpoint(ckpt_path)
    saved = ckpt["config"]
    protocol = args.protocol or saved["protocol"]

    print("=" * 74)
    print(f"TESTING  {ckpt_path.name}")
    print("=" * 74)
    print(f"  architecture      {ckpt['arch']}")
    print(f"  input channels    {ckpt['in_channels']} ({saved['leads']} lead)")
    print(f"  RR features       {ckpt['use_rr']}")
    print(f"  trained with      loss={saved['loss']}, seed={saved['seed']}, "
          f"protocol={saved['protocol']}-patient")
    print(f"  parameters        {count_parameters(model):,}")
    print(f"  best val macro F1 {ckpt['best_val_macro_f1']:.4f}")

    print(f"\nLoading {protocol}-patient test set...")
    train, val, test = get_split(protocol, seed=saved["seed"])
    ov = patient_overlap(train, test)
    print(f"  test beats        {len(test['labels']):,} "
          f"from {ov['n_test_patients']} patients")
    print(f"  patients also in training: {ov['n_shared_patients']} "
          f"({ov['pct_test_beats_from_seen_patients']}% of test beats)")

    cfg = ExperimentConfig(protocol=protocol, arch=ckpt["arch"],
                           loss=saved["loss"], leads=saved["leads"],
                           use_rr=ckpt["use_rr"], seed=saved["seed"])
    _, _, loader_te = make_loaders(train, val, test, cfg)

    device = get_device()
    model.to(device)
    y_true, y_pred = predict(model, loader_te, device)
    m = evaluate(y_true, y_pred)

    print("\n" + "-" * 74)
    print("RESULTS")
    print("-" * 74)
    print(f"  accuracy    {m['accuracy']:.4f}")
    print(f"  macro F1    {m['macro_f1']:.4f}")
    print(f"  Cohen kappa {m['kappa']:.4f}")
    print(f"\n  {'class':<7}{'Se':>9}{'P+':>9}{'F1':>9}{'support':>10}")
    for cls, pc in m["per_class"].items():
        print(f"  {cls:<7}{pc['sensitivity']:>9.4f}{pc['positive_predictivity']:>9.4f}"
              f"{pc['f1']:>9.4f}{pc['support']:>10,}")

    print("\n  Confusion matrix (counts)")
    print("  " + format_confusion_matrix(m["confusion_matrix"]).replace("\n", "\n  "))
    print("\n  Confusion matrix (row-normalised, % of each true class)")
    print("  " + format_confusion_matrix(m["confusion_matrix"], True).replace("\n", "\n  "))

    if args.save_figure:
        from src.plots import plot_confusion_matrix
        out = C.RESULTS / "confusion_matrix.png"
        plot_confusion_matrix(m["confusion_matrix"],
                              f"{ckpt['arch']} — {protocol}-patient test set", out)
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
