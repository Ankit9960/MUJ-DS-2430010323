#!/usr/bin/env python
"""Load an exported checkpoint and show that it works.

    python scripts/load_model.py models/E4_inter_proposed_seed0.pt

Demonstrates that the saved weights are self-contained: the architecture is
rebuilt from the checkpoint's own metadata, with no reference to the training
configuration that produced it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import build_model, count_parameters


def load(path: str | Path):
    """Rebuild the model and load its weights. Returns (model, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"],
                        ckpt["n_classes"], ckpt["dropout"],
                        use_rr=ckpt["use_rr"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"{path} not found. Run: python scripts/export_models.py")

    model, ckpt = load(path)
    cfg = ckpt["config"]

    print(f"Loaded {path.name}")
    print(f"  architecture      {ckpt['arch']}")
    print(f"  input channels    {ckpt['in_channels']} ({cfg['leads']} lead)")
    print(f"  RR features       {ckpt['use_rr']}")
    print(f"  classes           {', '.join(ckpt['classes'])}")
    print(f"  parameters        {count_parameters(model):,}")
    print(f"  protocol          {cfg['protocol']}-patient, loss={cfg['loss']}, seed={cfg['seed']}")
    print(f"  best val macro F1 {ckpt['best_val_macro_f1']:.4f}")

    # One forward pass on random input of the right shape proves it is usable.
    from src import config as C
    b = torch.randn(4, ckpt["in_channels"], C.BEAT_LEN)
    c = torch.randn(4, ckpt["in_channels"], C.CTX_LEN)
    r = torch.randn(4, C.N_RR_FEATURES)
    with torch.no_grad():
        logits = model(b, c, r)
    pred = logits.argmax(1)
    print(f"\n  forward pass OK   logits {tuple(logits.shape)} -> "
          f"predicted classes {[ckpt['classes'][i] for i in pred.tolist()]}")


if __name__ == "__main__":
    main()
