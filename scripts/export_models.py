#!/usr/bin/env python
"""Train and export model checkpoints to models/.

    python scripts/export_models.py          # the two headline configs (~4 min GPU)
    python scripts/export_models.py --all    # all 9 configurations (~27 min)

Checkpoints are trained with the same seeds and the same code path as the
reported results, so the weights correspond exactly to the numbers in
RESULTS.md. This script never touches results/ -- it only writes models/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import config as C
from src.train import ExperimentConfig, train_one
from src.utils import save_json, environment_report

MODELS = C.ROOT / "models"

# Imported here so the two scripts cannot drift apart.
from scripts.run_ablation import EXPERIMENTS  # noqa: E402

HEADLINE = ["E2_inter_baseline", "E4_inter_proposed"]


def load_split(protocol: str, seed: int):
    from src.splits import get_split
    return get_split(protocol, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="export every configuration, not just the headline two")
    ap.add_argument("--seeds", type=int, default=len(C.SEEDS))
    ap.add_argument("--epochs", type=int, default=C.DEFAULTS["epochs"])
    args = ap.parse_args()

    seeds = C.SEEDS[:max(args.seeds, 1)]
    wanted = None if args.all else set(HEADLINE)
    specs = [e for e in EXPERIMENTS if wanted is None or e["name"] in wanted]

    MODELS.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(f"Exporting {len(specs)} configurations x {len(seeds)} seeds -> {MODELS}")
    print("=" * 74)

    split_cache: dict = {}
    manifest = []

    for spec in specs:
        spec = dict(spec)
        label = spec.pop("label")
        spec.pop("is_base", None)
        notes = spec.pop("notes", "")
        name = spec["name"]
        print(f"\n{name} — {label}")

        for seed in seeds:
            key = (spec["protocol"], seed if spec["protocol"] == "intra" else 0)
            if key not in split_cache:
                split_cache[key] = load_split(spec["protocol"], key[1])
            train, val, test = split_cache[key]

            cfg = ExperimentConfig(seed=seed, epochs=args.epochs, notes=notes, **spec)
            out = MODELS / f"{name}_seed{seed}.pt"
            m = train_one(train, val, test, cfg, verbose=False, save_to=out)
            print(f"  seed {seed}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}"
                  f"  -> {out.name} ({out.stat().st_size/1e6:.2f} MB)")

            manifest.append(dict(
                file=out.name, config=name, label=label, seed=seed,
                accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                S_sensitivity=m["S_sensitivity"], V_sensitivity=m["V_sensitivity"],
                n_parameters=m["n_parameters"], epochs_run=m["epochs_run"],
            ))

    save_json(dict(models=manifest, seeds=seeds,
                   environment=environment_report()),
              MODELS / "manifest.json")

    print(f"\nWrote {len(manifest)} checkpoints + manifest.json to {MODELS}")
    print("Load one with:  python scripts/load_model.py models/<file>.pt")


if __name__ == "__main__":
    main()
