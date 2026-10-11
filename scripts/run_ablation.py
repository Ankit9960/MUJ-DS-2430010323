#!/usr/bin/env python
"""Step 2: the full experiment grid (CO3 + CO4 + CO5).

    python scripts/run_ablation.py                 # all 9 configs x 3 seeds
    python scripts/run_ablation.py --seeds 1       # fast pass, 1 seed each
    python scripts/run_ablation.py --only E4_inter_proposed
    python scripts/run_ablation.py --smoke-test    # synthetic, ~2 min, verifies wiring

Design: E1-E4 are the headline protocol comparison. A1-A5 are strict
one-factor-at-a-time changes from the proposed configuration (E4), so each row
of the ablation table isolates exactly one component.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.evaluate import aggregate_seeds
from src.train import ExperimentConfig, train_one
from src.utils import save_json, environment_report

BASE = dict(arch="multiscale", loss="focal", leads="dual", use_rr=True)

EXPERIMENTS = [
    # --- Headline: does the evaluation protocol change the conclusion? -----
    dict(name="E1_intra_baseline", protocol="intra", arch="baseline", loss="ce",
         leads="single", use_rr=False,
         label="Baseline CNN, intra-patient (leaky)",
         notes="Mirrors the common published setup: simple CNN, random beat split."),
    dict(name="E2_inter_baseline", protocol="inter", arch="baseline", loss="ce",
         leads="single", use_rr=False,
         label="Baseline CNN, inter-patient",
         notes="Identical model and hyper-parameters as E1; only the split changes."),
    dict(name="E3_intra_proposed", protocol="intra", **BASE,
         label="Proposed model, intra-patient (leaky)",
         notes="Upper bound the proposed model reaches under the leaky protocol."),
    dict(name="E4_inter_proposed", protocol="inter", **BASE,
         label="Proposed model, inter-patient  [PROPOSED]", is_base=True,
         notes="The configuration reported as this project's result."),

    # --- One-factor ablations from E4, all inter-patient -------------------
    dict(name="A1_loss_ce", protocol="inter", arch="multiscale", loss="ce",
         leads="dual", use_rr=True,
         label="− class-balanced focal loss (plain CE)",
         notes="Isolates the imbalance handling."),
    dict(name="A2_loss_weighted_ce", protocol="inter", arch="multiscale",
         loss="weighted_ce", leads="dual", use_rr=True,
         label="focal → inverse-frequency weighted CE",
         notes="Tests whether the focal term adds anything over plain reweighting."),
    dict(name="A3_no_context", protocol="inter", arch="baseline", loss="focal",
         leads="dual", use_rr=True,
         label="− rhythm-context branch (beat window only)",
         notes="Isolates the multi-scale contribution."),
    dict(name="A4_no_rr", protocol="inter", arch="multiscale", loss="focal",
         leads="dual", use_rr=False,
         label="− RR-interval features",
         notes="Isolates the explicit timing features."),
    dict(name="A5_single_lead", protocol="inter", arch="multiscale", loss="focal",
         leads="single", use_rr=True,
         label="− second lead (single-lead input)",
         notes="Isolates the value of the second channel."),
]


def load_split(protocol: str, seed: int, synthetic: bool):
    if synthetic:
        from src.synthetic import synthetic_split
        return synthetic_split(protocol, seed)
    from src.splits import get_split
    return get_split(protocol, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(C.SEEDS),
                    help="number of seeds per configuration (1 = fast pass)")
    ap.add_argument("--epochs", type=int, default=C.DEFAULTS["epochs"])
    ap.add_argument("--only", nargs="*", default=None, help="run only these config names")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:max(args.seeds, 1)]
    epochs = min(args.epochs, 4) if args.smoke_test else args.epochs
    experiments = [e for e in EXPERIMENTS
                   if args.only is None or e["name"] in args.only]

    print("=" * 74)
    print("STEP 2/3  Experiment grid")
    if args.smoke_test:
        print("*** SMOKE TEST — SYNTHETIC DATA. Not results. ***")
    print("=" * 74)
    print(f"{len(experiments)} configurations x {len(seeds)} seeds "
          f"= {len(experiments) * len(seeds)} runs, {epochs} epochs max each")
    print(f"Device: {environment_report().get('cuda_available') and 'GPU' or 'CPU'}")

    # Cache splits per (protocol, seed): loading is the slow part, not training.
    split_cache: dict = {}
    summary, t_start = {}, time.time()

    for i, spec in enumerate(experiments, 1):
        spec = dict(spec)
        label = spec.pop("label")
        is_base = spec.pop("is_base", False)
        notes = spec.pop("notes", "")
        name = spec["name"]

        print(f"\n[{i}/{len(experiments)}] {name} — {label}")
        print(f"    {notes}")
        runs = []
        for seed in seeds:
            key = (spec["protocol"], seed if spec["protocol"] == "intra" else 0)
            if key not in split_cache:
                split_cache[key] = load_split(spec["protocol"], key[1], args.smoke_test)
            train, val, test = split_cache[key]

            cfg = ExperimentConfig(seed=seed, epochs=epochs,
                                   synthetic=args.smoke_test, notes=notes, **spec)
            m = train_one(train, val, test, cfg, verbose=False)
            runs.append(m)
            print(f"    seed {seed}: acc={m['accuracy']:.4f}  "
                  f"macroF1={m['macro_f1']:.4f}  "
                  f"S_Se={m['S_sensitivity']:.4f}  V_Se={m['V_sensitivity']:.4f}  "
                  f"({m['train_seconds']}s, {m['epochs_run']} ep)")

            tag = "SMOKE_" if args.smoke_test else ""
            save_json(m, C.RAW_RESULTS / f"{tag}{name}_seed{seed}.json")

        agg = aggregate_seeds(runs)
        agg.update(name=name, label=label, is_base=is_base, notes=notes,
                   config=runs[0]["config"], n_parameters=runs[0]["n_parameters"],
                   protocol=spec["protocol"],
                   history=runs[0]["history"], synthetic=args.smoke_test)
        summary[name] = agg
        print(f"    => macroF1 {agg['macro_f1_mean']:.4f} "
              f"+/- {agg['macro_f1_std']:.4f}   "
              f"acc {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")

    tag = "SMOKE_" if args.smoke_test else ""
    out = C.RAW_RESULTS / f"{tag}ablation_summary.json"
    save_json(dict(experiments=summary, seeds=seeds, epochs=epochs,
                   environment=environment_report(),
                   total_minutes=round((time.time() - t_start) / 60, 1),
                   synthetic=args.smoke_test), out)

    print("\n" + "=" * 74)
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Wrote {out}")
    print("Next:  python scripts/make_report.py")


if __name__ == "__main__":
    main()
