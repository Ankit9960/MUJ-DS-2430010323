#!/usr/bin/env python
"""Step 3: turn raw results into figures, tables and RESULTS.md.

    python scripts/make_report.py
    python scripts/make_report.py --smoke-test   # report from the synthetic run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import sota
from src.evaluate import format_confusion_matrix
from src.plots import (plot_ablation, plot_confusion_matrix, plot_leakage_comparison,
                       plot_per_class_f1, plot_training_curves)
from src.utils import load_json

PCT = lambda v: f"{100 * v:.2f}"


def pm(agg: dict, key: str) -> str:
    """'92.31 ± 0.42' from the mean/std pair."""
    m, s = agg.get(f"{key}_mean"), agg.get(f"{key}_std", 0.0)
    return f"{100 * m:.2f} ± {100 * s:.2f}" if m is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    tag = "SMOKE_" if args.smoke_test else ""

    path = C.RAW_RESULTS / f"{tag}ablation_summary.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/run_ablation.py first.")
    blob = load_json(path)
    exp = blob["experiments"]

    print("=" * 74)
    print("STEP 3/3  Building report")
    print("=" * 74)

    figures = []

    # ---- Figure 2: the leakage comparison (headline CO5 result) ----------
    if "E2_inter_baseline" in exp and "E1_intra_baseline" in exp:
        figures.append(plot_leakage_comparison(
            exp["E1_intra_baseline"], exp["E2_inter_baseline"],
            C.FIGURES / f"{tag}fig2_leakage_baseline.png",
            "Baseline CNN: the same model scored two ways"))
    if "E3_intra_proposed" in exp and "E4_inter_proposed" in exp:
        figures.append(plot_leakage_comparison(
            exp["E3_intra_proposed"], exp["E4_inter_proposed"],
            C.FIGURES / f"{tag}fig3_leakage_proposed.png",
            "Proposed model: the same model scored two ways"))

    # ---- Figures 4-5: confusion matrices ---------------------------------
    for key, fname, title in [
        ("E2_inter_baseline", "fig4_cm_baseline_inter",
         "Baseline CNN — inter-patient (DS2)"),
        ("E4_inter_proposed", "fig5_cm_proposed_inter",
         "Proposed model — inter-patient (DS2)"),
        ("E3_intra_proposed", "fig6_cm_proposed_intra",
         "Proposed model — intra-patient (leaky split)"),
    ]:
        if key in exp:
            figures.append(plot_confusion_matrix(
                exp[key]["confusion_matrix_sum"], title,
                C.FIGURES / f"{tag}{fname}.png"))

    # ---- Figure 7: ablation ---------------------------------------------
    ablation_keys = ["E4_inter_proposed", "A1_loss_ce", "A2_loss_weighted_ce",
                     "A3_no_context", "A4_no_rr", "A5_single_lead"]
    rows = [dict(label=exp[k]["label"], is_base=exp[k].get("is_base", False),
                 macro_f1_mean=exp[k]["macro_f1_mean"],
                 macro_f1_std=exp[k]["macro_f1_std"],
                 S_sensitivity_mean=exp[k]["S_sensitivity_mean"],
                 S_sensitivity_std=exp[k]["S_sensitivity_std"])
            for k in ablation_keys if k in exp]
    if rows:
        figures.append(plot_ablation(
            rows, C.FIGURES / f"{tag}fig7_ablation_macro_f1.png",
            "macro_f1_mean", "Macro F1",
            "Ablation — each row removes one component from the proposed model"))
        figures.append(plot_ablation(
            rows, C.FIGURES / f"{tag}fig8_ablation_S_sensitivity.png",
            "S_sensitivity_mean", "S (SVEB) sensitivity",
            "Ablation — effect on the hardest class (supraventricular ectopic)"))

    # ---- Figure 9: per-class F1 ------------------------------------------
    sel = {exp[k]["label"]: exp[k] for k in
           ("E2_inter_baseline", "E4_inter_proposed") if k in exp}
    if sel:
        figures.append(plot_per_class_f1(
            sel, C.FIGURES / f"{tag}fig9_per_class_f1.png",
            "Per-class F1 — baseline vs proposed (inter-patient)"))

    # ---- Figure 10: training curve ---------------------------------------
    if "E4_inter_proposed" in exp and exp["E4_inter_proposed"].get("history"):
        figures.append(plot_training_curves(
            exp["E4_inter_proposed"]["history"],
            C.FIGURES / f"{tag}fig10_training_curve.png",
            "Proposed model — validation during training (seed 0)"))

    for f in figures:
        print(f"  wrote {f}")

    # ---------------------------------------------------------------------
    # RESULTS.md
    # ---------------------------------------------------------------------
    L = []
    add = L.append

    if args.smoke_test:
        add("> **SMOKE TEST OUTPUT — SYNTHETIC DATA.** These numbers verify that "
            "the pipeline runs. They are not results and must not be submitted.\n")

    add("# Results\n")
    add(f"Generated from `{path.name}`. "
        f"{blob.get('n_seeds', len(blob.get('seeds', [])))} seeds per configuration; "
        f"every score is mean ± standard deviation across seeds.\n")
    env = blob.get("environment", {})
    add(f"Environment: Python {env.get('python','?')}, torch {env.get('torch','?')}, "
        f"{'GPU: ' + env.get('gpu','') if env.get('cuda_available') else 'CPU'}. "
        f"Total runtime {blob.get('total_minutes','?')} min.\n")

    # --- CO3 --------------------------------------------------------------
    add("\n## CO3 — Model implemented and applied\n")
    add("| Configuration | Protocol | Accuracy | Macro F1 | Cohen κ | Params |")
    add("|---|---|---|---|---|---|")
    for k in ["E1_intra_baseline", "E2_inter_baseline",
              "E3_intra_proposed", "E4_inter_proposed"]:
        if k not in exp:
            continue
        e = exp[k]
        add(f"| {e['label']} | {e['protocol']}-patient | {pm(e,'accuracy')} | "
            f"{pm(e,'macro_f1')} | {pm(e,'kappa')} | {e.get('n_parameters',0):,} |")

    if "E4_inter_proposed" in exp:
        e = exp["E4_inter_proposed"]
        add("\n### Per-class performance — proposed model, inter-patient\n")
        add("| Class | Meaning | Sensitivity (Se) | Positive predictivity (P+) | F1 |")
        add("|---|---|---|---|---|")
        meanings = {"N": "Normal / BBB", "S": "Supraventricular ectopic",
                    "V": "Ventricular ectopic", "F": "Fusion", "Q": "Unknown / paced"}
        for cls in C.AAMI_CLASSES:
            m = e["per_class"][cls]
            add(f"| **{cls}** | {meanings[cls]} | "
                f"{100*m['sensitivity_mean']:.2f} ± {100*m['sensitivity_std']:.2f} | "
                f"{100*m['positive_predictivity_mean']:.2f} ± "
                f"{100*m['positive_predictivity_std']:.2f} | "
                f"{100*m['f1_mean']:.2f} ± {100*m['f1_std']:.2f} |")

        add("\n### Confusion matrix — proposed model, inter-patient (summed over seeds)\n")
        add("```")
        add(format_confusion_matrix(e["confusion_matrix_sum"]))
        add("")
        add(format_confusion_matrix(e["confusion_matrix_sum"], normalise=True))
        add("```")
        add(f"\n![Confusion matrix]({(C.FIGURES / f'{tag}fig5_cm_proposed_inter.png').relative_to(C.ROOT)})\n")

    # --- CO4 --------------------------------------------------------------
    add("\n## CO4 — Comparison with published work\n")
    add("All rows are the inter-patient (de Chazal DS1/DS2) protocol on MIT-BIH. "
        "Se = sensitivity, P+ = positive predictivity.\n")
    add(sota.markdown_table())

    if "E4_inter_proposed" in exp:
        e = exp["E4_inter_proposed"]
        fmt = lambda c: (f"{100*e['per_class'][c]['sensitivity_mean']:.1f} / "
                         f"{100*e['per_class'][c]['positive_predictivity_mean']:.1f}")
        add(f"| **This work** | 2026 | Multi-scale 1D CNN + class-balanced focal loss | "
            f"{100*e['accuracy_mean']:.1f} | {fmt('N')} | {fmt('S')} | "
            f"{fmt('V')} | {fmt('F')} |")

    add("\n**Sources**\n")
    add(sota.source_list())
    ev = sota.PROTOCOL_EVIDENCE
    add(f"\n**Independent evidence for the protocol effect.** {ev['source']} reports "
        f"the same CNN at **{ev['intra_accuracy']}%** intra-patient and "
        f"**{ev['inter_accuracy']}%** inter-patient — a {ev['delta']:.1f} point gap "
        f"attributable to the split alone. The review states: \"{ev['quote']}\"\n")
    add("\n> **Reading this table honestly.** Accuracy on this benchmark ranges from "
        "86% to 99.9% under nominally the same protocol. Entries above ~99% apply "
        "oversampling (e.g. SMOTE) or augmentation; if synthetic neighbours of test "
        "beats reach the training set, the inter-patient guarantee is weakened. The "
        "classical entries are the safest direct comparison for a clean "
        "re-implementation.\n")

    # --- CO5 --------------------------------------------------------------
    add("\n## CO5 — Innovation and ablation study\n")
    add("### Contribution 1 — quantifying the leakage\n")
    if "E1_intra_baseline" in exp and "E2_inter_baseline" in exp:
        a, b = exp["E1_intra_baseline"], exp["E2_inter_baseline"]
        d_acc = 100 * (a["accuracy_mean"] - b["accuracy_mean"])
        d_s = 100 * (a["S_sensitivity_mean"] - b["S_sensitivity_mean"])
        add(f"The same baseline CNN scores **{pm(a,'accuracy')}%** under a random "
            f"beat split and **{pm(b,'accuracy')}%** under the inter-patient split "
            f"— a drop of **{d_acc:.2f} points** from changing nothing but the "
            f"partition. Sensitivity on class S falls by **{d_s:.2f} points**, "
            f"because patient-specific morphology is exactly what a random split "
            f"lets the model memorise.\n")
        add(f"![Leakage, baseline]({(C.FIGURES / f'{tag}fig2_leakage_baseline.png').relative_to(C.ROOT)})\n")
    if "E3_intra_proposed" in exp and "E4_inter_proposed" in exp:
        add(f"![Leakage, proposed]({(C.FIGURES / f'{tag}fig3_leakage_proposed.png').relative_to(C.ROOT)})\n")

    add("\n### Contribution 2 — recovering part of the gap\n")
    add("| Variant | Macro F1 | Accuracy | S (SVEB) Se | V (VEB) Se | Δ Macro F1 vs proposed |")
    add("|---|---|---|---|---|---|")
    base_f1 = exp.get("E4_inter_proposed", {}).get("macro_f1_mean")
    for k in ablation_keys:
        if k not in exp:
            continue
        e = exp[k]
        delta = ("—" if e.get("is_base") or base_f1 is None
                 else f"{100*(e['macro_f1_mean'] - base_f1):+.2f}")
        add(f"| {e['label']} | {pm(e,'macro_f1')} | {pm(e,'accuracy')} | "
            f"{pm(e,'S_sensitivity')} | {pm(e,'V_sensitivity')} | {delta} |")

    add(f"\n![Ablation]({(C.FIGURES / f'{tag}fig7_ablation_macro_f1.png').relative_to(C.ROOT)})\n")
    add(f"![Ablation, S sensitivity]({(C.FIGURES / f'{tag}fig8_ablation_S_sensitivity.png').relative_to(C.ROOT)})\n")
    add("\nEach ablated row differs from the proposed configuration in exactly one "
        "component, so the Δ column attributes the change to that component alone.\n")

    out = C.ROOT / ("RESULTS_SMOKE.md" if args.smoke_test else "RESULTS.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  wrote {out}")
    print(f"\nDone. {len(figures)} figures in {C.FIGURES}")


if __name__ == "__main__":
    main()
