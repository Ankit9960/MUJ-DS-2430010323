# Results

Generated from `ablation_summary.json`. 3 seeds per configuration; every score is mean ± standard deviation across seeds.

Environment: Python 3.12.13, torch 2.10.0+cu128, GPU: Tesla T4. Total runtime 26.9 min.


## CO3 — Model implemented and applied

| Configuration | Protocol | Accuracy | Macro F1 | Cohen κ | Params |
|---|---|---|---|---|---|
| Baseline CNN, intra-patient (leaky) | intra-patient | 98.36 ± 0.10 | 71.40 ± 0.85 | 91.18 ± 0.53 | 44,069 |
| Baseline CNN, inter-patient | inter-patient | 87.16 ± 1.91 | 32.31 ± 1.02 | 43.82 ± 3.00 | 44,069 |
| Proposed model, intra-patient (leaky) | intra-patient | 98.63 ± 0.12 | 73.64 ± 1.54 | 92.98 ± 0.53 | 113,413 |
| Proposed model, inter-patient  [PROPOSED] | inter-patient | 84.65 ± 2.59 | 39.06 ± 0.76 | 44.55 ± 3.63 | 113,413 |

### Per-class performance — proposed model, inter-patient

| Class | Meaning | Sensitivity (Se) | Positive predictivity (P+) | F1 |
|---|---|---|---|---|
| **N** | Normal / BBB | 87.34 ± 3.31 | 96.52 ± 0.40 | 91.66 ± 1.74 |
| **S** | Supraventricular ectopic | 19.66 ± 5.95 | 31.27 ± 6.97 | 23.08 ± 3.31 |
| **V** | Ventricular ectopic | 91.66 ± 3.54 | 63.20 ± 4.78 | 74.59 ± 2.22 |
| **F** | Fusion | 29.38 ± 12.76 | 3.32 ± 1.68 | 5.96 ± 2.97 |
| **Q** | Unknown / paced | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Confusion matrix — proposed model, inter-patient (summed over seeds)

```
true\pred          N         S         V         F         Q
------------------------------------------------------------
N             115698      2560      3678     10529         6
S               2951      1083      1465         9         0
V                508        73      8835       223         0
F                715         0       107       342         0
Q                  9         0        12         0         0

true\pred          N         S         V         F         Q
------------------------------------------------------------
N               87.3       1.9       2.8       7.9       0.0
S               53.6      19.7      26.6       0.2       0.0
V                5.3       0.8      91.7       2.3       0.0
F               61.4       0.0       9.2      29.4       0.0
Q               42.9       0.0      57.1       0.0       0.0
```

![Confusion matrix](results/figures/fig5_cm_proposed_inter.png)


## CO4 — Comparison with published work

All rows are the inter-patient (de Chazal DS1/DS2) protocol on MIT-BIH. Se = sensitivity, P+ = positive predictivity.

| Study | Year | Method | Acc (%) | N Se/P+ | S Se/P+ | V Se/P+ | F Se/P+ |
|---|---|---|---|---|---|---|---|
| de Chazal et al. | 2004 | Linear discriminant (morphology + RR features) | 86.2 | 87.1 / 99.2 | 76.0 / 38.5 | 80.3 / 86.6 | 89.4 / 8.6 |
| Zhang et al. | 2014 | Disease-specific feature selection + SVM ensemble | 88.9 | 99.0 / 99.0 | 79.1 / 36.0 | 85.5 / 92.8 | 93.8 / 13.7 |
| Acharya et al. | 2014 | Nonlinear features + classifier ensemble | 93.0 | — | — | — | — |
| Garcia et al. | 2017 | Temporal vectorcardiogram + SVM | 92.4 | — | — | — | — |
| Mousavi & Afghah | 2019 | Sequence-to-sequence RNN with attention | 99.5 | 99.7 / 99.5 | 88.9 / 92.6 | 99.9 / 99.5 | — |
| Frontiers in Physiology | 2023 | MLP + weight capsule + seq2seq | 99.9 | 99.8 / 99.9 | 97.7 / 96.1 | 100.0 / 99.1 | 97.9 / 98.7 |
| **This work** | 2026 | Multi-scale 1D CNN + class-balanced focal loss | 84.6 | 87.3 / 96.5 | 19.7 / 31.3 | 91.7 / 63.2 | 29.4 / 3.3 |

**Sources**

- **de Chazal et al. (2004)** — IEEE Trans. Biomed. Eng. 51(7):1196-1206. PMID 15248536. _Defines the DS1/DS2 inter-patient protocol._
- **Zhang et al. (2014)** — Pattern Recognition 47(6):2154-2165. _Inter-patient, 4 classes reported._
- **Acharya et al. (2014)** — Reported in Mousavi & Afghah (ICASSP 2019), Table 3. _Per-class Se/P+ not reported in the comparison sources._
- **Garcia et al. (2017)** — Reported in Mousavi & Afghah (ICASSP 2019), Table 3. _Per-class Se/P+ not reported in the comparison sources._
- **Mousavi & Afghah (2019)** — ICASSP 2019, arXiv:1812.07421. _SMOTE oversampling applied to the training set. Treat as an optimistic upper reference._
- **Frontiers in Physiology (2023)** — Front. Physiol. 14:1247587. _Heavy resampling; near-ceiling scores on a protocol where classical methods reach ~87% invite scrutiny._

**Independent evidence for the protocol effect.** Systematic review, arXiv:2503.07276, reference [57] reports the same CNN at **99.48%** intra-patient and **88.34%** inter-patient — a 11.1 point gap attributable to the split alone. The review states: "Including heartbeats from the same patient in both phases introduces bias, as models memorize patient-specific characteristics during training, leading to inflated test performance (often approaching 100%)."


> **Reading this table honestly.** Accuracy on this benchmark ranges from 86% to 99.9% under nominally the same protocol. Entries above ~99% apply oversampling (e.g. SMOTE) or augmentation; if synthetic neighbours of test beats reach the training set, the inter-patient guarantee is weakened. The classical entries are the safest direct comparison for a clean re-implementation.


## CO5 — Innovation and ablation study

### Contribution 1 — quantifying the leakage

The same baseline CNN scores **98.36 ± 0.10%** under a random beat split and **87.16 ± 1.91%** under the inter-patient split — a drop of **11.20 points** from changing nothing but the partition. Sensitivity on class S falls by **66.77 points**, because patient-specific morphology is exactly what a random split lets the model memorise.

![Leakage, baseline](results/figures/fig2_leakage_baseline.png)

![Leakage, proposed](results/figures/fig3_leakage_proposed.png)


### Contribution 2 — recovering part of the gap

| Variant | Macro F1 | Accuracy | S (SVEB) Se | V (VEB) Se | Δ Macro F1 vs proposed |
|---|---|---|---|---|---|
| Proposed model, inter-patient  [PROPOSED] | 39.06 ± 0.76 | 84.65 ± 2.59 | 19.66 ± 5.95 | 91.66 ± 3.54 | — |
| − class-balanced focal loss (plain CE) | 43.34 ± 5.79 | 93.32 ± 1.66 | 38.14 ± 19.02 | 83.09 ± 4.06 | +4.28 |
| focal → inverse-frequency weighted CE | 43.86 ± 7.08 | 86.37 ± 3.80 | 43.95 ± 19.36 | 86.26 ± 4.89 | +4.80 |
| − rhythm-context branch (beat window only) | 41.74 ± 7.72 | 88.68 ± 5.32 | 30.54 ± 22.32 | 88.48 ± 8.92 | +2.68 |
| − RR-interval features | 30.85 ± 1.62 | 76.99 ± 2.91 | 9.77 ± 2.21 | 78.33 ± 9.38 | -8.21 |
| − second lead (single-lead input) | 44.36 ± 4.11 | 89.00 ± 0.96 | 44.50 ± 20.56 | 88.80 ± 2.59 | +5.31 |

![Ablation](results/figures/fig7_ablation_macro_f1.png)

![Ablation, S sensitivity](results/figures/fig8_ablation_S_sensitivity.png)


Each ablated row differs from the proposed configuration in exactly one component, so the Δ column attributes the change to that component alone.
