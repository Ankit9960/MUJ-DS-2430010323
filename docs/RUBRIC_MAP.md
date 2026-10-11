# Where each marking criterion is satisfied

A one-page index for the evaluator. Every row points at a file that exists in
this repository and a number that is generated, not typed.

---

## Submission checklist

| Required item | Where it is | Status |
|---|---|---|
| README | [`README.md`](../README.md) | ✔ |
| `requirements.txt` | [`requirements.txt`](../requirements.txt) | ✔ |
| Source code | [`src/`](../src), [`scripts/`](../scripts) | ✔ |
| Reproducible results | `results/raw/*.json` (config + metrics + env per run); fixed seeds; `RESULTS.md` generated from the JSONs | ✔ |
| Confusion matrix | `results/figures/fig4–fig6*.png`, plus text matrices in `RESULTS.md` § CO3 | ✔ |
| Performance metrics | Accuracy, macro F1, Cohen κ, and per-class sensitivity / positive predictivity / F1 — `RESULTS.md` § CO3 | ✔ |
| Ablation study | 6-row one-factor-at-a-time table + two figures — `RESULTS.md` § CO5 | ✔ |

---

## CO3 (10 marks) — Implement a DL model and apply it to a dataset

| Element | Evidence |
|---|---|
| Model implemented from scratch | [`src/models.py`](../src/models.py) — `BaselineCNN`, `MultiScaleCNN`; no pretrained weights, no model zoo |
| Custom loss implemented | [`src/losses.py`](../src/losses.py) — class-balanced focal loss written out, not imported |
| Applied to a real dataset | MIT-BIH Arrhythmia Database, 44 records, ~100k beats — [`src/data/`](../src/data) |
| Full training pipeline | [`src/train.py`](../src/train.py) — early stopping on validation macro-F1, LR scheduling, gradient clipping |
| Metrics reported | `RESULTS.md` § CO3: accuracy, macro F1, κ, and per-class Se / P+ / F1 |
| Confusion matrix | `fig4` (baseline), `fig5` (proposed), `fig6` (leaky split) + text versions |
| Correctness evidence | [`tests/test_pipeline.py`](../tests/test_pipeline.py) — 14 tests incl. label mapping, R-peak alignment, hand-worked metric check |

**Talking point for the viva.** The confusion matrix is not decoration here. Read
row S: the dominant off-diagonal cell is S→N. That is the model calling a
premature atrial beat normal, which is precisely the failure the rhythm-context
branch was built to address — and you can show it shrinking in the ablation.

---

## CO4 (5 marks) — Compare with state of the art on the same dataset

| Element | Evidence |
|---|---|
| Same dataset, same protocol | MIT-BIH under the de Chazal DS1/DS2 inter-patient split — the identical protocol every cited paper used |
| Comparison table | `RESULTS.md` § CO4 — 6 published methods, 2004–2023, with per-class Se/P+ |
| Sources cited | [`src/sota.py`](../src/sota.py) — every row carries its source and a protocol note |
| Critical reading | The table spans 86% → 99.9% accuracy on nominally one protocol; § CO4 explains why the high entries (SMOTE / augmentation) are not a like-for-like comparison |

**Talking point.** Anyone can paste a comparison table. The mark is in noticing
that the table is internally inconsistent — a 14-point spread on a fixed split
with a fixed test set means the protocols are not actually identical — and
saying which rows are safe to compare against.

---

## CO5 (5 marks) — Novel contribution, validated by ablation

Two contributions, one methodological and one architectural.

### Contribution 1 — quantify the leakage

Train the identical model under both protocols and measure the gap.

- Configurations `E1` vs `E2` (baseline model) and `E3` vs `E4` (proposed model)
- Figures `fig2`, `fig3`
- `src/data/splits.py::patient_overlap` reports, as a number in the results, how
  many test beats came from a patient the model trained on — the claim is
  evidenced, not asserted
- Corroborated independently by arXiv:2503.07276 (99.48% intra vs 88.34% inter)

### Contribution 2 — recover part of the gap

Multi-scale beat + rhythm-context architecture with class-balanced focal loss.

| Ablation row | Isolates |
|---|---|
| `A1` − focal loss (plain CE) | the imbalance handling |
| `A2` focal → inverse-frequency weighted CE | whether the focal term beats plain reweighting |
| `A3` − rhythm-context branch | the multi-scale idea itself |
| `A4` − RR-interval features | explicit timing information |
| `A5` − second lead | the value of the second channel |

Each row differs from `E4` in exactly one component, so the Δ column is
attributable. Figures `fig7` (macro F1) and `fig8` (S sensitivity).

**Talking point.** Report the ablation on **S sensitivity**, not just accuracy.
Overall accuracy moves by a point or two because class N dominates; S
sensitivity moves far more. That contrast — the headline metric barely reacting
while the clinically important one moves sharply — is itself the argument for
why this literature should stop leading with accuracy.

---

## Suggested report structure

1. **Introduction** — the task, why beat classification matters clinically
2. **Dataset** — MIT-BIH, AAMI classes, `fig1` class distribution, the imbalance
3. **Method** — architecture diagram, the loss, the two protocols
4. **Experimental setup** — the 9-configuration table, seeds, hardware
5. **Results (CO3)** — main table, per-class table, confusion matrices
6. **Comparison with published work (CO4)** — the SOTA table *and* the critique
7. **Ablation study (CO5)** — both contributions, `fig2`/`fig3`, `fig7`/`fig8`
8. **Limitations** — the list in `README.md`; do not omit this section
9. **Conclusion**

Sections 5–7 can be assembled almost verbatim from the generated `RESULTS.md`.
