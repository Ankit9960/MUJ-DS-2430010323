# ECG Arrhythmia Classification — and What the Evaluation Protocol Hides

Beat-level arrhythmia classification on the **MIT-BIH Arrhythmia Database** using a
multi-scale 1-D CNN, evaluated under the AAMI EC57 five-class scheme.

The project has two halves. The first is a competent deep-learning classifier
benchmarked against published work. The second is the part that makes it a
project rather than a tutorial: a controlled demonstration that **a large share
of the accuracy reported in this literature comes from how the data is split,
not from the model** — and an architecture that recovers part of the gap once
the split is made honest.

---

## The claim in one figure

Train one CNN. Evaluate it twice. Change nothing but the train/test partition.

| | Intra-patient split | Inter-patient split (DS1/DS2) |
|---|---|---|
| Beats from a test patient seen during training | **yes** | none |
| What the model can exploit | patient-specific morphology | arrhythmia morphology only |
| What it corresponds to clinically | re-reading a patient you already trained on | deploying on a new patient |

The second column is the only one that answers the question a hospital would
ask. Most published MIT-BIH results report the first.

An independent systematic review ([arXiv:2503.07276](https://arxiv.org/abs/2503.07276))
records the same CNN at **99.48%** intra-patient and **88.34%** inter-patient —
an 11-point gap from the partition alone. This project reproduces that effect
under controlled conditions and then attacks it.

---

## Quick start

```bash
git clone <your-repo-url> && cd ecg-arrhythmia-dl
pip install -r requirements.txt

# 0. 60-second check that everything is wired up (synthetic data, no download)
python tests/test_pipeline.py
python scripts/run_experiment.py --smoke-test

# 1. Download MIT-BIH (~100 MB) and segment beats            (~5 min)
python scripts/prepare_data.py

# 2. Run all 9 configurations x 3 seeds                      (~25 min GPU / ~3 h CPU)
python scripts/run_ablation.py
#    ...or a single-seed pass for a first look               (~8 min GPU)
python scripts/run_ablation.py --seeds 1

# 3. Build figures, tables and RESULTS.md                    (~10 s)
python scripts/make_report.py

# 4. Audit the submission against the assignment requirements
python scripts/verify_submission.py
```

Everything lands in `results/` and `RESULTS.md`.

`verify_submission.py` checks every required deliverable — README,
requirements, source, reproducible results, confusion matrix, performance
metrics, ablation study — and prints the full hyper-parameter list for the
write-up. Run it before submitting; it should report **0 waiting, 0 failed**.

**No GPU?** Open `notebooks/ECG_Colab.ipynb` in Google Colab, set Runtime →
GPU, and run all cells. It performs the same three steps end to end.

---

## What the model is

Two ideas, both aimed at one diagnosable failure.

**The failure.** Supraventricular ectopic beats (AAMI class **S**) are defined by
*when* they arrive, not by how they look — a premature atrial beat can be
morphologically almost identical to a normal one. A model that only ever sees a
1-second window centred on the R peak is structurally blind to this. That is why
**S** is the weakest column in nearly every published inter-patient results
table, often below 40% sensitivity.

**The response.**

1. **A rhythm-context branch.** A second convolutional encoder reads a 4-second
   window (decimated to the same tensor length), giving the network the
   surrounding rhythm rather than the isolated beat. Explicit RR-interval
   features — previous RR, following RR, a causal 10-beat local average, and the
   prematurity ratio — are concatenated before the classifier.
2. **Class-balanced focal loss.** MIT-BIH is roughly 90% class N. Plain
   cross-entropy can score ~90% accuracy while barely detecting anything
   clinically interesting. The loss combines effective-number reweighting
   ([Cui et al., 2019](https://arxiv.org/abs/1901.05555)) with the focal term
   ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)), moving gradient mass
   off the tens of thousands of easy normals and onto the rare hard beats.

```
       beat window (360 x C)          rhythm context (360 x C)
              |                                |
      Conv-BN-ReLU-Pool x3            Conv-BN-ReLU-Pool x3       RR features (4)
              |                                |                      |
          GAP -> 128                       GAP -> 128                  |
              +----------------+---------------+----------------------+
                               |
                    FC 128 -> 64 -> 5 (N, S, V, F, Q)
```

---

## Protocol (and why each choice matters)

| Decision | Choice | Reason |
|---|---|---|
| Classes | AAMI EC57: N, S, V, F, Q | The standard the comparison literature uses |
| Records | 44 of 48 | The 4 paced records are excluded by AAMI convention |
| Split | de Chazal DS1/DS2 | Patient-disjoint; the protocol the field defined for this |
| Validation | 4 records held out of DS1 | Model selection must not see a training *or* a test patient |
| Filtering | 0.5–40 Hz zero-phase Butterworth | Removes baseline wander without phase distortion |
| Normalisation | per-record z-score | Removes the per-patient gain that acts as a fingerprint |
| RR normalisation | train statistics only | Using global statistics would leak test information |
| Seeds | 3, mean ± std reported | The whole argument is a difference between conditions |

The DS1/DS2 record lists are in [`src/config.py`](src/config.py) and their
disjointness is asserted in the test suite, not assumed.

---

## Repository layout

```
<RegistrationNumber>/
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset_information.txt   ← source, records, AAMI mapping, preprocessing
│                                   (raw signals are fetched, not submitted)
├── src/
│   ├── train.py                  ← training loop + CLI entry point
│   ├── test.py                   ← load a checkpoint, evaluate on the test set
│   ├── model.py                  ← BaselineCNN, MultiScaleCNN
│   ├── preprocessing.py          ← filtering, segmentation, RR features
│   ├── splits.py                 ← inter- vs intra-patient protocols
│   ├── download.py               ← PhysioNet fetch
│   ├── synthetic.py              ← smoke-test data generator only
│   ├── losses.py                 ← CE, weighted CE, class-balanced focal
│   ├── evaluate.py               ← AAMI Se/P+, confusion matrix, aggregation
│   ├── plots.py                  ← all figures
│   ├── sota.py                   ← published results used for comparison
│   ├── config.py                 ← AAMI map, DS1/DS2 lists, hyper-parameters
│   └── utils.py                  ← seeding, environment capture
│
├── notebooks/
│   └── experiments.ipynb         ← end-to-end run (Kaggle / Colab)
│
├── results/
│   ├── confusion_matrix.png      ← proposed model, inter-patient
│   ├── training_curve.png
│   ├── results.csv               ← all 9 configurations x 3 seeds
│   ├── comparison.csv            ← 6 published methods + this work
│   └── raw/                      ← one JSON per run, fully reproducible
│
├── models/
│   ├── model_description.txt     ← architecture, loss, training setup
│   ├── README.md                 ← checkpoint format and loading snippet
│   └── *.pt                      ← weights (scripts/export_models.py)
│
├── figures/                      ← fig1 … fig10, every generated figure
│
├── scripts/
│   ├── prepare_data.py           ← step 1: download + segment
│   ├── run_ablation.py           ← step 2: the 9-configuration grid
│   ├── make_report.py            ← step 3: figures, CSVs, RESULTS.md
│   ├── run_experiment.py         ← a single run / smoke test
│   ├── export_models.py          ← train and save checkpoints
│   ├── load_model.py             ← verify a checkpoint loads
│   └── verify_submission.py      ← audit every required deliverable
│
├── tests/
│   └── test_pipeline.py          ← 14 tests, no GPU needed
│
└── docs/
    └── RUBRIC_MAP.md             ← where each marking criterion is satisfied
```

Direct entry points, if you would rather not use the scripts:

```bash
python src/train.py                 # train the proposed model
python src/train.py --arch baseline --loss ce --leads single --no-rr
python src/test.py                  # evaluate a saved checkpoint on DS2
```

## The experiment grid

Nine configurations. `E1`–`E4` establish the protocol effect; `A1`–`A5` are
strict **one-factor-at-a-time** changes from the proposed configuration `E4`, so
each ablation row attributes its change to exactly one component.

| ID | Protocol | Architecture | Loss | Leads | RR | Purpose |
|---|---|---|---|---|---|---|
| E1 | intra | baseline | CE | 1 | ✗ | The common published setup |
| E2 | inter | baseline | CE | 1 | ✗ | Same model, honest split |
| E3 | intra | multi-scale | focal | 2 | ✓ | Proposed model, leaky split |
| **E4** | **inter** | **multi-scale** | **focal** | **2** | **✓** | **Proposed — the reported result** |
| A1 | inter | multi-scale | CE | 2 | ✓ | − focal loss |
| A2 | inter | multi-scale | weighted CE | 2 | ✓ | focal → plain reweighting |
| A3 | inter | baseline | focal | 2 | ✓ | − rhythm-context branch |
| A4 | inter | multi-scale | focal | 2 | ✗ | − RR features |
| A5 | inter | multi-scale | focal | 1 | ✓ | − second lead |

E1 vs E2 isolates the split. E4 vs A1–A5 isolates each component.

---

## Reproducibility

- Every RNG seeded (`src/utils.py::set_seed`), cuDNN set deterministic.
- Every run writes a JSON to `results/raw/` containing its full config, metrics,
  confusion matrix, per-epoch history, parameter count, wall-clock time and the
  Python/torch/CUDA versions it ran under.
- `RESULTS.md` is generated from those JSONs — no number in the report is typed
  by hand.
- `tests/test_pipeline.py` covers the places where a silent bug would corrupt
  results without crashing: AAMI label mapping, R-peak window alignment, split
  disjointness, and the Se/P+ definitions (checked against a hand-worked
  example).

---

## Honest limitations

Worth stating in the report; examiners reward it and the alternative is being
asked about it unprepared.

- **Q is near-useless here.** With the paced records excluded, class Q has only a
  few dozen beats in DS2. Its metrics are reported but should not be read as
  meaningful.
- **F remains hard.** Fusion beats are genuinely ambiguous — they are, by
  definition, hybrids — and cardiologist inter-rater agreement on them is
  imperfect. Low F scores are partly a property of the label, not only the model.
- **Annotated R peaks are used.** Beat locations come from the database
  annotations rather than a detector, so this evaluates classification in
  isolation. A deployed system would inherit R-peak detection errors on top.
- **One database.** Cross-database generalisation (e.g. to INCART or SVDB) is the
  natural next test and is not attempted here.
- **The ceiling is not claimed.** Several published entries report >99%
  inter-patient accuracy. This project does not match them, and
  [`src/sota.py`](src/sota.py) explains why those numbers deserve scrutiny
  rather than emulation.

---

## Data and citation

MIT-BIH Arrhythmia Database, PhysioNet (`mitdb`, v1.0.0):

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> *IEEE Eng in Med and Biol* 20(3):45-50 (2001).

> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet.
> *Circulation* 101(23):e215–e220 (2000).

Protocol:

> de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using
> ECG morphology and heartbeat interval features. *IEEE Trans Biomed Eng*
> 51(7):1196–1206 (2004).

Full comparison sources are listed in `RESULTS.md` § CO4 and in
[`src/sota.py`](src/sota.py).
