"""Central configuration: AAMI class mapping, inter-patient splits, signal constants.

All protocol-defining constants live here so that every experiment is traceable
to a single source of truth (important for reproducibility claims in the report).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"        # wfdb records downloaded from PhysioNet
DATA_PROC = ROOT / "data" / "processed"  # cached beat arrays (.npz)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"      # top-level, per the required layout
TABLES = RESULTS / "tables"
RAW_RESULTS = RESULTS / "raw"
MODELS_DIR = ROOT / "models"    # trained checkpoints

for _p in (DATA_RAW, DATA_PROC, RESULTS, FIGURES, TABLES, RAW_RESULTS, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Signal constants (MIT-BIH Arrhythmia Database)
# --------------------------------------------------------------------------
FS = 360                  # native sampling rate (Hz)
BANDPASS = (0.5, 40.0)    # Butterworth bandpass cut-offs (Hz)
FILTER_ORDER = 3

# Beat window: 0.5 s before / 0.5 s after the annotated R peak -> 360 samples
BEAT_PRE = 180
BEAT_POST = 180
BEAT_LEN = BEAT_PRE + BEAT_POST          # 360

# Rhythm-context window: 2 s before / 2 s after, decimated by 4 -> 360 samples.
# Same tensor length as the beat window, so both branches share an architecture.
CTX_PRE = 720
CTX_POST = 720
CTX_DECIM = 4
CTX_LEN = (CTX_PRE + CTX_POST) // CTX_DECIM  # 360

N_RR_FEATURES = 4         # pre-RR, post-RR, local-RR, pre/local ratio

# --------------------------------------------------------------------------
# AAMI EC57 class mapping
# --------------------------------------------------------------------------
# N = normal / bundle-branch block, S = supraventricular ectopic,
# V = ventricular ectopic, F = fusion, Q = unknown / paced
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}

SYMBOL_TO_AAMI = {
    # N
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V
    "V": "V", "E": "V",
    # F
    "F": "F",
    # Q
    "/": "Q", "f": "Q", "Q": "Q",
}

# Annotation symbols that mark rhythm/quality events rather than heartbeats.
NON_BEAT_SYMBOLS = set('+~|x][!"')

# --------------------------------------------------------------------------
# Record lists
# --------------------------------------------------------------------------
# The four paced records are excluded by AAMI EC57 convention.
PACED_RECORDS = [102, 104, 107, 217]

# de Chazal et al. (2004) inter-patient division. DS1 and DS2 contain disjoint
# patients, which is the whole point of the protocol.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

ALL_RECORDS = sorted(DS1 + DS2)

# Records held out of DS1 to form an inter-patient validation set. Model
# selection must never see a test patient, and must not see a training patient
# either, or the validation score is itself leaked.
DS1_VAL = [114, 124, 205, 223]
DS1_TRAIN = [r for r in DS1 if r not in DS1_VAL]

# Record 114 has its two leads stored in reversed order relative to the rest.
REVERSED_LEAD_RECORDS = {114}

# --------------------------------------------------------------------------
# Default training hyper-parameters
# --------------------------------------------------------------------------
DEFAULTS = dict(
    epochs=30,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    patience=8,          # early stopping on validation macro-F1
    focal_gamma=2.0,
    cb_beta=0.9999,       # class-balanced loss (Cui et al., 2019)
    dropout=0.3,
)

SEEDS = [0, 1, 2]
