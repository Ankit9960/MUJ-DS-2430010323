"""Reproducibility helpers and small shared utilities."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run.

    Determinism matters here: the whole CO5 argument rests on a difference
    between conditions, so run-to-run noise must be measured (multiple seeds),
    not accidental.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0


def environment_report() -> dict:
    """Captured into every result file so runs are traceable."""
    import platform
    info = dict(python=platform.python_version(),
                platform=platform.platform(),
                numpy=np.__version__)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return info
