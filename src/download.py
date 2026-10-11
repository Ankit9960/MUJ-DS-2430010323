"""Fetch the MIT-BIH Arrhythmia Database (PhysioNet, mitdb v1.0.0).

Roughly 100 MB for the 44 non-paced records. Downloading is idempotent: records
already present on disk are skipped.
"""
from __future__ import annotations

import sys

from src import config as C

PHYSIONET_DB = "mitdb"
EXTENSIONS = (".dat", ".hea", ".atr")


def records_present(record_ids=None) -> list[int]:
    record_ids = record_ids or C.ALL_RECORDS
    return [r for r in record_ids
            if all((C.DATA_RAW / f"{r}{ext}").exists() for ext in EXTENSIONS)]


def download(record_ids=None, verbose: bool = True) -> None:
    try:
        import wfdb
    except ImportError:
        sys.exit("wfdb is not installed. Run:  pip install -r requirements.txt")

    record_ids = record_ids or C.ALL_RECORDS
    have = set(records_present(record_ids))
    todo = [r for r in record_ids if r not in have]

    if verbose:
        print(f"MIT-BIH: {len(have)} of {len(record_ids)} records already present.")
    if not todo:
        return

    if verbose:
        print(f"Downloading {len(todo)} records from PhysioNet -> {C.DATA_RAW}")
    wfdb.dl_database(PHYSIONET_DB, dl_dir=str(C.DATA_RAW),
                     records=[str(r) for r in todo], annotators=["atr"])

    missing = [r for r in record_ids if r not in set(records_present(record_ids))]
    if missing:
        raise RuntimeError(
            f"Download incomplete; missing records: {missing}. "
            "Check the network connection and re-run.")
    if verbose:
        print("Download complete.")


if __name__ == "__main__":
    download()
