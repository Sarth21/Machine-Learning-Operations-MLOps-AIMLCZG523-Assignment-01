#!/usr/bin/env python
"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/heart.csv.

Order tried: (1) ucimlrepo (id 45), (2) direct HTTPS from archive.ics.uci.edu,
(3) --synthetic stand-in (OFFLINE DRY RUNS ONLY, clearly marked in DATA_SOURCE.txt).
"""
import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from heartml import config as C  # noqa: E402


def from_ucimlrepo():
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=C.UCI_ID)
    df = pd.concat([ds.data.features.copy(), ds.data.targets.copy()], axis=1)
    df.columns = C.UCI_COLUMNS
    return df


def from_url():
    import requests

    resp = requests.get(C.UCI_CLEVELAND_URL, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), header=None, names=C.UCI_COLUMNS, na_values="?")


def write(df, source):
    C.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATA_RAW, index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    C.DATA_SOURCE_FILE.write_text(f"{source}\nrows={len(df)}\nretrieved={stamp}\n")
    print(f"Saved {len(df)} rows -> {C.DATA_RAW}  [{source}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing data/raw/heart.csv")
    ap.add_argument("--synthetic", action="store_true", help="synthetic stand-in (offline only)")
    args = ap.parse_args()
    if args.synthetic:
        from heartml.synthetic import make_synthetic_heart

        write(make_synthetic_heart(), "synthetic-standin (NOT real patient data)")
        print("WARNING: synthetic data - results are NOT clinically meaningful.", file=sys.stderr)
        return 0
    if C.DATA_RAW.exists() and not args.force:
        print(f"{C.DATA_RAW} already exists (use --force to re-download).")
        return 0
    errors = []
    for name, fn, label in [("ucimlrepo", from_ucimlrepo, "UCI Heart Disease id=45 via ucimlrepo"),
                            ("direct URL", from_url, "UCI processed.cleveland.data via HTTPS")]:
        try:
            write(fn(), label)
            return 0
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    print("Could not download the dataset:\n  " + "\n  ".join(errors), file=sys.stderr)
    print("Manual: get processed.cleveland.data from https://archive.ics.uci.edu/dataset/45/heart+disease"
          "\nand save as data/raw/heart.csv with header: " + ",".join(C.UCI_COLUMNS), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
