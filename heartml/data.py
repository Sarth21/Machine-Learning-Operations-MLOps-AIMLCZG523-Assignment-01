"""Data loading, validation and cleaning."""
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heartml import config as C

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


def load_raw(path=None):
    path = Path(path or C.DATA_RAW)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path, na_values=["?", "", " ", "NA", "nan"])


def validate_schema(df):
    missing = [c for c in C.FEATURES + [C.TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def clean(df):
    """Validate, coerce to numeric, null impossible values, drop duplicates, binarise target.

    Missing values are KEPT (NaN); imputation happens inside the sklearn pipeline so it is
    fitted on training folds only (no leakage).
    """
    validate_schema(df)
    out = df[C.FEATURES + [C.TARGET]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        if bad.any():
            log.warning("%d out-of-range values in %s set to NaN", int(bad.sum()), col)
            out.loc[bad, col] = np.nan
    for col, allowed in C.CATEGORICAL_VALUES.items():
        bad = out[col].notna() & ~out[col].isin(allowed)
        if bad.any():
            log.warning("%d invalid categories in %s set to NaN", int(bad.sum()), col)
            out.loc[bad, col] = np.nan
    out = out.drop_duplicates().reset_index(drop=True)
    out = out[out[C.TARGET].notna()].copy()
    out[C.TARGET] = (out[C.TARGET] > 0).astype(int)
    return out.reset_index(drop=True)


def split_xy(df):
    return df[C.FEATURES].copy(), df[C.TARGET].astype(int).copy()


def train_test(df, test_size=C.TEST_SIZE, seed=C.RANDOM_STATE):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def data_source():
    if C.DATA_SOURCE_FILE.exists():
        return C.DATA_SOURCE_FILE.read_text().strip().splitlines()[0]
    return "unknown"


def is_synthetic():
    return data_source().lower().startswith("synthetic")


def prepare(raw_path=None, out_path=None):
    df = clean(load_raw(raw_path))
    out_path = Path(out_path or C.DATA_CLEAN)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote cleaned dataset (%d rows) to %s", len(df), out_path)
    return df


def load_clean(path=None):
    path = Path(path or C.DATA_CLEAN)
    if not path.exists():
        return prepare()
    return pd.read_csv(path)
