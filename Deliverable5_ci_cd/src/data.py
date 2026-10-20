"""Load and validate the four UCI heart-disease files into one DataFrame.

Only *structural* work happens here (reading, checking, labelling the target).
Value cleaning (impossible zeros, missing values, scaling, encoding) lives in the
model pipeline -- see preprocessing.py -- so training and prediction share it.
"""

import hashlib

import pandas as pd

from src.config import DATA_DIR, RAW_COLUMNS, SOURCES, TARGET


def load_data(data_dir=DATA_DIR) -> pd.DataFrame:
    """Return all four sources combined, with a `source` column and a binary `target`."""
    frames = []
    for source, filename in SOURCES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        raw = pd.read_csv(path, header=None, na_values="?")
        if raw.shape[1] != len(RAW_COLUMNS):
            raise ValueError(f"{filename}: expected {len(RAW_COLUMNS)} columns, found {raw.shape[1]}")
        raw.columns = RAW_COLUMNS

        non_numeric = [c for c in RAW_COLUMNS if not pd.api.types.is_numeric_dtype(raw[c])]
        if non_numeric:
            raise ValueError(f"{filename}: non-numeric columns {non_numeric}")

        raw["source"] = source
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)
    df[TARGET] = (df["num"] > 0).astype(int)   # 0 = no disease, 1-4 = disease
    return df.drop(columns="num")


def data_fingerprint(data_dir=DATA_DIR) -> str:
    """SHA-256 over the raw files, so a saved model records exactly which data trained it."""
    digest = hashlib.sha256()
    for filename in sorted(SOURCES.values()):
        digest.update((data_dir / filename).read_bytes())
    return digest.hexdigest()[:16]
