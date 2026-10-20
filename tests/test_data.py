"""Tests for src/data.py: loading and validating the raw files."""

import pandas as pd
import pytest

from src.config import DATA_DIR, FEATURES, SOURCES, TARGET
from src.data import data_fingerprint, load_data

# ---------- Tests on the real data files (committed in source_data/) ----------

@pytest.fixture(scope="module")
def df():
    return load_data()


def test_all_four_sources_loaded(df):
    assert set(df["source"]) == set(SOURCES)


def test_expected_row_count(df):
    # Cleveland 303 + Hungary 294 + Switzerland 123 + VA 200
    assert len(df) == 920


def test_has_all_feature_columns_and_target(df):
    assert set(FEATURES + [TARGET, "source"]) <= set(df.columns)
    assert "num" not in df.columns, "the original 0-4 diagnosis must be replaced by the binary target"


def test_target_is_binary(df):
    assert set(df[TARGET].unique()) <= {0, 1}


def test_features_are_numeric(df):
    for col in FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_question_marks_became_missing_values(df):
    # '?' in the raw files must load as NaN, never as text
    assert df[FEATURES].isna().any().any()


def test_values_in_valid_ranges(df):
    assert df["age"].between(18, 100).all()
    assert df["sex"].isin([0, 1]).all()
    assert df["cp"].isin([1, 2, 3, 4]).all()
    assert df["ca"].dropna().isin([0, 1, 2, 3]).all()
    assert df["thal"].dropna().isin([3, 6, 7]).all()


def test_fingerprint_is_stable():
    assert data_fingerprint() == data_fingerprint()
    assert len(data_fingerprint()) == 16


# ---------- Tests for the validation errors, using temporary fake files ----------

def write_files(folder, row="63,1,1,145,233,1,2,150,0,2.3,3,0,6,0"):
    for filename in SOURCES.values():
        (folder / filename).write_text(row + "\n")


def test_missing_file_raises(tmp_path):
    write_files(tmp_path)
    (tmp_path / SOURCES["va"]).unlink()
    with pytest.raises(FileNotFoundError, match="processed.va.data"):
        load_data(tmp_path)


def test_wrong_column_count_raises(tmp_path):
    write_files(tmp_path, row="63,1,1,145")
    with pytest.raises(ValueError, match="expected 14 columns"):
        load_data(tmp_path)


def test_non_numeric_value_raises(tmp_path):
    write_files(tmp_path, row="63,1,1,145,abc,1,2,150,0,2.3,3,0,6,0")
    with pytest.raises(ValueError, match="non-numeric"):
        load_data(tmp_path)


def test_target_binarized_from_num(tmp_path):
    write_files(tmp_path, row="63,1,1,145,233,1,2,150,0,2.3,3,0,6,3")   # num = 3
    assert load_data(tmp_path)[TARGET].eq(1).all()


def test_real_data_dir_exists():
    assert DATA_DIR.exists(), f"Data folder not found: {DATA_DIR}"
