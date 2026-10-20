"""Tests for src/preprocessing.py: the transformations between raw values and the model."""

import numpy as np
import pandas as pd

from src.config import FEATURES
from src.preprocessing import build_preprocessor, clean_inputs


def test_clean_inputs_turns_impossible_zeros_into_missing(raw_patients):
    cleaned = clean_inputs(raw_patients)
    assert np.isnan(cleaned.loc[2, "chol"])
    assert np.isnan(cleaned.loc[2, "trestbps"])


def test_clean_inputs_keeps_real_zeros(raw_patients):
    # 0 is a valid value for e.g. oldpeak, ca, sex -- only chol/trestbps zeros are "missing"
    cleaned = clean_inputs(raw_patients)
    assert cleaned.loc[2, "oldpeak"] == 0.0
    assert cleaned.loc[0, "ca"] == 0.0


def test_clean_inputs_casts_to_float_and_orders_columns(raw_patients):
    shuffled = raw_patients[list(reversed(FEATURES))]
    cleaned = clean_inputs(shuffled)
    assert list(cleaned.columns) == FEATURES
    assert all(dtype == "float64" for dtype in cleaned.dtypes)


def test_clean_inputs_does_not_modify_the_caller_data(raw_patients):
    before = raw_patients.copy()
    clean_inputs(raw_patients)
    pd.testing.assert_frame_equal(raw_patients, before)


def test_preprocessor_output_has_no_missing_values(synthetic_training_data, raw_patients):
    X, _ = synthetic_training_data
    out = build_preprocessor().fit(X).transform(raw_patients)
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_preprocessor_output_shape_is_consistent(synthetic_training_data, raw_patients):
    X, _ = synthetic_training_data
    prep = build_preprocessor().fit(X)
    n_features = len(prep.get_feature_names_out())
    assert prep.transform(raw_patients).shape == (len(raw_patients), n_features)
    assert prep.transform(raw_patients.iloc[[0]]).shape == (1, n_features)


def test_preprocessor_one_hot_encodes_categories(synthetic_training_data):
    X, _ = synthetic_training_data
    names = build_preprocessor().fit(X).get_feature_names_out()
    assert any("cp_4" in n for n in names)
    assert not any(n.endswith("__cp") for n in names), "cp must be one-hot encoded, not used as a number"


def test_preprocessor_handles_unseen_category(synthetic_training_data, raw_patients):
    X, _ = synthetic_training_data
    prep = build_preprocessor().fit(X)
    odd = raw_patients.copy()
    odd.loc[0, "restecg"] = 9          # never seen in training
    assert prep.transform(odd).shape[0] == len(odd)


def test_numeric_features_are_scaled(synthetic_training_data):
    X, _ = synthetic_training_data
    prep = build_preprocessor().fit(X)
    out = pd.DataFrame(np.asarray(prep.transform(X), dtype=float), columns=prep.get_feature_names_out())
    age = out[[c for c in out.columns if c.endswith("age")][0]]
    assert abs(age.mean()) < 1e-6 and abs(age.std(ddof=0) - 1) < 1e-6


def test_drop_sparse_removes_sparse_columns(synthetic_training_data):
    X, _ = synthetic_training_data
    kept = build_preprocessor(drop_sparse=False).fit(X).get_feature_names_out()
    dropped = build_preprocessor(drop_sparse=True).fit(X).get_feature_names_out()
    assert len(dropped) < len(kept)
    source_columns = {n.split("__")[1].split("_")[0] for n in dropped}   # "cat__cp_4.0" -> "cp"
    assert not {"ca", "thal", "slope"} & source_columns
