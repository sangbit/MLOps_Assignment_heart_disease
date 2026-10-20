"""Shared test fixtures (pytest loads this file automatically)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURES
from src.preprocessing import build_pipeline


@pytest.fixture
def raw_patients() -> pd.DataFrame:
    """A small, hand-made set of raw patients, including the awkward cases on purpose."""
    return pd.DataFrame([
        {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
         "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6},
        {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
         "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3},
        # impossible zeros + missing values
        {"age": 54, "sex": 0, "cp": 3, "trestbps": 0, "chol": 0, "fbs": None, "restecg": 0,
         "thalach": 155, "exang": 0, "oldpeak": 0.0, "slope": None, "ca": None, "thal": None},
    ])[FEATURES]


@pytest.fixture
def synthetic_training_data():
    """200 random but learnable patients, so model tests don't depend on the real files."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "age": rng.normal(54, 9, n), "sex": rng.integers(0, 2, n),
        "cp": np.where(y == 1, 4, rng.choice([1, 2, 3], n)),
        "trestbps": rng.normal(132, 17, n), "chol": rng.normal(240, 50, n),
        "fbs": rng.integers(0, 2, n), "restecg": rng.choice([0, 1, 2], n),
        "thalach": rng.normal(150, 20, n) - 20 * y, "exang": y,
        "oldpeak": np.abs(rng.normal(1, 1, n)) + y, "slope": rng.choice([1, 2, 3], n),
        "ca": rng.choice([0, 1, 2, 3], n), "thal": rng.choice([3, 6, 7], n),
    }).astype(float)
    X.loc[::10, ["ca", "thal"]] = np.nan          # some missing values, like the real data
    return X, pd.Series(y, name="target")


@pytest.fixture
def fitted_pipeline(synthetic_training_data):
    X, y = synthetic_training_data
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
