"""The preprocessing pipeline: every transformation between raw patient values and the model.

Because all of it sits inside one scikit-learn Pipeline, the saved model file contains the
*fitted* preprocessing too (medians, most-frequent values, scaling statistics, category lists).
Training, batch prediction and the API therefore transform data identically -- no
training-serving skew.

    raw values ──► clean_inputs ──► impute ──► scale / one-hot ──► model
                   (zeros → NaN,     (median /    (numeric /
                    cast to float)    most freq)   categorical)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BINARY, CATEGORICAL, FEATURES, NUMERIC, SPARSE, ZERO_MEANS_MISSING


def clean_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Step 1: keep the expected columns, cast to float, and turn impossible zeros into NaN."""
    X = pd.DataFrame(X, columns=FEATURES).astype(float)
    X[ZERO_MEANS_MISSING] = X[ZERO_MEANS_MISSING].replace(0, np.nan)
    return X


def clean_feature_names(transformer, input_features=None):
    """Column names produced by clean_inputs: always FEATURES, whatever order the input had."""
    return np.asarray(FEATURES, dtype=object)


def build_preprocessor(drop_sparse: bool = False) -> Pipeline:
    """Cleaning + imputation + scaling + encoding, as one unfitted transformer."""
    numeric = [c for c in NUMERIC if not (drop_sparse and c in SPARSE)]
    categorical = [c for c in CATEGORICAL if not (drop_sparse and c in SPARSE)]

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary_steps = SimpleImputer(strategy="most_frequent")

    columns = ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
        ("bin", binary_steps, BINARY),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean_inputs, feature_names_out=clean_feature_names)),
        ("columns", columns),
    ])


def build_pipeline(model, drop_sparse: bool = False) -> Pipeline:
    """Full pipeline: preprocessing followed by the classifier."""
    return Pipeline([("prep", build_preprocessor(drop_sparse)), ("model", model)])
