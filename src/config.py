"""Project-wide constants: file locations, column names and feature groups.

Every other module imports from here, so a column or path is defined in exactly one place.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "source_data"
MODEL_DIR = PROJECT_ROOT / "models"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# The four UCI "processed" files: source name -> file name
SOURCES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

# Column order inside every raw file (14 columns, no header)
RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Feature groups decide how each column is prepared (see preprocessing.py)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

# Heavily missing outside Cleveland; params.yaml decides whether to keep them
SPARSE = ["ca", "thal", "slope"]

# A value of 0 is physiologically impossible for these: treat as "not recorded"
ZERO_MEANS_MISSING = ["trestbps", "chol"]

TARGET = "target"
