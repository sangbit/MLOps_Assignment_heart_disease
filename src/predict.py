"""Load the packaged model and predict heart-disease risk from raw patient values.

Run from the project root:
    python -m src.predict examples/sample_patients.json

Input: a JSON file with one patient (an object) or several (a list of objects), using the
13 raw feature names. Missing values can be null or left out; 0 for trestbps/chol means
"not recorded". No preprocessing is needed -- the saved pipeline does it.
"""

import argparse
import json

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_DIR

_model = None
_card = None


def load_model():
    """Load the saved pipeline once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "heart_disease_model.joblib")
    return _model


def load_model_card() -> dict:
    """Load the model's metadata (threshold, version info, metrics) once and reuse it."""
    global _card
    if _card is None:
        _card = json.loads((MODEL_DIR / "model_card.json").read_text())
    return _card


def predict(patients, threshold: float | None = None) -> list[dict]:
    """patients: a dict or list of dicts with raw feature values. Returns one result per patient."""
    if isinstance(patients, dict):
        patients = [patients]
    unknown = {k for p in patients for k in p} - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}. Expected: {FEATURES}")

    if threshold is None:
        threshold = load_model_card()["decision_threshold"]

    X = pd.DataFrame(patients).reindex(columns=FEATURES)
    proba = load_model().predict_proba(X)[:, 1]
    results = []
    for p in proba:
        disease = bool(p >= threshold)
        results.append({
            "prediction": int(disease),
            "label": "Disease" if disease else "No disease",
            "confidence": round(float(p if disease else 1 - p), 4),   # probability of the predicted class
            "disease_probability": round(float(p), 4),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart-disease risk from a JSON file.")
    parser.add_argument("input", help="JSON file with one patient object or a list of them")
    args = parser.parse_args()

    with open(args.input) as f:
        patients = json.load(f)
    for i, result in enumerate(predict(patients), start=1):
        print(f"Patient {i}: {result['label']:<10}  confidence {result['confidence']:.2f}  "
              f"(disease probability {result['disease_probability']:.2f})")


if __name__ == "__main__":
    main()
