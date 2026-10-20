"""Tests for the model pipeline, training helpers and the prediction interface."""

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import src.predict as predict_module
from src.preprocessing import build_pipeline
from src.train import check_quality_gate, make_model

# ---------- Model pipeline ----------

def test_pipeline_predicts_probabilities(fitted_pipeline, raw_patients):
    proba = fitted_pipeline.predict_proba(raw_patients)[:, 1]
    assert proba.shape == (len(raw_patients),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_learns_signal(fitted_pipeline, synthetic_training_data):
    X, y = synthetic_training_data
    assert (fitted_pipeline.predict(X) == y).mean() > 0.8


def test_training_is_reproducible(synthetic_training_data, raw_patients):
    X, y = synthetic_training_data
    a = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(raw_patients)
    b = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(raw_patients)
    np.testing.assert_array_equal(a, b)


def test_saved_pipeline_gives_identical_predictions(fitted_pipeline, raw_patients, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    np.testing.assert_array_equal(fitted_pipeline.predict_proba(raw_patients),
                                  joblib.load(path).predict_proba(raw_patients))


# ---------- Training helpers ----------

def test_make_model_builds_configured_models():
    lr = make_model({"seed": 1, "model": {"type": "logistic_regression", "params": {"C": 0.5}}})
    rf = make_model({"seed": 1, "model": {"type": "random_forest", "params": {"max_depth": 4}}})
    assert lr.C == 0.5
    assert rf.max_depth == 4 and rf.random_state == 1   # seed applied for reproducibility


def test_make_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model type"):
        make_model({"seed": 1, "model": {"type": "svm"}})


def test_quality_gate():
    gate = {"min_cv_roc_auc": 0.8, "min_test_roc_auc": 0.8, "min_test_recall": 0.75}
    good = {"cv_roc_auc_mean": 0.9, "test_roc_auc": 0.9, "test_recall": 0.8}
    bad = {"cv_roc_auc_mean": 0.9, "test_roc_auc": 0.9, "test_recall": 0.6}
    assert check_quality_gate(good, gate) == []
    assert len(check_quality_gate(bad, gate)) == 1


# ---------- Prediction interface (src/predict.py) ----------

@pytest.fixture
def saved_model_dir(fitted_pipeline, tmp_path, monkeypatch):
    """Save a model + model card to a temp folder and point predict.py at it."""
    joblib.dump(fitted_pipeline, tmp_path / "heart_disease_model.joblib")
    (tmp_path / "model_card.json").write_text(json.dumps({"decision_threshold": 0.5}))
    monkeypatch.setattr(predict_module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict_module, "_model", None)
    monkeypatch.setattr(predict_module, "_card", None)
    return tmp_path


def test_predict_single_patient(saved_model_dir, raw_patients):
    result = predict_module.predict(raw_patients.iloc[0].to_dict())
    assert len(result) == 1
    assert set(result[0]) == {"prediction", "label", "confidence", "disease_probability"}
    assert 0.5 <= result[0]["confidence"] <= 1
    assert result[0]["label"] in {"Disease", "No disease"}


def test_predict_several_patients_with_missing_fields(saved_model_dir):
    patients = [{"age": 60, "sex": 1, "cp": 4}, {"age": 45}]     # most fields missing
    assert len(predict_module.predict(patients)) == 2


def test_predict_rejects_unknown_fields(saved_model_dir):
    with pytest.raises(ValueError, match="Unknown fields"):
        predict_module.predict({"age": 60, "cholesterol": 200})


def test_predict_label_matches_threshold(saved_model_dir, raw_patients):
    for r in predict_module.predict(raw_patients.to_dict("records")):
        assert r["prediction"] == int(r["disease_probability"] >= 0.5)
