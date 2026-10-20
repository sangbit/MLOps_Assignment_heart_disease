"""Train the final model reproducibly and save it in a reusable format.

Run from the project root:
    python -m src.train

Reads params.yaml, trains the champion configuration, evaluates it (cross-validation +
held-out test set) and writes to models/:
    heart_disease_model.joblib   full pipeline: preprocessing + model (use this to predict)
    preprocessor.joblib          fitted preprocessing on its own (for reuse / inspection)
    model_card.json              metadata: params, metrics, input schema, data + code versions
"""

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import FEATURES, MODEL_DIR, PARAMS_FILE, PROJECT_ROOT, TARGET
from src.data import data_fingerprint, load_data
from src.preprocessing import build_pipeline

MODELS = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier}
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# Accepted input values, recorded in the model card (and reused by the API later)
INPUT_SCHEMA = {
    "age": "number, years", "sex": "0 = female, 1 = male",
    "cp": "1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic",
    "trestbps": "number, mm Hg (0 = not recorded)", "chol": "number, mg/dl (0 = not recorded)",
    "fbs": "0/1, fasting blood sugar > 120 mg/dl",
    "restecg": "0 normal, 1 ST-T abnormality, 2 LV hypertrophy",
    "thalach": "number, max heart rate", "exang": "0/1, exercise-induced angina",
    "oldpeak": "number, ST depression", "slope": "1 up, 2 flat, 3 down",
    "ca": "0-3, vessels coloured by fluoroscopy", "thal": "3 normal, 6 fixed defect, 7 reversible defect",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_model(cfg: dict):
    model_type = cfg["model"]["type"]
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type '{model_type}'. Use one of {list(MODELS)}")
    params = dict(cfg["model"].get("params") or {})
    if model_type == "random_forest":
        params.setdefault("random_state", cfg["seed"])
    return MODELS[model_type](**params)


def main() -> None:
    cfg = yaml.safe_load(PARAMS_FILE.read_text())
    seed, threshold = cfg["seed"], cfg["decision_threshold"]
    np.random.seed(seed)

    # 1. Data and split (stratified on hospital + diagnosis, same as Deliverables 2-3)
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    strata = df["source"] + "_" + y.astype(str)
    X_train, X_test, y_train, y_test, strata_train, _ = train_test_split(
        X, y, strata, test_size=cfg["test_size"], stratify=strata, random_state=seed)

    # 2. Cross-validation on the development set
    pipeline = build_pipeline(make_model(cfg), cfg["features"]["drop_sparse"])
    folds = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    cv = cross_validate(pipeline, X_train, y_train, cv=list(folds.split(X_train, strata_train)),
                        scoring=SCORING)
    metrics = {}
    for m in SCORING:
        metrics[f"cv_{m}_mean"] = round(float(cv[f"test_{m}"].mean()), 4)
        metrics[f"cv_{m}_std"] = round(float(cv[f"test_{m}"].std()), 4)

    # 3. Fit on the full development set, evaluate once on the test set
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics.update({
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    })

    # 4. Save: full pipeline, preprocessor on its own, and the model card
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "heart_disease_model.joblib")
    joblib.dump(pipeline.named_steps["prep"], MODEL_DIR / "preprocessor.joblib")

    card = {
        "model_name": "heart-disease-classifier",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_type": cfg["model"]["type"],
        "model_params": cfg["model"].get("params") or {},
        "decision_threshold": threshold,
        "features": FEATURES,
        "input_schema": INPUT_SCHEMA,
        "model_features_after_preprocessing":
            list(pipeline.named_steps["prep"].get_feature_names_out()),
        "metrics": metrics,
        "training": {"rows_train": len(X_train), "rows_test": len(X_test), "seed": seed,
                     "test_size": cfg["test_size"], "cv_folds": cfg["cv_folds"],
                     "drop_sparse": cfg["features"]["drop_sparse"]},
        "lineage": {"data_fingerprint": data_fingerprint(), "git_commit": git_commit()},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "intended_use": "Educational MLOps project. Not a medical device; not for clinical decisions.",
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

    # 5. Optionally record this run in MLflow (same experiment as Deliverable 3)
    if cfg.get("mlflow", {}).get("enabled"):
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name="final - packaged model"):
            mlflow.set_tags({"stage": "packaged", "model_type": cfg["model"]["type"],
                             **card["lineage"]})
            mlflow.log_params({**card["model_params"], **card["training"],
                               "decision_threshold": threshold})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(MODEL_DIR), artifact_path="packaged_model")
            mlflow.log_artifact(str(PARAMS_FILE))

    # 6. Report
    print(f"Model:            {cfg['model']['type']}  {card['model_params']}")
    lineage = card["lineage"]
    print(f"Data fingerprint: {lineage['data_fingerprint']}   git commit: {lineage['git_commit']}")
    print(f"CV  ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
          f"recall {metrics['cv_recall_mean']:.4f}")
    print(f"Test ROC-AUC {metrics['test_roc_auc']:.4f} | recall {metrics['test_recall']:.4f} | "
          f"precision {metrics['test_precision']:.4f} | accuracy {metrics['test_accuracy']:.4f}")
    print(f"Saved to {MODEL_DIR.relative_to(PROJECT_ROOT)}/: heart_disease_model.joblib, "
          "preprocessor.joblib, model_card.json")

    # 7. Quality gate
    failures = check_quality_gate(metrics, cfg.get("quality_gate", {}))
    write_ci_summary(card, failures)
    if failures:
        print("QUALITY GATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("Quality gate passed.")


def check_quality_gate(metrics: dict, gate: dict) -> list[str]:
    """Return a list of failed checks (empty list = passed)."""
    checks = {"min_cv_roc_auc": "cv_roc_auc_mean", "min_test_roc_auc": "test_roc_auc",
              "min_test_recall": "test_recall"}
    return [f"{metric} = {metrics[metric]:.4f} is below the minimum {gate[key]}"
            for key, metric in checks.items() if key in gate and metrics[metric] < gate[key]]


def write_ci_summary(card: dict, failures: list[str]) -> None:
    """On GitHub Actions, add a metrics table to the workflow run's summary page."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    m = card["metrics"]
    lines = [
        "## Model training results",
        f"**Model:** `{card['model_type']}` {card['model_params']}  ",
        f"**Data fingerprint:** `{card['lineage']['data_fingerprint']}` · "
        f"**Commit:** `{card['lineage']['git_commit']}`",
        "",
        "| Metric | Cross-validation (mean ± std) | Test set |",
        "|---|---|---|",
        *[f"| {name} | {m[f'cv_{k}_mean']:.4f} ± {m[f'cv_{k}_std']:.4f} | {m[f'test_{k}']:.4f} |"
          for name, k in [("Accuracy", "accuracy"), ("Precision", "precision"),
                          ("Recall", "recall"), ("ROC-AUC", "roc_auc")]],
        "",
        "**Quality gate:** " + ("❌ FAILED — " + "; ".join(failures) if failures else "✅ passed"),
    ]
    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
