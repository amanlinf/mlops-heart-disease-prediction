"""Leakage-safe selection, cross-validation, holdout evaluation and MLflow runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data import DEFAULT_CLEAN, FEATURES, TARGET
from src.model import candidate_models

# convert numpy types and nested containers to JSON serializable Python objects
def as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

# get the evaluation metrics for the model prediction
def evaluate(model: Any, x: pd.DataFrame, y: pd.Series) -> dict:
    # get model class prediction
    prediction = model.predict(x)
    # get model class probability, taking only the second column to get only the postive class probability for ROC-AUC calculations
    probability = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }

# saving test plots generated after the training
def save_test_plots(model: Any, x_test: pd.DataFrame, y_test: pd.Series, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap="Blues", values_format="d")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title("Test ROC Curve")
    plt.tight_layout()
    plt.savefig(output / "roc_curve.png", dpi=180)
    plt.close()

# model training logic
def run_training(
    data_path: Path = DEFAULT_CLEAN,
    output_dir: Path = Path("artifacts"),
    plot_dir: Path = Path("plots/evaluation"),
    mlflow_enabled: bool = True,
    # five fold cross validation
    folds: int = 5,
    seed: int = 42,) -> dict:
        mlflow = None
        if mlflow_enabled:
            import mlflow as tracking
            import mlflow.sklearn

            mlflow = tracking
            # MLFLOW_TRACKING_URI is not set as an env variable then, the storage location defaults to ./mlruns 
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(Path("mlruns").resolve())))
            mlflow.set_experiment("uci-heart-disease-binary-classification")

        df = pd.read_csv(data_path)
        if not set(FEATURES + [TARGET]).issubset(df.columns):
            raise ValueError("data does not include all the features and target columns")
        if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
            raise ValueError("contains unexpected value, need non null binary values")
        x = df[FEATURES]
        y = df[TARGET].astype(int)

        # Used 80-20 split for the training and testing data split
        # startify=y, tries to preserver the class distribution across training and testing split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, stratify=y, random_state=seed
        )
        # 5 fold CV is used, with four portions used to train and one to validate, with validation fold changing is each successive folds
        # metrics are averaged out across all the 5 folds
        cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = {
            "accuracy": "accuracy", "precision": "precision",
            "recall": "recall", "roc_auc": "roc_auc",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        candidates = {}
        selected = None
        winner_score = float("-inf")
        winning_name = ""

        # using GridSearchCV to calculate all the four metrics
        # also ROC-AUC metric is used for figuring out the best possible configuration
        for name, (pipeline, parameter_grid) in candidate_models(seed).items():
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring=scoring,
                refit="roc_auc",
                cv=cross_validation,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(x_train, y_train)
            best_index = search.best_index_
            cv_metrics = {
                metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
                for metric in scoring
            }
            candidates[name] = {
                "best_parameters": as_plain(search.best_params_),
                "cv_metrics": cv_metrics,
            }
            cv_csv = output_dir / f"cv_results_{name}.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_csv, index=False)
            # if mlflow is enabled logging and enabling the experiment tracking
            if mlflow is not None:
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
                    mlflow.log_params({"seed": seed, "folds": folds, "model_family": name})
                    mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
                    mlflow.log_artifact(str(cv_csv), artifact_path="cross_validation")
            if cv_metrics["roc_auc"] > winner_score:
                winner_score = cv_metrics["roc_auc"]
                selected = search.best_estimator_
                winning_name = name

        if selected is None:
            raise RuntimeError("No candidate model fitted is selected")
        test_metrics = evaluate(selected, x_test, y_test)
        save_test_plots(selected, x_test, y_test, plot_dir)
        # save the trained model in the output directory
        # joblib.dump serializes the fitted python model to disk
        # it saves the entire pipeline and not just the model itself
        # could be resued by the api layer using load and predict func
        model_path = output_dir / "model.joblib"
        joblib.dump(selected, model_path)
        # generating model checksum and metadata by computing sha256 checksum of the saved model file
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {
            "dataset_source": "UCI processed Cleveland Heart Disease (dataset ID 45)",
            "n_samples": int(len(df)),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "feature_names": FEATURES,
            "class_definition": "target=1 when UCI num > 0; otherwise 0",
            "selected_model": winning_name,
            "cv_selection_metric": "roc_auc",
            "candidate_results": candidates,
            "test_metrics": test_metrics,
            "model_sha256": model_hash,
            "random_state": seed,
            "cv_folds": folds,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        # logging the final model to mlflow
        if mlflow is not None:
            with mlflow.start_run(run_name=f"final_{winning_name}"):
                mlflow.log_params({
                    "selected_model": winning_name, "seed": seed, "folds": folds,
                    "selection_metric": "cv_roc_auc",
                })
                mlflow.log_metric("selected_cv_roc_auc", winner_score)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow.log_artifact(str(metadata_path))
                mlflow.log_artifacts(str(plot_dir), artifact_path="evaluation")
                mlflow.log_artifact(str(model_path), artifact_path="joblib")
                mlflow.sklearn.log_model(selected, artifact_path="model")
        print(json.dumps({"selected_model": winning_name, "test_metrics": test_metrics}, indent=2))
        return metadata

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--plots", type=Path, default=Path("plots/evaluation"))
    parser.add_argument("--no-mlflow", action="store_true", help="Local debugging only")
    args = parser.parse_args()
    # mlflow_enabled = not args.no_mlflow
    # if True -> during training, then mlflow and experiment tracking is enabled
    # if False -> during pytest, then mlflow and experiment tracking is disabled
    run_training(args.data, args.output, args.plots, mlflow_enabled=not args.no_mlflow)

if __name__ == "__main__":
    main()
