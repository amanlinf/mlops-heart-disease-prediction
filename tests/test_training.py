import json

from src.train import run_training

# test to check the model training phase, uses 2 fold CV and mlflow is disabled as well
def test_training_artifacts_and_metrics(tmp_path, sample_dataset):
    csv_path = tmp_path / "test_sample.csv"
    sample_dataset.to_csv(csv_path, index=False)
    artifacts = tmp_path / "artifacts"
    figures = tmp_path / "plots"
    result = run_training(
        csv_path, artifacts, figures, mlflow_enabled=False, folds=2
    )
    assert result["selected_model"] in {"logistic_regression", "random_forest"}
    assert (artifacts / "model.joblib").is_file()
    assert (artifacts / "metadata.json").is_file()
    assert (figures / "roc_curve.png").is_file()
    assert 0 <= result["test_metrics"]["roc_auc"] <= 1
    assert json.loads((artifacts / "metadata.json").read_text())["cv_folds"] == 2
