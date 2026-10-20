from fastapi.testclient import TestClient

from api.main import create_app
from src.model import candidate_models

# creates a temp logistic regression model and passes it directly to fastapi instead of using the trained model under artifacts/model.joblib
def make_client(sample_dataset):
    model, _ = candidate_models()["logistic_regression"]
    model.fit(sample_dataset.drop(columns="target"), sample_dataset["target"])
    return TestClient(create_app(model_override=model))

# checks health and prediction apis
def test_health_and_prediction(sample_dataset, patient_dict):
    with make_client(sample_dataset) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        response = client.post("/predict", json=patient_dict)
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in [0, 1]
        assert 0 <= body["probability_heart_disease"] <= 1
        assert 0.5 <= body["confidence"] <= 1
        assert body["model_version"] == "test-fixture"
        assert "X-Request-ID" in response.headers
        assert "diagnosis" in body["disclaimer"]

# check the negative case
def test_rejects_bad_request(sample_dataset, patient_dict):
    patient_dict["age"] = -3
    with make_client(sample_dataset) as client:
        response = client.post("/predict", json=patient_dict)
    assert response.status_code == 422

# checks the metric endpoint
def test_metrics(sample_dataset):
    with make_client(sample_dataset) as client:
        client.get("/health")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "heart_api_http_requests_total" in response.text
