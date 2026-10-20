from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.data import FEATURES

# setting up the logger for api server
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart_disease_prediction_api_logger")

# metric object from Prometheus Client Library
# metric for tracking total api request calls
HTTP_REQUESTS = Counter(
    "heart_api_http_requests_total", "API HTTP requests",
    ["method", "route", "status"],
)
# metric for tracking api latency
HTTP_LATENCY = Histogram(
    "heart_api_http_request_duration_seconds", "API latency in seconds",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# api request class for predict api
class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=15)
    slope: int = Field(ge=1, le=3)
    ca: int | None = Field(default=None, ge=0, le=4)
    thal: int | None = Field(default=None)

# Class for model prediction output
class PredictionResponse(BaseModel):
    prediction: int
    probability_heart_disease: float
    confidence: float
    model_version: str
    disclaimer: str

# used ASGI uvicorn server for handling networking, listens on TCP socket and handles HTTP requests
# used FastApi as an application framework
def create_app(model_override: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_override is not None:
            app.state.model = model_override
            app.state.version = "test-fixture"
        else:
            model_path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
            metadata_path = Path(os.getenv("METADATA_PATH", "artifacts/metadata.json"))
            if not model_path.is_file() or not metadata_path.is_file():
                raise RuntimeError("model artifact is not detected, run the src.train first and then restart the server")
            app.state.model = joblib.load(model_path)
            metadata = json.loads(metadata_path.read_text())
            app.state.version = metadata["model_sha256"][:12]
        # logging the model version loaded for api predictions 
        log.info("model_ready version=%s", app.state.version)
        yield

    # initializes the application
    app = FastAPI(
        title="Heart Disease Prediction Api Server",
        version="1.0.0",
        lifespan=lifespan,
        description="Classifier",
    )

    # middleware for tracking the api details such as status code, execution time etc
    @app.middleware("http")
    async def instrument(request: Request, call_next):
        request_id = str(uuid4())
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route_obj = request.scope.get("route")
            route = route_obj.path if route_obj is not None else "unmatched"
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            HTTP_LATENCY.labels(route).observe(elapsed)
            # logging the api call metrics
            log.info("request_id=%s method=%s route=%s status=%s duration_ms=%.2f",
                     request_id, request.method, route, status, elapsed * 1000)

    # health api
    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.version}

    # metric api, further scrapped by prometheus for metric dashboard
    # it uses the in-memory values stored inside the Counter and Histogram object to generate a report
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # prediction api
    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures, request: Request):
        record = patient.model_dump()
        frame = pd.DataFrame([record], columns=FEATURES)
        try:
            probability = float(request.app.state.model.predict_proba(frame)[0, 1])
        except Exception as exception:
            log.exception("prediction_failed")
            raise HTTPException(status_code=500, detail="Prediction unavailable") from exception
        prediction = int(probability >= 0.5)
        return PredictionResponse(
            prediction=prediction,
            probability_heart_disease=round(probability, 4),
            confidence=round(max(probability, 1 - probability), 4),
            model_version=request.app.state.version,
            disclaimer="For education only; not a medical diagnosis or calibrated clinical risk.",
        )

    return app

app = create_app()
