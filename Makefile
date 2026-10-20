.PHONY: install-dependencies data eda train mlflow-ui serve sample-api-run test

install-dependencies:
	python -m pip install -r requirements-dev.txt

data:
	python -m src.data

eda:
	python -m src.eda

train:
	python -m src.train

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

serve:
	uvicorn api.main:app --host 127.0.0.1 --port 8000

sample-api-run:
	python scripts/sample_api_prediction.py

test:
	python -m pytest -v