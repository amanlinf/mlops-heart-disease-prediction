.PHONY: install-dependencies data eda train mlflow-ui serve sample-api-run test lint docker-build install-kubectl-local install-minikube-local k8s-deploy k8s-status k8s-forward k8s-sample-api-run k8s-clean

# install python dependencies
install-dependencies:
	python -m pip install -r requirements-dev.txt

# data acquisiton, download dataset, laod raw data, clean dataset and store in directory 
data:
	python -m src.data

# generate eda data plots and store in directory
eda:
	python -m src.eda

# feature selection, pipeline config, train model, store artifacts and plots, implement experiment tracking
train:
	python -m src.train

# runs mlflow on ui
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# runs api server 
serve:
	uvicorn api.main:app --host 127.0.0.1 --port 8000

# makes sample request on the server
sample-api-run:
	python scripts/sample_api_prediction.py

# runs all the unit and api tests
test:
	python -m pytest -v

# implements linting
lint:
	ruff check .

# docker build command for generating fastapi server image
docker-build:
	docker build -t heart-predictor:1.0 .

# installing kubectl
install-kubectl-local:
	curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
	curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
	echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
	sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
	kubectl version --client

# installing minikube
install-minikube-local:
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube
	rm minikube-linux-amd64
	minikube version

# start minikube, build image, load image in minikube, and deploy application
k8s-deploy:
	minikube start --driver=docker
	kubectl get nodes
	docker build -t heart-predictor:1.0 .
	minikube image load heart-predictor:1.0
	kubectl apply -f k8s/
	kubectl rollout status deployment/heart-api --timeout=180s

# show kubernetes resources metrics
k8s-status:
	kubectl get deployments,pods,services

# expose Kubernetes service locally
k8s-forward:
	kubectl port-forward service/heart-pedictor-api 8080:80

# test running sample api in kubernetes
k8s-sample-api-run :
	python scripts/smoke_api.py --url http://localhost:8080

# remove k8s deployment and service resources
k8s-clean:
	kubectl delete -f k8s/ --ignore-not-found