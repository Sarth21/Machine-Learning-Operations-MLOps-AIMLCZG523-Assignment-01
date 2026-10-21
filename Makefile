.PHONY: help setup data data-synthetic clean-data eda train all test lint run docker-build docker-run smoke compose-up k8s-deploy k8s-status mlflow-ui diagrams report

PY ?= python
IMAGE ?= heart-disease-api:1.0.0

help:           ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'
setup:          ## create venv and install dev requirements
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt
data:           ## download the real UCI dataset
	$(PY) scripts/download_data.py --force
data-synthetic: ## OFFLINE dry-run only: synthetic stand-in
	$(PY) scripts/download_data.py --synthetic
clean-data:     ## clean raw data -> data/processed/heart_clean.csv
	$(PY) -c "import logging; logging.basicConfig(level=logging.INFO); from heartml import data; data.prepare()"
eda:            ## EDA figures -> reports/figures
	$(PY) -m heartml.eda
train:          ## tune, evaluate, track and save the model
	$(PY) -m heartml.train
all: clean-data eda train   ## full ML pipeline (after `make data`)
test:           ## unit tests with coverage
	$(PY) -m pytest --cov=heartml --cov=app --cov-report=term-missing

lint:           ## flake8
	$(PY) -m flake8 .
run:            ## run the API locally on :8000
	gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 "app.main:create_app()"
docker-build:   ## build the serving image
	docker build -t $(IMAGE) .
docker-run:     ## run the image on :8000
	docker run --rm -p 8000:8000 $(IMAGE)
smoke:          ## black-box checks against a running API
	$(PY) scripts/smoke_test.py http://localhost:8000
compose-up:     ## API + Prometheus + Grafana
	docker compose up --build
k8s-deploy:     ## apply manifests
	kubectl apply -k k8s/
k8s-status:
	kubectl -n heart-mlops get pods,svc,ingress,hpa
mlflow-ui:      ## browse experiment runs
	mlflow ui --backend-store-uri ./mlruns
diagrams:       ## regenerate architecture diagrams
	$(PY) docs/diagrams/build_diagrams.py
report:         ## rebuild the PDF report
	$(PY) reports/build_report.py
