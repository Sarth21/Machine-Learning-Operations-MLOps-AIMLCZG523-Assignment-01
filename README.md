# Heart Disease Risk Prediction - End-to-End MLOps

BITS Pilani MLOps (AIMLCZG523) Assignment 01. A scikit-learn classifier for the UCI Heart Disease
(Cleveland) dataset, packaged as a Flask API with experiment tracking (MLflow), unit tests, GitHub
Actions CI/CD, Docker, Kubernetes manifests and Prometheus/Grafana monitoring.

> **Important - data provenance.** The files in `data/`, `models/`, `reports/` and the generated PDF
> were produced offline on a **synthetic stand-in dataset** (same schema as UCI Cleveland). They prove the
> pipeline works but are **not clinically meaningful**. Before submitting, run steps 1-3 below on a machine
> with internet, re-capture the screenshots (`screenshots/README.md`) and rebuild the report.
> Also set your GitHub URL in `reports/build_report.py` (`REPO_URL`).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements-dev.txt

python scripts/download_data.py --force                 # 1. real UCI data
make all                                                # 2. clean -> EDA -> train (MLflow tracked)
make test                                               # 3. 48 unit tests (+ coverage)
make report                                             #    rebuild reports/MLOps_Assignment1_Report.pdf

make run                                                # API on :8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Training serving versions must match: `requirements-api.txt` pins the versions the shipped model was trained with.
If you retrain on a different Python/scikit-learn, update those pins (the metadata file records them).

## Docker / Kubernetes / monitoring
```bash
docker build -t heart-disease-api:1.0.0 . && docker run --rm -p 8000:8000 heart-disease-api:1.0.0
python scripts/smoke_test.py http://localhost:8000
docker compose up --build          # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)

minikube start && minikube addons enable metrics-server ingress
minikube image load heart-disease-api:1.0.0
kubectl apply -k k8s/ && minikube tunnel
```
The model file is baked into the image: run `make train` (or download the CI `training-outputs` artifact) before `docker build`.

## Layout
`heartml/` package - `app/` API - `tests/` - `notebooks/` - `scripts/` - `k8s/` - `monitoring/` - `docs/diagrams/` - `docs/evidence/` (captured local runs) - `reports/` (figures, metrics, PDF report) - `.github/workflows/ci.yml`

## API
`POST /predict` (object, list, or `{"instances":[...]}`, max 100) - `GET /health` `/ready` `/model-info` `/metrics`.
Responses: `{prediction, label, probability_disease, confidence, model_version, request_id}`. Invalid input -> 422 with details.
Patient features are not logged unless `LOG_FEATURES=true`. Educational project - not a medical device.

## What was verified offline vs. what you must verify
Verified here: data pipeline, training/tracking (local tracker fallback - MLflow itself was not installable offline), 48 unit tests,
live API smoke test, YAML syntax of all manifests/compose/workflow, diagram + report generation.
**Not run here** (no Docker, Kubernetes, GitHub or internet): `docker build`, the container, `kubectl apply`, the GitHub Actions
workflow, MLflow, Prometheus/Grafana. These files follow standard patterns but expect to fix small issues on first run.
