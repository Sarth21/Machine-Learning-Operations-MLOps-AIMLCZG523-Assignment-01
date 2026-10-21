"""Heart-disease risk prediction API (Flask).

GET / | /health (liveness) | /ready (readiness) | /model-info | /metrics (Prometheus)
POST /predict   JSON in -> prediction + confidence (single object, list, or {"instances": [...]})

Run: gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 "app.main:create_app()"
"""
import json
import logging
import os
import sys
import time
import uuid

from flask import Flask, g, jsonify, request

from app.metrics import Registry
from heartml import __version__
from heartml import config as C
from heartml.predict import load_metadata, load_model, predict_records
from heartml.schema import SAMPLE_PAYLOAD, validate_record

MAX_BATCH = 100


class JsonFormatter(logging.Formatter):
    """One JSON object per line (ready for Loki / ELK / CloudWatch)."""

    def format(self, record):
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                   + f".{int(record.msecs):03d}Z",
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_logger():
    logger = logging.getLogger("heart-api")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout)]
    if os.getenv("LOG_FILE"):
        os.makedirs(os.path.dirname(os.getenv("LOG_FILE")) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(os.getenv("LOG_FILE")))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    return logger


def create_app(model_path=None):
    app = Flask(__name__)
    app.json.sort_keys = False
    log = _build_logger()
    registry = Registry()
    log_features = os.getenv("LOG_FEATURES", "false").lower() == "true"

    model_path = model_path or os.getenv("MODEL_PATH") or str(C.MODEL_PATH)
    model = load_model(model_path)
    meta = load_metadata(model_path)
    model_version = meta.get("model_version", "unknown")
    registry.set_gauge("heart_model_info", 1, {"version": model_version, "model": meta.get("model_name", "unknown")},
                       help="Static info about the loaded model")
    log.info("model loaded", extra={"fields": {"model_path": model_path, "model_version": model_version}})

    @app.before_request
    def _start():
        g.t0 = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]

    @app.after_request
    def _finish(resp):
        elapsed = time.perf_counter() - g.get("t0", time.perf_counter())
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        registry.inc("http_requests_total", {"method": request.method, "endpoint": endpoint,
                                             "status": str(resp.status_code)}, help="Total HTTP requests")
        registry.observe("http_request_duration_seconds", elapsed,
                         {"method": request.method, "endpoint": endpoint}, help="HTTP request latency in seconds")
        if endpoint != "/metrics":
            log.info("request", extra={"fields": {
                "request_id": g.get("request_id"), "method": request.method, "path": request.path,
                "status": resp.status_code, "latency_ms": round(elapsed * 1000, 2),
                "remote_addr": request.headers.get("X-Forwarded-For", request.remote_addr)}})
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        return resp

    @app.get("/")
    def index():
        return jsonify({"service": "heart-disease-risk-api", "version": __version__,
                        "model_version": model_version,
                        "endpoints": {"POST /predict": "risk prediction", "GET /health": "liveness",
                                      "GET /ready": "readiness", "GET /model-info": "model metadata",
                                      "GET /metrics": "Prometheus metrics"},
                        "example_payload": SAMPLE_PAYLOAD})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/ready")
    def ready():
        return jsonify({"status": "ready", "model_version": model_version})

    @app.get("/model-info")
    def model_info():
        return jsonify({"model_name": meta.get("model_name"), "model_version": model_version,
                        "trained_at_utc": meta.get("trained_at_utc"), "features": C.FEATURES,
                        "feature_descriptions": C.FEATURE_DESCRIPTIONS,
                        "holdout_metrics": meta.get("holdout_metrics"), "cv_metrics": meta.get("cv_metrics"),
                        "data_source": meta.get("data_source"),
                        "decision_threshold": meta.get("decision_threshold", C.DECISION_THRESHOLD)})

    def _err(msg, code, reason=None, **extra):
        if reason:
            registry.inc("prediction_errors_total", {"reason": reason}, help="Rejected prediction requests")
        return jsonify({"error": msg, **extra, "request_id": g.request_id}), code

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return _err("request body must be valid JSON", 400, "bad_json")
        batch = False
        if isinstance(payload, dict) and "instances" in payload:
            records, batch = payload["instances"], True
        elif isinstance(payload, list):
            records, batch = payload, True
        else:
            records = [payload]
        if not isinstance(records, list) or not records:
            return _err("'instances' must be a non-empty list", 400, "bad_batch")
        if len(records) > MAX_BATCH:
            return _err(f"batch too large (max {MAX_BATCH})", 413, "bad_batch")
        clean, problems = [], []
        for i, rec in enumerate(records):
            c, errs = validate_record(rec)
            if errs:
                problems.append({"index": i, "errors": errs})
            clean.append(c)
        if problems:
            return _err("validation failed", 422, "validation", details=problems)
        t0 = time.perf_counter()
        try:
            results = predict_records(model, clean)
        except Exception:  # noqa: BLE001
            log.exception("inference failed", extra={"fields": {"request_id": g.request_id}})
            return _err("inference failed", 500, "inference")
        registry.observe("heart_inference_seconds", time.perf_counter() - t0,
                         help="Model inference latency in seconds")
        for rec, res in zip(clean, results):
            registry.inc("heart_predictions_total", {"label": str(res["prediction"])},
                         help="Predictions served, by predicted class")
            registry.observe("heart_prediction_confidence", res["confidence"],
                             buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0), help="Distribution of prediction confidence")
            fields = {"request_id": g.request_id, "prediction": res["prediction"],
                      "confidence": res["confidence"], "model_version": model_version}
            if log_features:  # off by default: patient data is sensitive
                fields["features"] = rec
            log.info("prediction", extra={"fields": fields})
        for r in results:
            r["model_version"] = model_version
        body = {"predictions": results} if batch else dict(results[0])
        body["request_id"] = g.request_id
        return jsonify(body)

    @app.get("/metrics")
    def metrics():
        return app.response_class(registry.render(), mimetype="text/plain; version=0.0.4; charset=utf-8")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def bad_method(_):
        return jsonify({"error": "method not allowed"}), 405

    return app


if __name__ == "__main__":  # local dev only; containers use gunicorn
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
