#!/usr/bin/env python
"""Black-box smoke test for a running API (local, Docker or Kubernetes).

    python scripts/smoke_test.py http://localhost:8000
Exits non-zero on the first failed check so it can gate CI/CD.
"""
import json
import sys
import time
import urllib.error
import urllib.request

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
          "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit(1)


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    for _ in range(40):  # wait up to ~40 s for start-up
        try:
            if call(base, "/health")[0] == 200:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    else:
        check("service becomes healthy", False)
    status, body = call(base, "/health")
    check("GET /health", status == 200, body.strip())
    check("GET /ready", call(base, "/ready")[0] == 200)
    status, body = call(base, "/predict", SAMPLE)
    out = json.loads(body)
    ok = status == 200 and out["prediction"] in (0, 1) and 0.5 <= out["confidence"] <= 1.0
    check("POST /predict (valid)", ok, body.strip())
    check("POST /predict (invalid -> 422)", call(base, "/predict", dict(SAMPLE, age=999))[0] == 422)
    status, body = call(base, "/predict", {"instances": [SAMPLE, SAMPLE]})
    check("POST /predict (batch)", status == 200 and len(json.loads(body)["predictions"]) == 2)
    status, body = call(base, "/metrics")
    check("GET /metrics", status == 200 and "http_requests_total" in body and "heart_predictions_total" in body)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
