#!/usr/bin/env python
"""Tiny traffic generator so the Grafana dashboard has data.

    python scripts/load_test.py http://localhost:8000 --n 300 --rps 15
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def record(rng):
    return {"age": rng.randint(30, 77), "sex": rng.randint(0, 1), "cp": rng.randint(1, 4),
            "trestbps": rng.randint(100, 190), "chol": rng.randint(150, 400), "fbs": rng.randint(0, 1),
            "restecg": rng.choice([0, 1, 2]), "thalach": rng.randint(90, 190), "exang": rng.randint(0, 1),
            "oldpeak": round(rng.uniform(0, 4.5), 1), "slope": rng.randint(1, 3), "ca": rng.randint(0, 3),
            "thal": rng.choice([3, 6, 7])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rps", type=float, default=10)
    ap.add_argument("--bad-rate", type=float, default=0.05, help="share of invalid requests")
    a = ap.parse_args()
    rng, codes = random.Random(1), {}
    for _ in range(a.n):
        rec = record(rng)
        if rng.random() < a.bad_rate:
            rec["age"] = 999
        req = urllib.request.Request(a.base.rstrip("/") + "/predict", json.dumps(rec).encode(),
                                     {"Content-Type": "application/json"})
        try:
            code = urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            code = e.code
        codes[code] = codes.get(code, 0) + 1
        time.sleep(1 / a.rps)
    print("status codes:", codes)


if __name__ == "__main__":
    main()
