"""Inference helpers shared by the API, notebooks and tests."""
import json
from pathlib import Path

import joblib
import pandas as pd

from heartml import config as C


def load_model(path=None):
    path = Path(path or C.MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model file {path} not found. Run `python -m heartml.train`.")
    return joblib.load(path)


def load_metadata(model_path=None):
    meta = Path(model_path).with_name("model_metadata.json") if model_path else C.METADATA_PATH
    return json.loads(meta.read_text()) if meta.exists() else {}


def predict_records(model, records, threshold=C.DECISION_THRESHOLD):
    proba = model.predict_proba(pd.DataFrame(records, columns=C.FEATURES))[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({"prediction": pred, "label": C.LABELS[pred],
                    "probability_disease": round(float(p), 4),
                    "confidence": round(float(max(p, 1 - p)), 4)})
    return out
