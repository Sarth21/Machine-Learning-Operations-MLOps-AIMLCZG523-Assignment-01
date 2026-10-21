"""Request validation for the prediction API (no Flask dependency, easy to unit test)."""
import math

from heartml import config as C


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(rec):
    """Return (clean_record, errors)."""
    if not isinstance(rec, dict):
        return None, ["each instance must be a JSON object"]
    errors = []
    unknown = sorted(set(rec) - set(C.FEATURES))
    if unknown:
        errors.append(f"unknown fields: {unknown}")
    clean = {}
    for name in C.FEATURES:
        if name not in rec:
            errors.append(f"missing field: {name}")
            continue
        value = rec[name]
        if value is None:
            if name in C.NULLABLE_FEATURES:
                clean[name] = float("nan")
            else:
                errors.append(f"{name} must not be null")
            continue
        if not _is_number(value):
            errors.append(f"{name} must be a number, got {type(value).__name__}")
            continue
        if name in C.NUMERIC_RANGES:
            lo, hi = C.NUMERIC_RANGES[name]
            if not lo <= value <= hi:
                errors.append(f"{name}={value} outside allowed range [{lo}, {hi}]")
                continue
        elif value not in C.CATEGORICAL_VALUES[name]:
            errors.append(f"{name}={value} not in allowed values {sorted(C.CATEGORICAL_VALUES[name])}")
            continue
        clean[name] = float(value)
    return clean, errors


SAMPLE_PAYLOAD = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                  "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                  "ca": 0, "thal": 6}
