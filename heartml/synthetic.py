"""Synthetic stand-in generator mimicking the UCI Cleveland schema.

ONLY for offline dry runs / CI-free development. scripts/download_data.py requires an explicit
--synthetic flag and writes a DATA_SOURCE.txt marker surfaced in logs, tracking and the report.
"""
import numpy as np
import pandas as pd

from heartml import config as C


def make_synthetic_heart(n=303, seed=7, missing=6):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(54, 9, n), 29, 77).round()
    sex = rng.binomial(1, 0.68, n)
    cp = rng.choice([1, 2, 3, 4], n, p=[0.08, 0.17, 0.28, 0.47])
    trestbps = np.clip(rng.normal(131, 17, n), 94, 200).round()
    chol = np.clip(rng.normal(246, 52, n), 126, 564).round()
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.choice([0, 1, 2], n, p=[0.50, 0.01, 0.49])
    thalach = np.clip(205 - 0.9 * age + rng.normal(0, 17, n), 71, 202).round()
    exang = rng.binomial(1, 0.33, n)
    oldpeak = np.clip(rng.gamma(1.1, 0.95, n), 0, 6.2).round(1)
    slope = rng.choice([1, 2, 3], n, p=[0.47, 0.46, 0.07])
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]).astype(float)
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39]).astype(float)
    logit = (0.04 * (age - 54) + 0.9 * sex + 0.55 * (cp == 4) + 0.012 * (trestbps - 131)
             + 0.003 * (chol - 246) - 0.03 * (thalach - 150) + 0.85 * exang + 0.55 * oldpeak
             + 0.3 * (slope >= 2) + 0.95 * ca + 1.0 * (thal == 7) + 0.3 * (thal == 6)
             + 0.2 * (restecg == 2))
    prob = 1 / (1 + np.exp(-2.6 * (logit - np.quantile(logit, 0.54))))
    disease = rng.binomial(1, prob)
    severity = np.where(disease == 1, rng.choice([1, 2, 3, 4], n, p=[0.5, 0.2, 0.2, 0.1]), 0)
    df = pd.DataFrame({"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                       "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                       "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
                       C.TARGET: severity})
    if missing:
        idx = rng.choice(n, size=missing, replace=False)
        df.loc[idx[: missing // 2], "ca"] = np.nan
        df.loc[idx[missing // 2:], "thal"] = np.nan
    return df
