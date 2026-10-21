"""Shared test helpers."""
import numpy as np

from heartml import data as D
from heartml.synthetic import make_synthetic_heart


def small_clean_df(n=160, seed=3):
    return D.clean(make_synthetic_heart(n=n, seed=seed, missing=4))


def quick_specs():
    """Tiny search spaces so training tests run in seconds."""
    from heartml import modeling as M

    specs = M.model_specs()
    specs["logistic_regression"] = (specs["logistic_regression"][0], {"clf__C": [0.1, 1.0]})
    specs["random_forest"] = (specs["random_forest"][0].set_params(n_estimators=20),
                              {"clf__max_depth": [3, None]})
    return specs


def rng_scores(n=50, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    return y, np.clip(y * 0.5 + r.normal(0.25, 0.2, n), 0, 1)
