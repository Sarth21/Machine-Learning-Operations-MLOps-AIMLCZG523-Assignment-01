"""Feature engineering + preprocessing: one pipeline used for training, CV and serving."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config as C


class FeatureAdder(BaseEstimator, TransformerMixin):
    """hr_reserve_ratio = thalach/(220-age); chol_per_age; high_bp = trestbps >= 140."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureAdder expects a pandas DataFrame")
        out = X.copy()
        predicted_max = (220.0 - out["age"]).where(lambda s: s > 0)
        out["hr_reserve_ratio"] = out["thalach"] / predicted_max
        out["chol_per_age"] = out["chol"] / out["age"].where(lambda s: s > 0)
        out["high_bp"] = (out["trestbps"] >= 140).astype(float).where(out["trestbps"].notna())
        return out


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", numeric, C.NUMERIC_FEATURES + C.ENGINEERED_NUMERIC),
         ("cat", categorical, C.CATEGORICAL_FEATURES + C.ENGINEERED_CATEGORICAL)],
        remainder="drop", verbose_feature_names_out=False)


def build_pipeline(estimator):
    return Pipeline([("features", FeatureAdder()), ("preprocess", build_preprocessor()),
                     ("clf", estimator)])


def feature_names(fitted_pipeline):
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
