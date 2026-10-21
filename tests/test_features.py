import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heartml import config as C
from heartml.features import FeatureAdder, build_pipeline, feature_names
from tests.helpers import small_clean_df


class TestFeatureAdder(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame([{"age": 60, "sex": 1, "cp": 4, "trestbps": 150, "chol": 240, "fbs": 0,
                                "restecg": 0, "thalach": 112, "exang": 1, "oldpeak": 1.5, "slope": 2,
                                "ca": 1, "thal": 7}])

    def test_adds_engineered_columns(self):
        out = FeatureAdder().fit_transform(self.X)
        for col in C.ENGINEERED_NUMERIC + C.ENGINEERED_CATEGORICAL:
            self.assertIn(col, out.columns)

    def test_values(self):
        out = FeatureAdder().fit_transform(self.X)
        self.assertAlmostEqual(out.loc[0, "hr_reserve_ratio"], 112 / 160)
        self.assertAlmostEqual(out.loc[0, "chol_per_age"], 4.0)
        self.assertEqual(out.loc[0, "high_bp"], 1.0)

    def test_does_not_mutate_input(self):
        before = self.X.copy()
        FeatureAdder().fit_transform(self.X)
        pd.testing.assert_frame_equal(self.X, before)

    def test_nan_propagates(self):
        X = self.X.copy()
        X["trestbps"] = np.nan
        self.assertTrue(np.isnan(FeatureAdder().fit_transform(X).loc[0, "high_bp"]))

    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            FeatureAdder().transform(np.zeros((2, 13)))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = small_clean_df()
        cls.X, cls.y = df[C.FEATURES], df[C.TARGET]
        cls.pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(cls.X, cls.y)

    def test_no_nans_reach_classifier(self):
        self.assertFalse(np.isnan(np.asarray(self.pipe[:-1].transform(self.X))).any())

    def test_numeric_columns_are_scaled(self):
        Xt = pd.DataFrame(self.pipe[:-1].transform(self.X), columns=feature_names(self.pipe))
        self.assertAlmostEqual(Xt["age"].mean(), 0, places=6)
        self.assertAlmostEqual(Xt["age"].std(ddof=0), 1, places=6)

    def test_onehot_columns_created(self):
        names = feature_names(self.pipe)
        self.assertTrue(any(n.startswith("cp_") for n in names))
        self.assertNotIn("cp", names)

    def test_unseen_category_is_ignored(self):
        row = self.X.iloc[[0]].copy()
        row["thal"] = 99.0
        self.assertEqual(self.pipe.predict_proba(row).shape, (1, 2))

    def test_missing_values_are_imputed_at_predict_time(self):
        row = self.X.iloc[[0]].copy()
        row["ca"] = np.nan
        row["thal"] = np.nan
        self.assertTrue(np.isfinite(self.pipe.predict_proba(row)).all())


if __name__ == "__main__":
    unittest.main()
