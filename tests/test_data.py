import unittest

import numpy as np
import pandas as pd

from heartml import config as C
from heartml import data as D
from heartml.synthetic import make_synthetic_heart


class TestClean(unittest.TestCase):
    def setUp(self):
        self.raw = make_synthetic_heart(n=120, seed=1, missing=4)

    def test_target_is_binary(self):
        self.assertEqual(set(D.clean(self.raw)[C.TARGET].unique()), {0, 1})

    def test_target_binarisation_rule(self):
        raw = self.raw.copy()
        raw[C.TARGET] = [0, 1, 2, 3, 4] * 24
        out = D.clean(raw)
        self.assertTrue(out[C.TARGET].isin([0, 1]).all())

    def test_question_marks_become_nan(self):
        raw = self.raw.astype(object)
        raw.loc[0, "ca"] = "?"
        raw.loc[1, "thal"] = "?"
        out = D.clean(raw)
        self.assertTrue(np.isnan(out.loc[0, "ca"]))
        self.assertTrue(np.isnan(out.loc[1, "thal"]))

    def test_missing_columns_raise(self):
        with self.assertRaises(D.DataValidationError):
            D.clean(self.raw.drop(columns=["chol"]))

    def test_impossible_values_flagged_missing(self):
        raw = self.raw.copy()
        raw.loc[0, "chol"] = 0
        raw.loc[1, "trestbps"] = 900
        raw.loc[2, "cp"] = 9
        out = D.clean(raw)
        self.assertTrue(np.isnan(out.loc[0, "chol"]))
        self.assertTrue(np.isnan(out.loc[1, "trestbps"]))
        self.assertTrue(np.isnan(out.loc[2, "cp"]))

    def test_duplicates_dropped(self):
        raw = pd.concat([self.raw, self.raw.iloc[:5]], ignore_index=True)
        self.assertEqual(len(D.clean(raw)), len(D.clean(self.raw)))

    def test_column_order_is_fixed(self):
        self.assertEqual(list(D.clean(self.raw[self.raw.columns[::-1]]).columns), C.FEATURES + [C.TARGET])

    def test_load_raw_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            D.load_raw("/nonexistent/heart.csv")


class TestSplit(unittest.TestCase):
    def test_stratified_split_preserves_ratio(self):
        df = D.clean(make_synthetic_heart(n=200, seed=2, missing=0))
        Xtr, Xte, ytr, yte = D.train_test(df, test_size=0.25, seed=1)
        self.assertEqual(len(Xtr) + len(Xte), len(df))
        self.assertAlmostEqual(ytr.mean(), yte.mean(), delta=0.05)

    def test_split_is_reproducible(self):
        df = D.clean(make_synthetic_heart(n=100, seed=2, missing=0))
        self.assertTrue((D.train_test(df, seed=5)[0].index == D.train_test(df, seed=5)[0].index).all())


if __name__ == "__main__":
    unittest.main()
