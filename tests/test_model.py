import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from heartml import config as C
from heartml import modeling as M
from heartml.predict import predict_records
from heartml.tracking import Tracker
from heartml.train import run_training
from tests.helpers import quick_specs, rng_scores, small_clean_df


class TestMetrics(unittest.TestCase):
    def test_evaluate_keys_and_ranges(self):
        y, s = rng_scores()
        res = M.evaluate(y, s)
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity"):
            self.assertTrue(0.0 <= res[k] <= 1.0)
        self.assertEqual(res["tn"] + res["fp"] + res["fn"] + res["tp"], len(y))

    def test_perfect_scores(self):
        res = M.evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["roc_auc"], 1.0)

    def test_pick_best_prefers_auc_then_recall(self):
        table = {"a": {"roc_auc": (0.90, 0), "recall": (0.5, 0)},
                 "b": {"roc_auc": (0.90, 0), "recall": (0.8, 0)},
                 "c": {"roc_auc": (0.85, 0), "recall": (0.99, 0)}}
        self.assertEqual(M.pick_best(table), "b")


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.df = small_clean_df(n=140)
        cls.results = run_training(
            df=cls.df, fig_dir=root / "fig", model_path=root / "m" / "model.joblib",
            metadata_path=root / "m" / "model_metadata.json", metrics_path=root / "metrics.json",
            models=["logistic_regression", "random_forest"], specs=quick_specs(),
            tracker=Tracker(tracking_uri=str(root / "runs"), backend="local"), quiet=True)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        for rel in ("m/model.joblib", "m/model_metadata.json", "metrics.json", "fig/roc_curves.png",
                    "fig/confusion_best.png", "fig/feature_importance.png"):
            self.assertTrue((self.root / rel).exists(), rel)

    def test_at_least_two_models_compared(self):
        self.assertGreaterEqual(len(self.results["cv"]), 2)

    def test_metrics_json_roundtrip(self):
        data = json.loads((self.root / "metrics.json").read_text())
        self.assertIn(data["selected_model"], data["cv"])
        self.assertTrue(0.5 <= data["test"][data["selected_model"]]["roc_auc"] <= 1.0)

    def test_saved_model_predicts_probabilities(self):
        model = joblib.load(self.root / "m" / "model.joblib")
        proba = model.predict_proba(self.df[C.FEATURES].head(10))
        self.assertEqual(proba.shape, (10, 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_predict_records_shape(self):
        model = joblib.load(self.root / "m" / "model.joblib")
        out = predict_records(model, self.df[C.FEATURES].head(3).to_dict("records"))
        self.assertEqual(len(out), 3)
        for r in out:
            self.assertIn(r["prediction"], (0, 1))
            self.assertGreaterEqual(r["confidence"], 0.5)

    def test_tracking_logged_params_metrics_and_plots(self):
        self.assertGreaterEqual(len(list((self.root / "runs").rglob("params.json"))), 3)
        metrics = list((self.root / "runs").rglob("metrics.json"))
        self.assertTrue(any("test_roc_auc" in json.loads(p.read_text()) for p in metrics))
        self.assertTrue(list((self.root / "runs").rglob("*.png")))

    def test_training_is_deterministic(self):
        X, y = self.df[C.FEATURES], self.df[C.TARGET]
        a = M.tune("logistic_regression", X, y, specs=quick_specs())
        b = M.tune("logistic_regression", X, y, specs=quick_specs())
        self.assertEqual(a.best_params_, b.best_params_)
        self.assertAlmostEqual(a.best_score_, b.best_score_)


if __name__ == "__main__":
    unittest.main()
