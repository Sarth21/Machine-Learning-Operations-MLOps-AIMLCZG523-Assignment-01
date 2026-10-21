import copy
import tempfile
import unittest
from pathlib import Path

from app.main import create_app
from app.metrics import Registry
from heartml import config as C
from heartml.schema import SAMPLE_PAYLOAD, validate_record
from heartml.tracking import Tracker
from heartml.train import run_training
from tests.helpers import quick_specs, small_clean_df


class TestSchema(unittest.TestCase):
    def test_valid_sample(self):
        clean, errors = validate_record(SAMPLE_PAYLOAD)
        self.assertEqual(errors, [])
        self.assertEqual(set(clean), set(C.FEATURES))

    def test_missing_and_unknown_fields(self):
        rec = copy.deepcopy(SAMPLE_PAYLOAD)
        del rec["age"]
        rec["bogus"] = 1
        _, errors = validate_record(rec)
        self.assertTrue(any("missing field: age" in e for e in errors))
        self.assertTrue(any("unknown fields" in e for e in errors))

    def test_type_and_range_errors(self):
        _, errors = validate_record(dict(SAMPLE_PAYLOAD, age="sixty", chol=9999, cp=0, sex=True))
        self.assertEqual(len(errors), 4)

    def test_nullable_features(self):
        clean, errors = validate_record(dict(SAMPLE_PAYLOAD, ca=None, thal=None))
        self.assertEqual(errors, [])
        self.assertNotEqual(clean["ca"], clean["ca"])  # NaN

    def test_non_nullable_null_rejected(self):
        self.assertTrue(validate_record(dict(SAMPLE_PAYLOAD, age=None))[1])

    def test_non_object_rejected(self):
        self.assertTrue(validate_record([1, 2, 3])[1])


class TestMetricsRegistry(unittest.TestCase):
    def test_render_counter_and_histogram(self):
        r = Registry()
        r.inc("hits_total", {"path": "/x"}, help="hits")
        r.inc("hits_total", {"path": "/x"})
        r.observe("lat_seconds", 0.03, buckets=(0.01, 0.1))
        text = r.render()
        self.assertIn('hits_total{path="/x"} 2', text)
        self.assertIn('lat_seconds_bucket{le="0.01"} 0', text)
        self.assertIn('lat_seconds_bucket{le="0.1"} 1', text)
        self.assertIn('lat_seconds_bucket{le="+Inf"} 1', text)
        self.assertIn("# TYPE hits_total counter", text)

    def test_label_escaping(self):
        r = Registry()
        r.inc("x_total", {"v": 'a"b'})
        self.assertIn('v="a\\"b"', r.render())


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        run_training(df=small_clean_df(n=140), fig_dir=root / "fig", model_path=root / "model.joblib",
                     metadata_path=root / "model_metadata.json", metrics_path=root / "metrics.json",
                     models=["logistic_regression"], specs=quick_specs(),
                     tracker=Tracker(tracking_uri=str(root / "runs"), backend="local"), quiet=True)
        cls.client = create_app(str(root / "model.joblib")).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_and_ready(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})
        self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_predict_single(self):
        r = self.client.post("/predict", json=SAMPLE_PAYLOAD)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn(body["prediction"], (0, 1))
        self.assertTrue(0.5 <= body["confidence"] <= 1.0)
        self.assertEqual(r.headers["X-Request-ID"], body["request_id"])

    def test_predict_batch(self):
        r = self.client.post("/predict", json={"instances": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]})
        self.assertEqual(len(r.get_json()["predictions"]), 2)
        self.assertEqual(self.client.post("/predict", json=[SAMPLE_PAYLOAD]).status_code, 200)

    def test_missing_values_allowed_for_ca_thal(self):
        self.assertEqual(self.client.post("/predict", json=dict(SAMPLE_PAYLOAD, ca=None, thal=None)).status_code, 200)

    def test_validation_error_422(self):
        r = self.client.post("/predict", json=dict(SAMPLE_PAYLOAD, age=500))
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["details"][0]["index"], 0)

    def test_bad_json_400(self):
        self.assertEqual(self.client.post("/predict", data="not json", content_type="application/json").status_code, 400)

    def test_empty_and_oversized_batches(self):
        self.assertEqual(self.client.post("/predict", json={"instances": []}).status_code, 400)
        self.assertEqual(self.client.post("/predict", json={"instances": [SAMPLE_PAYLOAD] * 101}).status_code, 413)

    def test_404_and_405(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/predict").status_code, 405)

    def test_model_info(self):
        self.assertEqual(self.client.get("/model-info").get_json()["features"], C.FEATURES)

    def test_metrics_endpoint_counts_requests(self):
        self.client.post("/predict", json=SAMPLE_PAYLOAD)
        self.client.post("/predict", json=dict(SAMPLE_PAYLOAD, age=500))
        text = self.client.get("/metrics").get_data(as_text=True)
        self.assertIn('http_requests_total{endpoint="/predict",method="POST",status="200"}', text)
        self.assertIn('status="422"', text)
        for key in ("heart_predictions_total", "http_request_duration_seconds_bucket", "heart_model_info"):
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main()
