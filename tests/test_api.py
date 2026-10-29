"""API contract and artifact-loading integration tests; no dataset scan or training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.ml.model_loader import load_xgboost_bundle


ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = ROOT / "artifacts/models/xgboost_20261029T152352Z"


class InferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_xgboost_bundle(MODEL_ARTIFACT)
        cls.features = cls.bundle["feature_names"]
        cls.labels = cls.bundle["label_classes"]

    @classmethod
    def record(cls):
        return {name: 0.0 for name in cls.features}

    def client(self, *, artifact_dir=MODEL_ARTIFACT, max_batch_size=64):
        app = create_app(Settings(artifact_dir=Path(artifact_dir), max_batch_size=max_batch_size))
        return TestClient(app)

    def test_valid_artifact_loads_once_without_retraining_and_is_ready(self):
        with patch("app.main.load_xgboost_bundle", wraps=load_xgboost_bundle) as loader:
            with self.client() as client:
                self.assertEqual(client.get("/ready").status_code, 200)
                self.assertEqual(client.get("/ready").json()["model_loaded"], True)
            loader.assert_called_once_with(MODEL_ARTIFACT)

    def test_health_readiness_and_safe_model_metadata(self):
        with self.client() as client:
            health = client.get("/health")
            ready = client.get("/ready")
            models = client.get("/models")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(ready.json(), {"status": "ready", "model_loaded": True})
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["model_name"], "xgboost")
        self.assertEqual(models.json()["supported_classes"], self.labels)
        self.assertEqual(models.json()["model_version"], self.bundle["model_version"])

    def test_cors_allows_only_local_vite_origin(self):
        with self.client() as client:
            allowed = client.options(
                "/predict",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            denied = client.options(
                "/predict",
                headers={
                    "Origin": "http://example.invalid",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertNotIn("access-control-allow-origin", denied.headers)

    def test_single_prediction_has_real_class_and_probability_mapping(self):
        with self.client() as client:
            response = client.post("/predict", json=self.record())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["predicted_class"], self.labels)
        self.assertEqual(list(result["probabilities"]), self.labels)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertEqual(result["is_attack"], result["predicted_class"] != "BENIGN")
        self.assertEqual(result["model_name"], "xgboost")
        self.assertEqual(result["model_version"], self.bundle["model_version"])
        self.assertTrue(result["request_id"])
        self.assertTrue(result["predicted_at"])
        self.assertNotIn("confidence", result)

    def test_batch_prediction_returns_one_result_per_record(self):
        with self.client() as client:
            response = client.post("/predict/batch", json={"records": [self.record(), self.record()]})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["model_name"], "xgboost")
        self.assertEqual(result["predictions"][0]["request_id"], result["request_id"])

    def test_missing_feature_is_rejected(self):
        record = self.record()
        record.pop(self.features[0])
        with self.client() as client:
            response = client.post("/predict", json=record)
        self.assertEqual(response.status_code, 422)
        self.assertIn(self.features[0], response.json()["detail"])

    def test_unexpected_feature_and_target_label_are_rejected(self):
        for key in ("extra_feature", "Label"):
            record = self.record()
            record[key] = "BENIGN"
            with self.subTest(key=key), self.client() as client:
                response = client.post("/predict", json=record)
            self.assertEqual(response.status_code, 422)
            self.assertIn(key, response.json()["detail"])

    def test_incorrect_type_and_boolean_are_rejected(self):
        for value in ("1.25", True, None, {"nested": 2}):
            record = self.record()
            record[self.features[0]] = value
            with self.subTest(value=value), self.client() as client:
                response = client.post("/predict", json=record)
            self.assertEqual(response.status_code, 422)

    def test_nan_and_infinity_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), 1e300):
            record = self.record()
            record[self.features[0]] = value
            body = json.dumps(record, allow_nan=True)
            with self.subTest(value=value), self.client() as client:
                response = client.post("/predict", content=body, headers={"content-type": "application/json"})
            self.assertEqual(response.status_code, 422)

    def test_oversized_batch_is_rejected_at_configured_limit(self):
        with self.client(max_batch_size=1) as client:
            response = client.post("/predict/batch", json={"records": [self.record(), self.record()]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("maximum is 1", response.json()["detail"])

    def test_empty_batch_is_rejected(self):
        with self.client() as client:
            response = client.post("/predict/batch", json={"records": []})
        self.assertEqual(response.status_code, 422)

    def test_missing_artifacts_keep_health_but_fail_readiness_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory, self.client(artifact_dir=directory) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/models").status_code, 503)
            prediction = client.post("/predict", json=self.record())
        self.assertEqual(prediction.status_code, 503)
        self.assertNotIn(directory, prediction.text)
        self.assertNotIn("Traceback", prediction.text)

    def test_missing_metadata_is_a_clear_loader_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                load_xgboost_bundle(directory)

    def test_incompatible_metadata_schema_is_rejected_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            source_metadata = MODEL_ARTIFACT / "metadata.json"
            (Path(directory) / "metadata.json").write_text(source_metadata.read_text(encoding="utf-8"), encoding="utf-8")
            incompatible = dict(self.bundle)
            incompatible["feature_names"] = [*self.features[:-1], "wrong_feature"]
            with patch("app.ml.model_loader.load_bundle", return_value=incompatible):
                with self.assertRaisesRegex(ValueError, "schema does not match"):
                    load_xgboost_bundle(directory)


if __name__ == "__main__":
    unittest.main()
