"""API contract: the paths and fields clients depend on. A deliberate change updates CONTRACT here."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sepsis_warning.data import DYNAMIC, STATIC, generate_synthetic, load_patients

HAVE_API = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))

CONTRACT = {
    "paths": {"/health": ["get"], "/model": ["get"], "/predict": ["post"], "/score": ["post"]},
    "ScoreRequest": {"required": ["hours", "patient_id"], "fields": ["hours", "patient_id"]},
    "PredictRequest": {"required": ["stays"], "fields": ["stays"]},
    "HourRecord": {"required": [], "fields": sorted(DYNAMIC + STATIC)},
    "StayPrediction": {
        "required": ["above_threshold", "hour", "hours_scored", "input_warnings", "last_alert_hour", "new_alert",
                     "patient_id", "prediction_id", "probability", "risk_tier"],
    },
    "HourScore": {"required": ["alert_event", "hour", "hourly_positive", "prediction_id", "probability", "risk_tier"]},
    "PredictResponse": {
        "required": ["alert_source", "bundle_sha256", "cooldown_hours", "disclaimer", "model_name", "model_version",
                     "predictions", "scored_at", "synthetic_model", "threshold"],
    },
    "ScoreResponse": {
        "required": ["alert_source", "bundle_sha256", "cooldown_hours", "disclaimer", "hours", "input_warnings",
                     "model_name", "model_version", "patient_id", "scored_at", "synthetic_model", "threshold"],
    },
}


@unittest.skipUnless(HAVE_API, "needs `pip install -e .[serve,test]`")
class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from test_service import make_bundle

        from sepsis_warning.api import create_app
        from sepsis_warning.service import ScoringService

        with tempfile.TemporaryDirectory() as temp:
            generate_synthetic(Path(temp), source_size=60, external_size=20)
            service = ScoringService(make_bundle(load_patients(Path(temp))))
        cls.schema = TestClient(create_app(service)).get("/openapi.json").json()

    def test_paths_and_methods(self):
        paths = {path: sorted(methods) for path, methods in self.schema["paths"].items()}
        self.assertEqual(paths, CONTRACT["paths"])

    def test_models_keep_their_fields(self):
        schemas = self.schema["components"]["schemas"]
        for name, expected in CONTRACT.items():
            if name == "paths":
                continue
            with self.subTest(model=name):
                self.assertEqual(sorted(schemas[name].get("required", [])), expected["required"])
                if "fields" in expected:
                    self.assertEqual(sorted(schemas[name]["properties"]), expected["fields"])
                else:
                    self.assertEqual(sorted(schemas[name]["properties"]), expected["required"])

    def test_errors_have_one_shape(self):
        from fastapi.testclient import TestClient
        from test_service import make_bundle

        from sepsis_warning.api import create_app
        from sepsis_warning.service import ScoringService

        with tempfile.TemporaryDirectory() as temp:
            generate_synthetic(Path(temp), source_size=60, external_size=20)
            client = TestClient(create_app(ScoringService(make_bundle(load_patients(Path(temp))))))
        for body, status in (({"patient_id": "b", "hours": [{"ICULOS": 1, "SepsisLabel": 1}]}, 422),
                             ({"patient_id": "b", "hours": []}, 422), ({"hours": [{"ICULOS": 1}]}, 422)):
            response = client.post("/score", json=body)
            self.assertEqual(response.status_code, status)
            self.assertEqual(sorted(response.json()), ["detail", "error"])


if __name__ == "__main__":
    unittest.main()
