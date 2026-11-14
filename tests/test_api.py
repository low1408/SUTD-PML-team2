"""L02 HTTP layer. Skipped unless the serve and test extras are installed (CI installs them)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sepsis_warning.data import generate_synthetic, load_patients

HAVE_API = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))


@unittest.skipUnless(HAVE_API, "needs `pip install -e .[serve,test]`")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from sepsis_warning.api import create_app
        from sepsis_warning.service import ScoringService
        from test_service import make_bundle, request_hours

        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        generate_synthetic(root, source_size=60, external_size=20)
        cls.patients = load_patients(root)
        cls.service = ScoringService(make_bundle(cls.patients))
        cls.client = TestClient(create_app(cls.service))
        cls.request_hours = staticmethod(request_hours)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_score_matches_the_service(self):
        patient = self.patients[-1]
        body = {"patient_id": "stay-1", "hours": self.request_hours(patient.frame)}
        response = self.client.post("/score", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), self.service.score("stay-1", body["hours"]))

    def test_invalid_input_is_422_with_a_reason(self):
        response = self.client.post("/score", json={"patient_id": "s", "hours": [{"ICULOS": 1, "SepsisLabel": 1}]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_input")
        self.assertIn("not model inputs", response.json()["detail"])

    def test_malformed_body_is_422(self):
        self.assertEqual(self.client.post("/score", json={"hours": []}).status_code, 422)
        self.assertEqual(self.client.post("/score", json={"patient_id": "s", "hours": "x"}).status_code, 422)

    def test_model_mismatch_is_409(self):
        from sepsis_warning.service import ModelMismatch

        original = self.service.score

        def broken(*_):
            raise ModelMismatch("saved feature columns are missing from this stay")

        self.service.score = broken
        try:
            response = self.client.post("/score", json={"patient_id": "s", "hours": [{"ICULOS": 1}]})
        finally:
            self.service.score = original
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "model_mismatch")


if __name__ == "__main__":
    unittest.main()
