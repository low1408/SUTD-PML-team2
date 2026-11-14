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

    def test_predict_returns_the_latest_hour_of_each_stay(self):
        stays = [{"patient_id": f"bed-{i}", "hours": self.request_hours(p.frame.iloc[:20])}
                 for i, p in enumerate(self.patients[:3])]
        response = self.client.post("/predict", json={"stays": stays})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual([p["patient_id"] for p in body["predictions"]], ["bed-0", "bed-1", "bed-2"])
        for stay, prediction in zip(stays, body["predictions"], strict=True):
            hours = self.service.score(stay["patient_id"], stay["hours"])["hours"]
            self.assertEqual(prediction["hour"], hours[-1]["hour"])
            self.assertEqual(prediction["probability"], hours[-1]["probability"])
            self.assertEqual(prediction["new_alert"], bool(hours[-1]["alert_event"]))
            self.assertEqual(prediction["hours_scored"], len(hours))
            alerts = [h["hour"] for h in hours if h["alert_event"]]
            self.assertEqual(prediction["last_alert_hour"], alerts[-1] if alerts else None)

    def test_predict_rejects_bad_batches(self):
        one = {"patient_id": "bed-1", "hours": [{"ICULOS": 1}]}
        duplicate = self.client.post("/predict", json={"stays": [one, one]})
        self.assertEqual(duplicate.status_code, 422)
        self.assertIn("unique", duplicate.json()["detail"])
        labelled = self.client.post("/predict", json={"stays": [one, {"patient_id": "bed-2",
                                                                     "hours": [{"ICULOS": 1, "SepsisLabel": 0}]}]})
        self.assertEqual(labelled.status_code, 422)
        self.assertIn("stays[1]", labelled.json()["detail"])
        self.assertEqual(self.client.post("/predict", json={"stays": []}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={"stays": [one] * 65}).status_code, 422)

    def test_model_without_selection_files(self):
        body = self.client.get("/model").json()
        self.assertEqual(body["model_name"], "logistic_regression")
        self.assertIsNone(body["selection"])


@unittest.skipUnless(HAVE_API, "needs `pip install -e .[serve,test]`")
class ModelCardTests(unittest.TestCase):
    """GET /model after train: the selected model and why, validation numbers only."""

    @classmethod
    def setUpClass(cls):
        from argparse import Namespace

        from fastapi.testclient import TestClient

        from sepsis_warning.api import create_app
        from sepsis_warning.cli import run_experiment
        from sepsis_warning.model_card import card_from_run_dir
        from sepsis_warning.service import ScoringService

        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        generate_synthetic(root / "data", source_size=80, external_size=20)
        (root / "data" / "SYNTHETIC.json").write_text("{}")
        cls.run_dir = root / "run"
        run_experiment(Namespace(
            data=root / "data", splits=None, output=cls.run_dir, seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
        ))
        cls.service = ScoringService.from_path(cls.run_dir / "model.joblib")
        cls.card = card_from_run_dir(cls.run_dir, cls.service.bundle)
        cls.client = TestClient(create_app(cls.service, cls.card))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_model_reports_the_selected_model_and_all_candidates(self):
        import json

        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        body = self.client.get("/model").json()
        selection = body["selection"]
        self.assertEqual(selection["selected_model"], metrics["selected_model"])
        self.assertEqual(body["model_name"], metrics["selected_model"])
        self.assertEqual(selection["threshold"], metrics["threshold"])
        self.assertEqual(sorted(c["model"] for c in selection["candidates"]),
                         sorted(c["model"] for c in metrics["validation_models"]))
        self.assertTrue(selection["candidates"][0]["selected"])
        self.assertEqual(sum(c["selected"] for c in selection["candidates"]), 1)
        self.assertEqual(selection["training"]["seed"], 42)

    def test_model_never_serves_held_out_results(self):
        text = self.client.get("/model").text
        for word in ("held_out", "internal_test", "external_test"):
            self.assertNotIn(word, text)

    def test_selection_files_for_another_bundle_are_refused(self):
        from sepsis_warning.model_card import CardMismatch, card_from_run_dir

        other = dict(self.service.bundle, threshold=self.service.bundle["threshold"] + 0.01)
        with self.assertRaises(CardMismatch):
            card_from_run_dir(self.run_dir, other)
        self.assertIsNone(card_from_run_dir(self.run_dir / "missing", self.service.bundle))


if __name__ == "__main__":
    unittest.main()
