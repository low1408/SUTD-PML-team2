"""L02 service core: schemas, errors, replay parity and record isolation (synthetic only)."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sepsis_warning.data import DYNAMIC, STATIC, Patient, generate_synthetic, load_patients
from sepsis_warning.features import META, feature_table
from sepsis_warning.modeling import candidate_models
from sepsis_warning.scoring import file_sha256, score_patients
from sepsis_warning.service import InvalidInput, ModelMismatch, ScoringService


def make_bundle(patients):
    """Small synthetic bundle in the same shape `train` saves. Not a real model."""
    table = feature_table(patients[:40], 6)
    columns = [name for name in table.columns if name not in META]
    model = candidate_models(0)["logistic_regression"]
    model.fit(table[columns], table["label"])
    return {
        "model": model, "model_name": "logistic_regression", "feature_columns": columns,
        "window": 6, "threshold": 0.5, "cooldown": 6, "synthetic": True,
    }


def request_hours(frame: pd.DataFrame) -> list[dict]:
    """Turn a stay into the JSON-shaped request body, null for missing."""
    hours = []
    for _, row in frame[DYNAMIC + STATIC].iterrows():
        hours.append({name: (None if pd.isna(value) else float(value)) for name, value in row.items()})
    return hours


class ScoringServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        generate_synthetic(cls.root / "data", source_size=60, external_size=20)
        cls.patients = load_patients(cls.root / "data")
        cls.bundle = make_bundle(cls.patients)
        cls.service = ScoringService(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def score(self, patient: Patient) -> pd.DataFrame:
        return pd.DataFrame(self.service.score("stay-1", request_hours(patient.frame))["hours"])

    def test_scores_match_batch_replay(self):
        patients = self.patients[-5:]
        batch = score_patients(self.bundle, patients)
        for patient in patients:
            expected = batch[batch["patient_id"] == patient.patient_id].reset_index(drop=True)
            served = self.score(patient)
            np.testing.assert_allclose(served["probability"], expected["probability"], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(served["hourly_positive"], expected["hourly_positive"])
            np.testing.assert_array_equal(served["alert_event"], expected["alert_event"])
            np.testing.assert_array_equal(served["hour"], expected["hour"])

    def test_prefix_request_matches_full_stay(self):
        patient = self.patients[-1]
        full = self.score(patient)
        prefix = pd.DataFrame(self.service.score("stay-1", request_hours(patient.frame.iloc[:10]))["hours"])
        pd.testing.assert_frame_equal(prefix, full.iloc[:10])

    def test_concurrent_requests_stay_isolated(self):
        patients = self.patients[-8:]
        sequential = [self.score(patient) for patient in patients]
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(self.score, patients * 3))
        for index, result in enumerate(parallel):
            pd.testing.assert_frame_equal(result, sequential[index % len(patients)])

    def test_missing_fields_are_treated_as_not_measured(self):
        hours = [{"ICULOS": 1, "HR": 80}, {"ICULOS": 2}]
        result = self.service.score("stay-1", hours)
        self.assertEqual([hour["hour"] for hour in result["hours"]], [1.0, 2.0])
        self.assertTrue(result["synthetic_model"])
        self.assertIn("not clinically validated", result["disclaimer"])

    def test_invalid_requests_are_rejected_with_a_reason(self):
        good = [{"ICULOS": 1, "HR": 80}]
        cases = {
            "patient_id": ("", good),
            "non-empty list": ("s", []),
            "not model inputs": ("s", [{"ICULOS": 1, "SepsisLabel": 0}]),
            "Unit1": ("s", [{"ICULOS": 1, "Unit1": 1}]),
            "unknown fields": ("s", [{"ICULOS": 1, "heart_rate": 80}]),
            "number or null": ("s", [{"ICULOS": 1, "HR": "80"}]),
            "must be finite": ("s", [{"ICULOS": 1, "HR": float("inf")}]),
            "ICULOS is required": ("s", [{"HR": 80}]),
            "consecutive hourly": ("s", [{"ICULOS": 1}, {"ICULOS": 3}]),
            "must be an object": ("s", [1]),
        }
        for message, (patient_id, hours) in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(InvalidInput, message):
                self.service.score(patient_id, hours)
        with self.assertRaisesRegex(InvalidInput, "number or null"):
            self.service.score("s", [{"ICULOS": 1, "HR": True}])

    def test_too_many_hours_is_rejected(self):
        hours = [{"ICULOS": hour} for hour in range(1, 1002)]
        with self.assertRaisesRegex(InvalidInput, "at most"):
            self.service.score("s", hours)

    def test_bundle_with_unknown_feature_columns_fails_at_startup(self):
        bundle = dict(self.bundle, feature_columns=self.bundle["feature_columns"] + ["HR__mean_99h"])
        with self.assertRaisesRegex(ModelMismatch, "does not match the current feature code"):
            ScoringService(bundle)

    def test_non_bundle_is_refused(self):
        with self.assertRaises(ModelMismatch):
            ScoringService({"model": None})

    def test_hash_check_on_load(self):
        path = self.root / "model.joblib"
        joblib.dump(self.bundle, path)
        digest = file_sha256(path)
        service = ScoringService.from_path(path, digest)
        self.assertEqual(service.health()["bundle_sha256"], digest)
        with self.assertRaisesRegex(ModelMismatch, "SHA-256"):
            ScoringService.from_path(path, "0" * 64)

    def test_health_reports_the_frozen_contract(self):
        health = self.service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["feature_count"], len(self.bundle["feature_columns"]))
        self.assertEqual(health["threshold"], 0.5)
        self.assertEqual(len(health["feature_columns_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
