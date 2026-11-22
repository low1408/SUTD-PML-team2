"""Phase 3 governance: promotion gates, approver identity, write-once publishing."""

import importlib.util
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from sepsis_warning.data import generate_synthetic
from sepsis_warning.gates import (
    MIN_UTILITY_GAIN,
    GateFailure,
    evaluate_gates,
    require_gates,
)

HAVE_MLFLOW = importlib.util.find_spec("mlflow") is not None


def summary(**changes):
    model = {"model": "logistic_regression", "normalized_hourly_utility": 0.40, "utility_ci_low": 0.33,
             "false_alert_events_per_nonseptic_patient_day": 0.18, "brier_score": 0.02}
    model.update({k: v for k, v in changes.items() if k in model})
    base = {
        "data_mode": changes.get("data_mode", "physionet"), "selected_model": "logistic_regression",
        "validation_alarm_budget": 0.2,
        "validation_models": [
            {"model": "constant_prior", "normalized_hourly_utility": 0.0, "brier_score": 0.021,
             "false_alert_events_per_nonseptic_patient_day": 0.0},
            model,
        ],
        "reference_baselines": {"sirs": {"normalized_hourly_utility": changes.get("sirs", 0.05)}},
    }
    return base


class GateTests(unittest.TestCase):
    def failed(self, selection, **options):
        return [r["gate"] for r in evaluate_gates(selection, **options) if not r["passed"]]

    def test_a_good_model_passes_every_gate(self):
        results = require_gates(summary(), require_real_data=True)
        self.assertEqual([r["gate"] for r in results],
                         ["utility_above_no_model", "beats_sirs", "within_alert_budget", "calibrated", "real_data"])

    def test_each_gate_fails_for_its_own_reason(self):
        self.assertEqual(self.failed(summary(normalized_hourly_utility=MIN_UTILITY_GAIN / 2, sirs=-1)),
                         ["utility_above_no_model"])
        self.assertEqual(self.failed(summary(utility_ci_low=-0.01)), ["utility_above_no_model"])
        self.assertEqual(self.failed(summary(sirs=0.45)), ["beats_sirs"])
        self.assertEqual(self.failed(summary(false_alert_events_per_nonseptic_patient_day=0.3)), ["within_alert_budget"])
        self.assertEqual(self.failed(summary(brier_score=0.05)), ["calibrated"])
        self.assertEqual(self.failed(summary(data_mode="synthetic_fixture"), require_real_data=True), ["real_data"])
        self.assertEqual(self.failed(summary(data_mode="synthetic_fixture")), [])

    def test_missing_evidence_fails_instead_of_being_skipped(self):
        self.assertEqual(self.failed(None), ["validation_summary"])
        no_sirs = summary()
        del no_sirs["reference_baselines"]
        self.assertEqual(self.failed(no_sirs), ["beats_sirs"])
        no_interval = summary()
        no_interval["validation_models"][1]["utility_ci_low"] = None
        self.assertEqual(self.failed(no_interval), ["utility_above_no_model"])
        with self.assertRaisesRegex(GateFailure, "beats_sirs"):
            require_gates(no_sirs)


@unittest.skipUnless(HAVE_MLFLOW, "needs the tracking extra")
class PromotionAndPublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sepsis_warning.cli import run_experiment
        from sepsis_warning.registry import register_bundle
        from sepsis_warning.tracking import log_run

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=120, external_size=20)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        cls.run_dir = cls.root / "run"
        run_experiment(Namespace(
            data=cls.data, splits=None, output=cls.run_dir, seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
        ))
        cls.uri = f"sqlite:///{cls.root / 'store' / 'mlflow.db'}"
        cls.version = register_bundle(cls.run_dir, cls.uri, log_run(cls.run_dir, cls.uri))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def client(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)

    def test_training_records_the_sirs_reference_and_who_registered(self):
        from sepsis_warning.registry import MODEL_NAME

        selection = json.loads((self.run_dir / "selection_metrics.json").read_text())
        self.assertIn("normalized_hourly_utility", selection["reference_baselines"]["sirs"])
        self.assertTrue(self.client().get_model_version(MODEL_NAME, self.version).tags["registered_by"].startswith("local:"))

    def test_promotion_runs_the_gates_and_records_the_approver(self):
        from sepsis_warning.registry import MODEL_NAME, promote, register_bundle

        version = register_bundle(self.run_dir, self.uri)          # no selection run: summary still attached
        change = promote(self.uri, version, "Reviewer", alias="staging")
        self.assertTrue(all(g["passed"] for g in change["gates"]))
        tags = self.client().get_model_version(MODEL_NAME, version).tags
        self.assertEqual(tags["staging_approved_by"], change["approved_by"])
        self.assertIn("beats_sirs", json.loads(tags["staging_gates"]))
        with self.assertRaisesRegex(GateFailure, "real_data"):
            promote(self.uri, self.version, "Reviewer", alias="strict", require_real_data=True)
        self.assertNotIn("strict_reviewed_by", self.client().get_model_version(MODEL_NAME, self.version).tags)

    def test_on_aws_the_registrant_cannot_approve(self):
        from sepsis_warning.registry import MODEL_NAME, promote

        trainer = "arn:aws:sts::123456789012:assumed-role/sepsis-training/job-1"
        self.client().set_model_version_tag(MODEL_NAME, self.version, "registered_by", trainer)
        with self.assertRaisesRegex(ValueError, "cannot also approve"):
            promote(self.uri, self.version, "Reviewer", alias="aws-same",
                    identity="arn:aws:sts::123456789012:assumed-role/sepsis-training/another-session")
        change = promote(self.uri, self.version, "Reviewer", alias="aws-other",
                         identity="arn:aws:iam::123456789012:user/approver")
        self.assertEqual(change["approved_by"], "arn:aws:iam::123456789012:user/approver")
        self.client().set_model_version_tag(MODEL_NAME, self.version, "registered_by", "local:restored")

    def test_publish_is_write_once_and_needs_the_frozen_evaluation_for_real_data(self):
        from test_aws import FakeS3

        from sepsis_warning.evaluation import EVALUATED_TAG
        from sepsis_warning.model_source import (
            PublishRefused,
            fetch_model,
            publish_model,
        )
        from sepsis_warning.registry import MODEL_NAME

        s3 = FakeS3(self.root / "s3")
        uri = f"models:/{MODEL_NAME}/{self.version}"
        first = publish_model(uri, self.uri, "bucket", client=s3)
        self.assertTrue(first["created"])
        self.assertFalse(publish_model(uri, self.uri, "bucket", client=s3)["created"])   # same bundle: no-op
        target = fetch_model(first["model_s3_uri"], first["sha256"], self.root / "dl", s3)
        self.assertTrue((target.parent / "manifest.json").is_file())
        s3.metadata[("bucket", f"models/v{self.version}/model.joblib")] = {"sha256": "f" * 64}
        with self.assertRaisesRegex(PublishRefused, "immutable"):
            publish_model(uri, self.uri, "bucket", client=s3)
        # A real-data version (tag flipped for the test) must be evaluated before it is published.
        client = self.client()
        client.set_model_version_tag(MODEL_NAME, self.version, "synthetic", "false")
        try:
            with self.assertRaisesRegex(PublishRefused, "frozen evaluation"):
                publish_model(uri, self.uri, "bucket2", client=s3)
            client.set_model_version_tag(MODEL_NAME, self.version, EVALUATED_TAG, "2026-11-15T00:00:00+00:00")
            self.assertTrue(publish_model(uri, self.uri, "bucket2", client=s3)["created"])
        finally:
            client.set_model_version_tag(MODEL_NAME, self.version, "synthetic", "true")
            client.delete_model_version_tag(MODEL_NAME, self.version, EVALUATED_TAG)


if __name__ == "__main__":
    unittest.main()
