"""Selection runs and the model registry against a real local MLflow (SQLite).

Skipped when the optional ``tracking`` extra is not installed.
"""

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None

from sepsis_warning.data import generate_synthetic, load_patients
from sepsis_warning.scoring import file_sha256
from sepsis_warning.service import ScoringService


@unittest.skipIf(mlflow is None, "MLflow is an optional install (tracking extra)")
class Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sepsis_warning.cli import run_experiment
        from sepsis_warning.registry import register_bundle
        from sepsis_warning.tracking import log_run

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "synthetic"
        generate_synthetic(cls.data, source_size=80, external_size=20)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        cls.run_dir = cls.root / "run"
        run_experiment(Namespace(
            data=cls.data, splits=None, output=cls.run_dir, seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
        ))
        cls.uri = f"sqlite:///{cls.root / 'store' / 'mlflow.db'}"
        cls.run_id = log_run(cls.run_dir, cls.uri)
        cls.version = register_bundle(cls.run_dir, cls.uri, cls.run_id)
        cls.client = MlflowClient(tracking_uri=cls.uri, registry_uri=cls.uri)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_selection_run_has_one_nested_run_per_candidate_and_no_test_metrics(self):
        candidates = json.loads((self.run_dir / "metrics.json").read_text())["validation_models"]
        children = self.client.search_runs(
            [self.client.get_run(self.run_id).info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{self.run_id}'")
        self.assertEqual(sorted(r.data.tags["candidate"] for r in children), sorted(c["model"] for c in candidates))
        for run in [self.client.get_run(self.run_id), *children]:
            self.assertFalse([k for k in run.data.metrics if "internal_test" in k or "external_test" in k])
        selected = [r for r in children if r.data.tags["selected"] == "true"]
        self.assertEqual(len(selected), 1)
        self.assertIn("validation.auroc", selected[0].data.metrics)

    def test_artifacts_stay_next_to_the_database(self):
        experiment = self.client.get_experiment(self.client.get_run(self.run_id).info.experiment_id)
        self.assertEqual(Path(experiment.artifact_location.replace("file://", "")).parent, self.root / "store")
        names = {item.path for item in self.client.list_artifacts(self.run_id)}
        self.assertNotIn("held_out_predictions.csv", names)
        self.assertNotIn("split_manifest.csv", names)
        self.assertIn("selection_metrics.json", names)

    def test_registered_version_records_the_bundle_hash_and_is_the_candidate(self):
        from sepsis_warning.registry import MODEL_NAME

        version = self.client.get_model_version(MODEL_NAME, self.version)
        self.assertEqual(version.tags["bundle_sha256"], file_sha256(self.run_dir / "model.joblib"))
        self.assertEqual(version.tags["synthetic"], "true")
        # Each new registration becomes the candidate.
        latest = max(int(v.version) for v in self.client.search_model_versions(f"name = '{MODEL_NAME}'"))
        self.assertEqual(int(self.client.get_model_version_by_alias(MODEL_NAME, "candidate").version), latest)

    def test_mlflow_pyfunc_scores_like_the_api(self):
        from sepsis_warning.registry import INPUT_COLUMNS, MODEL_NAME

        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
        stay = load_patients(self.data)[0].frame.iloc[:12]
        frame = stay[INPUT_COLUMNS].copy()
        frame.insert(0, "stay_id", "stay-1")
        predicted = model.predict(frame)
        service = ScoringService.from_path(self.run_dir / "model.joblib")
        records = stay[INPUT_COLUMNS].astype(object).where(stay[INPUT_COLUMNS].notna(), None).to_dict("records")
        expected = service.score("stay-1", records)["hours"]
        self.assertEqual(len(predicted), 12)
        for got, want in zip(predicted.to_dict("records"), expected, strict=True):
            self.assertAlmostEqual(got["probability"], want["probability"], places=12)
            self.assertEqual(got["alert_event"], want["alert_event"])

    def test_promote_needs_a_reviewer_and_records_the_previous_version(self):
        from sepsis_warning.registry import MODEL_NAME, promote, register_bundle

        with self.assertRaises(ValueError):
            promote(self.uri, self.version, reviewed_by=" ")
        first = promote(self.uri, self.version, reviewed_by="Kaiyuan")
        self.assertIsNone(first["previous"])
        second_version = register_bundle(self.run_dir, self.uri)
        second = promote(self.uri, second_version, reviewed_by="Mashie")
        self.assertEqual(second["previous"], self.version)
        tags = self.client.get_model_version(MODEL_NAME, second_version).tags
        self.assertEqual(tags["previous_production"], self.version)
        self.assertEqual(tags["production_reviewed_by"], "Mashie")
        with self.assertRaises(ValueError):
            promote(self.uri, second_version, reviewed_by="Mashie")

    def test_resolve_checks_the_hash_before_serving(self):
        from sepsis_warning.registry import MODEL_NAME, resolve_model
        from sepsis_warning.service import ModelMismatch

        path, expected, _tags = resolve_model(f"models:/{MODEL_NAME}/{self.version}", self.uri, self.root / "dl")
        self.assertEqual(file_sha256(path), expected)
        self.assertEqual(ScoringService.from_path(path, expected).health()["bundle_sha256"], expected)
        with self.assertRaises(ModelMismatch):
            ScoringService.from_path(path, "0" * 64)
        with self.assertRaises(ValueError):
            resolve_model("runs:/abc/model", self.uri)


if __name__ == "__main__":
    unittest.main()
