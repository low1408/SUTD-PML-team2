"""L01 run metadata and MLflow logging, without needing MLflow installed."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sepsis_warning.tracking import (
    LOGGABLE_ARTIFACTS, check_tracking_uri, dvc_evidence, log_run, run_metadata, split_evidence,
)

try:
    import yaml
except ImportError:
    yaml = None


def write_summary(directory: Path, seed=42) -> Path:
    path = directory / "splits.csv.summary.json"
    path.write_text(json.dumps({
        "seed": seed,
        "unit": "ICU stay (.psv record); not guaranteed person-disjoint",
        "source_parquet_sha256": "a" * 64,
        "splits_csv_sha256": "b" * 64,
        "counts": {"train": {"stays": 11183, "septic": 1}},
    }))
    return path


class FakeMlflow:
    """Records calls instead of talking to a tracking server."""

    def __init__(self):
        self.calls = []
        self.artifacts = []

    def set_tracking_uri(self, uri):
        self.calls.append(("uri", uri))

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    def start_run(self):
        fake = self

        class Run:
            info = SimpleNamespace(run_id="run-1")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fake.calls.append(("end", None))
                return False

        return Run()

    def set_tags(self, tags):
        self.calls.append(("tags", tags))

    def log_params(self, params):
        self.calls.append(("params", params))

    def log_metrics(self, metrics):
        self.calls.append(("metrics", metrics))

    def log_artifact(self, path):
        self.artifacts.append(Path(path).name)


class Tracking(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, **kwargs):
        return run_metadata(
            42, synthetic=True, window=6, budget=0.2, cooldown=6,
            feature_columns=["HR__last", "HR__mean_6h"], **kwargs,
        )

    def test_split_seed_and_hashes_are_recorded(self):
        metadata = self.metadata(splits_summary=write_summary(self.root))
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["split_seed"], 42)
        self.assertEqual(metadata["splits_csv_sha256"], "b" * 64)
        self.assertEqual(metadata["source_parquet_sha256"], "a" * 64)
        self.assertEqual(len(metadata["splits_summary_sha256"]), 64)
        self.assertIn("scikit-learn", metadata["packages"])
        json.dumps(metadata, allow_nan=False)

    def test_seed_mismatch_with_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not match split seed"):
            split_evidence(write_summary(self.root, seed=7), 42)

    def test_incomplete_split_summary_is_refused(self):
        path = self.root / "splits.csv.summary.json"
        path.write_text(json.dumps({"seed": 42}))
        with self.assertRaisesRegex(ValueError, "splits_csv_sha256"):
            split_evidence(path, 42)

    def test_feature_contract_hash_changes_with_columns(self):
        first = self.metadata()["feature_columns_sha256"]
        other = run_metadata(
            42, synthetic=True, window=6, budget=0.2, cooldown=6, feature_columns=["HR__mean_6h", "HR__last"],
        )["feature_columns_sha256"]
        self.assertNotEqual(first, other)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_dvc_lock_output_hashes_are_recorded(self):
        lock = self.root / "dvc.lock"
        lock.write_text(
            "schema: '2.0'\n"
            "stages:\n"
            "  prepare:\n"
            "    cmd: python -m sepsis_warning prepare\n"
            "    outs:\n"
            "    - path: data/physionet/combined.parquet\n"
            "      md5: 0123456789abcdef0123456789abcdef\n"
        )
        evidence = dvc_evidence(lock)
        self.assertEqual(
            evidence["dvc_outputs"], {"prepare": {"data/physionet/combined.parquet": "0123456789abcdef0123456789abcdef"}},
        )
        self.assertEqual(len(evidence["dvc_lock_sha256"]), 64)

    def test_only_local_tracking_stores_are_allowed(self):
        for uri in ["file:artifacts/mlruns", "sqlite:///artifacts/mlflow.db", "http://127.0.0.1:5000", "artifacts/mlruns"]:
            self.assertEqual(check_tracking_uri(uri), uri)
        for uri in ["https://mlflow.example.com", "databricks", "databricks-uc", "s3://bucket/mlruns"]:
            with self.assertRaisesRegex(ValueError, "approval"):
                check_tracking_uri(uri)

    def test_log_run_sends_aggregates_and_never_patient_level_files(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_text(json.dumps(self.metadata(splits_summary=write_summary(self.root))))
        (run_dir / "metrics.json").write_text(json.dumps({
            "selected_model": "logistic_regression", "threshold": 0.4,
            "held_out": {"internal_test": {"auroc": 0.7, "patients": 10, "median_inferred_lead_hours": None}},
        }))
        (run_dir / "config.json").write_text("{}")
        for private in ["held_out_predictions.csv", "split_manifest.csv", "replay.html", "model.joblib"]:
            (run_dir / private).write_text("private")
        fake = FakeMlflow()
        self.assertEqual(log_run(run_dir, "file:" + str(self.root / "mlruns"), mlflow_module=fake), "run-1")
        self.assertEqual(set(fake.artifacts), {"config.json", "metrics.json", "run_metadata.json"})
        self.assertTrue(set(fake.artifacts) <= LOGGABLE_ARTIFACTS)
        calls = dict(fake.calls)
        self.assertEqual(calls["params"]["seed"], "42")
        self.assertEqual(calls["params"]["splits_csv_sha256"], "b" * 64)
        self.assertEqual(calls["metrics"]["internal_test.auroc"], 0.7)
        self.assertNotIn("internal_test.median_inferred_lead_hours", calls["metrics"])
        self.assertEqual(calls["tags"]["research_only"], "true")

    def test_log_run_refuses_a_remote_store_before_reading_anything(self):
        fake = FakeMlflow()
        with self.assertRaises(ValueError):
            log_run(self.root / "missing", "https://mlflow.example.com", mlflow_module=fake)
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
