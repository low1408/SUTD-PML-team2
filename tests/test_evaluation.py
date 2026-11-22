"""Phase 1 guards: no test scoring in train, frozen evaluation once, manifest checks, data checks."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import joblib
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from sepsis_warning.cli import check_training_data, run_experiment
from sepsis_warning.data import generate_synthetic
from sepsis_warning.evaluation import (
    MARKER_FILE,
    AlreadyEvaluated,
    check_same_split,
    refuse_repeat,
    run_frozen_evaluation,
)
from sepsis_warning.manifest import (
    MANIFEST_FILE,
    ManifestMismatch,
    check_runtime,
    code_sha256,
    package_versions,
    recorded_sha256,
)
from sepsis_warning.model_source import load_serving_model
from sepsis_warning.scoring import file_sha256, load_bundle
from sepsis_warning.service import ModelMismatch, ScoringService

ROOT = Path(__file__).resolve().parents[1]
HAVE_PANDERA = importlib.util.find_spec("pandera") is not None
HAVE_MLFLOW = importlib.util.find_spec("mlflow") is not None


def train_args(data: Path, output: Path, **extra) -> Namespace:
    values = {"data": data, "splits": None, "output": output, "seed": 42, "window": 6, "budget": 0.2,
              "cooldown": 6, "splits_summary": None, "dvc_lock": None, "mlflow_uri": None,
              "register_model": False, **extra}
    return Namespace(**values)


class TrainedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=80, external_size=20)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        cls.run_dir = cls.root / "run"
        run_experiment(train_args(cls.data, cls.run_dir))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()


class TrainDoesNotTouchTestSplits(TrainedRun):
    def test_train_writes_no_test_results(self):
        names = {path.name for path in self.run_dir.iterdir()}
        self.assertNotIn("held_out_predictions.csv", names)
        self.assertIn("validation_predictions.csv", names)
        for name in ("metrics.json", "selection_metrics.json", "manifest.json", "replay.html"):
            text = (self.run_dir / name).read_text()
            for word in ("held_out", "internal_test", "external_test"):
                self.assertNotIn(word, text, f"{word} in {name}")
        replay = pd.read_csv(self.run_dir / "validation_predictions.csv")
        self.assertEqual(set(replay["split"]), {"validation"})
        manifest_rows = pd.read_csv(self.run_dir / "split_manifest.csv")
        validation_ids = set(manifest_rows.loc[manifest_rows["split"] == "validation", "patient_id"])
        self.assertEqual(set(replay["patient_id"]), validation_ids)

    def test_run_experiment_source_never_scores_test_splits(self):
        import inspect

        source = inspect.getsource(run_experiment)
        self.assertNotIn("internal_test", source)
        self.assertNotIn("external_test", source)

    def test_selection_records_the_rule_and_intervals(self):
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        self.assertEqual(metrics["selection"]["selected"], metrics["selected_model"])
        self.assertIn(metrics["selected_model"], metrics["selection"]["within_one_standard_error"])
        for row in metrics["validation_models"]:
            self.assertIn("utility_standard_error", row)
        columns = pd.read_csv(self.run_dir / "validation_models.csv").columns
        self.assertIn("utility_ci_low", columns)
        config = json.loads((self.run_dir / "config.json").read_text())
        metadata = json.loads((self.run_dir / "run_metadata.json").read_text())
        self.assertEqual(config["models_config_sha256"], metadata["models_config_sha256"])


class ManifestTests(TrainedRun):
    def test_manifest_describes_the_bundle_and_pins_its_hash(self):
        manifest = json.loads((self.run_dir / MANIFEST_FILE).read_text())
        bundle = load_bundle(self.run_dir / "model.joblib")
        self.assertEqual(manifest["bundle_sha256"], file_sha256(self.run_dir / "model.joblib"))
        self.assertEqual(recorded_sha256(self.run_dir / "model.joblib"), manifest["bundle_sha256"])
        embedded = bundle["manifest"]
        self.assertNotIn("bundle_sha256", embedded)
        self.assertEqual(embedded["model_name"], bundle["model_name"])
        self.assertEqual(embedded["packages"], package_versions())
        self.assertEqual(embedded["contract_code_sha256"], code_sha256())
        self.assertEqual(embedded["lineage"]["seed"], 42)
        self.assertIsNotNone(embedded["validation"]["normalized_hourly_utility"])

    def test_another_scikit_learn_minor_is_refused(self):
        bundle = joblib.load(self.run_dir / "model.joblib")
        manifest = bundle["manifest"]
        major, minor = manifest["packages"]["scikit-learn"].split(".")[:2]
        other = dict(manifest, packages=dict(manifest["packages"], **{"scikit-learn": f"{major}.{int(minor) + 1}.0"}))
        with self.assertRaisesRegex(ManifestMismatch, "scikit-learn"):
            check_runtime(other)
        with self.assertRaisesRegex(ModelMismatch, "retrain"):
            ScoringService(dict(bundle, manifest=other))
        patch = dict(manifest, packages=dict(manifest["packages"], **{"scikit-learn": f"{major}.{minor}.99"}))
        check_runtime(patch)  # a patch release is accepted
        with self.assertRaisesRegex(ManifestMismatch, "format"):
            check_runtime(dict(manifest, format=99))

    def test_health_returns_the_manifest(self):
        health = ScoringService.from_path(self.run_dir / "model.joblib").health()
        self.assertEqual(health["manifest"]["format"], 1)
        self.assertTrue(health["contract_code_matches"])

    def test_serving_needs_a_verified_bundle(self):
        bare = self.root / "bare"
        bare.mkdir(exist_ok=True)
        shutil.copyfile(self.run_dir / "model.joblib", bare / "model.joblib")
        with self.assertRaisesRegex(ValueError, "manifest.json"):
            load_serving_model(model=bare / "model.joblib")
        service, _, _ = load_serving_model(model=bare / "model.joblib", allow_unverified=True)
        self.assertEqual(service.bundle_sha256, file_sha256(bare / "model.joblib"))
        # A manifest that names another file's hash stops the load before unpickling.
        (bare / MANIFEST_FILE).write_text(json.dumps({"bundle_sha256": "0" * 64}))
        with self.assertRaises(ModelMismatch):
            load_serving_model(model=bare / "model.joblib")
        with self.assertRaisesRegex(ValueError, "manifest.json beside it"):
            load_bundle(bare / "model.joblib")


class FrozenEvaluationTests(TrainedRun):
    def test_evaluation_scores_both_test_splits_and_leaves_a_marker(self):
        output = self.root / "eval-1"
        summary = run_frozen_evaluation(data=self.data, model=self.run_dir / "model.joblib", output=output)
        self.assertEqual(set(summary["splits"]), {"internal_test", "external_test"})
        predictions = pd.read_csv(output / "held_out_predictions.csv")
        self.assertEqual(set(predictions["split"]), {"internal_test", "external_test"})
        self.assertTrue((output / "replay.html").is_file())
        marker = json.loads((self.run_dir / MARKER_FILE).read_text())
        self.assertEqual(marker["bundle_sha256"], file_sha256(self.run_dir / "model.joblib"))
        with self.assertRaisesRegex(ValueError, "not empty"):
            run_frozen_evaluation(data=self.data, model=self.run_dir / "model.joblib", output=output)

    def test_a_real_data_model_is_evaluated_once(self):
        refuse_repeat(None, synthetic=False, identity="m")           # first look
        refuse_repeat("2026-11-15T00:00:00+00:00", synthetic=True, identity="m")   # fixtures may repeat
        with self.assertRaisesRegex(AlreadyEvaluated, "already evaluated"):
            refuse_repeat("2026-11-15T00:00:00+00:00", synthetic=False, identity="m")

    def test_the_split_must_be_the_training_split(self):
        bundle = load_bundle(self.run_dir / "model.joblib")
        check_same_split(bundle["manifest"], self.data, None, 42)
        with self.assertRaisesRegex(ValueError, "seed"):
            check_same_split(bundle["manifest"], self.data, None, 7)
        with self.assertRaisesRegex(ValueError, "no manifest"):
            check_same_split(None, self.data, None, 42)
        # Parquet data: the saved split's hash must match the one in the manifest.
        parquet, splits = self.root / "combined.parquet", self.root / "splits.csv"
        parquet.write_bytes(b"x")
        splits.write_text("hospital,patient_id,split\n")
        with self.assertRaisesRegex(ValueError, "no split hash"):
            check_same_split(bundle["manifest"], parquet, splits, 42)
        pinned = dict(bundle["manifest"], lineage={"splits_csv_sha256": file_sha256(splits)})
        check_same_split(pinned, parquet, splits, 42)
        splits.write_text("hospital,patient_id,split\nA,p1,train\n")
        with self.assertRaisesRegex(ValueError, "not the split"):
            check_same_split(pinned, parquet, splits, 42)

    def test_exactly_one_model_source(self):
        with self.assertRaises(ValueError):
            run_frozen_evaluation(data=self.data, output=self.root / "eval-x")

    @unittest.skipUnless(HAVE_MLFLOW, "needs the tracking extra")
    def test_registered_version_is_tagged_and_logged_separately(self):
        from mlflow.tracking import MlflowClient

        from sepsis_warning.evaluation import EVALUATED_TAG, EVALUATION_EXPERIMENT
        from sepsis_warning.registry import MODEL_NAME, register_bundle
        from sepsis_warning.tracking import SELECTION_EXPERIMENT, log_run

        store = f"sqlite:///{self.root / 'store' / 'mlflow.db'}"
        version = register_bundle(self.run_dir, store, log_run(self.run_dir, store))
        summary = run_frozen_evaluation(data=self.data, model_uri=f"models:/{MODEL_NAME}/{version}",
                                        mlflow_uri=store, output=self.root / "eval-registry")
        client = MlflowClient(tracking_uri=store, registry_uri=store)
        self.assertEqual(client.get_model_version(MODEL_NAME, version).tags[EVALUATED_TAG], summary["evaluated_at"])
        evaluation = client.search_runs([client.get_experiment_by_name(EVALUATION_EXPERIMENT).experiment_id])
        self.assertEqual(len(evaluation), 1)
        self.assertIn("external_test.auroc", evaluation[0].data.metrics)
        for run in client.search_runs([client.get_experiment_by_name(SELECTION_EXPERIMENT).experiment_id]):
            self.assertFalse([k for k in run.data.metrics if "_test" in k])


@unittest.skipUnless(HAVE_PANDERA, "needs the checks extra")
class TrainingDataChecks(unittest.TestCase):
    def setUp(self):
        from sepsis_warning.prepare import prepare_parquet
        from sepsis_warning.splits import write_splits

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        generate_synthetic(root / "raw", source_size=80, external_size=20)
        self.parquet = root / "combined.parquet"
        prepare_parquet(root / "raw", self.parquet)
        write_splits(self.parquet, root / "splits.csv", 42)
        self.output = root / "out"
        self.output.mkdir()

    def args(self, **extra):
        return train_args(self.parquet, self.output, **extra)

    def test_clean_table_passes_and_writes_the_review_counts(self):
        check_training_data(self.args())
        self.assertTrue((self.output / "extreme_value_review_train.csv").is_file())

    def test_a_dropped_label_or_a_skipped_hour_stops_training(self):
        frame = pd.read_parquet(self.parquet)
        septic = frame.loc[frame["SepsisLabel"] == 1].index
        broken = frame.copy()
        broken.loc[septic[-1], "SepsisLabel"] = 0 if len(septic) > 1 else broken.loc[septic[-1], "SepsisLabel"]
        broken.loc[broken.index[1], "ICULOS"] = broken.loc[broken.index[1], "ICULOS"] + 5
        broken.to_parquet(self.parquet)
        with self.assertRaises(ValueError):
            check_training_data(self.args())
        check_training_data(self.args(skip_data_checks=True))  # explicit opt-out only


class NotebookGuard(unittest.TestCase):
    def check(self, cells):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "n.ipynb"
            path.write_text(json.dumps({"cells": cells}))
            return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_notebooks.py"), str(path)],
                                  capture_output=True, text=True, check=False)

    def test_clean_notebook_passes(self):
        result = self.check([{"cell_type": "code", "source": ["from sepsis_warning.features import build_features\n"],
                              "outputs": []}])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_outputs_and_copied_feature_code_fail(self):
        stored = self.check([{"cell_type": "code", "source": ["1 + 1"], "outputs": [{"text": "2"}]}])
        self.assertEqual(stored.returncode, 1)
        self.assertIn("stored outputs", stored.stdout)
        copied = self.check([{"cell_type": "code", "source": ["x = frame['HR'].ffill()\n"], "outputs": []},
                             {"cell_type": "code", "source": ["def build_features2(f):\n    return f\n"], "outputs": []}])
        self.assertEqual(copied.returncode, 1)
        self.assertIn("ffill", copied.stdout)
        self.assertIn("defines a feature", copied.stdout)


if __name__ == "__main__":
    unittest.main()
