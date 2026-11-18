"""Hyperparameter search: grouped folds, the train-only guarantee, reproducibility and MLflow logging."""

import importlib.util
import inspect
import json
import os
import tempfile
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from sepsis_warning import cli, tuning
from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.features import feature_table
from sepsis_warning.modeling import candidate_models, load_model_config

HAVE_MLFLOW = importlib.util.find_spec("mlflow") is not None


class SpaceTests(unittest.TestCase):
    def test_packaged_space_matches_models_in_the_default_config(self):
        space, sha = tuning.load_space()
        config, _ = load_model_config()
        self.assertEqual(len(sha), 64)
        self.assertTrue(set(space) <= set(config))
        self.assertNotIn("constant_prior", space)

    def test_bad_spaces_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "space.toml"
            for text in ("", "[m]\n", "[m]\nC = []\n", "[m]\nC = 3\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    tuning.load_space(path)

    def test_the_reference_is_first_and_nothing_repeats(self):
        space = {"C": [0.1, 0.3, 1.0], "depth": [4, 8]}
        base = {"C": 0.3, "depth": 8, "class_weight": "balanced"}
        rng = np.random.default_rng(0)
        configs = tuning.sample_trials(space, base, 4, rng)
        self.assertEqual(configs[0], base)
        self.assertEqual(len(configs), 5)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in configs}), 5)
        self.assertTrue(all(c["class_weight"] == "balanced" for c in configs))

    def test_a_small_space_is_searched_completely_and_sampling_is_reproducible(self):
        space = {"C": [0.1, 0.3, 1.0]}
        everything = tuning.sample_trials(space, {"C": 0.3}, 50, np.random.default_rng(1))
        self.assertEqual(sorted(c["C"] for c in everything), [0.1, 0.3, 1.0])
        big = {"a": list(range(6)), "b": list(range(6)), "c": list(range(6))}
        first = tuning.sample_trials(big, {"a": 0, "b": 0, "c": 0}, 5, np.random.default_rng(7))
        again = tuning.sample_trials(big, {"a": 0, "b": 0, "c": 0}, 5, np.random.default_rng(7))
        other = tuning.sample_trials(big, {"a": 0, "b": 0, "c": 0}, 5, np.random.default_rng(8))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        with self.assertRaises(ValueError):
            tuning.sample_trials(big, {}, -1, np.random.default_rng(0))


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=160, external_size=30)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        cls.cohorts = split_patients(load_patients(cls.data), 42)
        cls.progress = []
        cls.results, cls.summary, cls.tuned = tuning.run_search(
            cls.cohorts["train"], folds=3, trials=2, seed=42, only=["logistic_regression", "random_forest"],
            progress=cls.progress.append,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_folds_never_split_a_patient(self):
        table = feature_table(self.cohorts["train"], 6)
        folds = tuning.grouped_folds(table, 3, seed=42)
        seen = []
        for train_index, test_index in folds:
            train_ids = set(table.iloc[train_index]["patient_id"])
            test_ids = set(table.iloc[test_index]["patient_id"])
            self.assertFalse(train_ids & test_ids)
            seen.extend(test_index)
        self.assertEqual(sorted(seen), list(range(len(table))))   # every row is tested exactly once

    def test_only_the_training_split_is_searched(self):
        source = inspect.getsource(cli.run_tune)
        self.assertIn('cohorts["train"]', source)
        for other in ("validation", "calibration", "internal_test", "external_test"):
            self.assertNotIn(f'cohorts["{other}"]', source)
        train_stays = len(self.cohorts["train"])
        self.assertEqual(self.summary["stays_used"], train_stays)

    def test_every_model_is_tried_with_its_reference_first(self):
        self.assertEqual(set(self.results["model"]), {"logistic_regression", "random_forest"})
        for _, group in self.results.groupby("model"):
            self.assertEqual(int(group["trial"].iloc[0]), 0)
            self.assertTrue(bool(group["is_reference"].iloc[0]))
            self.assertEqual(len(group), 3)                 # the reference plus two sampled configurations
            self.assertTrue((group["folds_scored"] == 3).all())
        self.assertEqual(list(self.results.columns), tuning.RESULT_COLUMNS)

    def test_the_best_configuration_is_the_best_cross_validation_score(self):
        for model, best in self.summary["models"].items():
            rows = self.results[self.results["model"] == model]
            self.assertAlmostEqual(best["cv_ap_mean"], rows["cv_ap_mean"].max())
            self.assertGreaterEqual(best["ap_gain_over_reference"], 0)
            self.assertEqual(best["winner_is_reference"], best["ap_gain_over_reference"] == 0)
            self.assertEqual(self.tuned[model]["params"], best["params"])
        self.assertIn("optimistic", self.summary["note"])
        self.assertEqual(self.tuned["constant_prior"], load_model_config()[0]["constant_prior"])

    def test_the_search_is_reproducible(self):
        again, _, _ = tuning.run_search(self.cohorts["train"], folds=3, trials=2, seed=42,
                                        only=["logistic_regression"], progress=lambda _: None)
        first = self.results[self.results["model"] == "logistic_regression"]
        for column in ("params", "cv_ap_mean", "cv_auroc_mean"):
            self.assertEqual(list(again[column]), list(first[column]))

    def test_max_stays_uses_a_seeded_subset(self):
        _, small, _ = tuning.run_search(self.cohorts["train"], folds=2, trials=1, seed=3, max_stays=40,
                                        only=["logistic_regression"], progress=lambda _: None)
        self.assertEqual(small["stays_used"], 40)

    def test_bad_requests_are_refused(self):
        with self.assertRaises(ValueError):
            tuning.run_search(self.cohorts["train"], folds=1, progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "no model"):
            tuning.run_search(self.cohorts["train"], only=["constant_prior"], progress=lambda _: None)

    def test_tuned_models_toml_round_trips_and_builds_estimators(self):
        output = self.root / "out"
        tuning.write_outputs(output, self.results, self.summary, self.tuned, models_config_sha256="a" * 64,
                             space_sha256="b" * 64, data_note="synthetic_fixture")
        config, sha = load_model_config(output / "tuned_models.toml")
        self.assertEqual(len(sha), 64)
        for name, entry in self.tuned.items():
            self.assertEqual(config[name].get("params", {}), entry.get("params", {}))
        self.assertEqual(len(candidate_models(0, config)), 4)
        best = json.loads((output / "tuning_best.json").read_text())
        self.assertEqual(best["space_sha256"], "b" * 64)
        self.assertTrue(tomllib.loads((output / "tuned_models.toml").read_text()))

    def test_train_accepts_the_tuned_settings_and_records_their_hash(self):
        output = self.root / "tuned-out"
        tuning.write_outputs(output, self.results, self.summary, self.tuned, models_config_sha256="a" * 64,
                             space_sha256="b" * 64, data_note="synthetic_fixture")
        run = self.root / "run"
        cli.run_experiment(Namespace(
            data=self.data, splits=None, output=run, seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
            models_config=output / "tuned_models.toml"))
        metadata = json.loads((run / "run_metadata.json").read_text())
        self.assertEqual(metadata["models_config_sha256"], load_model_config(output / "tuned_models.toml")[1])
        self.assertNotEqual(metadata["models_config_sha256"], load_model_config()[1])

    @unittest.skipUnless(HAVE_MLFLOW, "needs the tracking extra")
    def test_every_trial_is_logged_to_mlflow_without_validation_or_test_metrics(self):
        from mlflow.tracking import MlflowClient

        from sepsis_warning.tracking import TUNING_EXPERIMENT, log_tuning

        output = self.root / "logged"
        tuning.write_outputs(output, self.results, self.summary, self.tuned, models_config_sha256="a" * 64,
                             space_sha256="b" * 64, data_note="synthetic_fixture")
        store = f"sqlite:///{self.root / 'store' / 'mlflow.db'}"
        parent = log_tuning(output, store)
        client = MlflowClient(tracking_uri=store, registry_uri=store)
        experiment = client.get_experiment_by_name(TUNING_EXPERIMENT)
        runs = client.search_runs([experiment.experiment_id])
        trials = [r for r in runs if r.data.tags.get("run_kind") == "tuning_trial"]
        self.assertEqual(len(trials), len(self.results))
        self.assertEqual(sum(r.data.tags["best"] == "true" for r in trials), len(self.summary["models"]))
        self.assertTrue(all(r.data.tags["mlflow.parentRunId"] == parent for r in trials))
        for run in runs:
            keys = " ".join(run.data.metrics)
            for word in ("validation", "calibration", "internal_test", "external_test"):
                self.assertNotIn(word, keys)
        names = {a.path for a in client.list_artifacts(parent)}
        self.assertEqual(names, set(__import__("sepsis_warning.tracking", fromlist=["x"]).TUNING_ARTIFACTS))


if __name__ == "__main__":
    unittest.main()
