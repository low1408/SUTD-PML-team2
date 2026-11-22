"""Model config, bootstrap intervals, the one-standard-error rule and NumPy candidates."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.features import META, feature_table
from sepsis_warning.modeling import (
    candidate_models,
    complexity_ranks,
    fit_calibrated,
    is_numpy_model,
    load_model_config,
)
from sepsis_warning.policy import (
    apply_policy,
    normalized_utility,
    select_candidate,
    tune_threshold,
    utility_interval,
)


class ModelConfigTests(unittest.TestCase):
    def test_default_config_enables_the_four_reference_models(self):
        config, sha = load_model_config()
        self.assertEqual(len(sha), 64)
        self.assertEqual(list(candidate_models(0, config)),
                         ["constant_prior", "logistic_regression", "random_forest", "hist_gradient_boosting"])
        with_numpy = candidate_models(0, config, include_numpy=True)
        self.assertEqual(len(with_numpy), 7)
        self.assertTrue(is_numpy_model(with_numpy["numpy_random_forest"]))
        self.assertFalse(is_numpy_model(with_numpy["random_forest"]))
        self.assertLess(complexity_ranks(config)["logistic_regression"],
                        complexity_ranks(config)["hist_gradient_boosting"])

    def test_settings_come_from_the_file_and_change_its_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "models.toml"
            path.write_text('[logistic_regression]\nkind = "sklearn.logistic_regression"\ncomplexity = 1\n'
                            'params = { C = 7.5, solver = "liblinear" }\n')
            config, sha = load_model_config(path)
            self.assertEqual(candidate_models(3, config)["logistic_regression"][-1].C, 7.5)
            self.assertNotEqual(sha, load_model_config()[1])

    def test_bad_configs_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "models.toml"
            for text in ('[x]\nkind = "sklearn.unknown"\ncomplexity = 1\n',
                         '[x]\nkind = "sklearn.random_forest"\n', ""):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_model_config(path)
            path.write_text('[x]\nkind = "numpy.random_forest"\nenabled = false\ncomplexity = 1\n')
            with self.assertRaises(ValueError):
                candidate_models(0, load_model_config(path)[0])


class SelectionRuleTests(unittest.TestCase):
    @staticmethod
    def row(model, utility, error=0.02, false_alerts=0.1, precision=0.5):
        return {"model": model, "normalized_hourly_utility": utility, "utility_standard_error": error,
                "false_alert_events_per_nonseptic_patient_day": false_alerts, "average_precision": precision}

    def test_the_simplest_model_within_one_standard_error_wins(self):
        rows = [self.row("constant_prior", 0.0), self.row("logistic_regression", 0.39),
                self.row("random_forest", 0.385), self.row("hist_gradient_boosting", 0.40, error=0.02)]
        selected, how = select_candidate(rows, complexity_ranks())
        self.assertEqual(selected["model"], "logistic_regression")
        self.assertEqual(how["best_by_utility"], "hist_gradient_boosting")
        self.assertNotIn("constant_prior", how["within_one_standard_error"])

    def test_a_clearly_better_complex_model_is_kept(self):
        rows = [self.row("logistic_regression", 0.30), self.row("hist_gradient_boosting", 0.40, error=0.02)]
        self.assertEqual(select_candidate(rows, complexity_ranks())[0]["model"], "hist_gradient_boosting")

    def test_without_an_interval_the_best_utility_wins(self):
        rows = [self.row("logistic_regression", 0.399, error=None), self.row("random_forest", 0.40, error=None)]
        self.assertEqual(select_candidate(rows, complexity_ranks())[0]["model"], "random_forest")


class IntervalAndNumpyCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        generate_synthetic(root, source_size=120, external_size=20)
        cohorts = split_patients(load_patients(root), 42)
        cls.tables = {name: feature_table(cohorts[name], 6) for name in ("train", "calibration", "validation")}
        cls.columns = [c for c in cls.tables["train"] if c not in META]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scored(self, estimator):
        model = fit_calibrated(estimator, self.tables["train"], self.tables["calibration"], self.columns)
        table = self.tables["validation"][META].copy()
        table["probability"] = model.predict_proba(self.tables["validation"][self.columns])[:, 1]
        return table

    def test_interval_brackets_the_utility_and_is_reproducible(self):
        table = self.scored(candidate_models(42)["logistic_regression"])
        metrics = tune_threshold(table, 0.2, 6)
        self.assertLessEqual(metrics["utility_ci_low"], metrics["normalized_hourly_utility"])
        self.assertGreaterEqual(metrics["utility_ci_high"], metrics["normalized_hourly_utility"])
        self.assertGreater(metrics["utility_standard_error"], 0)
        applied = apply_policy(table, metrics["threshold"], 6)
        self.assertEqual(utility_interval(applied, seed=1), utility_interval(applied, seed=1))
        self.assertAlmostEqual(normalized_utility(applied), metrics["normalized_hourly_utility"])

    def test_interval_resamples_whole_patients(self):
        table = apply_policy(self.scored(candidate_models(42)["logistic_regression"]), 0.5, 6)
        doubled = pd.concat([table, table.assign(patient_id=table["patient_id"] + "-copy")], ignore_index=True)
        # Twice the patients: the same utility, a smaller standard error.
        self.assertAlmostEqual(normalized_utility(doubled), normalized_utility(table))
        self.assertLess(utility_interval(doubled)["utility_standard_error"],
                        utility_interval(table)["utility_standard_error"])

    def test_numpy_candidates_rank_like_their_sklearn_counterparts(self):
        models = candidate_models(42, include_numpy=True)
        labels = self.tables["validation"]["label"].to_numpy()
        for numpy_name, sklearn_name in [("numpy_logistic_regression", "logistic_regression"),
                                         ("numpy_random_forest", "random_forest"),
                                         ("numpy_gradient_boosting", "hist_gradient_boosting")]:
            ours = roc_auc_score(labels, self.scored(models[numpy_name])["probability"])
            reference = roc_auc_score(labels, self.scored(models[sklearn_name])["probability"])
            self.assertLess(abs(ours - reference), 0.03, f"{numpy_name}: {ours:.3f} vs {reference:.3f}")
            probability = self.scored(models[numpy_name])["probability"].to_numpy()
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
