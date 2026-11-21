"""Tests for K04 before/after Platt scaling Brier scores and binned calibration curves."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from sepsis_warning.cli import run_experiment
from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.features import META, feature_table
from sepsis_warning.logistic import NumpyLogisticRegression
from sepsis_warning.modeling import candidate_models, fit_calibrated, load_model_config
from sepsis_warning.policy import calibration_curve_bins, calibration_diagnostics


class CalibrationCurveTests(unittest.TestCase):
    def test_bins_cover_unit_interval_including_one_and_handle_empty_bins(self):
        labels = [0, 0, 1, 1]
        probs = [0.05, 0.15, 0.85, 1.0]
        bins, ece = calibration_curve_bins(labels, probs, n_bins=5)

        self.assertEqual(len(bins), 5)
        self.assertEqual(sum(b["count"] for b in bins), 4)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[0]["positive_count"], 0)
        self.assertAlmostEqual(bins[0]["mean_predicted"], 0.10)
        self.assertAlmostEqual(bins[0]["observed_frequency"], 0.0)

        self.assertEqual(bins[1]["count"], 0)
        self.assertIsNone(bins[1]["mean_predicted"])
        self.assertIsNone(bins[1]["observed_frequency"])

        self.assertEqual(bins[4]["count"], 2)
        self.assertEqual(bins[4]["positive_count"], 2)
        self.assertAlmostEqual(bins[4]["mean_predicted"], 0.925)
        self.assertAlmostEqual(bins[4]["observed_frequency"], 1.0)

        expected_ece = (2 / 4) * abs(0.0 - 0.10) + (2 / 4) * abs(1.0 - 0.925)
        self.assertAlmostEqual(ece, expected_ece)

    def test_before_after_diagnostics_for_numpy_and_sklearn_models(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "data"
            generate_synthetic(data_dir, seed=7, source_size=80, external_size=10)
            cohorts = split_patients(load_patients(data_dir), seed=7)
            train_df = feature_table(cohorts["train"], 6)
            cal_df = feature_table(cohorts["calibration"], 6)
            val_df = feature_table(cohorts["validation"], 6)
            cols = [c for c in train_df if c not in META]

            numpy_est = NumpyLogisticRegression(C=0.3)
            numpy_cal = fit_calibrated(numpy_est, train_df, cal_df, cols)
            diag_np = calibration_diagnostics(
                val_df["label"],
                numpy_est.predict_proba(val_df[cols])[:, 1],
                numpy_cal.predict_proba(val_df[cols])[:, 1],
            )
            self.assertEqual(diag_np["n_samples"], len(val_df))
            self.assertEqual(diag_np["n_positives"], int(val_df["label"].sum()))
            self.assertEqual(diag_np["bin_strategy"], "uniform")
            self.assertEqual(diag_np["n_bins"], 10)
            self.assertAlmostEqual(diag_np["brier_delta"], diag_np["brier_after"] - diag_np["brier_before"])
            self.assertEqual(len(diag_np["bins_before"]), 10)
            self.assertEqual(len(diag_np["bins_after"]), 10)

            config, _ = load_model_config()
            sk_est = candidate_models(7, config)["logistic_regression"]
            sk_cal = fit_calibrated(sk_est, train_df, cal_df, cols)
            diag_sk = calibration_diagnostics(
                val_df["label"],
                sk_est.predict_proba(val_df[cols])[:, 1],
                sk_cal.predict_proba(val_df[cols])[:, 1],
            )
            self.assertEqual(sum(b["count"] for b in diag_sk["bins_after"]), len(val_df))

    def test_train_records_before_after_brier_and_calibration_curves(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            out_dir = root / "out"
            generate_synthetic(data_dir, seed=42, source_size=80, external_size=20)
            (data_dir / "SYNTHETIC.json").write_text("{}")
            args = Namespace(
                data=data_dir,
                splits=None,
                splits_summary=None,
                dvc_lock=None,
                models_config=None,
                include_numpy_models=True,
                output=out_dir,
                seed=42,
                window=6,
                budget=0.2,
                cooldown=6,
                mlflow_uri=None,
                register_model=False,
            )
            run_experiment(args)

            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))

            self.assertIn("calibration", metrics)
            self.assertEqual(metrics["calibration"], calibration)
            self.assertIn("numpy_logistic_regression", calibration)
            self.assertIn("hist_gradient_boosting", calibration)

            for row in metrics["validation_models"]:
                self.assertIn("uncalibrated_brier_score", row)
                cal = calibration[row["model"]]
                self.assertAlmostEqual(row["uncalibrated_brier_score"], cal["brier_before"])
                self.assertAlmostEqual(row["brier_score"], cal["brier_after"])
                self.assertEqual(cal["n_samples"], row["hours"])
                self.assertEqual(sum(b["count"] for b in cal["bins_after"]), row["hours"])
