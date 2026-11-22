"""M02: two-speed monitoring finds simulated shifts, stays quiet without one, and keeps its boundaries."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sepsis_warning import hospital_comparison
from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.features import META, feature_table
from sepsis_warning.modeling import candidate_models, fit_calibrated
from sepsis_warning.monitoring import (
    AGE_BINS,
    AGE_LABELS,
    ALERT,
    FAST_STATS,
    FLOORS,
    INSUFFICIENT,
    REQUIRED_COLUMNS,
    SLOW_STATS,
    WARN,
    MonitoringConfig,
    bin_shares,
    family_of,
    psi,
    quantile_edges,
    reference_from_cohorts,
    run_monitoring,
    subgroup_rows,
)
from sepsis_warning.policy import apply_policy, tune_threshold
from sepsis_warning.scoring import file_sha256, load_bundle, score_patients

# Small windows so the synthetic fixture yields several fast and slow windows.
CONFIG = MonitoringConfig(
    fast_window_hours=400, slow_window_hours=1600, min_observed_values=40, min_observed_stays=5,
    min_slow_stays=20, min_slow_septic_stays=4, min_subgroup_septic_stays=2, null_windows=60,
)
SMALL_FLAGS = [
    "--fast-window-hours", "400", "--slow-window-hours", "1600", "--min-observed-values", "40",
    "--min-observed-stays", "5", "--min-slow-stays", "20", "--min-slow-septic-stays", "4",
    "--min-subgroup-septic-stays", "2", "--null-windows", "60",
]


def statuses(table: pd.DataFrame, stat: str) -> list[str]:
    return table[f"status:{stat}"].tolist()


def family_alerts(table: pd.DataFrame, family: str) -> int:
    columns = [c for c in table if c.startswith("status:") and family_of(c[len("status:"):]) == family]
    return int((table[columns] == ALERT).any(axis=1).sum())


class Monitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=500, external_size=150)
        (cls.data / "SYNTHETIC.json").write_text('{"purpose": "test"}')
        cls.splits = split_patients(load_patients(cls.data), 42)
        tables = {name: feature_table(cls.splits[name], 6) for name in ["train", "calibration", "validation"]}
        columns = [c for c in tables["train"] if c not in META]
        model = fit_calibrated(candidate_models(42)["logistic_regression"], tables["train"], tables["calibration"], columns)
        validation = tables["validation"][META].copy()
        validation["probability"] = model.predict_proba(tables["validation"][columns])[:, 1]
        threshold = tune_threshold(validation, 0.2, 6)["threshold"]
        cls.bundle = {
            "model": model, "model_name": "logistic_regression", "feature_columns": columns,
            "window": 6, "threshold": threshold, "cooldown": 6, "synthetic": True,
        }
        cls.model_path = cls.root / "model.joblib"
        joblib.dump(cls.bundle, cls.model_path)
        cls.provenance = {
            "data_mode": "synthetic_fixture", "split_seed": 42, "model_name": "logistic_regression",
            "model_sha256": file_sha256(cls.model_path),
        }
        cls.reference = reference_from_cohorts(cls.splits, cls.bundle, CONFIG, cls.provenance)
        cls.internal = score_patients(cls.bundle, cls.splits["internal_test"], REQUIRED_COLUMNS)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rescore(self, table, probability):
        changed = table.drop(columns=["hourly_positive", "alert_event"]).assign(probability=probability)
        return apply_policy(changed, self.bundle["threshold"], self.bundle["cooldown"])

    # ------------------------------------------------------------ statistics

    def test_psi_is_zero_without_shift_and_large_with_one(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(80, 10, 20_000)
        edges = quantile_edges(reference, 10)
        shares = bin_shares(reference, edges)
        np.testing.assert_allclose(shares, 0.1, atol=1e-3)
        self.assertLess(psi(shares, bin_shares(rng.normal(80, 10, 5_000), edges)), 0.02)
        self.assertGreater(psi(shares, bin_shares(rng.normal(90, 10, 5_000), edges)), 0.2)
        # Missing values are ignored by PSI (missingness is its own check).
        with_nan = np.r_[reference[:1000], [np.nan] * 500]
        np.testing.assert_allclose(bin_shares(with_nan, edges).sum(), 1)

    def test_edges_handle_constant_binary_and_unobserved_values(self):
        self.assertIsNone(quantile_edges(np.array([np.nan, np.nan]), 10))
        constant = quantile_edges(np.array([5.0] * 10), 10)
        np.testing.assert_array_equal(bin_shares([5.0, 5.0], constant), [0, 1])
        binary = quantile_edges(np.array([0, 1] * 50, dtype=float), 10)
        shares = bin_shares(np.array([0, 1] * 50, dtype=float), binary)
        self.assertAlmostEqual(shares.max(), 0.5)
        self.assertEqual(psi(shares, shares), 0)

    def test_age_bands_match_the_bias_audit(self):
        self.assertEqual(AGE_BINS, hospital_comparison.AGE_BINS)
        self.assertEqual(AGE_LABELS, hospital_comparison.AGE_LABELS)

    def test_config_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            MonitoringConfig(fast_window_hours=0)
        with self.assertRaises(ValueError):
            MonitoringConfig(fast_window_hours=500, slow_window_hours=100)
        with self.assertRaises(ValueError):
            MonitoringConfig(warn_quantile=0.99, alert_quantile=0.95)

    # ------------------------------------------------------------ reference

    def test_reference_reads_only_train_and_validation_and_is_deterministic(self):
        # Any access to the test cohorts would raise KeyError.
        only = {"train": self.splits["train"], "validation": self.splits["validation"]}
        reference = reference_from_cohorts(only, self.bundle, CONFIG, self.provenance)
        self.assertIn("train split", reference["reference_data"]["inputs"])
        self.assertIn("validation split", reference["reference_data"]["scores_alerts_performance"])
        self.assertEqual(reference["performance"]["stays"], len(self.splits["validation"]))
        # Same inputs and seed give byte-identical JSON (and no NaN).
        first = json.dumps(self.reference, allow_nan=False, sort_keys=True)
        self.assertEqual(first, json.dumps(reference, allow_nan=False, sort_keys=True))

    def test_thresholds_respect_floors_and_never_retrain(self):
        thresholds = self.reference["thresholds"]
        self.assertEqual(set(thresholds), set(FAST_STATS + SLOW_STATS))
        for stat, entry in thresholds.items():
            warn_floor, alert_floor = FLOORS[family_of(stat)]
            self.assertGreaterEqual(entry["warn"], warn_floor, stat)
            if alert_floor is None:
                self.assertIsNone(entry["alert"], stat)
            else:
                self.assertGreaterEqual(entry["alert"], max(alert_floor, entry["warn"]), stat)
        self.assertFalse(self.reference["policy"]["automatic_retraining"])
        self.assertEqual(set(self.reference["checks"]["schema"]), {"speed", "owner", "owner_role", "watches"})

    # ------------------------------------------------------------ no shift

    def test_no_alarm_on_unshifted_source_hospital_stays(self):
        results = run_monitoring(self.internal, self.reference)
        fast, slow = results["fast"], results["slow"]
        self.assertGreaterEqual(len(fast), 5)
        self.assertGreaterEqual(len(slow), 2)
        status_columns = [c for c in fast if c.startswith("status:")]
        self.assertFalse((fast[status_columns] == ALERT).any().any())
        self.assertFalse((slow[[c for c in slow if c.startswith("status:")]] == ALERT).any().any())
        self.assertTrue((slow["status:auroc_drop"] != INSUFFICIENT).all())

    # ------------------------------------------------------------ simulated shifts

    def test_unit_change_is_detected_after_two_windows(self):
        shifted = self.internal.assign(Temp=self.internal["Temp"] * 1.8 + 32)  # Celsius charted as Fahrenheit
        fast = run_monitoring(shifted, self.reference)["fast"]
        temp = statuses(fast, "psi:Temp")
        evaluated = [s for s in temp if s != INSUFFICIENT]
        self.assertEqual(evaluated[0], WARN, "persistence: the first breach is only a warning")
        self.assertIn(ALERT, evaluated)

    def test_missingness_shift_is_detected(self):
        rng = np.random.default_rng(1)
        hidden = rng.random(len(self.internal)) < 0.6
        shifted = self.internal.assign(HR=self.internal["HR"].mask(hidden))
        fast = run_monitoring(shifted, self.reference)["fast"]
        self.assertIn(ALERT, statuses(fast, "missing:HR"))

    def test_schema_breaks_alert_immediately(self):
        dropped = run_monitoring(self.internal.drop(columns=["MAP"]), self.reference)["fast"]
        self.assertTrue(all(s == ALERT for s in statuses(dropped, "schema_errors")))
        self.assertIn("MAP", dropped.loc[0, "schema_missing_columns"])
        self.assertEqual(statuses(dropped, "psi:MAP")[0], INSUFFICIENT)
        gap = self.internal.drop(index=self.internal.index[10]).reset_index(drop=True)
        fast = run_monitoring(gap, self.reference)["fast"]
        self.assertEqual(int((fast["schema_hour_gaps"] > 0).sum()), 1)
        self.assertEqual(statuses(fast, "schema_errors").count(ALERT), 1)

    def test_variable_never_seen_in_training_only_warns(self):
        shifted = self.internal.assign(EtCO2=35.0)  # never observed in the synthetic A train split
        fast = run_monitoring(shifted, self.reference)["fast"]
        self.assertTrue(all(s == WARN for s in statuses(fast, "new_variables")))
        self.assertEqual(fast.loc[0, "new_variable_names"], "EtCO2")

    def test_score_and_calibration_shift_without_ranking_change(self):
        # Monotone shift: every score rises, the ranking is unchanged, calibration is broken.
        shifted = self.rescore(self.internal, 0.2 + 0.8 * self.internal["probability"])
        results = run_monitoring(shifted, self.reference)
        self.assertIn(ALERT, statuses(results["fast"], "score_psi"))
        self.assertIn(ALERT, statuses(results["slow"], "calibration_log_oe_change"))
        self.assertNotIn(ALERT, statuses(results["slow"], "auroc_drop"))

    def test_lost_ranking_is_a_slow_performance_alarm(self):
        rng = np.random.default_rng(2)
        shuffled = self.rescore(self.internal, rng.permutation(self.internal["probability"].to_numpy()))
        slow = run_monitoring(shuffled, self.reference)["slow"]
        self.assertIn(ALERT, statuses(slow, "auroc_drop"))
        self.assertIn(ALERT, statuses(slow, "utility_drop"))

    def test_slow_checks_need_settled_labels(self):
        unlabeled = self.internal.assign(label=np.nan)
        results = run_monitoring(unlabeled, self.reference)
        slow = results["slow"]
        for stat in SLOW_STATS:
            self.assertTrue(all(s == INSUFFICIENT for s in statuses(slow, stat)), stat)
        self.assertTrue((slow["slow_note"] == "labels not settled").all())
        # Fast checks still run without labels.
        self.assertTrue(all(s != INSUFFICIENT for s in statuses(results["fast"], "score_psi")))

    def test_small_subgroups_are_flagged_not_scored(self):
        strict = MonitoringConfig(**{**CONFIG.__dict__, "min_subgroup_septic_stays": 10_000})
        rows = subgroup_rows(self.internal, strict)
        self.assertEqual({r["attribute"] for r in rows}, {"sex", "age_band"})
        self.assertTrue(all(not r["sufficient"] and r["normalized_hourly_utility"] is None for r in rows))
        self.assertEqual(sum(r["stays"] for r in rows if r["attribute"] == "sex"), len(self.splits["internal_test"]))

    # ------------------------------------------------------------ commands

    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, "-m", "sepsis_warning", *arguments],
                              capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})

    def test_cli_freezes_reference_then_replays_hospital_b(self):
        reference_dir, replay_dir = self.root / "cli_reference", self.root / "cli_replay"
        setup = self.run_cli("monitor-setup", "--data", str(self.data), "--model", str(self.model_path),
                             "--output", str(reference_dir), *SMALL_FLAGS)
        self.assertEqual(setup.returncode, 0, setup.stderr)
        reference_path = reference_dir / "monitoring_reference.json"
        reference = json.loads(reference_path.read_text())
        self.assertEqual(reference["model_sha256"], file_sha256(self.model_path))
        self.assertEqual(reference["config"]["fast_window_hours"], 400)

        replay = self.run_cli("monitor-replay", "--data", str(self.data), "--model", str(self.model_path),
                              "--reference", str(reference_path), "--split", "external_test",
                              "--output", str(replay_dir), "--after-freeze")
        self.assertEqual(replay.returncode, 0, replay.stderr)
        written = {p.name for p in replay_dir.iterdir()}
        self.assertEqual(written, {"fast_windows.csv", "slow_windows.csv", "slow_subgroups.csv", "summary.json",
                                   "monitoring_report.md", "monitoring_overview.png", "cross_source.json"})
        summary = json.loads((replay_dir / "summary.json").read_text())
        self.assertFalse(summary["automatic_retraining"])
        self.assertEqual(summary["split"], "external_test")
        # The synthetic B has a +4 bpm HR offset and 10 points more missingness.
        self.assertGreater(summary["families"]["missingness"]["alert"], 0)
        self.assertIn("HR", summary["inputs_with_any_psi_alert"])
        # Aggregate outputs only: no record identifiers anywhere.
        record_id = re.compile(r"synthetic_\d{4}")
        for path in [*replay_dir.glob("*"), reference_path]:
            if path.suffix != ".png":
                self.assertIsNone(record_id.search(path.read_text()), path.name)

        again = self.run_cli("monitor-replay", "--data", str(self.data), "--model", str(self.model_path),
                             "--reference", str(reference_path), "--output", str(replay_dir), "--after-freeze")
        self.assertEqual(again.returncode, 2)
        self.assertIn("not empty", again.stderr)

    def test_replay_refuses_a_different_model_and_non_test_splits(self):
        reference_dir = self.root / "guard_reference"
        reference_dir.mkdir()
        reference_path = reference_dir / "monitoring_reference.json"
        reference_path.write_text(json.dumps({**self.reference, "reference_version": 1}))
        tampered = self.root / "tampered.joblib"
        shutil.copy(self.model_path, tampered)
        with open(tampered, "ab") as handle:
            handle.write(b"\0")
        result = self.run_cli("monitor-replay", "--data", str(self.data), "--model", str(tampered),
                              "--reference", str(reference_path), "--split", "external_test", "--after-freeze",
                              "--output", str(self.root / "guard_out"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("SHA-256", result.stderr)
        with self.assertRaises(ValueError):
            load_bundle(tampered, expected_sha256=self.reference["model_sha256"])
        wrong_split = self.run_cli("monitor-replay", "--data", str(self.data), "--model", str(self.model_path),
                                   "--reference", str(reference_path), "--split", "validation",
                                   "--output", str(self.root / "guard_out2"))
        self.assertEqual(wrong_split.returncode, 2)

    def test_setup_refuses_too_small_validation_windows(self):
        result = self.run_cli("monitor-setup", "--data", str(self.data), "--model", str(self.model_path),
                              "--output", str(self.root / "too_big"), "--fast-window-hours", "1000000",
                              "--slow-window-hours", "1000000")
        self.assertEqual(result.returncode, 2)
        self.assertIn("fewer than", result.stderr)


if __name__ == "__main__":
    unittest.main()
