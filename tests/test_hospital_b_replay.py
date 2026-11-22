"""M01: hospital B replay gives cross-source metrics with stay-grouped uncertainty, and never retunes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.cross_source import (
    METRICS,
    _bootstrap,
    cross_source_report,
    timing_limitations,
)
from sepsis_warning.data import generate_synthetic
from sepsis_warning.policy import apply_policy


def stays_table(n_stays: int, septic_every: int, seed: int) -> pd.DataFrame:
    """Synthetic hourly tables: alerts track the label imperfectly, with a stable seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(n_stays):
        hours = np.arange(1, 25)
        label = np.zeros(24, dtype=int)
        if stay % septic_every == 0:
            label[12:] = 1
        probability = np.clip(0.2 + 0.6 * label + rng.normal(0, 0.15, 24), 0, 1)
        rows.append(pd.DataFrame({"patient_id": f"s{stay}", "hospital": "A", "hour": hours,
                                  "label": label, "probability": probability}))
    table = pd.concat(rows, ignore_index=True)
    return apply_policy(table, 0.5, 6)


class CrossSourceTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_stays_not_hours(self):
        table = stays_table(40, 4, seed=1)
        groups = [g for _, g in table.groupby("patient_id", sort=False)]
        draws = _bootstrap(groups, 30, np.random.default_rng(0))
        self.assertEqual(set(draws), set(METRICS))
        self.assertTrue(all(len(v) > 0 for v in draws.values()))

    def test_report_has_points_intervals_and_a_paired_difference(self):
        reference = stays_table(60, 4, seed=1)
        replay = stays_table(50, 3, seed=2)
        report = cross_source_report(reference, replay, draws=40, seed=7)
        for source in ("reference_A_validation", "replay_B_external_test"):
            self.assertIn("normalized_hourly_utility", report["metrics_point"][source])
            interval = report["metrics_interval_95"][source]["normalized_hourly_utility"]
            self.assertLessEqual(interval["low"], interval["high"])
        diff = report["difference_B_minus_A"]["normalized_hourly_utility"]
        self.assertAlmostEqual(diff, report["metrics_point"]["replay_B_external_test"]["normalized_hourly_utility"]
                               - report["metrics_point"]["reference_A_validation"]["normalized_hourly_utility"])
        self.assertIn("paired", report["method"])
        self.assertIn("No threshold was changed", report["caveat"])

    def test_same_seed_same_report(self):
        reference, replay = stays_table(40, 4, 1), stays_table(40, 4, 2)
        self.assertEqual(cross_source_report(reference, replay, draws=20, seed=3),
                         cross_source_report(reference, replay, draws=20, seed=3))

    def test_timing_counts_censored_and_unscorable_stays(self):
        early = stays_table(8, 2, seed=3)
        early.loc[(early["patient_id"] == "s0") & (early["hour"] == 1), "label"] = 1   # left-censored
        early.loc[early["patient_id"] == "s2", "label"] = 0
        early.loc[(early["patient_id"] == "s2") & (early["hour"] >= 22), "label"] = 1  # onset+6 past the end
        timing = timing_limitations(early)
        self.assertEqual(timing["stays"], 8)
        self.assertEqual(timing["left_censored_septic_stays"], 1)
        self.assertGreaterEqual(timing["septic_stays_ending_before_onset"], 1)
        self.assertEqual(timing["stays_eligible_for_early_detection"],
                         timing["septic_stays"] - timing["left_censored_septic_stays"]
                         - timing["septic_stays_ending_before_onset"])


class FreezeGateTests(unittest.TestCase):
    def test_test_splits_need_the_freeze_flag(self):
        from sepsis_warning.ward import check_split

        for split in ("internal_test", "external_test"):
            with self.assertRaisesRegex(ValueError, "after the model is frozen"):
                check_split(split, after_freeze=False)
            check_split(split, after_freeze=True)
        check_split("validation", after_freeze=False)

    def test_cli_refuses_hospital_b_before_freeze(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-m", "sepsis_warning", "monitor-replay", "--data", temp,
                 "--model", "x.joblib", "--reference", "x.json", "--split", "external_test",
                 "--output", str(Path(temp) / "out")], capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("after the model is frozen", result.stderr)


class ReplayEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from argparse import Namespace

        from sepsis_warning.cli import run_experiment
        from sepsis_warning.monitoring import (
            MonitoringConfig,
            replay_monitoring,
            setup_monitoring,
        )

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=200, external_size=60)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        run_experiment(Namespace(data=cls.data, splits=None, output=cls.root / "run", seed=42, window=6,
                                 budget=0.2, cooldown=6, splits_summary=None, dvc_lock=None,
                                 mlflow_uri=None, register_model=False))
        config = MonitoringConfig(fast_window_hours=400, slow_window_hours=1600, min_observed_values=40,
                                  min_observed_stays=5, min_slow_stays=20, min_slow_septic_stays=4,
                                  min_subgroup_septic_stays=2, null_windows=60)
        cls.reference = cls.root / "reference"
        setup_monitoring(cls.data, cls.root / "run" / "model.joblib", cls.reference, config, 42)
        cls.summary = replay_monitoring(cls.data, cls.root / "run" / "model.joblib",
                                        cls.reference / "monitoring_reference.json", "external_test",
                                        cls.root / "replay")
        cls.threshold_before = json.loads((cls.reference / "monitoring_reference.json").read_text())["policy"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_replay_writes_cross_source_aggregates_only(self):
        out = self.root / "replay"
        self.assertTrue((out / "cross_source.json").is_file())
        cross = json.loads((out / "cross_source.json").read_text())
        self.assertEqual(set(cross["metrics_point"]), {"reference_A_validation", "replay_B_external_test"})
        text = (out / "cross_source.json").read_text()
        self.assertNotIn("patient_id", text)
        for name in out.iterdir():
            if name.suffix == ".csv":
                self.assertNotIn("patient_id", name.read_text().splitlines()[0])

    def test_the_frozen_policy_is_the_one_replayed(self):
        bundle_threshold = json.loads((self.root / "run" / "metrics.json").read_text())["threshold"]
        reference = json.loads((self.reference / "monitoring_reference.json").read_text())
        self.assertEqual(reference["model_sha256"], self.summary["model_sha256"])
        self.assertEqual(self.threshold_before, reference["policy"])
        self.assertIsNotNone(bundle_threshold)

    def test_internal_test_replay_has_no_cross_source_block(self):
        from sepsis_warning.monitoring import replay_monitoring

        out = self.root / "replay-internal"
        summary = replay_monitoring(self.data, self.root / "run" / "model.joblib",
                                    self.reference / "monitoring_reference.json", "internal_test", out)
        self.assertNotIn("cross_source", summary)
        self.assertFalse((out / "cross_source.json").exists())


if __name__ == "__main__":
    unittest.main()
