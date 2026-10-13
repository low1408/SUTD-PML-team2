"""Checks of leakage boundaries and alert accounting, not model accuracy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.data import generate_synthetic, load_patients, read_patient, split_patients
from sepsis_warning.features import build_features
from sepsis_warning.policy import (
    alert_events, apply_policy, normalized_utility, summarize, tune_threshold, utility_components,
)


class ResearchInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        generate_synthetic(cls.root, source_size=100, external_size=20)
        cls.patients = load_patients(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_feature_prefix_matches_streaming_history(self):
        frame = self.patients[0].frame
        full = build_features(frame)
        for length in [1, 3, 12, len(frame)]:
            pd.testing.assert_frame_equal(full.iloc[:length], build_features(frame.iloc[:length]))

    def test_future_changes_do_not_change_past_features(self):
        frame = self.patients[0].frame.copy()
        before = build_features(frame).iloc[:10]
        frame.loc[10:, "HR"] = 9999
        frame.loc[10:, "Lactate"] = 9999
        frame["SepsisLabel"] = 1 - frame["SepsisLabel"]
        frame["Unit1"] = 999
        pd.testing.assert_frame_equal(before, build_features(frame).iloc[:10])

    def test_missing_value_is_not_backfilled(self):
        frame = self.patients[0].frame.iloc[:3].copy()
        frame["HR"] = [np.nan, np.nan, 80]
        features = build_features(frame)
        self.assertTrue(features.loc[:1, "HR__last"].isna().all())
        self.assertEqual(features.loc[2, "HR__hours_since_seen"], 0)
        self.assertEqual(features.loc[0, "HR__never_seen"], 1)

    def test_split_is_disjoint_and_b_is_external(self):
        splits = split_patients(self.patients, 42)
        ids = [p.patient_id for group in splits.values() for p in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(p.hospital == "B" for p in splits["external_test"]))
        self.assertTrue(all(p.hospital == "A" for name, group in splits.items() if name != "external_test" for p in group))

    def test_cooldown_uses_hours_and_resets_per_patient(self):
        events = alert_events([1, 2, 6, 7, 13], [0.9] * 5, 0.5, 6)
        np.testing.assert_array_equal(events, [1, 0, 0, 1, 1])
        table = pd.DataFrame({"patient_id": ["a", "b"], "hour": [1, 1], "probability": [0.9, 0.9]})
        self.assertEqual(apply_policy(table, 0.5, 6)["alert_event"].sum(), 2)

    def test_utility_matches_official_published_example(self):
        positive, negative = utility_components([0, 0, 0, 0, 1, 1])
        decision = [0, 0, 1, 1, 1, 1]
        score = np.where(decision, positive, negative).sum()
        self.assertAlmostEqual(score, 3.388888888888889)

    def test_utility_normalization_and_late_cutoff(self):
        label = np.r_[np.zeros(18), np.ones(22)].astype(int)
        positive, negative = utility_components(label)
        table = pd.DataFrame({"patient_id": "a", "hour": np.arange(40), "label": label})
        table["hourly_positive"] = positive > negative
        self.assertAlmostEqual(normalized_utility(table), 1)
        table["hourly_positive"] = 0
        self.assertAlmostEqual(normalized_utility(table), 0)
        self.assertTrue((positive[28:] == 0).all())
        self.assertTrue((negative[28:] == 0).all())

    def test_no_alert_fallback_for_zero_budget_even_at_probability_one(self):
        table = pd.DataFrame({
            "patient_id": ["a"] * 24 + ["b"] * 24,
            "hour": list(range(1, 25)) * 2,
            "label": [0] * 24 + [0] * 12 + [1] * 12,
            "probability": [1.0] * 48,
        })
        selected = tune_threshold(table, budget=0, cooldown=6)
        self.assertEqual(selected["threshold"], 1.01)
        self.assertEqual(selected["false_alert_events_per_nonseptic_patient_day"], 0)

    def test_hourly_positives_are_separate_from_notifications(self):
        table = pd.DataFrame({"patient_id": "a", "hour": np.arange(1, 25), "label": 0, "probability": 0.9})
        predicted = apply_policy(table, 0.5, 6)
        self.assertEqual(predicted["hourly_positive"].sum(), 24)
        metrics = summarize(predicted)
        self.assertEqual(metrics["false_alert_events_per_nonseptic_patient_day"], 4)
        self.assertIsNone(metrics["normalized_hourly_utility"])

    def test_lead_time_excludes_left_censored_records(self):
        table = pd.DataFrame({"patient_id": "a", "hour": np.arange(1, 25), "label": 1, "probability": 0.9})
        metrics = summarize(apply_policy(table, 0.5, 6))
        self.assertEqual(metrics["left_censored_septic_patients"], 1)
        self.assertIsNone(metrics["median_inferred_lead_hours"])

    def test_unlabelled_patient_can_be_scored(self):
        path = self.root / "unlabelled.psv"
        self.patients[0].frame.drop(columns="SepsisLabel").to_csv(path, sep="|", index=False)
        self.assertNotIn("SepsisLabel", read_patient(path, require_labels=False))
        with self.assertRaises(ValueError):
            read_patient(path)


if __name__ == "__main__":
    unittest.main()
