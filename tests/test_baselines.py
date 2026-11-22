"""SIRS / always-no baseline: rule correctness, causality, missing handling."""

import unittest

import numpy as np
import pandas as pd

from sepsis_warning.baselines import (
    BASELINE_THRESHOLD,
    auroc,
    average_precision,
    baseline_table,
    hourly_classification_metrics,
    sirs_count,
    sirs_threshold_sweep,
)
from sepsis_warning.data import DYNAMIC, STATIC, Patient
from sepsis_warning.policy import apply_policy


def make_patient(hours, label_from=None, **values):
    """All-missing patient for `hours` hours; overwrite chosen columns."""
    frame = pd.DataFrame(np.nan, index=range(hours), columns=DYNAMIC + STATIC)
    frame["ICULOS"] = np.arange(1, hours + 1)
    frame["SepsisLabel"] = 0
    if label_from is not None:
        frame.loc[label_from:, "SepsisLabel"] = 1
    for column, series in values.items():
        frame[column] = series
    return frame


class SirsBaseline(unittest.TestCase):
    def test_fires_at_exact_expected_hour(self):
        nan = np.nan
        frame = make_patient(
            6,
            HR=[80, 95, nan, nan, nan, nan],        # abnormal from row 1 on (ffill)
            Temp=[37, nan, nan, 38.5, nan, nan],    # abnormal from row 3 on
            Resp=[18, nan, nan, nan, nan, 22],      # abnormal at row 5
        )
        np.testing.assert_array_equal(sirs_count(frame), [0, 1, 1, 2, 2, 3])
        table = baseline_table([Patient("A:p1", "A", frame)], "sirs")
        alerts = apply_policy(table, BASELINE_THRESHOLD, cooldown=6)
        np.testing.assert_array_equal(alerts["hourly_positive"], [0, 0, 0, 1, 1, 1])
        # First alert at row 3; cooldown suppresses rows 4 and 5.
        np.testing.assert_array_equal(alerts["alert_event"], [0, 0, 0, 1, 0, 0])

    def test_or_criteria_and_thresholds_are_strict(self):
        frame = make_patient(
            5,
            HR=[90, 90, 91, 90, 90],                # boundary 90 is NOT > 90
            Temp=[38, 36, 37, 35.9, 37],            # 38 and 36 are normal
            PaCO2=[40, 40, 40, 40, 31],             # PaCO2 < 32 meets resp criterion
            WBC=[12, 4, 8, 8, 3.9],                 # 12 and 4 are normal
        )
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 1, 1, 2])

    def test_future_hours_never_change_earlier_scores(self):
        rng = np.random.default_rng(0)
        frame = make_patient(
            30, HR=rng.normal(90, 10, 30), Temp=rng.normal(37.5, 1, 30),
            Resp=rng.normal(20, 4, 30), WBC=rng.normal(10, 4, 30),
        )
        frame.loc[rng.random(30) < 0.5, "Temp"] = np.nan
        before = sirs_count(frame)[:15]
        changed = frame.copy()
        changed.loc[15:, ["HR", "Temp", "Resp", "PaCO2", "WBC"]] = [200, 42, 40, 10, 30]
        changed.loc[15:, "SepsisLabel"] = 1
        np.testing.assert_array_equal(before, sirs_count(changed)[:15])
        for length in [1, 7, 15]:
            np.testing.assert_array_equal(before[:length], sirs_count(frame.iloc[:length]))

    def test_missing_values_never_count_as_abnormal(self):
        frame = make_patient(4)                     # nothing ever observed
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 0, 0])
        # A value observed later is not back-filled into earlier hours.
        frame = make_patient(4, HR=[np.nan, np.nan, 120, np.nan], Temp=[np.nan, np.nan, 39, np.nan])
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 2, 2])

    def test_always_no_never_alerts(self):
        frame = make_patient(10, label_from=4, HR=[120] * 10, Temp=[39] * 10)
        table = apply_policy(baseline_table([Patient("A:p", "A", frame)], "always_no"), BASELINE_THRESHOLD, 6)
        self.assertEqual(table["alert_event"].sum(), 0)
        metrics = hourly_classification_metrics(table["label"], table["hourly_positive"])
        # Course convention: 0/0 precision is reported as 0 and flagged.
        self.assertEqual(metrics["hourly_precision"], 0)
        self.assertTrue(metrics["no_predicted_positives"])
        self.assertEqual(metrics["hourly_recall"], 0)
        self.assertEqual(metrics["hourly_f1"], 0)
        self.assertEqual(metrics["hourly_specificity"], 1)
        # Hours 4-9 are positive (6 of 10), so "never alert" is right on the 4 negatives.
        self.assertAlmostEqual(metrics["hourly_accuracy"], 0.4)

    def test_hand_written_metrics(self):
        # y:    1 1 0 0 1 0
        # yhat: 1 0 1 0 1 0   -> TP 2, FN 1, FP 1, TN 2
        metrics = hourly_classification_metrics([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
        self.assertEqual((metrics["hourly_tp"], metrics["hourly_fp"], metrics["hourly_fn"], metrics["hourly_tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(metrics["hourly_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["hourly_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["hourly_f1"], 2 * 2 / (2 * 2 + 1 + 1))
        self.assertAlmostEqual(metrics["hourly_specificity"], 2 / 3)
        self.assertAlmostEqual(metrics["hourly_accuracy"], 4 / 6)
        self.assertFalse(metrics["no_predicted_positives"])

    def test_threshold_sweep_trades_recall_for_fewer_alerts(self):
        rng = np.random.default_rng(3)
        patients = []
        for i in range(30):
            septic_from = 20 if i % 3 == 0 else None
            frame = make_patient(
                30, label_from=septic_from, HR=rng.normal(92, 12, 30), Temp=rng.normal(37.6, 0.9, 30),
                Resp=rng.normal(20, 4, 30), WBC=rng.normal(10, 4, 30),
            )
            patients.append(Patient(f"A:p{i}", "A", frame))
        sweep = sirs_threshold_sweep(baseline_table(patients, "sirs"), cooldown=6)
        self.assertEqual(list(sweep["alert_at_criteria"]), [1, 2, 3, 4])
        # Raising the threshold never flags more hours, so recall cannot rise (L4).
        self.assertTrue((np.diff(sweep["hourly_alert_rate"]) <= 0).all())
        self.assertTrue((np.diff(sweep["hourly_recall"]) <= 1e-12).all())
        self.assertTrue((np.diff(sweep["hourly_specificity"]) >= -1e-12).all())
        # The ">= 2" row equals scoring the official rule directly.
        direct = apply_policy(baseline_table(patients, "sirs"), BASELINE_THRESHOLD, 6)
        official = hourly_classification_metrics(direct["label"], direct["hourly_positive"])
        self.assertAlmostEqual(sweep.loc[1, "hourly_f1"], official["hourly_f1"])


    def test_ranking_metrics_by_hand(self):
        # Positives at 0.9 and 0.4; negatives at 0.8 and 0.1.
        # Pairs (pos > neg): 0.9>0.8, 0.9>0.1, 0.4>0.1 -> 3 of 4.
        self.assertAlmostEqual(auroc([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]), 0.75)
        # Ranking: P N P N -> precision 1 at recall .5, 2/3 at recall 1.
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]), 0.5 + 0.5 * 2 / 3)
        # All tied: AUROC 0.5, AP = prevalence.
        self.assertAlmostEqual(auroc([1, 0, 0, 0], [1, 1, 1, 1]), 0.5)
        self.assertAlmostEqual(average_precision([1, 0, 0, 0], [1, 1, 1, 1]), 0.25)
        self.assertIsNone(auroc([0, 0], [0.1, 0.2]))

    def test_ranking_metrics_match_reference_with_ties(self):
        from sklearn.metrics import average_precision_score, roc_auc_score  # reference only
        rng = np.random.default_rng(1)
        labels = rng.random(2000) < 0.05
        scores = rng.integers(0, 5, 2000) / 4 + labels * rng.random(2000) * 0.3  # heavy ties, like SIRS
        self.assertAlmostEqual(auroc(labels, scores), roc_auc_score(labels, scores))
        self.assertAlmostEqual(average_precision(labels, scores), average_precision_score(labels, scores))


if __name__ == "__main__":
    unittest.main()
