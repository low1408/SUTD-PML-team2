"""Checks of leakage boundaries and alert accounting, not model accuracy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.data import (
    DYNAMIC, STATIC, generate_synthetic, load_patients, read_patient, split_patients,
)
from sklearn.ensemble import HistGradientBoostingClassifier

from sepsis_warning.features import (
    apply_training_fallback, build_features, features_for_windows, fit_training_fallback,
    select_saved_features,
)
from sepsis_warning.modeling import candidate_models
from sepsis_warning.prepare import prepare_parquet
from sepsis_warning.policy import (
    UTILITY_BENCHMARK_HOURS, UTILITY_BENCHMARK_PATIENTS, alert_events, apply_policy,
    benchmark_utility, normalized_utility, summarize, tune_threshold, utility_benchmark_table,
    utility_components,
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

    def test_next_hour_does_not_change_current_features(self):
        frame = self.patients[0].frame.iloc[:6].copy()
        before = build_features(frame).iloc[:4]
        frame.loc[4:, "HR"] = 9999
        frame.loc[4:, "Lactate"] = 9999
        frame.loc[4:, "SepsisLabel"] = 1
        frame.loc[4:, "Unit1"] = 999
        pd.testing.assert_frame_equal(before, build_features(frame).iloc[:4])

    def test_a_leaking_fill_changes_earlier_rows(self):
        """Backward fill reads a later hour, so the same corruption check must fail."""
        frame = self.patients[0].frame.iloc[:6].copy().reset_index(drop=True)
        frame["HR"] = [np.nan, np.nan, np.nan, np.nan, 80.0, 90.0]

        def leaking_fill(table):
            leaked = table.copy()
            leaked["HR"] = leaked["HR"].bfill()
            return leaked

        before = leaking_fill(frame).iloc[:4]
        corrupted = frame.copy()
        corrupted.loc[4:, "HR"] = 9999
        corrupted.loc[4:, "SepsisLabel"] = 1
        corrupted.loc[4:, "Unit1"] = 999
        after = leaking_fill(corrupted).iloc[:4]
        self.assertFalse(before["HR"].equals(after["HR"]))
        pd.testing.assert_frame_equal(build_features(frame).iloc[:4], build_features(corrupted).iloc[:4])

    def test_lab_recency_covers_first_hour_gap_and_never_seen(self):
        frame = self.patients[0].frame.iloc[:6].copy()
        frame["Lactate"] = [np.nan, np.nan, 1.5, np.nan, np.nan, 2.0]
        frame["Fibrinogen"] = np.nan
        features = build_features(frame)
        self.assertEqual(features.loc[0, "Lactate__never_seen"], 1)
        self.assertEqual(features.loc[0, "Lactate__missing_now"], 1)
        self.assertTrue(pd.isna(features.loc[0, "Lactate__hours_since_seen"]))
        self.assertEqual(features.loc[2, "Lactate__hours_since_seen"], 0)
        self.assertEqual(features.loc[2, "Lactate__missing_now"], 0)
        self.assertEqual(features.loc[2, "Lactate__never_seen"], 0)
        self.assertEqual(features.loc[4, "Lactate__hours_since_seen"], frame.loc[4, "ICULOS"] - frame.loc[2, "ICULOS"])
        self.assertEqual(features.loc[4, "Lactate__missing_now"], 1)
        self.assertTrue(features["Fibrinogen__never_seen"].eq(1).all())
        self.assertTrue(features["Fibrinogen__missing_now"].eq(1).all())
        self.assertTrue(features["Fibrinogen__hours_since_seen"].isna().all())

    def test_stay_starting_after_hour_one_uses_iculos(self):
        frame = self.patients[0].frame.iloc[:6].copy().reset_index(drop=True)
        frame["ICULOS"] = [8, 9, 10, 12, 13, 14]
        frame["Lactate"] = [np.nan, 1.5, np.nan, np.nan, 2.0, np.nan]
        full = build_features(frame)
        pd.testing.assert_frame_equal(full.iloc[:3], build_features(frame.iloc[:3]))
        self.assertTrue(pd.isna(full.loc[0, "Lactate__hours_since_seen"]))
        self.assertEqual(float(full.loc[1, "Lactate__hours_since_seen"]), 0.0)
        # ICU hour 12 is three hours after the lactate at hour 9, and only two rows later.
        self.assertEqual(
            float(full.loc[3, "Lactate__hours_since_seen"]),
            float(frame.loc[3, "ICULOS"] - frame.loc[1, "ICULOS"]),
        )

    def test_scoring_a_prefix_matches_the_full_stay(self):
        frame = next(patient.frame for patient in self.patients if patient.frame["SepsisLabel"].nunique() == 2)
        full = build_features(frame)
        model = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=0)
        model.fit(full, frame["SepsisLabel"].to_numpy())
        prefix = build_features(frame.iloc[:8])
        full_scores = model.predict_proba(full.iloc[:8])[:, 1]
        prefix_scores = model.predict_proba(prefix)[:, 1]
        np.testing.assert_allclose(prefix_scores, full_scores)

    def test_saved_feature_columns_must_match_this_stay(self):
        features = build_features(self.patients[0].frame.iloc[:4])
        columns = list(features.columns)
        selected = select_saved_features(features, columns)
        self.assertEqual(list(selected.columns), columns)
        with self.assertRaises(ValueError):
            select_saved_features(features, [])
        with self.assertRaises(ValueError):
            select_saved_features(features, columns + ["HR__mean_12h"])

    def test_validation_rows_do_not_change_the_training_fallback(self):
        train = pd.DataFrame({"HR": [80.0, 100.0], "Fibrinogen": [np.nan, np.nan]})
        validation = pd.DataFrame({"HR": [0.0, 0.0, 0.0], "Fibrinogen": [9.0, 9.0, 9.0]})
        fitted = fit_training_fallback([train])
        self.assertEqual(float(fitted["HR"]), 90.0)
        self.assertTrue(pd.isna(fitted["Fibrinogen"]))
        pd.testing.assert_series_equal(fitted, fit_training_fallback([train]))
        leaked = fit_training_fallback([train, validation])
        self.assertNotEqual(float(leaked["HR"]), float(fitted["HR"]))
        self.assertEqual(float(leaked["Fibrinogen"]), 9.0)
        held_out = pd.DataFrame({"HR": [np.nan], "Fibrinogen": [np.nan]})
        applied = apply_training_fallback(held_out, fitted)
        self.assertEqual(float(applied.loc[0, "HR"]), 90.0)
        self.assertTrue(pd.isna(applied.loc[0, "Fibrinogen"]))

    def test_outage_keeps_the_last_value_and_leaves_raw_data_unchanged(self):
        frame = pd.DataFrame({"HR": [80.0, np.nan, np.nan, 100.0]})
        original = frame.copy()
        applied = apply_training_fallback(frame, pd.Series({"HR": 90.0}))
        self.assertEqual(applied["HR"].tolist(), [80.0, 80.0, 80.0, 100.0])
        pd.testing.assert_frame_equal(frame, original)

    def test_leading_gap_uses_the_training_fallback_not_the_future(self):
        frame = pd.DataFrame({"HR": [np.nan, np.nan, 80.0]})
        applied = apply_training_fallback(frame, pd.Series({"HR": 90.0}))
        self.assertEqual(applied["HR"].tolist(), [90.0, 90.0, 80.0])

    def test_fallback_resets_at_each_patient(self):
        frame = pd.DataFrame({
            "patient_id": ["b", "b", "a", "a"],
            "HR": [50.0, 50.0, np.nan, np.nan],
        })
        applied = apply_training_fallback(frame, pd.Series({"HR": 90.0}))
        self.assertEqual(applied.loc[applied["patient_id"] == "a", "HR"].tolist(), [90.0, 90.0])
        self.assertEqual(applied.loc[applied["patient_id"] == "b", "HR"].tolist(), [50.0, 50.0])
        reversed_stays = pd.DataFrame({
            "patient_id": ["a", "a", "b", "b"],
            "HR": [80.0, 80.0, np.nan, np.nan],
        })
        applied = apply_training_fallback(reversed_stays, pd.Series({"HR": 90.0}))
        self.assertEqual(applied.loc[applied["patient_id"] == "b", "HR"].tolist(), [90.0, 90.0])

    def test_training_model_imputer_ignores_validation_rows(self):
        train = pd.DataFrame({"HR__last": [80.0, 100.0, np.nan], "label": [0, 1, 0]})
        validation = pd.DataFrame({"HR__last": [0.0, 0.0], "label": [0, 1]})
        model = candidate_models(0)["logistic_regression"]
        model.fit(train[["HR__last"]], train["label"])
        training_median = float(model.named_steps["simpleimputer"].statistics_[0])
        self.assertEqual(training_median, 90.0)
        blank = pd.DataFrame({"HR__last": [np.nan, np.nan]})
        filled = model.named_steps["simpleimputer"].transform(blank)
        self.assertTrue(np.all(filled == 90.0))
        combined = pd.concat([train, validation], ignore_index=True)
        refit = candidate_models(0)["logistic_regression"]
        refit.fit(combined[["HR__last"]], combined["label"])
        self.assertNotEqual(float(refit.named_steps["simpleimputer"].statistics_[0]), training_median)

    def test_shared_builder_supplies_every_window(self):
        frame = self.patients[0].frame.iloc[:12]
        six = build_features(frame, 6)
        wide = features_for_windows(frame, (3, 6, 12))
        pd.testing.assert_frame_equal(wide[six.columns], six)
        self.assertIn("HR__mean_3h", wide.columns)
        self.assertIn("HR__mean_12h", wide.columns)
        self.assertEqual(str(wide["HR__last"].dtype), "float32")
        source = (Path(__file__).resolve().parents[1] / "scripts" / "causal_features.py").read_text(encoding="utf-8")
        self.assertIn("build_features", source)
        self.assertNotIn("def causal_rolling_features", source)
        self.assertNotIn("def utility_components", source)
        self.assertNotIn(".rolling(", source)

    def test_utility_benchmark_records_runtime_without_a_speed_limit(self):
        """Time hourly utility on 500 patients x 48 hours. No pass/fail speed limit."""
        table = utility_benchmark_table()
        first = benchmark_utility(table)
        second = benchmark_utility(table)
        self.assertEqual(first["patients"], UTILITY_BENCHMARK_PATIENTS)
        self.assertEqual(first["hours_per_patient"], UTILITY_BENCHMARK_HOURS)
        self.assertEqual(first["patient_hours"], UTILITY_BENCHMARK_PATIENTS * UTILITY_BENCHMARK_HOURS)
        self.assertGreater(first["elapsed_seconds"], 0)
        self.assertEqual(first["normalized_utility"], second["normalized_utility"])
        self.assertEqual(first["normalized_utility"], normalized_utility(table))
        changed = table.copy()
        changed.loc[UTILITY_BENCHMARK_HOURS, "hourly_positive"] = 1
        self.assertNotEqual(normalized_utility(changed), first["normalized_utility"])
        print(
            "Utility benchmark: "
            f"{first['patients']} patients x {first['hours_per_patient']} hours "
            f"= {first['patient_hours']} patient-hours in {first['elapsed_seconds']:.4f} seconds"
        )

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

    def test_prepare_combines_validated_records_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for hospital, patient_id in [("A", "a001"), ("B", "b001")]:
                directory = root / f"training_set{hospital}"
                directory.mkdir()
                frame = pd.DataFrame(1.0, index=range(2), columns=DYNAMIC + STATIC)
                frame["ICULOS"] = [1, 2]
                frame["SepsisLabel"] = [0, 1]
                frame.to_csv(directory / f"{patient_id}.psv", sep="|", index=False)

            output = root / "combined.parquet"
            summary = prepare_parquet(root, output)
            combined = pd.read_parquet(output)

            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["hourly_rows"], 4)
            self.assertEqual(summary["records_by_hospital"], {"A": 1, "B": 1})
            self.assertEqual(summary["hourly_rows_by_hospital"], {"A": 2, "B": 2})
            self.assertEqual(len(combined), 4)
            self.assertEqual(
                combined.groupby("hospital")["patient_id"].nunique().to_dict(),
                {"A": 1, "B": 1},
            )
            self.assertEqual(set(combined["patient_id"]), {"a001", "b001"})


if __name__ == "__main__":
    unittest.main()
