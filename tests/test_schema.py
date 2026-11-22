"""E03: valid data passes, each broken rule fails by name, review flags never reject."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from sepsis_warning.data import DYNAMIC, STATIC
from sepsis_warning.schema import MIN_CELL, extreme_value_counts, validate_hourly


def stay(pid="p1", hospital="A", hours=3, labels=None):
    frame = pd.DataFrame({name: 1.0 for name in DYNAMIC + STATIC}, index=range(hours))
    frame["HR"] = 80.0
    frame["ICULOS"] = np.arange(1.0, hours + 1)
    frame["SepsisLabel"] = labels if labels is not None else [0.0] * hours
    frame.insert(0, "patient_id", pid)
    frame.insert(1, "hospital", hospital)
    return frame


@unittest.skipUnless(importlib.util.find_spec("pandera"), "Pandera is an optional install")
class HardRules(unittest.TestCase):
    def assert_fails(self, frame, message):
        with self.assertRaisesRegex(ValueError, message):
            validate_hourly(frame)

    def test_valid_data_passes(self):
        validate_hourly(pd.concat([stay("p1"), stay("p2", "B", labels=[0.0, 1.0, 1.0])], ignore_index=True))

    def test_blank_clinical_values_are_allowed(self):
        frame = stay()
        frame["WBC"] = np.nan
        frame.loc[1, "HR"] = np.nan
        validate_hourly(frame)

    def test_missing_column_fails(self):
        self.assert_fails(stay().drop(columns=["Lactate"]), "missing column Lactate")

    def test_every_missing_column_is_named(self):
        self.assert_fails(stay().drop(columns=["Lactate", "HR", "Temp"]), "missing column HR, Lactate, Temp")

    def test_hour_below_1_fails(self):
        frame = stay()
        frame["ICULOS"] = [-1.0, 0.0, 1.0]
        self.assert_fails(frame, r"ICULOS: whole_hour_from_1 \(2 rows\)")

    def test_part_hour_fails(self):
        frame = stay()
        frame["ICULOS"] = [1.5, 2.5, 3.5]
        self.assert_fails(frame, r"ICULOS: whole_hour_from_1 \(3 rows\)")

    def test_text_in_a_number_column_fails(self):
        frame = stay()
        frame["HR"] = frame["HR"].astype(object)
        frame.loc[0, "HR"] = "abc"
        self.assert_fails(frame, "HR: numeric_type")

    def test_infinite_value_fails(self):
        frame = stay()
        frame.loc[1, "Temp"] = np.inf
        self.assert_fails(frame, r"Temp: no_infinite_values \(1 rows\)")

    def test_label_other_than_0_or_1_fails(self):
        self.assert_fails(stay(labels=[0.0, 2.0, 2.0]), r"SepsisLabel: isin\(\[0, 1\]\)")

    def test_label_dropping_back_to_0_fails(self):
        self.assert_fails(stay(labels=[0.0, 1.0, 0.0]), "label_never_drops_back_to_0")

    def test_gap_in_hours_fails(self):
        frame = stay()
        frame.loc[2, "ICULOS"] = 5.0
        self.assert_fails(frame, "hours_are_consecutive")

    def test_blank_hour_fails(self):
        frame = stay()
        frame.loc[1, "ICULOS"] = np.nan
        self.assert_fails(frame, "ICULOS")

    def test_rules_apply_within_each_stay_only(self):
        # A positive stay followed by a new stay starting at 0 and hour 1 is fine.
        validate_hourly(pd.concat([stay("p1", labels=[0.0, 1.0, 1.0]), stay("p2")], ignore_index=True))

    def test_extreme_values_are_not_rejected(self):
        frame = stay()
        frame["Glucose"] = 5000.0
        validate_hourly(frame)

    def test_error_shows_no_patient_id_or_reading(self):
        frame = stay(pid="secret123")
        frame.loc[1, "Temp"] = np.inf
        frame.loc[2, "SepsisLabel"] = 7.0
        with self.assertRaises(ValueError) as caught:
            validate_hourly(frame)
        self.assertNotIn("secret123", str(caught.exception))
        self.assertNotIn("inf", str(caught.exception).replace("infinite", ""))
        self.assertNotIn("7.0", str(caught.exception))


class ReviewFlags(unittest.TestCase):
    def table(self, frame):
        return extreme_value_counts(frame).set_index("variable")

    def test_counts_readings_outside_the_limits_and_skips_blanks(self):
        frame = stay(hours=40)
        frame["Glucose"] = [900.0] * 12 + [100.0] * 16 + [5.0] * 11 + [np.nan]
        row = self.table(frame).loc["Glucose"]
        self.assertEqual(row["above_high"], 12)
        self.assertEqual(row["below_low"], 11)
        self.assertEqual(row["readings"], 39)

    def test_a_reading_on_the_limit_is_not_flagged(self):
        frame = stay(hours=MIN_CELL)
        frame["HR"] = 250.0
        self.assertEqual(self.table(frame).loc["HR", "above_high"], 0)

    def test_small_counts_are_hidden(self):
        frame = stay(hours=20)
        frame["Glucose"] = [900.0] * (MIN_CELL - 1) + [100.0] * (20 - MIN_CELL + 1)
        self.assertEqual(self.table(frame).loc["Glucose", "above_high"], f"<{MIN_CELL}")

    def test_report_holds_no_identifiers(self):
        self.assertNotIn("secret123", extreme_value_counts(stay(pid="secret123")).to_csv())


if __name__ == "__main__":
    unittest.main()
