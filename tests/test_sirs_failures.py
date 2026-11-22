"""Where SIRS fails: hand-checkable totals, and nothing patient-level gets out."""

import unittest

import numpy as np
import pandas as pd

from sepsis_warning.baselines import SIRS_COLUMNS
from sepsis_warning.data import Patient
from sepsis_warning.sirs_failures import (
    MIN_CELL,
    by_hour_of_stay,
    caught_vs_missed,
    failure_tables,
    false_alarm_causes,
    hourly_table,
    missed_stays,
)


def stay(name, hr, resp=None, wbc=None, labels=None, age=60.0, gender=1.0):
    n = len(hr)
    frame = pd.DataFrame({column: np.nan for column in SIRS_COLUMNS}, index=range(n))
    frame["HR"] = hr
    frame["Temp"] = 37.0
    frame["Resp"] = resp if resp is not None else [12.0] * n
    frame["WBC"] = wbc if wbc is not None else [np.nan] * n
    frame["ICULOS"] = range(1, n + 1)
    frame["SepsisLabel"] = labels if labels is not None else [0] * n
    frame["Age"] = age
    frame["Gender"] = gender
    return Patient(f"A:{name}", "A", frame)


class Failures(unittest.TestCase):
    def test_false_alarm_causes_counts_the_criteria_that_fired(self):
        patients = [stay("a", [100.0, 100.0, 70.0], resp=[25.0, 25.0, 12.0])]
        causes = false_alarm_causes(hourly_table(patients))
        self.assertEqual(causes.iloc[0]["criteria_met"], "hr + resp")
        self.assertEqual(causes.iloc[0]["false_alarm_hours"], 2)
        self.assertEqual(causes.iloc[0]["share"], 1.0)

    def test_septic_hours_split_into_caught_and_missed(self):
        # Hour 1: HR + Resp met, caught. Hour 2: HR only, missed. WBC never measured.
        patient = stay("a", [100.0, 100.0], resp=[25.0, 12.0], labels=[1, 1])
        table = caught_vs_missed(hourly_table([patient])).set_index("criterion")
        self.assertEqual(table.loc["resp", "met_in_caught_hours"], 1.0)
        self.assertEqual(table.loc["resp", "met_in_missed_hours"], 0.0)
        self.assertEqual(table.loc["wbc", "never_measured_in_missed_hours"], 1.0)
        self.assertEqual(table.loc["wbc", "caught_hours"], 1)
        self.assertEqual(table.loc["wbc", "missed_hours"], 1)

    def test_a_reading_counts_as_measured_only_from_that_hour_on(self):
        patient = stay("a", [70.0, 70.0, 70.0], wbc=[np.nan, 9.0, np.nan])
        self.assertEqual(hourly_table([patient])["never_sirs_wbc"].tolist(), [True, False, False])

    def test_small_groups_are_blanked(self):
        few = [stay(f"s{i}", [100.0] * 8, resp=[25.0] * 8) for i in range(MIN_CELL - 1)]
        hours = hourly_table(few)
        self.assertTrue(by_hour_of_stay(hours)["alert_rate"].isna().all())

    def test_large_enough_groups_show_rates(self):
        many = [stay(f"s{i}", [100.0] * 8, resp=[25.0] * 8) for i in range(MIN_CELL)]
        table = by_hour_of_stay(hourly_table(many)).set_index("icu_hour")
        self.assertEqual(table.loc["1-6", "stays"], MIN_CELL)
        self.assertEqual(table.loc["1-6", "alert_rate"], 1.0)

    def test_warned_stays_use_the_12_hours_before_onset(self):
        # First positive row is hour 15, so onset is hour 21 and the window is hours 9-20.
        def septic(name, hr):
            return stay(name, hr, resp=[25.0] * 24, labels=[0] * 14 + [1] * 10)
        warned = [septic(f"w{i}", [100.0] * 24) for i in range(MIN_CELL)]
        silent = [septic(f"n{i}", [70.0] * 24) for i in range(MIN_CELL)]
        table = missed_stays(warned + silent).set_index("outcome")
        self.assertEqual(table.loc["warned", "stays"], MIN_CELL)
        self.assertEqual(table.loc["not warned", "stays"], MIN_CELL)
        self.assertEqual(table.loc["not warned", "median_peak_criteria_in_window"], 1)
        self.assertEqual(table.loc["warned", "median_hours_before_onset"], 20)

    def test_no_stay_identifier_appears_in_any_output(self):
        patients = [stay(f"secret{i}", [100.0] * 24, resp=[25.0] * 24, labels=[0] * 14 + [1] * 10) for i in range(MIN_CELL)]
        for name, table in failure_tables(patients).items():
            self.assertNotIn("secret", table.to_csv(), name)
            self.assertFalse({"stay", "patient_id"} & set(table.columns), name)


if __name__ == "__main__":
    unittest.main()
