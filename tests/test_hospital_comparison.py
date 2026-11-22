"""Hospital A vs B comparison: aggregates are correct and SIRS stays per-patient."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.baselines import sirs_count
from sepsis_warning.data import DYNAMIC, STATIC
from sepsis_warning.hospital_comparison import (
    add_sirs,
    missingness,
    restrict_a_to_train,
    run_comparison,
    stay_table,
    summary_table,
)


def stay(hospital, pid, hours, septic_from=None, age=60, gender=1, **values):
    frame = pd.DataFrame(np.nan, index=range(hours), columns=DYNAMIC + STATIC + ["Unit1", "Unit2"])
    frame["ICULOS"] = np.arange(1, hours + 1)
    frame["Age"], frame["Gender"], frame["HospAdmTime"] = age, gender, -1
    frame["SepsisLabel"] = 0
    if septic_from is not None:
        frame.loc[septic_from:, "SepsisLabel"] = 1
    for column, series in values.items():
        frame[column] = series
    frame["patient_id"], frame["hospital"] = pid, hospital
    return frame


def fixture():
    return pd.concat([
        # Ends with abnormal HR and Temp ...
        stay("A", "p1", 3, HR=[80, 120, 120], Temp=[37, 39, 39]),
        # ... which must NOT carry into the next stay, which never measures them.
        stay("A", "p2", 4, septic_from=2, age=85, gender=0),
        stay("B", "p3", 2, septic_from=0, HR=[100, np.nan], Resp=[25, np.nan]),
        stay("B", "p4", 5, age=30, gender=0),
    ], ignore_index=True)


class HospitalComparison(unittest.TestCase):
    def test_sirs_does_not_leak_between_stays(self):
        hours = add_sirs(fixture())
        np.testing.assert_array_equal(hours.loc[hours.patient_id == "p2", "sirs_count"], [0, 0, 0, 0])
        np.testing.assert_array_equal(hours.loc[hours.patient_id == "p4", "sirs_count"], [0] * 5)
        # Grouped version equals the single-patient reference implementation.
        for _, frame in fixture().groupby("patient_id"):
            pid = frame["patient_id"].iloc[0]
            np.testing.assert_array_equal(
                hours.loc[hours.patient_id == pid, "sirs_count"], sirs_count(frame.reset_index(drop=True))
            )

    def test_stay_table_and_summary(self):
        hours = add_sirs(fixture())
        stays = stay_table(hours).set_index("patient_id")
        self.assertEqual(stays.loc["p2", "hours"], 4)
        self.assertEqual(stays.loc["p2", "first_positive_iculos"], 3)
        self.assertEqual(stays.loc["p2", "age_group"], "80+")
        self.assertEqual(stays.loc["p4", "age_group"], "<45")
        summary = summary_table(hours, stays.reset_index())
        self.assertEqual(summary.loc["ICU stays", "A"], 2)
        self.assertAlmostEqual(summary.loc["Septic stays (%)", "B"], 50)
        self.assertAlmostEqual(summary.loc["Positive-label hours (%)", "A"], 100 * 2 / 7)
        # p3 is positive from its first row (left-censored); p2 is not.
        self.assertAlmostEqual(summary.loc["Septic stays positive from first hour (%)", "B"], 100)
        self.assertAlmostEqual(summary.loc["Septic stays positive from first hour (%)", "A"], 0)
        self.assertAlmostEqual(summary.loc["Male (%)", "A"], 50)

    def test_missingness_fraction(self):
        missing = missingness(fixture())
        self.assertAlmostEqual(missing.loc["HR", "A"], 4 / 7)
        self.assertAlmostEqual(missing.loc["HR", "B"], 6 / 7)
        self.assertEqual(missing.loc["EtCO2", "A"], 1.0)

    def test_exploration_uses_only_a_train_plus_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = Path(tmp) / "splits.csv"
            pd.DataFrame({
                "patient_id": ["p1", "p2", "p3", "p4"], "hospital": ["A", "A", "B", "B"],
                "septic": [0, 1, 1, 0], "split": ["train", "internal_test", "external_test", "external_test"],
            }).to_csv(splits, index=False)
            kept = restrict_a_to_train(fixture(), splits)
            self.assertEqual(sorted(kept["patient_id"].unique()), ["p1", "p3", "p4"])  # p2 (A test) dropped

    def test_end_to_end_writes_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combined.parquet"
            fixture().to_parquet(path)
            run_comparison(path, Path(tmp) / "figures")
            written = sorted(p.name for p in (Path(tmp) / "figures").iterdir())
            self.assertEqual(sum(name.endswith(".png") for name in written), 7)
            self.assertIn("README.md", written)
            self.assertIn("hospital_summary.csv", written)


if __name__ == "__main__":
    unittest.main()
