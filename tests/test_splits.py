"""E04: the saved split is reproducible, complete, and independent of load order."""

import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.prepare import prepare_parquet
from sepsis_warning.cli import run_baselines, run_experiment
from sepsis_warning.splits import load_split_patients, load_splits, write_splits


def assignment(splits):
    return {p.patient_id: name for name, cohort in splits.items() for p in cohort}


class Splits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        generate_synthetic(root / "raw", source_size=120, external_size=30)
        cls.patients = load_patients(root / "raw")
        cls.parquet = root / "combined.parquet"
        prepare_parquet(root / "raw", cls.parquet)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_load_order_does_not_change_the_split(self):
        shuffled = self.patients[:]
        random.Random(1).shuffle(shuffled)
        self.assertEqual(assignment(split_patients(self.patients, 42)), assignment(split_patients(shuffled, 42)))

    def test_saved_split_matches_psv_split_and_is_reproducible(self):
        first, second = self.root / "s1.csv", self.root / "s2.csv"
        summary = write_splits(self.parquet, first, 42)
        write_splits(self.parquet, second, 42)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertNotIn(b"\r", first.read_bytes(), "line endings must not depend on the operating system")
        # Same assignment as splitting the .psv-loaded patients directly.
        self.assertEqual(load_splits(first), assignment(split_patients(self.patients, 42)))
        saved = json.loads((self.root / "s1.csv.summary.json").read_text())
        self.assertEqual(saved["seed"], 42)
        self.assertEqual(saved["splits_csv_sha256"], summary["splits_csv_sha256"])

    def test_every_stay_once_and_b_is_external(self):
        path = self.root / "s3.csv"
        write_splits(self.parquet, path, 42)
        table = pd.read_csv(path, dtype=str)
        self.assertEqual(len(table), len(self.patients))
        self.assertFalse(table.duplicated(["hospital", "patient_id"]).any())
        self.assertTrue((table.loc[table.hospital == "B", "split"] == "external_test").all())
        self.assertFalse((table.loc[table.hospital == "A", "split"] == "external_test").any())

    def test_loading_saved_split_gives_the_same_stays_and_rows(self):
        path = self.root / "s4.csv"
        write_splits(self.parquet, path, 42)
        loaded = load_split_patients(self.parquet, path)
        self.assertEqual(assignment(loaded), assignment(split_patients(self.patients, 42)))
        from_psv = {p.patient_id: p for p in self.patients}
        for cohort in loaded.values():
            for patient in cohort:
                self.assertEqual(patient.septic, from_psv[patient.patient_id].septic)
                pd.testing.assert_frame_equal(
                    patient.frame, from_psv[patient.patient_id].frame, check_dtype=False, check_exact=True,
                )

    def test_files_that_do_not_belong_together_are_refused(self):
        path = self.root / "s5.csv"
        write_splits(self.parquet, path, 42)
        table = pd.read_csv(path, dtype=str)
        short, extra, bad = self.root / "short.csv", self.root / "extra.csv", self.root / "bad.csv"
        table.iloc[1:].to_csv(short, index=False)
        ghost = pd.DataFrame([{"patient_id": "p999999", "hospital": "A", "septic": "0", "split": "train"}])
        pd.concat([table, ghost]).to_csv(extra, index=False)
        table.assign(split="holdout").to_csv(bad, index=False)
        for wrong, message in [(short, "not in"), (extra, "are not in"), (bad, "unknown split")]:
            with self.assertRaisesRegex(ValueError, message):
                load_split_patients(self.parquet, wrong)

    def test_baseline_gives_the_same_numbers_from_parquet_and_psv(self):
        write_splits(self.parquet, self.root / "splits.csv", 42)
        (self.root / "raw" / "SYNTHETIC.json").write_text("{}")
        common = {"seed": 42, "cooldown": 6, "splits": None}
        from_psv = run_baselines(Namespace(
            data=self.root / "raw", output=self.root / "b1", figures=self.root / "f1", **common))
        from_parquet = run_baselines(Namespace(
            data=self.parquet, output=self.root / "b2", figures=self.root / "f2", **common))
        pd.testing.assert_frame_equal(from_psv, from_parquet)

    def test_train_runs_from_parquet(self):
        write_splits(self.parquet, self.root / "splits.csv", 42)
        out_parquet = self.root / "train_parquet"
        run_experiment(Namespace(
            data=self.parquet, splits=None, output=out_parquet, seed=42, window=6,
            budget=0.2, cooldown=6, splits_summary=None, dvc_lock=None, mlflow_uri=None,
        ))
        self.assertTrue((out_parquet / "model.joblib").exists())
        self.assertTrue((out_parquet / "validation_models.csv").exists())
        self.assertTrue((out_parquet / "metrics.json").exists())
        self.assertTrue((out_parquet / "split_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
