"""Evaluation script: table content, figures with recreate commands, no test data touched."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from sepsis_warning import cli, model_report


def synthetic_run(folder: Path, stays: int = 40, hours: int = 20) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for stay in range(stays):
        septic = stay % 4 == 0
        for hour in range(hours):
            label = int(septic and hour >= 12)
            base = 0.6 if label else 0.03
            rows.append({"patient_id": f"s{stay}", "hospital": "A", "hour": hour, "label": label,
                         "p_good": min(1, max(0, base + rng.normal(0, 0.05))),
                         "p_weak": float(rng.uniform(0, 0.3)), "sirs_score": float(label * 2 + rng.integers(0, 2))})
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "validation_scores.csv", index=False)
    pd.DataFrame([
        {"model": "good", "threshold": 0.3, "brier_score": 0.01, "normalized_hourly_utility": 0.8,
         "false_alert_events_per_nonseptic_patient_day": 0.1, "early_detection_recall": 0.9},
        {"model": "weak", "threshold": 0.25, "brier_score": 0.05, "normalized_hourly_utility": 0.1,
         "false_alert_events_per_nonseptic_patient_day": 0.2, "early_detection_recall": 0.1},
    ]).to_csv(folder / "validation_models.csv", index=False)
    (folder / "selection_metrics.json").write_text(json.dumps({
        "selected_model": "good",
        "reference_baselines": {"sirs": {"threshold": 2.0, "normalized_hourly_utility": 0.3,
                                         "false_alert_events_per_nonseptic_patient_day": 0.05,
                                         "early_detection_recall": 0.5}}}))
    return folder


class ModelReportTests(unittest.TestCase):
    def test_table_and_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            out = Path(temp) / "out"
            manifest = model_report.build_report(run, out, bootstrap=30)
            table = pd.read_csv(out / "metrics_table.csv")
            self.assertEqual(set(table["model"]), {"good", "weak", "sirs_rule"})
            row = table.set_index("model").loc["good"]
            self.assertGreater(row["recall"], 0.9)
            self.assertLessEqual(row["f1_ci_low"], row["f1"])
            self.assertGreaterEqual(row["f1_ci_high"], row["f1"])
            self.assertEqual(table.set_index("model").loc["good", "role"], "selected")
            for figure in manifest["figures"]:
                self.assertTrue((out / figure["file"]).stat().st_size > 1000)
                self.assertIn("python -m sepsis_warning model-report", figure["recreate"])
            self.assertIn("F1", (out / "FIGURES.md").read_text())
            self.assertTrue(manifest["skipped"])  # no loss history in this run
            self.assertTrue((out / "environment.json").is_file())

    def test_counts_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            run = model_report._read_run(synthetic_run(Path(temp) / "run"))
            table = model_report.metrics_table(run, draws=0).set_index("model")
            scores = run["scores"]
            predicted = scores["p_good"] >= 0.3
            self.assertEqual(int(table.loc["good", "tp"]), int((predicted & (scores["label"] == 1)).sum()))
            self.assertEqual(int(table.loc["good", "fp"]), int((predicted & (scores["label"] == 0)).sum()))
            self.assertTrue(pd.isna(table.loc["good", "f1_ci_low"]))

    def test_refuses_missing_inputs_and_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty"
            empty.mkdir()
            with self.assertRaises(model_report.ReportInputError):
                model_report.build_report(empty, Path(temp) / "o")
            run = synthetic_run(Path(temp) / "run")
            busy = Path(temp) / "busy"
            busy.mkdir()
            (busy / "x").write_text("x")
            with self.assertRaises(ValueError):
                model_report.build_report(run, busy)

    def test_cli_command(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            out = Path(temp) / "out"
            argv = ["sepsis_warning", "model-report", "--run", str(run), "--output", str(out), "--bootstrap", "10"]
            with mock.patch("sys.argv", argv):
                cli.main()
            self.assertTrue((out / "fig_pr_curves.png").is_file())

    def test_evaluation_folder_needed_for_test_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            run = synthetic_run(Path(temp) / "run")
            with self.assertRaises(model_report.ReportInputError):
                model_report.build_report(run, Path(temp) / "o", evaluation_dir=Path(temp) / "nope", bootstrap=5)


if __name__ == "__main__":
    unittest.main()
