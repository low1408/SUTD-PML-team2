"""Evidently monitoring: the frames, the report job and the UI sync. The Evidently-dependent tests need the tools image."""

import datetime
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning import evidently_frames as frames
from sepsis_warning import evidently_job as job

HAVE_EVIDENTLY = importlib.util.find_spec("evidently") is not None
HAVE_PARQUET = importlib.util.find_spec("pyarrow") is not None


def records(rng, count, shift=0.0, bundle="a" * 64, missing_rate=0.2, tag=""):
    rows = []
    for i in range(count):
        probability = float(np.clip(rng.beta(2, 20) + shift, 0, 1))
        rows.append({
            "record_type": "prediction", "prediction_id": f"p{tag}{shift}-{i}", "stay": f"s{i}", "hour": 5.0,
            "probability": probability, "above_threshold": probability >= 0.1, "new_alert": bool(rng.random() < 0.03 + shift),
            "alert_source": "ledger", "input_warnings": 0,
            "missing_now": [name for name in ("Lactate", "Fibrinogen") if rng.random() < missing_rate],
            "model_name": "m", "model_version": "1", "bundle_sha256": bundle[:12],
        })
    return pd.DataFrame(rows)


class FakeS3:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, bucket, key, data: bytes):
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        base = self.root / Bucket
        keys = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        return {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)], "IsTruncated": False}

    def get_object(self, Bucket, Key):
        return {"Body": type("B", (), {"read": lambda self_: (self.root / Bucket / Key).read_bytes()})()}

    def download_file(self, bucket, key, filename):
        path = self.root / bucket / key
        if not path.is_file():
            error = Exception("404")
            error.response = {"Error": {"Code": "404"}}
            raise error
        shutil.copy(path, filename)

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.put(bucket, key, Path(filename).read_bytes())

    def keys(self, bucket, prefix=""):
        return [item["Key"] for item in self.list_objects_v2(bucket, prefix)["Contents"]]


class FramesTests(unittest.TestCase):
    def test_the_live_and_reference_frames_have_the_same_columns_and_no_identifiers(self):
        live = frames.prediction_frame(records(np.random.default_rng(0), 30))
        scored = pd.DataFrame({"probability": [0.1, 0.4], "hourly_positive": [0, 1], "alert_event": [0, 1],
                               **{name: [None, 1.0] for name in frames.DYNAMIC}})
        reference = frames.reference_frame(scored)
        self.assertEqual(list(live.columns), list(reference.columns))
        self.assertEqual(list(live.columns), frames.COLUMNS)
        for forbidden in ("stay", "patient_id", "hour", "prediction_id"):
            self.assertNotIn(forbidden, live.columns)
        self.assertEqual(reference["missing_HR"].tolist(), [1, 0])      # NaN now means missing

    def test_missing_now_names_become_flags(self):
        live = frames.prediction_frame(pd.DataFrame([{"probability": 0.2, "above_threshold": True, "new_alert": False,
                                                      "input_warnings": 2, "missing_now": ["Lactate"]}]))
        self.assertEqual((live["missing_Lactate"][0], live["missing_HR"][0], live["input_warnings"][0], live["above_threshold"][0]), (1, 0, 2.0, 1))
        self.assertTrue(frames.prediction_frame(pd.DataFrame()).empty)


class MetricsTests(unittest.TestCase):
    def summary(self, rows=500, share=0.4, score_drifted=True, ratio=2.0):
        return {"rows": {"reference": 1000, "current": rows}, "drifted_share": share,
                "score_drift": {"value": 0.01, "drifted": score_drifted}, "alert_rate": {"ratio": ratio}}

    def test_metrics_for_the_alarm_are_published_only_with_enough_rows(self):
        names = {m["MetricName"] for m in job.cloudwatch_metrics(self.summary(), "sepsis-dev")}
        self.assertEqual(names, {"Rows", "DriftedColumnsShare", "ScoreDriftValue", "ScoreDrifted", "AlertRateRatio"})
        thin = {m["MetricName"] for m in job.cloudwatch_metrics(self.summary(rows=50), "sepsis-dev")}
        self.assertEqual(thin, {"Rows"})                 # too little data is not "ok": no drift value at all
        self.assertTrue(all(m["Dimensions"] == [{"Name": "Environment", "Value": "sepsis-dev"}] for m in job.cloudwatch_metrics(self.summary(), "sepsis-dev")))

    def test_extract_reads_the_drift_verdicts(self):
        snapshot = {"metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 2.0, "share": 0.5}},
            {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "probability", "method": "K-S p_value", "threshold": 0.05}, "value": 0.001},
            {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "missing_HR", "method": "Jensen-Shannon distance", "threshold": 0.1}, "value": 0.05},
        ]}
        result = job.extract(snapshot)
        self.assertEqual((result["drifted_columns"], result["drifted_share"]), (2.0, 0.5))
        self.assertTrue(result["columns"]["probability"]["drifted"])
        self.assertFalse(result["columns"]["missing_HR"]["drifted"])


class LogWindowTests(unittest.TestCase):
    def test_the_log_window_reads_the_last_days_and_picks_the_serving_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            s3 = FakeS3(Path(temp))
            now = datetime.datetime(2026, 11, 26, 3, tzinfo=datetime.UTC)
            for day, bundle in ((26, "a" * 64), (25, "a" * 64), (20, "b" * 64)):
                lines = "\n".join(json.dumps({**row, "bundle_sha256": bundle[:12]}) for row in records(np.random.default_rng(day), 5, tag=f"day{day}").to_dict("records"))
                s3.put("bkt", f"prediction-log/2026/11/{day}/09/part-1", lines.encode())
            window = job.read_log_window(s3, "bkt", 2, now)
            self.assertEqual(len(window), 10)                      # the 20th is outside the two-day window
            self.assertEqual(job.serving_bundle(window), "a" * 12)
            self.assertIsNone(job.serving_bundle(pd.DataFrame()))

    def test_a_missing_reference_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "monitor_reference"):
                job.load_reference(FakeS3(Path(temp)), "bkt", "c" * 64)


@unittest.skipUnless(HAVE_EVIDENTLY and HAVE_PARQUET, "needs the tools image (evidently, pyarrow)")
class EvidentlyRunTests(unittest.TestCase):
    def test_drift_is_detected_and_the_report_and_workspace_are_written(self):
        rng = np.random.default_rng(1)
        reference = frames.prediction_frame(records(rng, 600))
        steady = frames.prediction_frame(records(rng, 400))
        shifted = frames.prediction_frame(records(rng, 400, shift=0.25, missing_rate=0.8))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            calm = job.compare(reference, steady, temp / "calm")
            drift = job.compare(reference, shifted, temp / "drift", temp / "workspace", run_name="test run")
            self.assertTrue((temp / "drift" / "report.html").stat().st_size > 1000)
            self.assertEqual(json.loads((temp / "drift" / "metrics.json").read_text())["rows"]["current"], 400)
            self.assertGreater(drift["drifted_share"], calm["drifted_share"])
            self.assertTrue(drift["score_drift"]["drifted"])
            self.assertFalse(calm["score_drift"]["drifted"])
            self.assertTrue(any((temp / "workspace").rglob("*")))             # the UI has something to show
            from evidently.ui.workspace import Workspace
            project = Workspace(str(temp / "workspace")).search_project(job.PROJECT_NAME)[0]
            self.assertGreaterEqual(sum(len(tab.panels) for tab in project.dashboard.model().tabs), 3)   # not an empty dashboard

    def test_the_whole_report_job_against_a_fake_bucket_publishes_metrics(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            s3, now = FakeS3(temp / "s3"), datetime.datetime(2026, 11, 26, 3, tzinfo=datetime.UTC)
            lines = "\n".join(json.dumps(row) for row in records(rng, 300, shift=0.2, missing_rate=0.7).to_dict("records"))
            s3.put("bkt", "prediction-log/2026/11/26/09/part-1", lines.encode())
            reference_path = temp / "reference.parquet"
            frames.prediction_frame(records(rng, 600)).to_parquet(reference_path)
            s3.put("bkt", f"monitoring/evidently/{'a' * 12}/reference.parquet", reference_path.read_bytes())
            sent = []
            cloudwatch = type("Cw", (), {"put_metric_data": lambda self, **kw: sent.append(kw)})()
            summary = job.report("bkt", 1, "sepsis-dev", now=now, s3=s3, cloudwatch=cloudwatch, out=lambda *_: None)
            self.assertEqual(summary["bundle"], "a" * 12)
            self.assertEqual(sent[0]["Namespace"], "SepsisWarning/Evidently")
            keys = [k for k in s3.keys("bkt", "reports/monitoring/")]
            self.assertTrue(any(k.endswith("report.html") for k in keys) and any(k.endswith("metrics.json") for k in keys))
            self.assertTrue(s3.keys("bkt", "monitoring/evidently/workspace/"))      # the UI reads this


class UiTests(unittest.TestCase):
    def test_the_ui_serves_the_workspace_and_refreshes_it_until_stopped(self):
        started = []

        class Process:
            def __init__(self, cmd):
                started.append(cmd)
                self.running = True

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.running = False

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as temp:
            s3 = FakeS3(Path(temp))
            s3.put("bkt", "monitoring/evidently/workspace/project/snapshot.json", b"{}")
            ticks = iter([False, False, True])
            code = job.ui("bkt", 8000, every=1, s3=s3, popen=Process, sleep=lambda _: None, stop=lambda: next(ticks))
        self.assertEqual(code, 0)
        self.assertEqual(started[0][:2], ["evidently", "ui"])
        self.assertIn("--host", started[0])
        self.assertEqual(started[0][started[0].index("--port") + 1], "8000")

    def test_the_ui_restarts_when_a_refresh_brings_a_new_project(self):
        started = []

        class Process:
            def __init__(self, cmd):
                started.append(cmd)
                self.running = True

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.running = False

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as temp:
            s3 = FakeS3(Path(temp))
            ticks = iter([False, False, False, True])
            added = []

            def sleep(_):
                if not added:
                    added.append(True)
                    s3.put("bkt", "monitoring/evidently/workspace/project/snapshot.json", b"{}")

            job.ui("bkt", 8000, every=1, s3=s3, popen=Process, sleep=sleep, stop=lambda: next(ticks))
        self.assertEqual(len(started), 2)


if __name__ == "__main__":
    unittest.main()
