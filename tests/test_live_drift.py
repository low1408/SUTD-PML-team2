"""Phase 4: live drift checks on the prediction log, and the scheduled drift job."""

import gzip
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from sepsis_warning.cli import run_experiment
from sepsis_warning.data import (
    DYNAMIC,
    STATIC,
    generate_synthetic,
    load_patients,
    split_patients,
)
from sepsis_warning.live_drift import (
    emf_record,
    live_checks,
    read_prediction_log,
    run_live_drift,
)
from sepsis_warning.monitoring import REFERENCE_FILE, MonitoringConfig, setup_monitoring
from sepsis_warning.service import ScoringService
from sepsis_warning.telemetry import Telemetry


def records(frame):
    inputs = frame[DYNAMIC + STATIC]
    return inputs.astype(object).where(inputs.notna(), None).to_dict(orient="records")


class LiveDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate_synthetic(cls.data, source_size=200, external_size=20)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        run_experiment(Namespace(
            data=cls.data, splits=None, output=cls.root / "run", seed=42, window=6, budget=0.2, cooldown=6,
            splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
        ))
        cls.model = cls.root / "run" / "model.joblib"
        config = MonitoringConfig(
            fast_window_hours=400, slow_window_hours=1600, min_observed_values=40, min_observed_stays=5,
            min_slow_stays=20, min_slow_septic_stays=4, min_subgroup_septic_stays=2, null_windows=60,
        )
        setup_monitoring(cls.data, cls.model, cls.root / "reference", config, 42)
        cls.reference_path = cls.root / "reference" / REFERENCE_FILE
        cls.reference = json.loads(cls.reference_path.read_text())
        cls.stays = split_patients(load_patients(cls.data), 42)["validation"]
        cls.normal = cls.stream(cls.root / "normal.jsonl")
        cls.shifted = cls.stream(cls.root / "shifted.jsonl", drop=("HR", "Resp", "Temp"), max_hours=20)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def stream(cls, path, drop=(), max_hours=None):
        """Play the validation stays hour by hour through the service, as a ward would."""
        service = ScoringService.from_path(cls.model)
        telemetry = Telemetry(stream=io.StringIO(), prediction_log=path)
        service.on_prediction = telemetry.prediction
        for index, stay in enumerate(cls.stays):
            hours = records(stay.frame.iloc[:max_hours] if max_hours else stay.frame)
            for record in hours:
                for name in drop:
                    record[name] = None
            for now in range(1, len(hours) + 1):
                service.score(f"bed-{index}", hours[:now])
        return path

    def test_the_log_has_one_record_per_stay_hour_and_retries_count_once(self):
        frame = read_prediction_log([self.normal])
        self.assertEqual(len(frame), sum(len(s.frame) for s in self.stays))
        doubled = self.root / "doubled.jsonl"
        doubled.write_text(self.normal.read_text() * 2)
        self.assertEqual(len(read_prediction_log([doubled])), len(frame))
        text = self.normal.read_text()
        self.assertNotIn("bed-0", text)
        self.assertNotIn('"HR":', text)

    def test_an_unchanged_stream_is_ok(self):
        result = run_live_drift([self.normal], self.reference_path, self.root / "out")
        # Whole validation stays, hour by hour: exactly what the reference was frozen from.
        self.assertEqual(result["status"], "ok", result)
        self.assertLess(result["checks"]["score_psi"]["value"], 0.01)
        self.assertAlmostEqual(result["checks"]["alert_rate_ratio"]["value"], 1.0, places=6)
        self.assertTrue((self.root / "out" / "live_drift.json").is_file())

    def test_a_broken_feed_raises_the_missingness_alarm(self):
        result = live_checks(read_prediction_log([self.shifted]), self.reference)
        self.assertEqual(result["status"], "alarm")
        self.assertEqual(result["checks"]["max_missingness_shift"]["status"], "alarm")
        self.assertEqual({item["input"] for item in result["largest_missingness_shifts"][:3]}, {"HR", "Resp", "Temp"})
        self.assertEqual(emf_record(result, "test")["DriftLevel"], 2)

    def test_too_little_data_or_another_bundle_is_not_judged(self):
        frame = read_prediction_log([self.normal])
        self.assertEqual(live_checks(frame.head(50), self.reference)["status"], "insufficient_data")
        other = dict(self.reference, model_sha256="f" * 64)
        self.assertEqual(live_checks(frame, other)["status"], "reference_mismatch")
        self.assertEqual(live_checks(frame.head(0), self.reference)["status"], "insufficient_data")
        self.assertEqual(emf_record(live_checks(frame.head(50), self.reference), "t")["DriftLevel"], -1)

    def test_cloudwatch_firehose_batches_are_read(self):
        lines = self.normal.read_text().splitlines()[:5]
        batch = json.dumps({"messageType": "DATA_MESSAGE", "logEvents": [{"message": line} for line in lines]
                            + [{"message": "not json"}, {"message": json.dumps({"record_type": "request"})}]})
        frame = read_prediction_log([gzip.compress((batch + batch).encode())])   # two batches back to back
        self.assertEqual(len(frame), 5)

    def test_scheduled_job_reads_the_day_and_stores_the_result(self):
        from sepsis_warning.drift_handler import handler

        objects = {"prediction-log/2026/11/14/09/part-1": gzip.compress(self.normal.read_bytes()),
                   "prediction-log/2026/11/13/09/other-day": self.shifted.read_bytes(),
                   "monitoring/reference.json": self.reference_path.read_bytes()}
        stored = {}

        class FakeS3:
            def get_object(self, Bucket, Key):
                return {"Body": io.BytesIO(objects[Key])}

            def list_objects_v2(self, Bucket, Prefix, **_):
                return {"Contents": [{"Key": key} for key in objects if key.startswith(Prefix)]}

            def put_object(self, Bucket, Key, Body):
                stored[Key] = json.loads(Body)

        env = {"PREDICTION_LOG_S3": "s3://bucket/prediction-log/", "REFERENCE_S3": "s3://bucket/monitoring/reference.json"}
        with mock.patch.dict(os.environ, env), mock.patch("builtins.print") as printed:
            outcome = handler({"date": "2026-11-14"}, None, client=FakeS3())
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(stored["drift-results/2026-11-14.json"]["rows"], outcome["rows"])
        metric = json.loads(printed.call_args.args[0])
        self.assertEqual(metric["DriftLevel"], 0)
        self.assertEqual(metric["_aws"]["CloudWatchMetrics"][0]["Namespace"], "SepsisWarning/Drift")


if __name__ == "__main__":
    unittest.main()
