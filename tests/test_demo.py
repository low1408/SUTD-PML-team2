"""The predict demo: freeze rule, replay arguments and the workflow. No AWS."""

import datetime
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_demo as demo  # noqa: E402


class DemoTests(unittest.TestCase):
    def test_held_out_splits_need_a_recorded_frozen_evaluation(self):
        for split in ("internal_test", "external_test"):
            for registry in (None, {}, {"frozen_evaluated_at": None}):
                with self.assertRaisesRegex(ValueError, "no recorded frozen evaluation"):
                    demo.require_frozen(split, registry)
            demo.require_frozen(split, {"frozen_evaluated_at": "2026-11-25T01:00:00+00:00"})
        demo.require_frozen("validation", None)          # validation was used for selection: always allowed
        demo.require_frozen("train", {})

    def test_the_replay_command_scores_through_the_signed_endpoint_and_marks_the_freeze_for_held_out_splits(self):
        base = dict(api_url="https://x.lambda-url.r.on.aws", data=Path("d.parquet"), splits=Path("s.csv"), beds=12,
                    seed=42, output=Path("w.html"))
        validation = demo.ward_argv(split="validation", **base)
        self.assertIn("--sign", validation)
        self.assertEqual(validation[validation.index("--api-url") + 1], "https://x.lambda-url.r.on.aws")
        self.assertNotIn("--after-freeze", validation)
        self.assertIn("--after-freeze", demo.ward_argv(split="external_test", **base))

    def test_the_page_key_is_under_the_private_demo_prefix(self):
        key = demo.report_key(datetime.datetime(2026, 11, 25, 3, 4, 5, tzinfo=datetime.UTC), "validation")
        self.assertEqual(key, "reports/demo/20261125T030405Z/ward-validation.html")

    def test_a_missing_registry_file_is_an_answer_not_a_crash(self):
        class S3:
            def get_object(self, Bucket, Key):
                error = Exception("NoSuchKey")
                error.response = {"Error": {"Code": "NoSuchKey"}}
                raise error

        self.assertIsNone(demo.read_json(S3(), "b", "models/v1/registry.json"))

    def test_the_workflow_is_manual_keeps_inputs_out_of_scripts_and_never_uploads_the_page_as_an_artifact(self):
        text = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["split"]["options"], ["validation", "internal_test", "external_test"])
        self.assertNotIn("upload-artifact", text)
        for step in workflow["jobs"]["demo"]["steps"]:
            self.assertNotIn("${{ inputs.", step.get("run", ""))
        self.assertIn("demo) cmd_demo", (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8"))


class TrafficTests(unittest.TestCase):
    """Many stays through /score for the monitor: the ward page stops at 24 beds and always shows the same stays."""

    @staticmethod
    def cohort(n=40, hours=100):
        import pandas as pd

        from sepsis_warning.data import DYNAMIC, STATIC, Patient
        columns = DYNAMIC + STATIC + ["SepsisLabel"]
        return [Patient(f"p{i:03d}", "A", pd.DataFrame({c: [float(i)] * hours for c in columns})) for i in range(n)]

    def test_the_sample_has_the_requested_size_and_no_stay_twice(self):
        chosen = demo.pick_traffic(self.cohort(40), 25, 72, seed=1)
        self.assertEqual(len(chosen), 25)
        self.assertEqual(len({p.patient_id for p in chosen}), 25)

    def test_a_small_split_gives_all_its_stays_and_hours_are_cut(self):
        chosen = demo.pick_traffic(self.cohort(10), 300, 72, seed=1)
        self.assertEqual(len(chosen), 10)
        self.assertTrue(all(len(p.frame) == 72 for p in chosen))

    def test_the_seed_decides_which_stays_and_the_same_seed_gives_the_same_ones(self):
        cohort = self.cohort(60)
        ids = lambda seed: [p.patient_id for p in demo.pick_traffic(cohort, 20, 72, seed)]  # noqa: E731
        self.assertEqual(ids(1), ids(1))
        self.assertNotEqual(ids(1), ids(2))

    def test_nothing_to_send_is_refused(self):
        with self.assertRaises(ValueError):
            demo.pick_traffic(self.cohort(5), 0, 72, 1)

    def test_every_stay_is_one_request_and_the_counts_add_up(self):
        calls = []

        def post(patient_id, hours):
            calls.append((patient_id, len(hours)))
            return {"hours": [{"alert_event": i == 0} for i in range(3)]}

        result = demo.send_traffic(demo.pick_traffic(self.cohort(12), 12, 72, 1), post, workers=3)
        self.assertEqual((result["sent"], result["ok"], result["failed"]), (12, 12, 0))
        self.assertEqual((result["hours"], result["alerts"]), (36, 12))
        self.assertEqual(len(calls), 12)
        self.assertEqual(len({pid for pid, _ in calls}), 12)
        self.assertTrue(all(pid.startswith("traffic-") for pid, _ in calls))        # no record id leaves the machine
        self.assertTrue(all(n == 72 for _, n in calls))

    def test_a_failed_request_is_counted_and_does_not_stop_the_others(self):
        def post(patient_id, hours):
            if patient_id.endswith("0003"):
                raise RuntimeError("HTTP 502")
            return {"hours": []}

        result = demo.send_traffic(demo.pick_traffic(self.cohort(8), 8, 72, 1), post, workers=2)
        self.assertEqual((result["ok"], result["failed"]), (7, 1))
        self.assertIn("HTTP 502", result["first_error"])

    def test_the_same_freeze_and_real_model_rules_apply_as_for_the_ward(self):
        import inspect
        source = inspect.getsource(demo.traffic)
        self.assertIn("require_frozen(split, registry)", source)
        self.assertIn('health.get("synthetic_model")', source)

    def test_the_workflow_offers_traffic_and_passes_the_numbers_through_the_environment(self):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8"))
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["mode"]["options"], ["ward", "traffic"])
        self.assertEqual(inputs["mode"]["default"], "ward")
        self.assertEqual(inputs["stays"]["default"], "300")
        text = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
        for name in ("MODE", "STAYS", "SEED"):
            self.assertIn(f"{name}: ${{{{ inputs.{name.lower()} }}}}", text)
        script = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn('--mode "${MODE:-ward}"', script)


if __name__ == "__main__":
    unittest.main()
