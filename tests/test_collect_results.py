"""collect_results copies only the aggregate results of a finished run, scrubbed, and records where they came from."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("collect_results", ROOT / "scripts" / "collect_results.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

EXEC = "a37825bf-462e-4c29-a23f-256987b37802"
BUCKET = "sepsis-dev-artifacts-nyz0wipac1sq-artifactbucket-kekhzo7000uf"
ACCOUNT = "849996547916"
REGISTRY = {"approved_by": f"arn:aws:sts::{ACCOUNT}:assumed-role/ApproverTaskRole/x", "reviewed_by": "leothelucifer",
            "frozen_evaluated_at": "2026-11-25T08:03:09+00:00", "synthetic": False}


class FakeS3:
    def __init__(self, store):
        self.store = dict(store)

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, ContinuationToken=None):
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        if Delimiter:
            prefixes, contents = set(), []
            for key in keys:
                rest = key[len(Prefix):]
                if Delimiter in rest:
                    prefixes.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
                else:
                    contents.append({"Key": key})
            return {"Contents": contents, "CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)], "IsTruncated": False}
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.store[Key])}

    def head_object(self, Bucket, Key):
        return {"Metadata": {"sha256": "a" * 64}}


class Client:
    def __init__(self, **methods):
        self.__dict__.update(methods)


def session(store, executions=None):
    s3 = FakeS3(store)
    outputs = [{"OutputKey": "ArtifactBucketName", "OutputValue": BUCKET}, {"OutputKey": "PipelineArn", "OutputValue": "arn:machine"}]
    clients = {
        "s3": s3,
        "cloudformation": Client(describe_stacks=lambda StackName: {"Stacks": [{"Outputs": outputs}]}),
        "stepfunctions": Client(list_executions=lambda stateMachineArn, statusFilter, maxResults: {
            "executions": [{"name": EXEC}] if executions is None else executions}),
    }
    return Client(client=lambda name: clients[name])


def full_store():
    text = lambda s: s.encode()  # noqa: E731
    return {
        f"reports/{EXEC}/metrics_table.md": text(f"| model |\n bucket {BUCKET} account {ACCOUNT}\n"),
        f"reports/{EXEC}/metrics_table.csv": text("model,ap\nrandom_forest,0.12\n"),
        f"reports/{EXEC}/fig_pr_curves.png": b"\x89PNG....",
        f"reports/{EXEC}/fig_failure_examples.png": b"\x89PNG-individual-stays",
        f"reports/{EXEC}/replay.html": text("<html>patient trajectories</html>"),
        f"training-output/{EXEC}/training/model/selection_metrics.json": text('{"selected_model": "random_forest"}'),
        f"training-output/{EXEC}/training/model/validation_models.csv": text("model,ap\n"),
        f"training-output/{EXEC}/training/model/model.joblib": b"EXECUTABLE",
        f"training-output/{EXEC}/training/model/held_out_predictions.csv": text("patient,score\n"),
        f"training-output/{EXEC}/tuning/model/tuning_results.csv": text("model,trial\n"),
        f"training-output/{EXEC}/tuning/model/tuned_models.toml": text("[random_forest]\n"),
        f"training-output/{EXEC}/publish/model/evaluation_metrics.json": text('{"splits": {}}'),
        f"training-output/{EXEC}/prepared/combined.parquet": b"PAR1",
        "models/v1/model.joblib": b"MODEL",
        "models/v1/registry.json": json.dumps(REGISTRY).encode(),
        "reports/monitoring/20261125T101500Z/metrics.json": text('{"drifted_share": 0.1}'),
        "reports/monitoring/20261125T101500Z/report.html": text("<html>evidently</html>"),
        "reports/demo/20261125T100902Z/ward-external_test.html": text("<html>ward</html>"),
    }


def run(store=None, **kwargs):
    with tempfile.TemporaryDirectory() as temp:
        out = []
        result = cr.collect(session(full_store() if store is None else store), "sepsis-dev", root=Path(temp), out=out.append, **kwargs)
        written = sorted(str(p.relative_to(temp)) for p in Path(temp).rglob("*") if p.is_file())
        texts = {str(p.relative_to(temp)): p.read_text(errors="replace") for p in Path(temp).rglob("*") if p.is_file() and p.suffix != ".png"}
        return result, written, texts, out


class ClassifyTests(unittest.TestCase):
    def test_aggregate_files_go_to_the_folders_the_report_uses(self):
        cases = {
            f"reports/{EXEC}/metrics_table.md": "reports/results/metrics_table.md",
            f"reports/{EXEC}/fig_calibration.png": "reports/results/fig_calibration.png",
            f"training-output/{EXEC}/training/model/selection_metrics.json": "reports/results/run/selection_metrics.json",
            f"training-output/{EXEC}/tuning/model/tuning_results.csv": "reports/results/tuning/tuning_results.csv",
            f"training-output/{EXEC}/publish/model/evaluation_metrics.json": "reports/results_test/evaluation_metrics.json",
        }
        for key, destination in cases.items():
            self.assertEqual(cr.classify(key, EXEC)[0], destination, key)

    def test_patient_level_executable_and_replay_files_are_never_copied(self):
        for key in (f"training-output/{EXEC}/training/model/model.joblib", f"training-output/{EXEC}/x/held_out_predictions.csv",
                    f"training-output/{EXEC}/x/split_manifest.csv", f"reports/{EXEC}/replay.html", f"training-output/{EXEC}/prepared/combined.parquet",
                    "reports/demo/20261125T100902Z/ward-external_test.html", "reports/monitoring/20261125T101500Z/report.html"):
            self.assertIsNone(cr.classify(key, EXEC)[0], key)
            self.assertIsNone(cr.classify(key, EXEC, include_failure=True)[0], key)

    def test_the_failure_example_figure_needs_an_opt_in(self):
        key = f"reports/{EXEC}/fig_failure_examples.png"
        self.assertIsNone(cr.classify(key, EXEC)[0])
        self.assertEqual(cr.classify(key, EXEC, include_failure=True)[0], "reports/results/fig_failure_examples.png")

    def test_another_run_and_unlisted_files_are_not_taken(self):
        self.assertIsNone(cr.classify("reports/other-run-123456/metrics_table.md", EXEC)[0])
        self.assertIsNone(cr.classify(f"reports/{EXEC}/notes.txt", EXEC)[0])
        self.assertIsNone(cr.classify(f"training-output/{EXEC}/training/model/anything_else.csv", EXEC)[0])

    def test_two_keys_for_one_destination_keep_the_first_and_say_so(self):
        keys = [f"training-output/{EXEC}/training/model/config.json", f"training-output/{EXEC}/training/output/config.json"]
        chosen, skipped = cr.plan(keys, EXEC)
        self.assertEqual(list(chosen.values()), [keys[0]])
        self.assertIn("duplicate", skipped[0][1])


class ScrubTests(unittest.TestCase):
    def test_bucket_and_account_ids_become_placeholders(self):
        text = cr.scrub(f"s3://{BUCKET}/x arn:aws:iam::{ACCOUNT}:role/r id 1234567890123", BUCKET)
        self.assertNotIn(BUCKET, text)
        self.assertNotIn(ACCOUNT, text)
        self.assertIn("<artifact-bucket>", text)
        self.assertIn("<aws-account>", text)
        self.assertIn("1234567890123", text)   # thirteen digits are not an account id


class CollectTests(unittest.TestCase):
    def test_it_writes_the_allowed_files_scrubbed_and_nothing_else(self):
        result, written, texts, out = run()
        self.assertEqual(result["execution"], EXEC)
        self.assertEqual(result["version"], 1)
        self.assertEqual(written, sorted([
            "reports/results/PROVENANCE.md", "reports/results/fig_pr_curves.png", "reports/results/metrics_table.csv",
            "reports/results/metrics_table.md", "reports/results/monitoring/metrics.json", "reports/results/registry.json",
            "reports/results/run/selection_metrics.json", "reports/results/run/validation_models.csv",
            "reports/results/tuning/tuned_models.toml", "reports/results/tuning/tuning_results.csv",
            "reports/results_test/evaluation_metrics.json"]))
        everything = "\n".join(texts.values())
        self.assertNotIn(BUCKET, everything)
        self.assertNotIn(ACCOUNT, everything)
        self.assertNotIn("EXECUTABLE", everything)

    def test_provenance_names_the_run_the_approval_and_every_file(self):
        _, _, texts, _ = run(source={"commit": "abc123", "run_url": "https://github.com/o/r/actions/runs/1"})
        page = texts["reports/results/PROVENANCE.md"]
        for needle in (EXEC, "leothelucifer", "2026-11-25T08:03:09", "abc123", "actions/runs/1", "metrics_table.md", "Skipped on purpose",
                       "model.joblib", "held_out_predictions.csv"):
            self.assertIn(needle, page)
        self.assertNotIn(ACCOUNT, page)

    def test_a_dry_run_lists_files_and_writes_nothing(self):
        _, written, _, out = run(dry_run=True)
        self.assertEqual(written, [])
        self.assertTrue(any("metrics_table.md" in line for line in out))

    def test_the_failure_figure_is_copied_only_when_asked(self):
        self.assertNotIn("reports/results/fig_failure_examples.png", run()[1])
        self.assertIn("reports/results/fig_failure_examples.png", run(include_failure=True)[1])

    def test_monitoring_can_be_left_out_and_is_noted_when_missing(self):
        self.assertNotIn("reports/results/monitoring/metrics.json", run(include_monitoring=False)[1])
        store = {k: v for k, v in full_store().items() if "monitoring" not in k}
        _, written, texts, _ = run(store)
        self.assertNotIn("reports/results/monitoring/metrics.json", written)
        self.assertIn("no drift report yet", texts["reports/results/PROVENANCE.md"])

    def test_a_run_without_the_frozen_evaluation_warns(self):
        store = {k: v for k, v in full_store().items() if "evaluation_metrics" not in k}
        _, written, _, out = run(store)
        self.assertNotIn("reports/results_test/evaluation_metrics.json", written)
        self.assertTrue(any("frozen evaluation has not finished" in line for line in out))

    def test_the_latest_succeeded_run_is_used_and_none_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "SUCCEEDED"):
                cr.collect(session(full_store(), executions=[]), "sepsis-dev", root=Path(temp), out=lambda *_: None)

    def test_a_bad_run_id_is_refused(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "run id"):
            cr.collect(session(full_store()), "sepsis-dev", root=Path(temp), execution="../../etc", out=lambda *_: None)

    def test_an_oversized_file_is_skipped_not_copied(self):
        store = full_store()
        store[f"reports/{EXEC}/fig_big.png"] = b"x" * (cr.MAX_FILE_BYTES + 1)
        _, written, texts, _ = run(store)
        self.assertNotIn("reports/results/fig_big.png", written)
        self.assertIn("larger than", texts["reports/results/PROVENANCE.md"])

    def test_the_model_version_can_be_chosen_and_must_be_a_number(self):
        self.assertEqual(run(model_version=1)[0]["version"], 1)
        self.assertEqual(cr.main(["--stack", "s", "--region", "r", "--model-version", "v1"]), 1)


if __name__ == "__main__":
    unittest.main()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "collect-results.yml").read_text(encoding="utf-8"))
        self.steps = self.workflow["jobs"]["collect"]["steps"]
        self.text = (ROOT / ".github" / "workflows" / "collect-results.yml").read_text(encoding="utf-8")

    def test_it_is_started_by_hand_only_and_can_do_a_dry_run(self):
        self.assertEqual(list(self.workflow[True]), ["workflow_dispatch"])
        inputs = self.workflow[True]["workflow_dispatch"]["inputs"]
        self.assertIs(inputs["dry_run"]["default"], False)
        self.assertIs(inputs["include_failure_examples"]["default"], False)

    def test_it_can_push_and_open_a_pull_request_but_keeps_the_keyless_aws_login(self):
        permissions = self.workflow["permissions"]
        self.assertEqual((permissions["contents"], permissions["pull-requests"], permissions["id-token"]), ("write", "write", "write"))
        self.assertNotIn("secrets.AWS", self.text)

    def test_no_input_is_pasted_into_a_shell_script(self):
        for step in self.steps:
            self.assertNotIn("${{", step.get("run", ""), step.get("name"))

    def test_only_the_two_results_folders_are_ever_staged_and_checked(self):
        self.assertIn("git add -A reports/results reports/results_test", self.text)
        self.assertNotIn("git add -A .", self.text)
        self.assertIn("psv|joblib|pkl|pickle|parquet", self.text)

    def test_nothing_is_committed_on_a_dry_run(self):
        by_name = {s["name"]: s for s in self.steps if "name" in s}
        for name in ("Check what changed before committing", "Commit, push and open the pull request"):
            self.assertEqual(by_name[name]["if"], "inputs.dry_run != true")

    def test_the_collection_runs_before_the_commit(self):
        names = [s.get("name", "") for s in self.steps]
        self.assertLess(names.index("Collect the aggregate results from S3"), names.index("Commit, push and open the pull request"))
