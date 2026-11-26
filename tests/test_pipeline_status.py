"""The pipeline workflow starts a run and ends; status and stop answer the questions that come up during a long run."""

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from scripts import aws_pipeline_status as ps

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(UTC)


class FakeStepFunctions:
    def __init__(self, executions, events=()):
        self.executions, self.events, self.stopped = executions, list(events), []

    def list_executions(self, stateMachineArn, maxResults):
        return {"executions": self.executions[:maxResults]}

    def get_execution_history(self, executionArn, reverseOrder=False, maxResults=50):
        return {"events": self.events}

    def stop_execution(self, executionArn, error, cause):
        self.stopped.append((executionArn, error))


class FakeEcs:
    def __init__(self, tasks=()):
        self.tasks = list(tasks)

    def list_tasks(self, cluster, desiredStatus):
        return {"taskArns": [t["taskArn"] for t in self.tasks]}

    def describe_tasks(self, cluster, tasks):
        return {"tasks": self.tasks}


def execution(state="RUNNING", started=NOW - timedelta(hours=1, minutes=5), stopped=None):
    item = {"name": "run-1", "executionArn": "arn:run-1", "status": state, "startDate": started}
    if stopped:
        item["stopDate"] = stopped
    return item


def tune_task():
    return {"taskArn": "arn:task/1", "taskDefinitionArn": "arn:aws:ecs:r:1:task-definition/sepsis-dev-training:7",
            "cpu": "4096", "memory": "16384", "startedAt": NOW - timedelta(minutes=50)}


class StatusTests(unittest.TestCase):
    def run_status(self, stepfunctions, ecs):
        out = []
        ps.status(stepfunctions, ecs, "machine", "cluster", out=out.append, now=NOW)
        return "\n".join(out)

    def test_a_running_run_shows_its_step_its_age_and_what_bills(self):
        text = self.run_status(
            FakeStepFunctions([execution()], events=[{"stateEnteredEventDetails": {"name": "Tune"}}]), FakeEcs([tune_task()]))
        self.assertIn("RUNNING, in step Tune", text)
        self.assertIn("running for 1h 05m", text)
        self.assertIn("4 vCPU, 16 GB", text)
        self.assertIn("what bills", text)

    def test_waiting_for_approval_says_what_to_do_next(self):
        text = self.run_status(
            FakeStepFunctions([execution()], events=[{"stateEnteredEventDetails": {"name": "WaitForApproval"}}]), FakeEcs())
        self.assertIn("promote", text)
        self.assertIn("Nothing is running on the training cluster", text)

    def test_a_failed_run_shows_the_reason(self):
        events = [{"taskFailedEventDetails": {"error": "States.TaskFailed", "cause": "exit code 1 in step Tune"}}]
        text = self.run_status(
            FakeStepFunctions([execution("FAILED", stopped=NOW)], events=events), FakeEcs())
        self.assertIn("FAILED", text)
        self.assertIn("exit code 1 in step Tune", text)

    def test_no_run_yet(self):
        self.assertIn("No pipeline run yet", self.run_status(FakeStepFunctions([]), FakeEcs()))


class StopTests(unittest.TestCase):
    def test_a_running_run_is_stopped(self):
        sf, out = FakeStepFunctions([execution()]), []
        ps.stop(sf, "machine", out=out.append)
        self.assertEqual(sf.stopped, [("arn:run-1", "StoppedByUser")])

    def test_nothing_is_stopped_when_nothing_runs(self):
        sf, out = FakeStepFunctions([execution("SUCCEEDED")]), []
        ps.stop(sf, "machine", out=out.append)
        self.assertEqual(sf.stopped, [])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8"))
        self.steps = {s.get("name"): s for s in self.workflow["jobs"]["pipeline"]["steps"]}

    def test_the_run_is_started_and_not_followed_past_the_one_hour_login(self):
        start = self.steps["Start the pipeline"]
        self.assertEqual(start["env"]["WAIT"], "0")
        self.assertIn("start", start["if"])

    def test_status_and_stop_are_actions(self):
        self.assertEqual(self.workflow[True]["workflow_dispatch"]["inputs"]["action"]["options"], ["start", "status", "stop"])
        self.assertIn("status", self.steps["Show or stop the pipeline run"]["run"])

    def test_images_and_data_copy_run_only_when_starting(self):
        for name in ("Build the images for this commit (skipped when they already exist)",
                     "Copy the PhysioNet data into our bucket (S3 to S3, no web download)"):
            self.assertIn("inputs.action == 'start'", self.steps[name]["if"])


if __name__ == "__main__":
    unittest.main()
