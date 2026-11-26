"""The automatic CI and the manual deploy are wired as designed (docs/cicd.md)."""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def load(name):
    document = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    document["on"] = document.pop(True, document.get("on"))   # PyYAML reads the key `on` as True
    return document


RULES = yaml.safe_load((ROOT / "ci" / "change_rules.json").read_text(encoding="utf-8"))


class TestsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tests = load("tests.yml")
        self.jobs = self.tests["jobs"]

    def test_it_runs_on_changes_and_keeps_the_manual_button(self):
        triggers = self.tests["on"]
        for name in ("pull_request", "push", "schedule", "workflow_dispatch"):
            self.assertIn(name, triggers)
        self.assertIn("run_everything", triggers["workflow_dispatch"]["inputs"])

    def test_the_change_job_has_one_output_per_ci_job_in_the_rules(self):
        self.assertEqual(set(self.jobs["changes"]["outputs"]), set(RULES["all_ci_jobs"]))

    def test_every_scoped_job_waits_for_the_classification_and_checks_its_own_flag(self):
        mapping = {"unit": "unit", "docker-image": "docker", "cloudformation": "cloudformation", "audit": "audit", "workflows": "workflows"}
        for job, flag in mapping.items():
            self.assertEqual(self.jobs[job]["needs"], "changes", job)
            self.assertEqual(self.jobs[job]["if"], f"needs.changes.outputs.{flag} == 'true'", job)

    def test_the_patient_data_guard_always_runs(self):
        self.assertNotIn("changes", str(self.jobs["guard"].get("needs", "")))
        self.assertIn("schedule", self.jobs["guard"]["if"])   # skipped only for the weekly audit run

    def test_ci_ok_waits_for_every_other_job_and_runs_even_when_some_are_skipped(self):
        others = set(self.jobs) - {"ci-ok"}
        self.assertEqual(set(self.jobs["ci-ok"]["needs"]), others)
        self.assertEqual(self.jobs["ci-ok"]["if"], "always()")

    def test_the_weekly_scan_runs_only_the_audit(self):
        text = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
        schedule = text[text.index("schedule)"):text.index(";;", text.index("schedule)"))]
        self.assertIn("ci_audit=true", schedule)
        self.assertNotIn("ci_unit=true", schedule)

    def test_the_audit_covers_all_four_locked_targets(self):
        self.assertIn("scoring lambda training tools", (WORKFLOWS / "tests.yml").read_text(encoding="utf-8"))


class ImagesWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.images = load("images.yml")
        self.text = (WORKFLOWS / "images.yml").read_text(encoding="utf-8")

    def test_it_builds_on_merge_for_image_inputs_only_and_keeps_the_manual_button(self):
        self.assertEqual(self.images["on"]["push"]["branches"], ["main"])
        self.assertIn("Dockerfile", self.images["on"]["push"]["paths"])
        self.assertIn("requirements/**", self.images["on"]["push"]["paths"])
        self.assertIn("workflow_dispatch", self.images["on"])

    def test_it_can_be_switched_off_and_skips_when_there_is_no_stack(self):
        self.assertIn("vars.AUTO_PREBUILD != 'false'", self.images["jobs"]["images"]["if"])
        self.assertIn("describe-stacks", self.text)
        self.assertIn("steps.stack.outputs.exists == 'true'", self.text)

    def test_it_only_writes_images_and_never_changes_the_stack_or_services(self):
        for forbidden in ("aws_deploy.sh stack", "aws_deploy.sh serve", "aws_deploy.sh retrain", "aws_deploy.sh publish",
                          "cloudformation deploy", "power", "update-service"):
            self.assertNotIn(forbidden, self.text, forbidden)
        self.assertIn("aws_deploy.sh images", self.text)

    def test_it_uses_the_short_lived_role_only(self):
        self.assertEqual(self.images["permissions"]["id-token"], "write")
        self.assertNotIn("secrets.AWS", self.text)


class AwsWorkflowsStayManualTests(unittest.TestCase):
    def test_everything_that_changes_aws_is_started_by_hand(self):
        for name in ("deploy", "pipeline", "promote", "approve", "demo", "monitor", "power", "teardown", "verify", "collect-results"):
            self.assertEqual(list(load(f"{name}.yml")["on"]), ["workflow_dispatch"], name)

    def test_every_workflow_keeps_a_manual_option(self):
        for path in WORKFLOWS.glob("*.yml"):
            self.assertIn("workflow_dispatch", load(path.name)["on"], path.name)


class DeployScriptTests(unittest.TestCase):
    SCRIPT = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")

    def function(self, name):
        start = self.SCRIPT.index(f"{name}() {{")
        return self.SCRIPT[start:self.SCRIPT.index("\n}\n", start)]

    def test_retrain_copies_the_data_switches_mlflow_on_and_does_not_wait(self):
        body = self.function("cmd_retrain")
        self.assertLess(body.index("cmd_fetch_data"), body.index("require-on"))
        self.assertLess(body.index("require-on"), body.index("cmd_pipeline"))
        self.assertIn("DATA_SOURCE=physionet", body)
        self.assertIn("WAIT=0", body)

    def test_the_new_commands_are_reachable(self):
        for command in ("plan) cmd_plan", "record-state) cmd_record_state", "retrain) cmd_retrain"):
            self.assertIn(command, self.SCRIPT)

    def test_the_plan_passes_the_manual_options_through(self):
        body = self.function("cmd_plan")
        for option in ("--mode", "--retrain", "--dry-run", "--head"):
            self.assertIn(option, body)

    def test_a_forced_prepare_reaches_the_pipeline_input(self):
        self.assertIn('"force": "%s"', self.SCRIPT)
        template = (ROOT / "infra" / "cloudformation" / "pipeline-ecs.yaml").read_text(encoding="utf-8")
        self.assertIn("HP_force", template)


if __name__ == "__main__":
    unittest.main()
