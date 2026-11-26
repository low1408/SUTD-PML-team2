"""Each role may do what its code does (see docs/iam-review.md). The monitor reference step once failed with AccessDenied on a prefix
no policy named; these tests spell out what every role needs, tied to constants in the code where there are any."""

import json
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFN = ROOT / "infra" / "cloudformation"
sys.path.insert(0, str(ROOT / "src"))


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {f"!{suffix}": value}


_Loader.add_multi_constructor("!", _tag)


def load(name):
    return yaml.load((CFN / name).read_text(encoding="utf-8"), Loader=_Loader)


def text(resource) -> str:
    """A resource entry as plain text: Fn::Sub strings keep their path, references keep their name."""
    if isinstance(resource, dict):
        return " ".join(text(v) for v in resource.values())
    if isinstance(resource, list):
        return " ".join(text(v) for v in resource)
    return str(resource)


def grants(template: str, role: str) -> list[dict]:
    policies = load(template)["Resources"][role]["Properties"].get("Policies", [])
    found = []
    for policy in policies:
        for statement in policy["PolicyDocument"]["Statement"]:
            if not isinstance(statement, dict) or "Effect" not in statement:
                continue                      # a conditional (Fn::If) statement is not a plain grant
            actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
            resources = statement["Resource"] if isinstance(statement["Resource"], list) else [statement["Resource"]]
            found.append({"effect": statement["Effect"], "actions": actions, "resources": [text(r) for r in resources],
                          "condition": statement.get("Condition")})
    return found


def allows(template: str, role: str, action: str, resource_end: str = "") -> bool:
    for grant in grants(template, role):
        if grant["effect"] != "Allow":
            continue
        if any(a == action or a == action.split(":")[0] + ":*" or (a.endswith("*") and action.startswith(a[:-1])) for a in grant["actions"]):
            if any(r.endswith(resource_end) or resource_end == "" for r in grant["resources"]):
                return True
    return False


class TrainingTaskRoleTests(unittest.TestCase):
    """Prepare, tune, train, evaluate and publish, and the monitor reference step all run as this role."""

    ROLE = ("ecs-platform.yaml", "TaskRole")

    def test_it_reads_what_the_steps_read(self):
        for prefix in ("data/*", "tuning/*", "raw/*", "models/*"):
            self.assertTrue(allows(*self.ROLE, "s3:GetObject", prefix), prefix)

    def test_it_writes_what_the_steps_write(self):
        for prefix in ("data/*", "tuning/*", "reports/*", "training-output/*", "models/*", "monitoring/evidently/*"):
            self.assertTrue(allows(*self.ROLE, "s3:PutObject", prefix), prefix)

    def test_it_lists_the_bucket_and_uses_the_key(self):
        self.assertTrue(allows(*self.ROLE, "s3:ListBucket"))
        for action in ("kms:Decrypt", "kms:GenerateDataKey"):
            self.assertTrue(allows(*self.ROLE, action), action)


class ApproverTaskRoleTests(unittest.TestCase):
    def test_promote_writes_only_its_own_output_and_cannot_publish_or_train(self):
        role = ("ecs-platform.yaml", "ApproverTaskRole")
        self.assertTrue(allows(*role, "s3:PutObject", "training-output/*"))
        for prefix in ("models/*", "data/*", "reports/*"):
            self.assertFalse(allows(*role, "s3:PutObject", prefix), prefix)    # the approver is a different, narrower principal


class EvidentlyRolesTests(unittest.TestCase):
    def test_the_report_reads_the_log_and_the_reference_and_writes_reports_and_the_workspace(self):
        role = ("evidently-ecs.yaml", "ReportTaskRole")
        for prefix in ("prediction-log/*", "monitoring/evidently/*"):
            self.assertTrue(allows(*role, "s3:GetObject", prefix), prefix)
        for prefix in ("reports/monitoring/*", "monitoring/evidently/workspace/*"):
            self.assertTrue(allows(*role, "s3:PutObject", prefix), prefix)
        self.assertTrue(allows(*role, "s3:ListBucket"))

    def test_the_metrics_namespace_in_the_policy_is_the_one_the_code_publishes_to(self):
        from sepsis_warning import evidently_job
        grant = next(g for g in grants("evidently-ecs.yaml", "ReportTaskRole") if "cloudwatch:PutMetricData" in g["actions"])
        self.assertEqual(grant["condition"]["StringEquals"]["cloudwatch:namespace"], evidently_job.NAMESPACE)

    def test_the_ui_lists_the_prefix_the_code_lists(self):
        """sync_down lists '<prefix>/'; the policy condition must match that form."""
        role = ("evidently-ecs.yaml", "UiTaskRole")
        self.assertTrue(allows(*role, "s3:GetObject", "monitoring/evidently/workspace/*"))
        listing = next(g for g in grants(*role) if "s3:ListBucket" in g["actions"])
        pattern = listing["condition"]["StringLike"]["s3:prefix"]
        self.assertTrue(re.fullmatch(pattern.replace("*", ".*"), "monitoring/evidently/workspace/"))

    def test_the_ui_cannot_write(self):
        self.assertFalse(allows("evidently-ecs.yaml", "UiTaskRole", "s3:PutObject"))


class MlflowRolesTests(unittest.TestCase):
    def test_the_server_stores_artifacts_under_its_own_prefix_only(self):
        role = ("mlflow-ecs.yaml", "TaskRole")
        for action in ("s3:GetObject", "s3:PutObject"):
            self.assertTrue(allows(*role, action, "mlflow-artifacts/*"), action)
        self.assertFalse(allows(*role, "s3:PutObject", "models/*"))

    def test_the_task_can_read_the_database_password_it_is_given(self):
        self.assertTrue(allows("mlflow-ecs.yaml", "ExecutionRole", "secretsmanager:GetSecretValue"))


class ServingRolesTests(unittest.TestCase):
    def test_the_function_reads_the_model_and_keeps_the_alert_ledger(self):
        role = ("serverless.yaml", "FunctionRole")
        self.assertTrue(allows(*role, "s3:GetObject", "models/*"))
        for action in ("dynamodb:GetItem", "dynamodb:UpdateItem", "kms:Decrypt", "logs:PutLogEvents"):
            self.assertTrue(allows(*role, action), action)

    def test_the_function_cannot_write_models_or_data(self):
        self.assertFalse(allows("serverless.yaml", "FunctionRole", "s3:PutObject"))

    def test_the_prediction_log_is_written_by_firehose_not_by_the_function(self):
        self.assertTrue(allows("observability.yaml", "FirehoseRole", "s3:PutObject", "prediction-log/*"))

    def test_the_drift_function_reads_the_log_and_writes_its_results(self):
        role = ("observability.yaml", "DriftRole")
        self.assertTrue(allows(*role, "s3:GetObject", "prediction-log/*"))
        self.assertTrue(allows(*role, "s3:PutObject", "drift-results/*"))


class PipelineRoleTests(unittest.TestCase):
    ROLE = ("pipeline-ecs.yaml", "PipelineRole")

    def test_it_runs_tasks_registers_definitions_and_notifies(self):
        for action in ("ecs:RunTask", "ecs:RegisterTaskDefinition", "ecs:StopTask", "ecs:DescribeTasks", "sns:Publish",
                       "events:PutRule"):
            self.assertTrue(allows(*self.ROLE, action), action)

    def test_it_may_hand_the_task_and_execution_roles_to_ecs_and_no_other_role(self):
        passes = [g for g in grants(*self.ROLE) if "iam:PassRole" in g["actions"]]
        self.assertEqual(len(passes), 1)
        joined = " ".join(passes[0]["resources"])
        self.assertIn("ExecutionRoleArn", joined)
        self.assertIn("TaskRoleArn", joined)
        self.assertNotIn("ApproverTaskRoleArn", joined)     # the pipeline never runs the approver's tasks


class DriftRetrainInputTests(unittest.TestCase):
    def test_the_drift_alarm_starts_the_pipeline_with_every_key_the_pipeline_reads(self):
        """A missing key fails the first step. The ECS pipeline reads these from its input."""
        rule = load("observability.yaml")["Resources"]["RetrainOnDriftRule"]["Properties"]["Targets"][0]["Input"]["!Sub"]
        given = set(json.loads(rule.replace("${RetrainImageTag}", "training-x")))
        needed = {"image_tag", "git_commit", "require_real_data", "prepare", "data_source", "limit", "trials", "force"}
        self.assertEqual(needed - given, set())


if __name__ == "__main__":
    unittest.main()
