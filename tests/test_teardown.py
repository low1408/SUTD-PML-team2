"""Teardown: finds retained resources in nested stacks, plans by default, deletes only when confirmed."""

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aws_teardown", ROOT / "scripts" / "aws_teardown.py")
teardown = importlib.util.module_from_spec(spec)
sys.modules["aws_teardown"] = teardown
spec.loader.exec_module(teardown)


class FakeCloudFormation:
    def __init__(self):
        self.deleted = []
        self.stacks = {
            "root": [
                {"ResourceType": "AWS::CloudFormation::Stack", "PhysicalResourceId": "nested-artifacts",
                 "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::IAM::Role", "PhysicalResourceId": "role", "ResourceStatus": "CREATE_COMPLETE"},
            ],
            "nested-artifacts": [
                {"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": "art-bucket", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::KMS::Key", "PhysicalResourceId": "key-1", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::DynamoDB::Table", "PhysicalResourceId": "table", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::Logs::LogGroup", "PhysicalResourceId": "/aws/lambda/x", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::ECR::Repository", "PhysicalResourceId": "repo", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": "gone", "ResourceStatus": "DELETE_COMPLETE"},
            ],
        }

    def list_stack_resources(self, StackName, **_):
        return {"StackResourceSummaries": self.stacks[StackName]}

    def delete_stack(self, StackName):
        self.deleted.append(StackName)

    def get_waiter(self, _):
        return Namespace(wait=lambda **kw: None)


class FakeS3:
    def __init__(self):
        self.calls = []

    def get_paginator(self, _):
        return Namespace(paginate=lambda Bucket: iter([{
            "Versions": [{"Key": "a", "VersionId": "1"}, {"Key": "a", "VersionId": "2"}],
            "DeleteMarkers": [{"Key": "b", "VersionId": "3"}]}]))

    def delete_objects(self, Bucket, Delete):
        self.calls.append(("objects", Bucket, len(Delete["Objects"])))

    def delete_bucket(self, Bucket):
        self.calls.append(("bucket", Bucket))


class Recorder:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getattr__(self, method):
        return lambda **kw: self.log.append((self.name, method, kw))


class FakeSession:
    def __init__(self):
        self.cfn, self.s3, self.log = FakeCloudFormation(), FakeS3(), []

    def client(self, name):
        return {"cloudformation": self.cfn, "s3": self.s3}.get(name) or Recorder(name, self.log)


ARGS = dict(stack="sepsis-dev", region="ap-southeast-2", templates_bucket="sepsis-dev-1-r-cfn")


class TeardownTests(unittest.TestCase):
    def test_collects_retained_resources_from_nested_stacks(self):
        found = teardown.collect(FakeCloudFormation(), "root")
        self.assertEqual(found, [("AWS::S3::Bucket", "art-bucket"), ("AWS::ECR::Repository", "repo"),
                                 ("AWS::DynamoDB::Table", "table"), ("AWS::Logs::LogGroup", "/aws/lambda/x"),
                                 ("AWS::KMS::Key", "key-1")])

    def test_plan_only_deletes_nothing(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=False, confirm=""), session), 0)
        self.assertEqual((session.cfn.deleted, session.s3.calls, session.log), ([], [], []))

    def test_wrong_confirmation_deletes_nothing(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-prod"), session), 2)
        self.assertEqual((session.cfn.deleted, session.s3.calls), ([], []))

    def test_confirmed_run_deletes_stack_then_leftovers(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-dev"), session), 0)
        self.assertEqual(session.cfn.deleted, ["sepsis-dev"])
        self.assertIn(("objects", "art-bucket", 3), session.s3.calls)
        self.assertIn(("bucket", "art-bucket"), session.s3.calls)
        self.assertIn(("bucket", "sepsis-dev-1-r-cfn"), session.s3.calls)
        methods = [(n, m) for n, m, _ in session.log]
        self.assertIn(("ecr", "delete_repository"), methods)
        self.assertIn(("dynamodb", "delete_table"), methods)
        self.assertIn(("logs", "delete_log_group"), methods)
        key = [kw for n, m, kw in session.log if m == "schedule_key_deletion"][0]
        self.assertEqual(key["PendingWindowInDays"], 7)
        self.assertNotIn(("bucket", "gone"), session.s3.calls)

    def test_missing_stack_is_not_an_error(self):
        session = FakeSession()

        def missing(StackName, **_):
            raise RuntimeError("An error occurred (ValidationError): Stack with id x does not exist")

        session.cfn.list_stack_resources = missing
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-dev"), session), 0)
        self.assertEqual((session.cfn.deleted, session.s3.calls), ([], []))

    def test_script_and_workflow_require_confirmation(self):
        script = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("CONFIRM_DESTROY", script)
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "teardown.yml").read_text(encoding="utf-8"))
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["scope"]["default"], "plan only")
        self.assertEqual(inputs["scope"]["options"], ["plan only", "stack only", "everything"])
        self.assertEqual(workflow["jobs"]["teardown"]["environment"], "${{ inputs.environment }}")
        self.assertEqual(workflow["permissions"]["id-token"], "write")


if __name__ == "__main__":
    unittest.main()
