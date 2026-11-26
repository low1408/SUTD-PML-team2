"""The ECS Fargate training backend: pipeline definition, platform template and wiring. Shape checks, no AWS."""

import json
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFN = ROOT / "infra" / "cloudformation"


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):  # CloudFormation tags (!Ref, !Sub, ...) become plain values
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_Loader.add_multi_constructor("!", _tag)


def load(name: str) -> dict:
    return yaml.load((CFN / name).read_text(encoding="utf-8"), Loader=_Loader)


def machine(name: str = "pipeline-ecs.yaml") -> dict:
    return load(name)["Resources"]["TrainingPipeline"]["Properties"]["Definition"]


def env_of(state: dict) -> dict:
    items = state["Parameters"]["Overrides"]["ContainerOverrides"][0]["Environment"]
    return {item["Name"]: item.get("Value", item.get("Value.$")) for item in items}


class EcsPipelineTests(unittest.TestCase):
    def test_the_states_run_in_the_same_order_as_the_sagemaker_pipeline(self):
        states = machine()["States"]
        self.assertEqual(machine()["StartAt"], "RegisterTaskDefinition")
        self.assertEqual(states["RegisterTaskDefinition"]["Next"], "CheckPrepare")
        self.assertEqual(states["CheckPrepare"]["Default"], "Tune")
        chain = [("PrepareData", "PreparedOk", "Tune"), ("Tune", "TunedOk", "Train"),
                 ("Train", "TrainedOk", "WaitForApproval"), ("EvaluateAndPublish", "PublishedOk", "NotifyPublished")]
        for step, check, following in chain:
            self.assertEqual(states[step]["Next"], check, step)
            # A non-zero exit code stops the pipeline; anything else (including a result without the field, because the
            # Step Functions page does not document that shape) continues, with failures raised by the integration caught.
            self.assertEqual(states[check]["Choices"][0]["Next"], "NotifyFailure", check)
            self.assertEqual(states[check]["Default"], following, check)
            condition = json.dumps(states[check]["Choices"][0])
            self.assertIn("ExitCode", condition)
            self.assertIn("IsPresent", condition)
            self.assertNotIn("ResultSelector", states[step])        # nothing depends on an undocumented output shape
        self.assertEqual(states["WaitForApproval"]["Next"], "EvaluateAndPublish")
        # Same state names as the SageMaker pipeline for everything a person or a script refers to.
        original = machine("pipeline.yaml")["States"]
        self.assertTrue({"CheckPrepare", "PrepareData", "Tune", "Train", "WaitForApproval", "EvaluateAndPublish",
                         "NotifyPublished", "NotifyFailure", "Stopped"} <= set(original) & set(states))

    def test_every_task_failure_ends_in_the_failure_notice(self):
        for name, state in machine()["States"].items():
            if state["Type"] == "Task" and name not in ("NotifyFailure", "NotifyPublished"):
                self.assertEqual(state["Catch"][0]["Next"], "NotifyFailure", name)

    def test_steps_run_the_ecs_entry_point_with_the_same_modes_and_channels(self):
        states = machine()["States"]
        container = machine()["States"]["RegisterTaskDefinition"]["Parameters"]["ContainerDefinitions"][0]
        self.assertEqual(container["EntryPoint"], ["python", "-m", "sepsis_warning.ecs_job"])
        self.assertEqual(container["LogConfiguration"]["LogDriver"], "awslogs")
        self.assertEqual(env_of(states["PrepareData"])["HP_mode"], "prepare")
        self.assertNotIn("CH_data", env_of(states["PrepareData"]))               # it makes the data
        self.assertEqual(env_of(states["Tune"])["HP_mode"], "tune")
        self.assertEqual(env_of(states["EvaluateAndPublish"])["HP_mode"], "evaluate_publish")
        train = env_of(states["Train"])
        self.assertEqual({"CH_data", "CH_config"} <= set(train), True)
        self.assertIn("tuning/", train["CH_config"])
        self.assertEqual(train["HP_require_gates"], "true")
        for name in ("PrepareData", "Tune", "Train", "EvaluateAndPublish"):
            self.assertIn("training-output/{}/", env_of(states[name])["JOB_OUTPUT"], name)

    def test_the_download_and_the_publish_are_not_on_spot_and_the_work_is(self):
        states = machine()["States"]
        provider = lambda n: states[n]["Parameters"]["CapacityProviderStrategy"][0]["CapacityProvider"]  # noqa: E731
        self.assertEqual(provider("PrepareData"), "FARGATE")
        self.assertEqual(provider("EvaluateAndPublish"), "FARGATE")
        self.assertEqual(provider("Tune"), "${WorkProvider}")
        self.assertEqual(provider("Train"), "${WorkProvider}")
        self.assertNotIn("LaunchType", states["Tune"]["Parameters"])      # it cannot be combined with a capacity provider

    def test_tasks_get_a_public_ip_only_because_the_vpc_has_no_nat(self):
        network = machine()["States"]["Tune"]["Parameters"]["NetworkConfiguration"]["AwsvpcConfiguration"]
        self.assertEqual(network["AssignPublicIp"], "ENABLED")

    def test_states_format_has_one_argument_per_placeholder(self):
        for name, state in machine()["States"].items():
            values = dict(state.get("Parameters", {}))
            for item in state.get("Parameters", {}).get("Overrides", {}).get("ContainerOverrides", [{}])[0].get("Environment", []):
                values[f"{name}.env.{item['Name']}"] = item.get("Value.$", "")
            for key, value in values.items():
                if isinstance(value, str) and "States.Format(" in value:
                    inner = value[value.index("States.Format(") + len("States.Format("):].rstrip().rstrip(")")
                    template, arguments = re.match(r"'(.*)'\s*,\s*(.*)$", inner, flags=re.S).groups()
                    self.assertEqual(template.count("{}"), len([a for a in arguments.split(",") if a.strip()]), f"{name}.{key}")

    def test_the_pipeline_role_may_run_only_this_clusters_tasks_and_pass_only_the_two_task_roles(self):
        statements = load("pipeline-ecs.yaml")["Resources"]["PipelineRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        run = next(s for s in statements if s["Action"] == "ecs:RunTask")
        self.assertIn("ecs:cluster", json.dumps(run["Condition"]))
        passrole = next(s for s in statements if s["Action"] == "iam:PassRole")
        self.assertEqual(len(passrole["Resource"]), 2)
        self.assertEqual(passrole["Condition"]["StringEquals"]["iam:PassedToService"], "ecs-tasks.amazonaws.com")


class EcsPlatformTests(unittest.TestCase):
    def test_no_standing_costs_and_no_inbound_access(self):
        text = (CFN / "ecs-platform.yaml").read_text(encoding="utf-8")
        for resource in ("AWS::EC2::NatGateway", "AWS::EC2::VPCEndpoint", "AWS::EC2::EIP", "AWS::ElasticLoadBalancingV2",
                         "AWS::RDS::", "AWS::EFS::"):
            self.assertNotIn(resource, text, f"{resource} costs money while idle")
        group = load("ecs-platform.yaml")["Resources"]["TaskSecurityGroup"]["Properties"]
        self.assertNotIn("SecurityGroupIngress", group)
        # HTTPS to anywhere (ECR, S3), and the MLflow port inside the VPC only.
        ports = {(r["FromPort"], r["ToPort"]): r["CidrIp"] for r in group["SecurityGroupEgress"]}
        self.assertEqual(set(ports), {(443, 443), (5000, 5000)})
        self.assertEqual(ports[(5000, 5000)], "VpcCidr")

    def test_the_training_cluster_does_not_clash_with_the_serving_cluster(self):
        serving = load("service.yaml")["Resources"]["Cluster"]["Properties"]["ClusterName"]
        training = load("ecs-platform.yaml")["Resources"]["Cluster"]["Properties"]["ClusterName"]
        self.assertNotEqual(serving, training)
        pipeline = json.dumps(load("pipeline-ecs.yaml")["Resources"]["PipelineRole"])
        self.assertIn("-training-cluster/", pipeline)

    def test_the_cluster_offers_fargate_and_fargate_spot_and_insights_are_off(self):
        cluster = load("ecs-platform.yaml")["Resources"]["Cluster"]["Properties"]
        self.assertEqual(cluster["CapacityProviders"], ["FARGATE", "FARGATE_SPOT"])
        self.assertEqual(cluster["ClusterSettings"], [{"Name": "containerInsights", "Value": "disabled"}])

    def test_the_task_role_has_the_same_data_permissions_as_the_sagemaker_role(self):
        mlops = (CFN / "mlops.yaml").read_text(encoding="utf-8")
        ecs = (CFN / "ecs-platform.yaml").read_text(encoding="utf-8")
        for prefix in ("data", "tuning", "reports", "raw", "models", "training-output"):
            self.assertIn(f"${{ArtifactBucketArn}}/{prefix}/*", ecs, prefix)
        self.assertIn("training-output/*", mlops)
        # The execution role can pull the image and write logs, nothing else.
        policy = load("ecs-platform.yaml")["Resources"]["ExecutionRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        actions = {a for s in policy for a in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]])}
        self.assertFalse(any(a.startswith("s3:") for a in actions))


class RootWiringTests(unittest.TestCase):
    def test_the_backend_is_a_parameter_and_only_one_pipeline_is_created(self):
        root = load("root.yaml")
        self.assertEqual(root["Parameters"]["TrainingBackend"]["Default"], "sagemaker")
        self.assertEqual(root["Parameters"]["TrainingBackend"]["AllowedValues"], ["sagemaker", "ecs"])
        resources = root["Resources"]
        self.assertEqual(resources["Pipeline"]["Condition"], "UseSageMakerTraining")
        self.assertEqual(resources["PipelineEcs"]["Condition"], "UseEcsTraining")
        self.assertEqual(resources["EcsPlatform"]["Condition"], "NeedsEcsPlatform")
        self.assertEqual(resources["PipelineEcs"]["Properties"]["TemplateURL"], "pipeline-ecs.yaml")

    def test_every_environment_trains_on_ecs_because_a_new_account_has_no_sagemaker_quota(self):
        """staging and prod once kept the SageMaker defaults, and a deploy there would have needed the quota (see tests/test_params.py)."""
        for name in ("dev.json", "staging.json", "prod.json"):
            params = json.loads((ROOT / "infra" / "params" / name).read_text(encoding="utf-8"))
            self.assertEqual(params.get("TrainingBackend"), "ecs", name)

    def test_deploy_script_accepts_the_backend_override(self):
        text = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn('TrainingBackend=$TRAINING_BACKEND', text)



class TrainingTaskRoleTests(unittest.TestCase):
    def statements(self):
        role = load("ecs-platform.yaml")["Resources"]["TaskRole"]["Properties"]["Policies"]
        return [s for policy in role for s in policy["PolicyDocument"]["Statement"]]

    def writable(self):
        found = []
        for statement in self.statements():
            if not isinstance(statement, dict):     # a conditional statement (Fn::If) is not a plain grant
                continue
            actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
            if "s3:PutObject" in actions:
                resources = statement["Resource"] if isinstance(statement["Resource"], list) else [statement["Resource"]]
                found += [json.dumps(r) for r in resources]
        return " ".join(found)

    def test_the_monitor_reference_step_may_save_the_evidently_reference(self):
        """The first real run of monitor reference failed with AccessDenied on monitoring/evidently/<bundle>/reference.json."""
        self.assertIn("monitoring/evidently/*", self.writable())

    def test_the_other_prefixes_the_steps_write_stay_writable(self):
        for prefix in ("data/*", "tuning/*", "reports/*", "training-output/*", "models/*"):
            self.assertIn(prefix, self.writable(), prefix)


if __name__ == "__main__":
    unittest.main()


class MlflowOnEcsTests(unittest.TestCase):
    def test_the_database_is_private_encrypted_and_has_a_generated_password(self):
        db = load("mlflow-ecs.yaml")["Resources"]["Database"]["Properties"]
        self.assertEqual(db["Engine"], "postgres")
        self.assertIs(db["PubliclyAccessible"], False)
        self.assertIs(db["StorageEncrypted"], True)
        self.assertIs(db["ManageMasterUserPassword"], True)       # no password in the template or the parameters
        self.assertNotIn("MasterUserPassword", db)

    def test_there_is_no_load_balancer_and_the_server_is_reachable_only_from_training_tasks_by_default(self):
        text = (CFN / "mlflow-ecs.yaml").read_text(encoding="utf-8")
        for resource in ("AWS::ElasticLoadBalancingV2", "AWS::EC2::NatGateway", "AWS::EFS::", "0.0.0.0/0\n      Description: The"):
            self.assertNotIn(resource, text)
        group = load("mlflow-ecs.yaml")["Resources"]["MlflowSecurityGroup"]["Properties"]
        self.assertEqual(len(group["SecurityGroupIngress"]), 1)
        self.assertIn("SourceSecurityGroupId", group["SecurityGroupIngress"][0])
        self.assertNotIn("CidrIp", group["SecurityGroupIngress"][0])

    def test_the_service_carries_the_tags_the_power_workflow_uses(self):
        service = load("mlflow-ecs.yaml")["Resources"]["Service"]
        tags = {t["Key"]: t["Value"] for t in service["Properties"]["Tags"]}
        self.assertEqual((tags["power:group"], tags["power:desired"]), ("mlflow", "1"))
        self.assertEqual(service["Condition"], "HasImage")          # no image yet, no service
        db_tags = {t["Key"]: t["Value"] for t in load("mlflow-ecs.yaml")["Resources"]["Database"]["Properties"]["Tags"]}
        self.assertEqual(db_tags["power:group"], "database")

    def test_the_password_comes_from_secrets_manager_and_artifacts_go_to_s3(self):
        definition = load("mlflow-ecs.yaml")["Resources"]["TaskDefinition"]["Properties"]["ContainerDefinitions"][0]
        self.assertEqual([s["Name"] for s in definition["Secrets"]], ["DB_PASSWORD"])
        names = {e["Name"] for e in definition["Environment"]}
        self.assertNotIn("DB_PASSWORD", names)
        self.assertEqual(definition["Command"], ["sepsis_warning.mlflow_server"])

    def test_root_wires_the_flags_and_the_rules(self):
        root = load("root.yaml")
        self.assertEqual(root["Parameters"]["MlflowBackend"]["AllowedValues"], ["sagemaker", "ecs"])
        self.assertEqual(root["Parameters"]["EnableSageMakerMlflowApp"]["Default"], "true")
        self.assertEqual(root["Resources"]["MlflowEcs"]["Condition"], "UseEcsMlflow")
        self.assertIn("EcsMlflowNeedsEcsTraining", root["Rules"])
        self.assertIn("SageMakerMlflowNeedsTheApp", root["Rules"])
        mlops = load("mlops.yaml")
        self.assertEqual(mlops["Resources"]["MlflowApp"]["Condition"], "HasMlflowApp")
        self.assertEqual(mlops["Outputs"]["MlflowAppArn"]["Condition"], "HasMlflowApp")

    def test_dev_uses_the_ecs_server_and_drops_the_sagemaker_app(self):
        dev = json.loads((ROOT / "infra" / "params" / "dev.json").read_text(encoding="utf-8"))
        self.assertEqual((dev["MlflowBackend"], dev["EnableSageMakerMlflowApp"], dev["TrainingBackend"]), ("ecs", "false", "ecs"))

    def test_the_tools_image_is_built_locked_and_scanned(self):
        self.assertIn("FROM base AS tools", (ROOT / "Dockerfile").read_text(encoding="utf-8"))
        self.assertIn("tools:ToolsRepositoryUri", (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8"))
        lock = (ROOT / "requirements" / "tools.txt").read_text(encoding="utf-8")
        for package in ("mlflow==", "psycopg2-binary==", "evidently=="):
            self.assertIn(package, lock)
        ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        self.assertIn("scoring lambda training tools", ci)


class EvidentlyOnEcsTests(unittest.TestCase):
    def test_the_ui_has_no_inbound_rule_and_carries_the_power_tags(self):
        template = load("evidently-ecs.yaml")
        group = template["Resources"]["UiSecurityGroup"]["Properties"]
        self.assertNotIn("SecurityGroupIngress", group)
        self.assertEqual({t["Key"]: t["Value"] for t in group["Tags"]}["power:ui-port"], "8000")
        tags = {t["Key"]: t["Value"] for t in template["Resources"]["UiService"]["Properties"]["Tags"]}
        self.assertEqual((tags["power:group"], tags["power:desired"]), ("evidently", "1"))
        self.assertEqual(template["Resources"]["UiService"]["Condition"], "HasImage")

    def test_the_ui_can_only_read_and_the_report_job_cannot_touch_models_or_data(self):
        roles = load("evidently-ecs.yaml")["Resources"]
        ui = json.dumps(roles["UiTaskRole"]["Properties"]["Policies"])
        self.assertNotIn("s3:PutObject", ui)
        report = json.dumps(roles["ReportTaskRole"]["Properties"]["Policies"])
        for forbidden in ("/models/", "/data/", "/raw/", "mlflow-artifacts"):
            self.assertNotIn(forbidden, report)
        self.assertIn("cloudwatch:namespace", report)
        self.assertIn("SepsisWarning/Evidently", report)

    def test_the_report_runs_on_demand_not_on_a_schedule_and_the_alarm_emails_the_owner(self):
        text = (CFN / "evidently-ecs.yaml").read_text(encoding="utf-8")
        self.assertNotIn("AWS::Events::Rule", text)
        self.assertNotIn("ScheduleExpression", text)
        alarm = load("evidently-ecs.yaml")["Resources"]["DriftAlarm"]["Properties"]
        self.assertEqual((alarm["Namespace"], alarm["MetricName"]), ("SepsisWarning/Evidently", "DriftedColumnsShare"))
        self.assertEqual(alarm["AlarmActions"], ["AlarmTopicArn"])
        self.assertEqual(alarm["TreatMissingData"], "notBreaching")

    def test_root_flag_rule_and_outputs(self):
        root = load("root.yaml")
        self.assertEqual(root["Parameters"]["EnableEvidently"]["Default"], "false")
        self.assertEqual(root["Resources"]["EvidentlyEcs"]["Condition"], "UseEvidently")
        self.assertIn("EvidentlyNeedsTheEcsPlatform", root["Rules"])
        self.assertIn("EvidentlyReportFamily", root["Outputs"])
        dev = json.loads((ROOT / "infra" / "params" / "dev.json").read_text(encoding="utf-8"))
        self.assertEqual(dev["EnableEvidently"], "true")

    def test_the_monitor_workflow_is_manual_and_uses_the_two_commands(self):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "monitor.yml").read_text(encoding="utf-8"))
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["action"]["options"], ["report", "reference"])
        text = (ROOT / ".github" / "workflows" / "monitor.yml").read_text(encoding="utf-8")
        self.assertIn("monitor-reference", text)
        self.assertIn("monitor-report", text)
        for step in workflow["jobs"]["monitor"]["steps"]:
            self.assertNotIn("${{ inputs.", step.get("run", ""))
        script = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("monitor-reference) cmd_monitor_reference", script)
        self.assertIn("monitor-report) cmd_monitor_report", script)

    def test_the_tools_lock_uses_the_pandas_evidently_can_import(self):
        lock = (ROOT / "requirements" / "tools.txt").read_text(encoding="utf-8")
        version = re.search(r"^pandas==(\d+)\.", lock, flags=re.M).group(1)
        self.assertEqual(version, "2")          # evidently 0.7 does not import under pandas 3
        script = (ROOT / "scripts" / "lock_requirements.sh").read_text(encoding="utf-8")
        self.assertIn('pandas<3', script)


class FailFastTests(unittest.TestCase):
    def test_both_ecs_services_fail_the_deployment_instead_of_waiting_for_a_steady_state_that_never_comes(self):
        for template, name in (("mlflow-ecs.yaml", "Service"), ("evidently-ecs.yaml", "UiService")):
            config = load(template)["Resources"][name]["Properties"]["DeploymentConfiguration"]
            self.assertEqual(config["DeploymentCircuitBreaker"], {"Enable": True, "Rollback": False}, template)

    def test_root_exposes_the_serving_mode_so_verify_knows_which_images_to_expect(self):
        self.assertIn("ServingMode", load("root.yaml")["Outputs"])
        self.assertIn('outputs.get("ServingMode", "lambda")', (ROOT / "scripts" / "aws_verify.py").read_text(encoding="utf-8"))

    def test_the_deploy_workflow_builds_the_images_side_by_side(self):
        self.assertIn('IMAGES_PARALLEL: "1"', (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8"))
