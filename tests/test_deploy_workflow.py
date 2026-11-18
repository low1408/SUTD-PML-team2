"""The deploy workflow lets you pick the environment and region, and uses no stored AWS keys."""

import unittest
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


class DeployWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.text = PATH.read_text(encoding="utf-8")
        self.workflow = yaml.safe_load(self.text)
        self.inputs = self.workflow[True]["workflow_dispatch"]["inputs"]

    def test_environment_and_region_are_chosen_at_start(self):
        self.assertEqual(self.inputs["environment"]["options"], ["dev", "staging", "prod"])
        regions = self.inputs["region"]["options"]
        self.assertIn("ap-southeast-1", regions)
        self.assertIn("ap-southeast-2", regions)
        self.assertEqual(self.workflow["jobs"]["deploy"]["environment"], "${{ inputs.environment }}")

    def test_first_deploy_can_stop_before_publishing(self):
        steps = self.inputs["steps"]
        self.assertEqual(steps["default"], "stack and images only")
        gated = [s["name"] for s in self.workflow["jobs"]["deploy"]["steps"] if s.get("if") == "inputs.steps == 'full deploy'"]
        self.assertEqual(gated, ["Publish the promoted model and serve it", "Smoke test the deployed API"])

    def test_credentials_come_from_oidc_only(self):
        self.assertEqual(self.workflow["permissions"]["id-token"], "write")
        self.assertNotIn("secrets.AWS", self.text)
        self.assertNotIn("aws-access-key-id", self.text)
        self.assertIn("steps.target.outputs.role", self.text)

    def test_role_name_matches_the_bootstrap_template(self):
        template = (PATH.parents[2] / "infra" / "bootstrap" / "github-oidc.yaml").read_text(encoding="utf-8")
        for environment in ("dev", "staging", "prod"):
            self.assertIn(f"github-deploy-{environment}-${{AWS::Region}}", template)
        self.assertIn("role/github-deploy-${ENVIRONMENT}-${region}", self.text)

    def test_explicit_role_takes_precedence_over_account_role(self):
        explicit = self.text.index('if [ -n "$ROLE_ARN_VAR" ]; then')
        account = self.text.index('elif [ -n "$ACCOUNT_ID" ]; then')
        self.assertLess(explicit, account)
        self.assertIn('role="$ROLE_ARN_VAR"', self.text)

    def test_untrusted_text_goes_through_env_not_inline(self):
        for step in self.workflow["jobs"]["deploy"]["steps"]:
            self.assertNotIn("inputs.model_uri", step.get("run", ""))


if __name__ == "__main__":
    unittest.main()
