"""What a change needs (ci/change_rules.json): the classifier shared by CI and the deploy planner."""

import json
import subprocess
import unittest
from pathlib import Path

from scripts import ci_changes as cc

ROOT = Path(__file__).resolve().parents[1]
RULES = cc.load_rules()


class RulesFileTests(unittest.TestCase):
    def test_every_rule_category_has_an_entry_and_every_name_is_known(self):
        categories = set(RULES["categories"])
        self.assertTrue({r["category"] for r in RULES["rules"]} <= categories)
        self.assertIn("other", categories)
        for name, entry in RULES["categories"].items():
            self.assertTrue(set(entry["ci"]) <= set(RULES["all_ci_jobs"]), name)
            self.assertTrue(set(entry["deploy"]) <= set(RULES["all_deploy_stages"]), name)

    def test_every_file_in_the_repository_gets_a_category(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        self.assertGreater(len(tracked), 100)
        for path in tracked:
            self.assertIn(cc.classify(path, RULES), RULES["categories"], path)

    def test_source_is_never_left_to_the_fallback(self):
        # A module nobody listed must still be training code (retrain is the cautious reading), never 'other'.
        for path in (ROOT / "src" / "sepsis_warning").glob("*.py"):
            self.assertNotEqual(cc.classify(f"src/sepsis_warning/{path.name}", RULES), "other", path.name)

    def test_the_rules_file_is_valid_json(self):
        json.loads((ROOT / "ci" / "change_rules.json").read_text(encoding="utf-8"))


class ClassifyTests(unittest.TestCase):
    def test_examples(self):
        examples = {
            "docs/ecs-runbook.md": "docs", "README.md": "docs",
            "tests/test_api.py": "tests",
            "src/sepsis_warning/service.py": "serving", "src/sepsis_warning/lambda_app.py": "serving",
            "src/sepsis_warning/features.py": "features",
            "src/sepsis_warning/tuning.py": "training", "src/sepsis_warning/sagemaker_train.py": "training",
            "src/sepsis_warning/mlflow_server.py": "tools", "src/sepsis_warning/evidently_job.py": "tools",
            "infra/cloudformation/mlflow-ecs.yaml": "infra", "scripts/aws_deploy.sh": "infra",
            "Dockerfile": "image", "requirements/tools.txt": "dependencies", "pyproject.toml": "dependencies",
            ".github/workflows/deploy.yml": "workflows", "ci/change_rules.json": "ci",
            "something/unknown.bin": "other",
        }
        for path, expected in examples.items():
            self.assertEqual(cc.classify(path, RULES), expected, path)


class PlanTests(unittest.TestCase):
    def test_docs_only_needs_no_checks_and_no_deploy(self):
        result = cc.plan(["docs/a.md", "README.md"], RULES)
        self.assertEqual((result["ci"], result["deploy"]), ([], []))

    def test_serving_code_needs_tests_docker_and_a_serve(self):
        result = cc.plan(["src/sepsis_warning/api.py"], RULES)
        self.assertEqual(result["ci"], ["unit", "docker"])
        self.assertEqual(result["deploy"], ["serve"])

    def test_training_code_needs_a_retrain_but_not_a_stack_update(self):
        result = cc.plan(["src/sepsis_warning/tuning.py"], RULES)
        self.assertEqual(result["deploy"], ["retrain"])

    def test_features_change_both_the_model_and_the_service(self):
        self.assertEqual(cc.plan(["src/sepsis_warning/features.py"], RULES)["deploy"], ["retrain", "serve"])

    def test_infra_needs_the_template_checks_and_a_stack_update(self):
        result = cc.plan(["infra/cloudformation/root.yaml"], RULES)
        self.assertIn("cloudformation", result["ci"])
        self.assertEqual(result["deploy"], ["stack"])

    def test_dependencies_need_the_audit_and_rebuilt_images(self):
        result = cc.plan(["requirements/lambda.txt"], RULES)
        self.assertIn("audit", result["ci"])
        self.assertEqual(result["deploy"], ["stack", "serve"])

    def test_workflow_changes_only_check_the_workflows(self):
        self.assertEqual(cc.plan([".github/workflows/tests.yml"], RULES)["ci"], ["workflows"])

    def test_several_files_combine_in_a_fixed_order(self):
        result = cc.plan(["tests/t.py", "infra/x.yaml", "src/sepsis_warning/api.py"], RULES)
        self.assertEqual(result["ci"], ["unit", "docker", "cloudformation"])
        self.assertEqual(result["deploy"], ["stack", "serve"])

    def test_an_unknown_change_runs_everything(self):
        result = cc.plan(None, RULES)
        self.assertFalse(result["known"])
        self.assertEqual(result["ci"], RULES["all_ci_jobs"])
        self.assertEqual(result["deploy"], RULES["all_deploy_stages"])

    def test_no_files_changed_runs_nothing(self):
        result = cc.plan([], RULES)
        self.assertEqual((result["ci"], result["deploy"]), ([], []))


class ChangedFilesTests(unittest.TestCase):
    def test_a_missing_or_zero_base_means_unknown(self):
        self.assertIsNone(cc.changed_files("", "HEAD"))
        self.assertIsNone(cc.changed_files("0" * 40, "HEAD"))

    def test_a_base_that_git_does_not_know_means_unknown(self):
        self.assertIsNone(cc.changed_files("deadbeef" * 5, "HEAD"))

    def test_a_real_range(self):
        # A throwaway repository, so the test does not depend on how much history the CI checkout has.
        import functools
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            def git(*args):
                subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=temp, check=True,
                               capture_output=True)

            git("init", "-q")
            (Path(temp) / "a.txt").write_text("1", encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "-m", "one")
            (Path(temp) / "b.txt").write_text("2", encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "-m", "two")
            run = functools.partial(subprocess.run, cwd=temp)
            self.assertEqual(cc.changed_files("HEAD~1", "HEAD", run=run), ["b.txt"])


class OutputTests(unittest.TestCase):
    def test_outputs_have_one_line_per_job_and_stage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            out, summary = Path(temp) / "out", Path(temp) / "summary"
            cc.write_outputs(cc.plan(["src/sepsis_warning/api.py"], RULES), RULES, str(out), str(summary))
            lines = out.read_text().splitlines()
            self.assertIn("ci_unit=true", lines)
            self.assertIn("ci_cloudformation=false", lines)
            self.assertIn("deploy_serve=true", lines)
            self.assertIn("categories=serving", lines)
            self.assertIn("After merging", summary.read_text())


if __name__ == "__main__":
    unittest.main()
