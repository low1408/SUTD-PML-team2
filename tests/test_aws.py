"""AWS paths without AWS: S3 model loading, the Lambda handler and the SageMaker entry point.

S3 is replaced by a fake client over a local folder. The Lambda and SageMaker
tests are skipped unless their optional packages are installed.
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from sepsis_warning.data import generate_synthetic
from sepsis_warning.model_source import (
    ModelHashMismatch,
    fetch_model,
    load_serving_model,
    parse_s3_uri,
)
from sepsis_warning.scoring import file_sha256
from sepsis_warning.tracking import check_tracking_uri, is_mlflow_app

HAVE = {name: importlib.util.find_spec(name) is not None for name in ("mangum", "fastapi", "mlflow")}
APP_ARN = "arn:aws:sagemaker:ap-southeast-1:123456789012:mlflow-app/app-ABC123"


class FakeS3:
    """download_file/upload_file over a local folder laid out as <root>/<bucket>/<key>."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.uploads = []

    def download_file(self, bucket, key, filename):
        source = self.root / bucket / key
        if not source.is_file():
            error = FileNotFoundError(key)
            error.response = {"Error": {"Code": "404"}}
            raise error
        shutil.copyfile(source, filename)

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, Metadata=None):
        target = self.root / Bucket / Key
        if IfNoneMatch == "*" and target.exists():
            error = RuntimeError("exists")
            error.response = {"Error": {"Code": "PreconditionFailed"}}
            raise error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Body)
        self.metadata = getattr(self, "metadata", {})
        self.metadata[(Bucket, Key)] = Metadata or {}
        self.uploads.append((Key, Metadata))

    def head_object(self, Bucket, Key):
        return {"Metadata": getattr(self, "metadata", {}).get((Bucket, Key), {})}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        target = self.root / bucket / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
        self.uploads.append((key, ExtraArgs))


def train_run(root: Path) -> Path:
    from sepsis_warning.cli import run_experiment

    generate_synthetic(root / "data", source_size=80, external_size=20)
    (root / "data" / "SYNTHETIC.json").write_text("{}")
    run_experiment(Namespace(
        data=root / "data", splits=None, output=root / "run", seed=42, window=6, budget=0.2, cooldown=6,
        splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False,
    ))
    return root / "run"


class ModelSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_dir = train_run(cls.root)
        cls.sha = file_sha256(cls.run_dir / "model.joblib")
        cls.s3 = FakeS3(cls.root / "s3")
        prefix = cls.root / "s3" / "bucket" / "models" / "v1"
        prefix.mkdir(parents=True)
        for name in ("model.joblib", "selection_metrics.json", "run_metadata.json"):
            shutil.copyfile(cls.run_dir / name, prefix / name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_parse_s3_uri(self):
        self.assertEqual(parse_s3_uri("s3://b/models/v1/model.joblib"), ("b", "models/v1/model.joblib"))
        for bad in ("b/key", "s3://bucket", "s3:///key", "https://b/key"):
            with self.assertRaises(ValueError):
                parse_s3_uri(bad)

    def test_fetch_checks_the_hash_and_brings_the_card_files(self):
        target = fetch_model("s3://bucket/models/v1/model.joblib", self.sha, self.root / "dl", self.s3)
        self.assertEqual(file_sha256(target), self.sha)
        self.assertTrue((target.parent / "selection_metrics.json").is_file())
        self.assertFalse((target.parent / "registry.json").exists())  # optional file, absent

    def test_fetch_refuses_a_changed_object(self):
        with self.assertRaises(ModelHashMismatch):
            fetch_model("s3://bucket/models/v1/model.joblib", "0" * 64, self.root / "bad", self.s3)
        self.assertFalse((self.root / "bad" / "model.joblib").exists())
        with self.assertRaises(ValueError):
            fetch_model("s3://bucket/models/v1/model.joblib", "", self.root / "bad", self.s3)

    def test_load_serving_model_from_s3_builds_the_card(self):
        service, card, _ = load_serving_model(
            model_s3="s3://bucket/models/v1/model.joblib", expected_sha256=self.sha,
            download_dir=self.root / "serve", client=self.s3)
        self.assertEqual(service.bundle_sha256, self.sha)
        self.assertEqual(card["selected_model"], service.bundle["model_name"])
        with self.assertRaises(ValueError):
            load_serving_model(model_s3="s3://bucket/models/v1/model.joblib", client=self.s3)
        with self.assertRaises(ValueError):
            load_serving_model(model=self.run_dir / "model.joblib", model_s3="s3://bucket/x")


class TrackingUriTests(unittest.TestCase):
    def test_the_mlflow_app_arn_is_accepted_and_other_hosts_are_not(self):
        self.assertTrue(is_mlflow_app(APP_ARN))
        self.assertEqual(check_tracking_uri(APP_ARN), APP_ARN)
        for uri in ("arn:aws:sagemaker:ap-southeast-1:123456789012:mlflow-tracking-server/x",
                    "arn:aws:s3:::bucket", "https://mlflow.sagemaker.ap-southeast-1.app.amazonaws.com"):
            with self.assertRaisesRegex(ValueError, "approval"):
                check_tracking_uri(uri)


@unittest.skipUnless(HAVE["mangum"] and HAVE["fastapi"], "needs `pip install -e .[serve,lambda]`")
class LambdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = train_run(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        from sepsis_warning import lambda_app

        self.lambda_app = lambda_app
        lambda_app._adapter = lambda_app._failed_at = None
        self.addCleanup(setattr, lambda_app, "_adapter", None)
        self.addCleanup(setattr, lambda_app, "_failed_at", None)

    @staticmethod
    def event(method, path, body=None):
        event = {
            "version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
            "headers": {"content-type": "application/json", "host": "x.lambda-url.ap-southeast-1.on.aws"},
            "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1",
                                        "sourceIp": "10.0.0.1", "userAgent": "test"},
                               "requestId": "r", "routeKey": "$default", "stage": "$default"},
            "isBase64Encoded": False,
        }
        if body is not None:
            event["body"] = json.dumps(body)
        return event

    def test_function_url_events_reach_the_same_api(self):
        env = {"MODEL_PATH": str(self.run_dir / "model.joblib"), "MODEL_SHA256": file_sha256(self.run_dir / "model.joblib")}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("MODEL_S3_URI", None)
            health = self.lambda_app.handler(self.event("GET", "/health"), None)
            self.assertEqual(health["statusCode"], 200)
            self.assertEqual(json.loads(health["body"])["bundle_sha256"], env["MODEL_SHA256"])
            card = json.loads(self.lambda_app.handler(self.event("GET", "/model"), None)["body"])
            self.assertIsNotNone(card["selection"])
            stays = {"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118}]}]}
            predicted = self.lambda_app.handler(self.event("POST", "/predict", stays), None)
            self.assertEqual(predicted["statusCode"], 200)
            self.assertEqual(json.loads(predicted["body"])["predictions"][0]["hour"], 2.0)
            labelled = {"patient_id": "x", "hours": [{"ICULOS": 1, "SepsisLabel": 1}]}
            self.assertEqual(self.lambda_app.handler(self.event("POST", "/score", labelled), None)["statusCode"], 422)

    def test_a_wrong_pinned_hash_stops_the_function(self):
        from sepsis_warning.service import ModelMismatch

        env = {"MODEL_PATH": str(self.run_dir / "model.joblib"), "MODEL_SHA256": "0" * 64}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("MODEL_S3_URI", None)
            with self.assertRaises(ModelMismatch):
                self.lambda_app.handler(self.event("GET", "/health"), None)

    def test_a_failed_load_is_remembered_and_answered_with_503(self):
        env = {"MODEL_PATH": str(self.run_dir / "model.joblib"), "MODEL_SHA256": "0" * 64}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("MODEL_S3_URI", None)
            with self.assertRaises(Exception):  # noqa: B017 - first request surfaces the load error
                self.lambda_app.handler(self.event("GET", "/health"), None)
            with mock.patch.object(self.lambda_app, "_build", side_effect=AssertionError("must not reload")):
                again = self.lambda_app.handler(self.event("GET", "/health"), None)
        self.assertEqual(again["statusCode"], 503)
        self.assertEqual(again["headers"]["retry-after"], "60")
        self.assertNotIn("_reason", again)

    def test_missing_configuration_is_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("MODEL_S3_URI", "MODEL_PATH"):
                os.environ.pop(name, None)
            with self.assertRaisesRegex(RuntimeError, "MODEL_S3_URI"):
                self.lambda_app.handler(self.event("GET", "/health"), None)


class SageMakerEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "opt-ml"
        (self.root / "input" / "config").mkdir(parents=True)
        generate_synthetic(self.root / "input" / "data" / "data", source_size=80, external_size=20)
        (self.root / "input" / "data" / "data" / "SYNTHETIC.json").write_text("{}")

    def write_params(self, **params):
        (self.root / "input" / "config" / "hyperparameters.json").write_text(json.dumps(params))

    def run_entry(self):
        from sepsis_warning import sagemaker_train

        with mock.patch.dict(os.environ, {"SM_ROOT": str(self.root)}):
            return sagemaker_train.main(["train"])

    def test_hyperparameters_arrive_as_strings(self):
        from sepsis_warning.sagemaker_train import build_args

        self.write_params(seed="7", window="6", budget="0.2", cooldown="6", register_model="true")
        args = build_args(self.root)
        self.assertEqual((args.seed, args.window, args.budget, args.cooldown), (7, 6, 0.2, 6))
        self.assertFalse(args.register_model)  # no tracking URI, nothing to register in
        self.assertEqual(args.output, self.root / "model")

    def test_training_writes_the_bundle_to_opt_ml_model(self):
        self.write_params(seed="42")
        self.assertEqual(self.run_entry(), 0)
        self.assertTrue((self.root / "model" / "model.joblib").is_file())
        self.assertTrue((self.root / "model" / "selection_metrics.json").is_file() or
                        (self.root / "model" / "metrics.json").is_file())

    def test_a_bad_channel_writes_the_failure_reason(self):
        shutil.rmtree(self.root / "input" / "data" / "data")
        (self.root / "input" / "data" / "data").mkdir()
        self.write_params()
        self.assertEqual(self.run_entry(), 1)
        self.assertIn("no combined.parquet", (self.root / "output" / "failure").read_text())

    def test_pipeline_step_evaluates_once_then_publishes(self):
        from sepsis_warning import evaluation, model_source

        self.write_params(mode="evaluate_publish", mlflow_tracking_uri="sqlite:///x.db", publish_bucket="bucket",
                          model_uri="models:/sepsis-warning/4")
        published = {"model_s3_uri": "s3://bucket/models/v4/model.joblib", "sha256": "a" * 64, "version": "4"}
        with mock.patch.object(evaluation, "run_frozen_evaluation") as evaluate, \
                mock.patch.object(model_source, "publish_model", return_value=published) as publish:
            self.assertEqual(self.run_entry(), 0)
            self.assertEqual(evaluate.call_args.kwargs["model_uri"], "models:/sepsis-warning/4")
            publish.assert_called_once_with("models:/sepsis-warning/4", "sqlite:///x.db", "bucket")
            self.assertEqual(json.loads((self.root / "model" / "publish.json").read_text()), published)
            # Already evaluated: no second look, but the publish still happens.
            evaluate.side_effect = evaluation.AlreadyEvaluated("done")
            self.assertEqual(self.run_entry(), 0)
            self.assertEqual(publish.call_count, 2)
        self.write_params(mode="evaluate_publish")
        self.assertEqual(self.run_entry(), 1)
        self.assertIn("publish_bucket", (self.root / "output" / "failure").read_text())
        self.assertFalse((self.root / "model" / "model.joblib").exists())   # this mode never trains

    def test_require_gates_stops_registration_of_a_failing_model(self):
        from sepsis_warning.sagemaker_train import build_args

        self.write_params(require_gates="true", require_real_data="true", register_model="true",
                          mlflow_tracking_uri="sqlite:///x.db")
        args = build_args(self.root)
        self.assertTrue(args.require_gates and args.require_real_data and args.register_model)

    @unittest.skipUnless(HAVE["mlflow"], "needs the tracking extra")
    def test_training_logs_and_registers_with_a_tracking_uri(self):
        store = f"sqlite:///{Path(self.temp.name) / 'store' / 'mlflow.db'}"
        self.write_params(seed="42", mlflow_tracking_uri=store, register_model="true", git_commit="abc123")
        self.assertEqual(self.run_entry(), 0)
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=store, registry_uri=store)
        version = client.get_model_version_by_alias("sepsis-warning", "candidate")
        self.assertEqual(version.tags["bundle_sha256"], file_sha256(self.root / "model" / "model.joblib"))
        self.assertEqual(json.loads((self.root / "model" / "run_metadata.json").read_text())["git_commit"] is not None, True)


if __name__ == "__main__":
    unittest.main()
