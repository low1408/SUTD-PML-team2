#!/usr/bin/env python3
"""Start a SageMaker AI training job for `train` and log it to the serverless MLflow App.

    python scripts/sagemaker_train.py --stack sepsis-research --image-tag training-<git sha> --wait

The job runs the `training` image (Dockerfile target) on one instance, reads
`s3://<bucket>/data/physionet/` (combined.parquet, splits.csv and its summary),
writes the run to `s3://<bucket>/training-output/<job>/output/model.tar.gz`,
logs the selection run to the MLflow App and registers the selected bundle as
`@candidate`. Nothing runs, and nothing is billed, between jobs.

Cost controls: managed spot capacity by default (usually far cheaper than
on-demand; the job waits for capacity up to --max-wait), a hard --max-runtime,
one instance, encrypted volume and output with the stack's KMS key.

Bucket, role, image repository, KMS key and MLflow App ARN come from the
CloudFormation stack outputs (`--stack`), or can be given directly.
`--dry-run` prints the request without calling AWS. Needs boto3 (aws extra).
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

OUTPUT_KEYS = {
    "ArtifactBucketName": "bucket",
    "SageMakerExecutionRoleArn": "role_arn",
    "TrainingRepositoryUri": "repository_uri",
    "ArtifactKeyArn": "kms_key_arn",
    "MlflowAppArn": "mlflow_app_arn",
}


def stack_outputs(stack: str, region: str | None) -> dict:
    import boto3

    cloudformation = boto3.client("cloudformation", region_name=region)
    outputs = cloudformation.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    found = {OUTPUT_KEYS[o["OutputKey"]]: o["OutputValue"] for o in outputs if o["OutputKey"] in OUTPUT_KEYS}
    missing = sorted(set(OUTPUT_KEYS.values()) - set(found))
    if missing:
        raise SystemExit(f"stack {stack} is missing outputs for {missing}; deploy it with EnableTraining=true")
    return found


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_request(args, config: dict) -> dict:
    job = args.job_name or f"sepsis-train-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    bucket = config["bucket"]
    stopping = {"MaxRuntimeInSeconds": args.max_runtime}
    if args.spot:
        # Must be larger than MaxRuntimeInSeconds (managed spot training guide).
        stopping["MaxWaitTimeInSeconds"] = max(args.max_wait, args.max_runtime + 600)
    hyperparameters = {
        "seed": str(args.seed), "window": str(args.window), "budget": str(args.budget),
        "cooldown": str(args.cooldown), "register_model": "true" if args.register else "false",
        "git_commit": args.git_commit if args.git_commit is not None else git_commit(),
    }
    if config.get("mlflow_app_arn") and not args.no_mlflow:
        hyperparameters["mlflow_tracking_uri"] = config["mlflow_app_arn"]
    return {
        "TrainingJobName": job,
        "AlgorithmSpecification": {
            "TrainingImage": f"{config['repository_uri']}:{args.image_tag}",
            "TrainingInputMode": "File",
        },
        "RoleArn": config["role_arn"],
        "HyperParameters": hyperparameters,
        "InputDataConfig": [{
            "ChannelName": "data",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{bucket}/{args.data_prefix.strip('/')}/",
                "S3DataDistributionType": "FullyReplicated",
            }},
            "InputMode": "File",
        }],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/training-output/", "KmsKeyId": config["kms_key_arn"]},
        "ResourceConfig": {
            "InstanceType": args.instance_type, "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_gb, "VolumeKmsKeyId": config["kms_key_arn"],
        },
        "StoppingCondition": stopping,
        "EnableManagedSpotTraining": bool(args.spot),
        "Environment": {"MLFLOW_DISABLE_TELEMETRY": "true", "MLFLOW_DISABLE_AGENT_HINT": "1"},
        "Tags": [{"Key": "Project", "Value": "sepsis-warning-research"}],
    }


def wait(client, job: str) -> int:
    last = None
    while True:
        described = client.describe_training_job(TrainingJobName=job)
        status = (described["TrainingJobStatus"], described.get("SecondaryStatus"))
        if status != last:
            print(f"{datetime.now(UTC):%H:%M:%S} {status[0]} / {status[1]}", flush=True)
            last = status
        if status[0] in ("Completed", "Failed", "Stopped"):
            if status[0] == "Completed":
                billable = described.get("BillableTimeInSeconds")
                total = described.get("TrainingTimeInSeconds")
                print(f"Model: {described['ModelArtifacts']['S3ModelArtifacts']}")
                print(f"Billable seconds: {billable} of {total} (spot savings apply to the difference)")
                return 0
            print(f"Reason: {described.get('FailureReason', 'not given')}", file=sys.stderr)
            return 1
        time.sleep(30)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", default="sepsis-research", help="CloudFormation stack name")
    parser.add_argument("--region", default=None)
    parser.add_argument("--image-tag", required=True, help="Tag of the training image pushed to TrainingRepositoryUri")
    parser.add_argument("--data-prefix", default="data/physionet", help="S3 prefix (in the artifact bucket) of the data")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--volume-gb", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=3600, help="Hard stop in seconds")
    parser.add_argument("--max-wait", type=int, default=7200, help="Spot: seconds to wait for capacity plus runtime")
    parser.add_argument("--on-demand", dest="spot", action="store_false", help="Do not use spot capacity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--cooldown", type=int, default=6)
    parser.add_argument("--no-register", dest="register", action="store_false", help="Log the run but do not register")
    parser.add_argument("--no-mlflow", action="store_true", help="Train without logging to the MLflow App")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--wait", action="store_true", help="Follow the job until it ends")
    parser.add_argument("--dry-run", action="store_true", help="Print the request and stop")
    for key in sorted(set(OUTPUT_KEYS.values())):
        parser.add_argument(f"--{key.replace('_', '-')}", default=None, help="Overrides the stack output")
    args = parser.parse_args(argv)

    given = {key: getattr(args, key) for key in OUTPUT_KEYS.values() if getattr(args, key)}
    config = given if len(given) == len(OUTPUT_KEYS) else {**stack_outputs(args.stack, args.region), **given}
    request = build_request(args, config)
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    import boto3

    client = boto3.client("sagemaker", region_name=args.region)
    client.create_training_job(**request)
    print(f"Started {request['TrainingJobName']} on {args.instance_type}"
          f" ({'spot' if args.spot else 'on-demand'}, max {args.max_runtime} s)", flush=True)
    return wait(client, request["TrainingJobName"]) if args.wait else 0


if __name__ == "__main__":
    sys.exit(main())
