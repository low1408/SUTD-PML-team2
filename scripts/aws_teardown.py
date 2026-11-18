#!/usr/bin/env python3
"""Tear a deployment down completely, including the resources the stack keeps on purpose.

    python scripts/aws_teardown.py --stack sepsis-dev --region ap-southeast-2            # plan only
    python scripts/aws_teardown.py --stack sepsis-dev --region ap-southeast-2 --yes --confirm sepsis-dev

Deleting the CloudFormation stack leaves the artifact and log buckets, the KMS key, the ECR
repositories, the DynamoDB table and the log groups (their DeletionPolicy is Retain, so a
mistaken stack delete cannot lose a model or data). This script lists them first, then deletes
the stack and removes those leftovers:

* S3 buckets: every object version and delete marker is removed, then the bucket (this includes
  every published model and training output in the artifact bucket, and the template bucket
  made by `aws_deploy.sh stack`)
* ECR repositories (with their images), the DynamoDB table, the log groups
* the KMS key is scheduled for deletion after 7 days (the shortest AWS allows)

Without ``--yes`` it only prints the plan. With ``--yes`` the stack name must be repeated in
``--confirm``. It never touches anything that is not part of the named stack, plus the
``<stack>-<account>-<region>-cfn`` template bucket. Needs boto3 (aws extra). This cannot be undone.
"""

import argparse
import sys

RETAINED_TYPES = ("AWS::S3::Bucket", "AWS::ECR::Repository", "AWS::DynamoDB::Table", "AWS::Logs::LogGroup",
                  "AWS::KMS::Key")
ORDER = {kind: index for index, kind in enumerate(RETAINED_TYPES)}


def collect(cloudformation, stack: str) -> list[tuple[str, str]]:
    """(type, physical id) of every retained-type resource in the stack and its nested stacks."""
    found: list[tuple[str, str]] = []
    token = None
    while True:
        page = cloudformation.list_stack_resources(StackName=stack, **({"NextToken": token} if token else {}))
        for item in page["StackResourceSummaries"]:
            kind, physical = item["ResourceType"], item.get("PhysicalResourceId")
            if not physical or item["ResourceStatus"] == "DELETE_COMPLETE":
                continue
            if kind == "AWS::CloudFormation::Stack":
                found += collect(cloudformation, physical)
            elif kind in RETAINED_TYPES:
                found.append((kind, physical))
        token = page.get("NextToken")
        if not token:
            return sorted(set(found), key=lambda pair: (ORDER[pair[0]], pair[1]))


def empty_bucket(s3, bucket: str) -> int:
    """Delete every object version and delete marker. Returns how many were removed."""
    removed = 0
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket):
        batch = [{"Key": v["Key"], "VersionId": v["VersionId"]}
                 for v in page.get("Versions", []) + page.get("DeleteMarkers", [])]
        for start in range(0, len(batch), 1000):
            chunk = batch[start:start + 1000]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})
            removed += len(chunk)
    return removed


def _missing(error) -> bool:
    code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
    return code in {"NoSuchBucket", "404", "NotFound", "RepositoryNotFoundException", "ResourceNotFoundException",
                    "NotFoundException", "NoSuchEntity"}


def purge(clients: dict, resources: list[tuple[str, str]], log=print) -> None:
    """Delete the retained resources. Missing ones are skipped, so a rerun finishes the job."""
    s3, ecr, dynamodb, logs, kms = (clients[k] for k in ("s3", "ecr", "dynamodb", "logs", "kms"))
    for kind, name in resources:
        try:
            if kind == "AWS::S3::Bucket":
                log(f"  bucket {name}: removed {empty_bucket(s3, name)} object versions")
                s3.delete_bucket(Bucket=name)
            elif kind == "AWS::ECR::Repository":
                ecr.delete_repository(repositoryName=name, force=True)
            elif kind == "AWS::DynamoDB::Table":
                try:
                    dynamodb.update_table(TableName=name, DeletionProtectionEnabled=False)
                except Exception:  # noqa: BLE001  protection may already be off; delete reports real problems
                    pass
                dynamodb.delete_table(TableName=name)
            elif kind == "AWS::Logs::LogGroup":
                logs.delete_log_group(logGroupName=name)
            elif kind == "AWS::KMS::Key":
                kms.schedule_key_deletion(KeyId=name, PendingWindowInDays=7)
            log(f"  deleted {kind.split('::')[-1]} {name}")
        except Exception as error:  # noqa: BLE001  botocore ClientError and friends
            if _missing(error):
                log(f"  already gone: {name}")
            else:
                raise


def run(args, session=None) -> int:
    import boto3

    session = session or boto3.Session(region_name=args.region)
    cloudformation = session.client("cloudformation")
    clients = {name: session.client(name) for name in ("s3", "ecr", "dynamodb", "logs", "kms")}
    try:
        resources = collect(cloudformation, args.stack)
    except Exception as error:  # noqa: BLE001  botocore ClientError
        if "does not exist" in str(error):
            print(f"Stack {args.stack} does not exist in {args.region}: nothing to tear down (credentials work).")
            return 0
        raise
    if args.templates_bucket:
        resources.append(("AWS::S3::Bucket", args.templates_bucket))
    print(f"Stack {args.stack} in {args.region} and these leftovers would be removed:")
    for kind, name in resources:
        print(f"  {kind.split('::')[-1]:<12} {name}")
    if not args.yes:
        print("Plan only. Add --yes --confirm <stack name> to delete all of it (cannot be undone).")
        return 0
    if args.confirm != args.stack:
        print(f"--confirm must be exactly the stack name ({args.stack}); nothing was deleted.", file=sys.stderr)
        return 2
    print("Deleting the stack...")
    cloudformation.delete_stack(StackName=args.stack)
    cloudformation.get_waiter("stack_delete_complete").wait(StackName=args.stack)
    print("Removing retained resources...")
    purge(clients, resources)
    print("Done. The KMS key is scheduled for deletion in 7 days; the stack and its data are gone.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--templates-bucket", default=None, help="Bucket made by `aws_deploy.sh stack` for templates")
    parser.add_argument("--yes", action="store_true", help="Actually delete (default is a plan only)")
    parser.add_argument("--confirm", default="", help="Repeat the stack name to confirm")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
