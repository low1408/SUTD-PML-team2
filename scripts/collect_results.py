#!/usr/bin/env python3
"""Pull the aggregate results of a finished pipeline run from S3 into the repository folders the report uses.

    python scripts/collect_results.py --stack sepsis-dev --region ap-southeast-2 [--execution <id>] [--model-version N] [--dry-run]

What it copies (an allowlist: anything else is skipped, with the reason):

    reports/<execution>/            tables and figures of the training report        -> reports/results/
    training-output/<execution>/    the run's aggregate files (selection, tuning)    -> reports/results/run/, reports/results/tuning/
    training-output/<execution>/    evaluation_metrics.json (the one frozen test)    -> reports/results_test/
    models/v<N>/registry.json       who approved the model, when, the frozen eval    -> reports/results/registry.json
    reports/monitoring/<latest>/    metrics.json of the newest drift report          -> reports/results/monitoring/

What it never copies: the model (``model.joblib``), the ward replays (``reports/demo/``, ``replay.html``), the prepared data,
per-patient files (``held_out_predictions.csv``, ``split_manifest.csv``), the failure-example figure (individual stays; opt in with
``--include-failure-examples``) and the Evidently ``report.html``. The same rule as ``tracking.py``: aggregate outputs only.

Text files have the artifact bucket name and the 12-digit AWS account id replaced by placeholders. A ``PROVENANCE.md`` records the
execution, the model version, the commit, and the SHA-256 and size of every file, plus what was skipped. Needs boto3.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SELECTION_FILES = {"selection_metrics.json", "validation_models.csv", "run_metadata.json", "candidate_details.json",
                   "manifest.json", "config.json", "extreme_value_review_train.csv"}
TUNING_FILES = {"tuning_results.csv", "tuning_best.json", "tuned_models.toml"}
REPORT_FILES = {"metrics_table.csv", "metrics_table.md", "subgroup_table.csv", "subgroup_table.md", "feature_importance.csv",
                "feature_importance.md", "environment.json", "figures_manifest.json", "FIGURES.md"}
FAILURE_FIGURE = "fig_failure_examples.png"
NEVER = ("model.joblib", ".joblib", ".pkl", ".pickle", ".parquet", ".psv", "replay.html", "held_out", "split_manifest", "ward-",
         "report.html", "/demo/")
TEXT_SUFFIXES = {".json", ".md", ".csv", ".toml"}
MAX_FILE_BYTES = 3_000_000
MAX_TOTAL_BYTES = 15_000_000
EXECUTION = re.compile(r"^[A-Za-z0-9._-]{8,80}$")


def classify(key: str, execution: str, *, include_failure: bool = False) -> tuple[str | None, str]:
    """(destination relative to the repository, reason). Destination None = skipped, and the reason says why."""
    name = key.rsplit("/", 1)[-1]
    if not name:
        return None, "a folder"
    lowered = key.lower()
    for fragment in NEVER:
        if fragment.lower() in lowered:
            return None, f"never copied ({fragment.strip('/')}: patient-level, executable or a replay)"
    if key.startswith(f"reports/{execution}/") and key.count("/") == 2:
        if name in REPORT_FILES:
            return f"reports/results/{name}", "report table"
        if name == FAILURE_FIGURE:
            return (f"reports/results/{name}", "figure") if include_failure else (None, "individual stays: opt in with --include-failure-examples")
        if name.startswith("fig_") and name.endswith(".png"):
            return f"reports/results/{name}", "figure"
        return None, "not on the report allowlist"
    if key.startswith(f"training-output/{execution}/"):
        if name in SELECTION_FILES and "/tuning/" not in key and "/publish/" not in key:
            return f"reports/results/run/{name}", "selection output"
        if name in TUNING_FILES and "/tuning/" in key:
            return f"reports/results/tuning/{name}", "tuning output"
        if name == "evaluation_metrics.json" and "/publish/" in key:
            return "reports/results_test/evaluation_metrics.json", "frozen test evaluation (aggregates)"
        return None, "not on the allowlist"
    return None, "outside the run"


def plan(keys: list[str], execution: str, *, include_failure: bool = False) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """({destination: key}, [(key, why skipped)]). When two keys map to the same destination the shorter path wins."""
    chosen: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []
    for key in sorted(keys):
        destination, reason = classify(key, execution, include_failure=include_failure)
        if destination is None:
            if reason != "a folder":
                skipped.append((key, reason))
        elif destination in chosen:
            skipped.append((key, f"duplicate of {chosen[destination]}"))
        else:
            chosen[destination] = key
    return chosen, skipped


def scrub(text: str, bucket: str) -> str:
    """Placeholders for the artifact bucket name and any 12-digit AWS account id."""
    if bucket:
        text = text.replace(bucket, "<artifact-bucket>")
    return re.sub(r"(?<!\d)\d{12}(?!\d)", "<aws-account>", text)


def latest_execution(stepfunctions, machine_arn: str) -> dict:
    found = stepfunctions.list_executions(stateMachineArn=machine_arn, statusFilter="SUCCEEDED", maxResults=1).get("executions", [])
    if not found:
        raise ValueError("no pipeline run has SUCCEEDED yet: finish the pipeline (promote, approve) first")
    return found[0]


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, **({"ContinuationToken": token} if token else {})}
        page = s3.list_objects_v2(**kwargs)
        keys += [item["Key"] for item in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")


def newest_monitoring_metrics(s3, bucket: str) -> str | None:
    keys = [k for k in list_keys(s3, bucket, "reports/monitoring/") if k.endswith("/metrics.json")]
    return max(keys) if keys else None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance(*, execution: str, version, registry: dict, files: list[dict], skipped: list[tuple[str, str]], collected_at: str,
               source: dict) -> str:
    lines = ["# Results from AWS", "",
             "Collected by the `collect-results` workflow from the private artifact bucket. Aggregate outputs only: no model, no patient-level "
             "files, no ward replays.", "",
             "| | |", "|---|---|",
             f"| Pipeline run | `{execution}` |", f"| Model version | {version if version is not None else 'not recorded'} |",
             f"| Collected | {collected_at} |"]
    for label, key in (("Commit", "commit"), ("Workflow run", "run_url")):
        if source.get(key):
            lines.append(f"| {label} | {source[key]} |")
    for label, key in (("Approved by", "approved_by"), ("Reviewed by", "reviewed_by"), ("Frozen evaluation at", "frozen_evaluated_at"),
                       ("Bundle SHA-256", "sha256")):
        if registry.get(key):
            lines.append(f"| {label} | `{registry[key]}` |")
    lines += ["", "## Files", "", "| File | Size (bytes) | SHA-256 |", "|---|---:|---|"]
    lines += [f"| `{f['path']}` | {f['bytes']} | `{f['sha256'][:16]}…` |" for f in files]
    if skipped:
        lines += ["", "## Skipped on purpose", "", "| S3 key | Why |", "|---|---|"]
        lines += [f"| `{key}` | {why} |" for key, why in skipped]
    return "\n".join(lines) + "\n"


def collect(session, stack: str, *, root: Path, execution: str | None = None, model_version: int | None = None,
            include_monitoring: bool = True, include_failure: bool = False, dry_run: bool = False, source: dict | None = None,
            out=print) -> dict:
    from aws_published_model import find, versions

    cloudformation, s3 = session.client("cloudformation"), session.client("s3")
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
    bucket = outputs.get("ArtifactBucketName")
    if not bucket or not outputs.get("PipelineArn"):
        raise ValueError("the stack has no artifact bucket or pipeline yet")
    if execution is None:
        execution = latest_execution(session.client("stepfunctions"), outputs["PipelineArn"])["name"]
    if not EXECUTION.match(execution):
        raise ValueError(f"execution {execution!r} does not look like a run id")
    out(f"Run {execution}")

    keys = list_keys(s3, bucket, f"reports/{execution}/") + list_keys(s3, bucket, f"training-output/{execution}/")
    chosen, skipped = plan(keys, execution, include_failure=include_failure)

    published = versions(s3, bucket)
    version = model_version if model_version is not None else (max(published) if published else None)
    registry: dict = {}
    if version is not None and version in published:
        registry = find(s3, bucket, version).get("registry") or {}
        chosen["reports/results/registry.json"] = f"models/v{version}/registry.json"
    if include_monitoring:
        newest = newest_monitoring_metrics(s3, bucket)
        if newest:
            chosen["reports/results/monitoring/metrics.json"] = newest
        else:
            skipped.append(("reports/monitoring/", "no drift report yet (run monitor first)"))
    if not any(d.startswith("reports/results_test/") for d in chosen):
        out("WARNING: no evaluation_metrics.json found: the frozen evaluation has not finished for this run")

    files, total = [], 0
    for destination, key in sorted(chosen.items()):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if len(body) > MAX_FILE_BYTES:
            skipped.append((key, f"larger than {MAX_FILE_BYTES} bytes"))
            continue
        if Path(destination).suffix in TEXT_SUFFIXES:
            body = scrub(body.decode("utf-8"), bucket).encode("utf-8")
        total += len(body)
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"more than {MAX_TOTAL_BYTES} bytes in total: something other than aggregate results was selected")
        files.append({"path": destination, "bytes": len(body), "sha256": sha256(body), "body": body})
    for item in files:
        out(f"  {item['path']}  ({item['bytes']} bytes)")
    for key, why in skipped:
        out(f"  skipped {key}: {why}")
    if not files:
        raise ValueError("nothing to collect for this run")
    if not dry_run:
        for item in files:
            target = root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item["body"])
        page = provenance(execution=execution, version=version, registry=json.loads(scrub(json.dumps(registry), bucket)), files=[
            {k: v for k, v in f.items() if k != "body"} for f in files], skipped=[(scrub(k, bucket), w) for k, w in skipped],
            collected_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), source=source or {})
        (root / "reports" / "results" / "PROVENANCE.md").write_text(page, encoding="utf-8")
    return {"execution": execution, "version": version, "files": [f["path"] for f in files], "skipped": len(skipped)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--execution", default="", help="pipeline run id (default: the latest run that SUCCEEDED)")
    parser.add_argument("--model-version", default="", help="published model version (default: the newest)")
    parser.add_argument("--no-monitoring", action="store_true")
    parser.add_argument("--include-failure-examples", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied; write nothing")
    parser.add_argument("--root", default=".", help="repository root to write into")
    args = parser.parse_args(argv)
    if args.model_version and not args.model_version.isdigit():
        print("FAIL: --model-version must be a whole number", file=sys.stderr)
        return 1
    import boto3

    server, repo, run_id = os.environ.get("GITHUB_SERVER_URL", ""), os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_RUN_ID", "")
    source = {"commit": os.environ.get("GITHUB_SHA", ""), "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else ""}
    try:
        result = collect(boto3.Session(region_name=args.region), args.stack, root=Path(args.root), execution=args.execution or None,
                         model_version=int(args.model_version) if args.model_version else None,
                         include_monitoring=not args.no_monitoring, include_failure=args.include_failure_examples,
                         dry_run=args.dry_run, source=source)
    except (ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Collected {len(result['files'])} files from run `{result['execution']}`"
                         f"{' (dry run: nothing written)' if args.dry_run else ''}\n")
    out_file = os.environ.get("GITHUB_OUTPUT")
    if out_file:
        with open(out_file, "a", encoding="utf-8") as handle:
            handle.write(f"execution={result['execution']}\nversion={result['version'] if result['version'] is not None else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
