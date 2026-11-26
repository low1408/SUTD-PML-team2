"""Evidently monitoring on ECS: the report job and the live UI (the "tools" image).

    python -m sepsis_warning.evidently_job report --bucket B --days 1 --environment sepsis-dev
    python -m sepsis_warning.evidently_job ui --bucket B

``report`` compares the scoring service's prediction log for the last days with the frozen reference of the model that
produced it (``monitoring/evidently/<bundle>/reference.parquet``, built by the ``monitor_reference`` pipeline mode from
hospital A validation). It runs Evidently's data-drift preset on the columns of ``evidently_frames`` (the score, whether
it crossed the threshold or alerted, range warnings and which inputs were missing), then

* writes ``report.html`` and ``metrics.json`` to ``reports/monitoring/<time>/``,
* adds the run to the Evidently workspace in ``monitoring/evidently/workspace/`` (what the UI shows),
* publishes CloudWatch metrics in ``SepsisWarning/Evidently`` (the share of drifted columns, the score's drift value,
  the alert-rate ratio, the number of rows) so an alarm can email the owner.

A fast drift result is a prompt to investigate, not proof that performance dropped: labels arrive later, and the
slower label-based checks (``monitor-replay``) are the ones that speak to performance. Nothing here retrains or
changes the model or its threshold.

``ui`` runs ``evidently ui`` on the workspace and refreshes it from S3 every minute, so a run added by the report job
shows up without restarting the service.
"""

import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from .evidently_frames import COLUMNS, FLAGS, MISSING, NUMERIC, prediction_frame
from .prediction_log import read_prediction_log

NAMESPACE = "SepsisWarning/Evidently"
WORKSPACE_PREFIX = "monitoring/evidently/workspace"
PROJECT_NAME = "sepsis-warning drift"
MIN_ROWS = 200
S3_CLIENT = None   # tests set a stand-in
CLOUDWATCH_CLIENT = None


def _s3(client=None):
    client = client or S3_CLIENT
    if client is None:
        import boto3

        client = boto3.client("s3")
    return client


def extract(snapshot_json: dict) -> dict:
    """The numbers worth keeping from an Evidently snapshot: drifted share and per column value and verdict."""
    summary = {"drifted_columns": None, "drifted_share": None, "columns": {}}
    for metric in snapshot_json.get("metrics", []):
        config = metric.get("config", {})
        kind = str(config.get("type", "")).rsplit(":", 1)[-1]
        if kind == "DriftedColumnsCount":
            value = metric["value"]
            summary["drifted_columns"], summary["drifted_share"] = float(value["count"]), float(value["share"])
        elif kind == "ValueDrift":
            value, threshold, method = float(metric["value"]), float(config.get("threshold", 0.05)), str(config.get("method", ""))
            # p-value tests drift below the threshold; distance measures drift at or above it.
            drifted = value < threshold if "p_value" in method else value >= threshold
            summary["columns"][config["column"]] = {"value": value, "method": method, "threshold": threshold, "drifted": bool(drifted)}
    return summary


def compare(reference: pd.DataFrame, current: pd.DataFrame, output: Path, workspace: Path | None = None,
            run_name: str = "") -> dict:
    """Run the drift preset; write ``report.html`` and ``metrics.json`` into ``output``; optionally add to a workspace."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=FLAGS + MISSING)
    snapshot = Report([DataDriftPreset()]).run(Dataset.from_pandas(current[COLUMNS], data_definition=definition),
                                               Dataset.from_pandas(reference[COLUMNS], data_definition=definition))
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "report.html"))
    summary = extract(json.loads(snapshot.json()))
    reference_rate, live_rate = float(reference["new_alert"].mean()), float(current["new_alert"].mean())
    summary.update({
        "rows": {"reference": int(len(reference)), "current": int(len(current))},
        "alert_rate": {"reference": reference_rate, "current": live_rate,
                       "ratio": (live_rate / reference_rate) if reference_rate > 0 else None},
        "score_drift": summary["columns"].get("probability"),
    })
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if workspace is not None:
        add_to_workspace(workspace, snapshot, run_name)
    return summary


def add_to_workspace(workspace: Path, snapshot, run_name: str = "") -> None:
    from evidently.ui.workspace import Workspace

    workspace.mkdir(parents=True, exist_ok=True)
    ws = Workspace.create(str(workspace))
    found = ws.search_project(PROJECT_NAME)
    project = found[0] if found else ws.create_project(PROJECT_NAME)
    ensure_dashboard(project)
    ws.add_run(project.id, snapshot, name=run_name or None)


def ensure_dashboard(project) -> None:
    """Give the project's Dashboard tab panels. A new project has none and the UI says the dashboard is empty."""
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import counter_panel, line_plot_panel

    if any(tab.panels for tab in project.dashboard.model().tabs):
        return
    drifted = "evidently:metric_v2:DriftedColumnsCount"
    share = PanelMetric(legend="share of columns", metric=drifted, metric_labels={"value_type": "share"})
    count = PanelMetric(legend="columns", metric=drifted, metric_labels={"value_type": "count"})
    project.dashboard.add_panel(counter_panel(title="Share of drifted columns (latest run)", size="half", values=[share],
                                              aggregation="last"))
    project.dashboard.add_panel(counter_panel(title="Drifted columns (latest run)", size="half", values=[count], aggregation="last"))
    project.dashboard.add_panel(line_plot_panel(title="Share of drifted columns over runs", size="full", values=[share]))
    project.save()


def cloudwatch_metrics(summary: dict, environment: str) -> list[dict]:
    """The metrics an alarm can use. Nothing is published for a window with too few rows to judge."""
    data = [{"MetricName": "Rows", "Value": float(summary["rows"]["current"]), "Unit": "Count"}]
    if summary["rows"]["current"] < MIN_ROWS or summary["drifted_share"] is None:
        return [dict(item, Dimensions=[{"Name": "Environment", "Value": environment}]) for item in data]
    data.append({"MetricName": "DriftedColumnsShare", "Value": float(summary["drifted_share"]), "Unit": "None"})
    score = summary.get("score_drift")
    if score is not None:
        data.append({"MetricName": "ScoreDriftValue", "Value": float(score["value"]), "Unit": "None"})
        data.append({"MetricName": "ScoreDrifted", "Value": 1.0 if score["drifted"] else 0.0, "Unit": "None"})
    ratio = summary["alert_rate"]["ratio"]
    if ratio is not None:
        data.append({"MetricName": "AlertRateRatio", "Value": float(ratio), "Unit": "None"})
    return [dict(item, Dimensions=[{"Name": "Environment", "Value": environment}]) for item in data]


def publish_metrics(summary: dict, environment: str, client=None) -> list[dict]:
    items = cloudwatch_metrics(summary, environment)
    client = client or CLOUDWATCH_CLIENT
    if client is None:
        import boto3

        client = boto3.client("cloudwatch")
    client.put_metric_data(Namespace=NAMESPACE, MetricData=items)
    return items


# ---------------------------------------------------------------- S3 plumbing
def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys += [item["Key"] for item in page.get("Contents", []) if not item["Key"].endswith("/")]
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")


def read_log_window(s3, bucket: str, days: int, now: datetime.datetime, prefix: str = "prediction-log") -> pd.DataFrame:
    chunks = []
    for back in range(days):
        day = now - datetime.timedelta(days=back)
        for key in list_keys(s3, bucket, f"{prefix}/{day:%Y/%m/%d}/"):
            chunks.append(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    return read_prediction_log(chunks) if chunks else pd.DataFrame()


def serving_bundle(records: pd.DataFrame) -> str | None:
    """The bundle the log mostly comes from (first 12 hex characters), so the matching reference is used."""
    if records.empty or "bundle_sha256" not in records:
        return None
    counts = Counter(records["bundle_sha256"].dropna().astype(str))
    return counts.most_common(1)[0][0] if counts else None


def load_reference(s3, bucket: str, bundle: str) -> pd.DataFrame:
    key = f"monitoring/evidently/{bundle[:12]}/reference.parquet"
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "reference.parquet"
        try:
            s3.download_file(bucket, key, str(target))
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"no Evidently reference for bundle {bundle[:12]} at s3://{bucket}/{key}; build it with the "
                             "monitor_reference pipeline mode (aws_deploy.sh monitor-reference) for the deployed model") from error
        return pd.read_parquet(target)


def sync_down(s3, bucket: str, prefix: str, target: Path) -> int:
    count = 0
    for key in list_keys(s3, bucket, prefix.rstrip("/") + "/"):
        destination = target / key[len(prefix.rstrip("/")) + 1:]
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(destination))
        count += 1
    return count


def sync_up(s3, bucket: str, prefix: str, source: Path) -> int:
    count = 0
    for path in sorted(Path(source).rglob("*")):
        if path.is_file():
            s3.upload_file(str(path), bucket, f"{prefix.rstrip('/')}/{path.relative_to(source).as_posix()}")
            count += 1
    return count


def report(bucket: str, days: int, environment: str, now: datetime.datetime | None = None, s3=None, cloudwatch=None,
           out=print) -> dict:
    s3 = _s3(s3)
    now = now or datetime.datetime.now(datetime.UTC)
    records = read_log_window(s3, bucket, days, now)
    bundle = serving_bundle(records)
    if bundle is None:
        raise ValueError(f"no prediction records in s3://{bucket}/prediction-log/ for the last {days} day(s)")
    current = prediction_frame(records)
    reference = load_reference(s3, bucket, bundle)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory(prefix="sepsis-evidently-") as temp:
        temp = Path(temp)
        workspace = temp / "workspace"
        sync_down(s3, bucket, WORKSPACE_PREFIX, workspace)        # add to the existing history, not a fresh one
        summary = compare(reference, current, temp / "report", workspace, run_name=f"{stamp} bundle {bundle[:12]}")
        summary["bundle"] = bundle[:12]
        (temp / "report" / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        sync_up(s3, bucket, f"reports/monitoring/{stamp}", temp / "report")
        sync_up(s3, bucket, WORKSPACE_PREFIX, workspace)
    published = publish_metrics(summary, environment, cloudwatch)
    out(f"Evidently report: s3://{bucket}/reports/monitoring/{stamp}/report.html ({summary['rows']['current']} rows, "
        f"{summary['drifted_share']} of columns drifted); published {len(published)} metric(s)")
    if summary["rows"]["current"] < MIN_ROWS:
        out(f"Only {summary['rows']['current']} rows: below {MIN_ROWS}, so no drift metrics were published (do not read it as ok).")
    return summary


def tree_signature(root: Path) -> str:
    """A fingerprint of every file under ``root`` (path and content), to see whether a refresh changed anything."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ui(bucket: str, port: int = 8000, every: int = 60, s3=None, popen=subprocess.Popen, sleep=None, stop=lambda: False) -> int:
    """Serve the workspace and keep it in step with S3 until the task is stopped.

    The Evidently UI reads its projects when it starts and may not notice files that appear later, so when a refresh
    changes the workspace (the first report run, or a new run) the UI process is restarted on the new files. Without this
    the page stays on "no projects" although the workspace in S3 holds the report.
    """
    import time

    s3, sleep = _s3(s3), sleep or time.sleep
    workspace = Path(tempfile.mkdtemp(prefix="evidently-workspace-"))
    command = ["evidently", "ui", "--workspace", str(workspace), "--host", "0.0.0.0", "--port", str(port)]
    count = sync_down(s3, bucket, WORKSPACE_PREFIX, workspace)
    seen = tree_signature(workspace)
    print(f"workspace: {count} file(s) from s3://{bucket}/{WORKSPACE_PREFIX}/", flush=True)
    process = popen(command)
    signal.signal(signal.SIGTERM, lambda *_: process.terminate())
    while process.poll() is None and not stop():
        sleep(every)
        try:
            count = sync_down(s3, bucket, WORKSPACE_PREFIX, workspace)
        except Exception as error:  # noqa: BLE001 - keep serving what is there
            print(f"workspace refresh failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            continue
        current = tree_signature(workspace)
        if current != seen:
            seen = current
            print(f"workspace changed ({count} file(s)): restarting the UI so it loads the new project", flush=True)
            process.terminate()
            process.wait()
            process = popen(command)
    if process.poll() is None:
        process.terminate()
    return process.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("report", help="compare the last days of the prediction log with the frozen reference")
    run.add_argument("--bucket", default=os.environ.get("ARTIFACT_BUCKET"))
    run.add_argument("--days", type=int, default=int(os.environ.get("REPORT_DAYS", "1")))
    run.add_argument("--environment", default=os.environ.get("ENVIRONMENT_NAME", "sepsis-dev"))
    serve = sub.add_parser("ui", help="serve the Evidently workspace kept in S3")
    serve.add_argument("--bucket", default=os.environ.get("ARTIFACT_BUCKET"))
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not args.bucket:
        print("a bucket is required (--bucket or ARTIFACT_BUCKET)", file=sys.stderr)
        return 2
    try:
        if args.command == "report":
            report(args.bucket, args.days, args.environment)
            return 0
        return ui(args.bucket, args.port)
    except ValueError as error:
        print(f"evidently_job: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
