"""Scheduled drift job for AWS Lambda (same image as the scoring function, other handler).

Reads one day of the prediction log from S3, runs ``live_drift.live_checks``
against the frozen monitoring reference, prints the result as a CloudWatch
metric line and stores it next to the log. A CloudWatch alarm on
``DriftLevel`` notifies the owners; nothing is retrained or changed here.

Environment: ``PREDICTION_LOG_S3`` (``s3://bucket/prediction-log/``),
``REFERENCE_S3`` (``s3://bucket/monitoring/reference.json``), ``ENVIRONMENT_NAME``.
The event may carry ``{"date": "YYYY-MM-DD"}``; the default is yesterday (UTC).
"""

import datetime
import json
import os

from .live_drift import emf_record, live_checks, read_prediction_log
from .model_source import parse_s3_uri


def handler(event, context, client=None):
    if client is None:
        import boto3

        client = boto3.client("s3")
    day = (event or {}).get("date") or (datetime.datetime.now(datetime.UTC).date()
                                         - datetime.timedelta(days=1)).isoformat()
    bucket, prefix = parse_s3_uri(os.environ["PREDICTION_LOG_S3"].rstrip("/") + "/")
    reference_bucket, reference_key = parse_s3_uri(os.environ["REFERENCE_S3"])
    reference = json.loads(client.get_object(Bucket=reference_bucket, Key=reference_key)["Body"].read())

    day_prefix = f"{prefix}{day.replace('-', '/')}/"     # Firehose writes yyyy/MM/dd/HH/
    chunks, token = [], None
    while True:
        page = client.list_objects_v2(Bucket=bucket, Prefix=day_prefix,
                                      **({"ContinuationToken": token} if token else {}))
        for item in page.get("Contents", []):
            chunks.append(client.get_object(Bucket=bucket, Key=item["Key"])["Body"].read())
        token = page.get("NextContinuationToken")
        if not token:
            break
    result = live_checks(read_prediction_log(chunks), reference)
    result["date"] = day
    print(json.dumps(emf_record(result, os.environ.get("ENVIRONMENT_NAME", "lambda")), allow_nan=False), flush=True)
    client.put_object(Bucket=bucket, Key=f"drift-results/{day}.json",
                      Body=json.dumps(result, indent=2, allow_nan=False).encode())
    return {"date": day, "status": result["status"], "rows": result["rows"]}
