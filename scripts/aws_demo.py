#!/usr/bin/env python3
"""Replay real stays through the deployed scoring endpoint and keep the ward page in S3 (the predict demo).

    python scripts/aws_demo.py --stack sepsis-dev --region ap-southeast-2 --split validation

What it does, in order:

1. calls the deployed function URL (SigV4-signed, the way a hospital system would) for ``/health`` and reports which
   model is serving;
2. for a held-out split (``internal_test`` or ``external_test``, i.e. hospital B) it reads the published model's
   ``registry.json`` and refuses unless that model has a recorded frozen evaluation, so test data is only looked at
   after the model was frozen, and passes ``--after-freeze`` to the replay itself;
3. downloads the prepared PhysioNet data from the artifact bucket, sends the stays hour by hour to ``POST /score``
   and writes the ward replay page (alerts against the recorded onset);
4. uploads the page to ``s3://<bucket>/reports/demo/<time>/`` and prints where it is.

The page holds patient-level trajectories from PhysioNet, so it stays in the private encrypted bucket. It is
never written into the repository and never uploaded as a workflow artifact.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HELD_OUT = ("internal_test", "external_test")
SPLITS = ("train", "calibration", "validation") + HELD_OUT


def require_frozen(split: str, registry: dict | None) -> None:
    """A held-out split may only be replayed for a model with a recorded frozen evaluation."""
    if split not in HELD_OUT:
        return
    if not registry or not registry.get("frozen_evaluated_at"):
        raise ValueError(
            f"refusing to replay {split}: the serving model has no recorded frozen evaluation. Approve it in the pipeline "
            "(the one evaluation of the test splits runs after the approval) before showing test data")


def ward_argv(*, api_url: str, data: Path, splits: Path, split: str, beds: int, seed: int, output: Path) -> list[str]:
    argv = ["ward", "--api-url", api_url, "--sign", "--data", str(data), "--splits", str(splits), "--split", split,
            "--beds", str(beds), "--seed", str(seed), "--output", str(output)]
    if split in HELD_OUT:
        argv.append("--after-freeze")
    return argv


def report_key(now: datetime.datetime, split: str) -> str:
    return f"reports/demo/{now.strftime('%Y%m%dT%H%M%SZ')}/ward-{split}.html"


def read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as error:  # noqa: BLE001 - a missing file is an answer, not a crash
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def run(session, stack: str, region: str, *, split: str, beds: int, seed: int, data_folder: str, out=print) -> str:
    from aws_verify import signed_health

    cloudformation, s3 = session.client("cloudformation"), session.client("s3")
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
    api, bucket = outputs.get("ApiBaseUrl"), outputs.get("ArtifactBucketName")
    if not api or not bucket:
        raise ValueError("the stack has no scoring endpoint yet: run the deploy workflow with steps 'full deploy' first")
    health = signed_health(session, region)(api.rstrip("/") + "/health")
    out(f"Serving {health.get('model_name')} version {health.get('model_version')} "
        f"(bundle {str(health.get('bundle_sha256'))[:12]}, synthetic={health.get('synthetic_model')})")
    if health.get("synthetic_model"):
        raise ValueError("the endpoint serves a synthetic model; this demo replays real PhysioNet stays")
    version = health.get("model_version")
    registry = read_json(s3, bucket, f"models/v{version}/registry.json") if version else None
    require_frozen(split, registry)
    with tempfile.TemporaryDirectory(prefix="sepsis-demo-") as temp:
        temp = Path(temp)
        for name in ("combined.parquet", "splits.csv"):
            s3.download_file(bucket, f"data/{data_folder}/{name}", str(temp / name))
        page = temp / "ward.html"
        argv = ward_argv(api_url=api.rstrip("/"), data=temp / "combined.parquet", splits=temp / "splits.csv",
                         split=split, beds=beds, seed=seed, output=page)
        from sepsis_warning import cli

        old_argv = sys.argv
        sys.argv = ["sepsis-warning", *argv]
        try:
            cli.main()
        except SystemExit as stop:      # the CLI reports a bad request with parser.error()
            if stop.code not in (0, None):
                raise RuntimeError(f"the ward replay was refused (exit {stop.code}); see the message above") from stop
        finally:
            sys.argv = old_argv
        key = report_key(datetime.datetime.now(datetime.UTC), split)
        s3.upload_file(str(page), bucket, key, ExtraArgs={"ContentType": "text/html"})
    location = f"s3://{bucket}/{key}"
    out(f"Ward replay saved to {location} (private; open it with: aws s3 cp {location} . )")
    return location


# ---- traffic: many stays through the endpoint, no page (feeds the prediction log that Evidently reads)

def pick_traffic(cohort: list, count: int, max_hours: int, seed: int) -> list:
    """``count`` different stays chosen at random with ``seed`` (all of them when the split is smaller), each cut to ``max_hours``.

    The ward replay shows at most 24 beds and always the same seeded mix; the monitor needs hundreds of requests, one per stay,
    so this picks a plain random sample instead, and a different seed gives a different sample.
    """
    from dataclasses import replace

    import numpy as np

    if count < 1 or max_hours < 1:
        raise ValueError("stays and max-hours must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cohort))[:count]
    return [replace(cohort[i], frame=cohort[i].frame.iloc[:max_hours].reset_index(drop=True)) for i in sorted(order)]


def send_traffic(stays: list, post, workers: int = 4) -> dict:
    """Score each stay with ``post(patient_id, hours)`` (one request per stay) on a few threads. Returns the counts."""
    from sepsis_warning.data import DYNAMIC, STATIC

    def one(item):
        index, stay = item
        inputs = stay.frame[DYNAMIC + STATIC].astype(object).where(stay.frame[DYNAMIC + STATIC].notna(), None)
        try:
            response = post(f"traffic-{index:04d}", inputs.to_dict(orient="records"))
        except Exception as error:  # noqa: BLE001 - one failed request must not stop the others
            return {"ok": False, "error": f"{type(error).__name__}: {str(error)[:120]}"}
        hours = response.get("hours", [])
        return {"ok": True, "hours": len(hours), "alerts": sum(bool(h.get("alert_event")) for h in hours)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, enumerate(stays, start=1)))
    failed = [r for r in results if not r["ok"]]
    return {"sent": len(results), "ok": len(results) - len(failed), "failed": len(failed),
            "hours": sum(r.get("hours", 0) for r in results), "alerts": sum(r.get("alerts", 0) for r in results),
            "first_error": failed[0]["error"] if failed else ""}


def traffic(session, stack: str, region: str, *, split: str, stays: int, seed: int, max_hours: int, workers: int,
            data_folder: str, out=print) -> dict:
    """Send ``stays`` real stays of a split through the deployed ``/score`` and write nothing but the prediction log.

    The same rules as the ward replay: a real model only, and a held-out split only for a model with a recorded frozen evaluation.
    No page is made and nothing is uploaded: the stays are scored and forgotten, so the bucket gets only the service's own log.
    """
    from aws_verify import signed_health

    cloudformation, s3 = session.client("cloudformation"), session.client("s3")
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
    api, bucket = outputs.get("ApiBaseUrl"), outputs.get("ArtifactBucketName")
    if not api or not bucket:
        raise ValueError("the stack has no scoring endpoint yet: run the deploy workflow first")
    health = signed_health(session, region)(api.rstrip("/") + "/health")
    out(f"Serving {health.get('model_name')} version {health.get('model_version')} "
        f"(bundle {str(health.get('bundle_sha256'))[:12]}, synthetic={health.get('synthetic_model')})")
    if health.get("synthetic_model"):
        raise ValueError("the endpoint serves a synthetic model; this sends real PhysioNet stays")
    version = health.get("model_version")
    registry = read_json(s3, bucket, f"models/v{version}/registry.json") if version else None
    require_frozen(split, registry)
    with tempfile.TemporaryDirectory(prefix="sepsis-traffic-") as temp:
        temp = Path(temp)
        for name in ("combined.parquet", "splits.csv"):
            s3.download_file(bucket, f"data/{data_folder}/{name}", str(temp / name))
        from argparse import Namespace

        from sepsis_warning import cli
        from sepsis_warning.ward import _post_json

        cohorts, _ = cli.load_cohorts(Namespace(data=temp / "combined.parquet", splits=temp / "splits.csv", seed=seed))
        chosen = pick_traffic(cohorts[split], stays, max_hours, seed)
        out(f"Sending {len(chosen)} stays of {split} to /score on {workers} threads ...")
        base = api.rstrip("/")
        result = send_traffic(chosen, lambda pid, hours: _post_json(f"{base}/score", {"patient_id": pid, "hours": hours}, 60, True),
                              workers)
    out(f"Done: {result['ok']} of {result['sent']} requests scored ({result['hours']} hours, {result['alerts']} alert events)"
        + (f"; {result['failed']} failed, first: {result['first_error']}" if result["failed"] else ""))
    if result["failed"] > max(1, result["sent"] // 10):
        raise RuntimeError(f"more than a tenth of the requests failed: {result['first_error']}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--mode", choices=["ward", "traffic"], default="ward",
                        help="ward: the replay page (at most 24 beds). traffic: many stays through /score, no page")
    parser.add_argument("--beds", type=int, default=12)
    parser.add_argument("--stays", type=int, default=300, help="traffic: how many stays to send")
    parser.add_argument("--max-hours", type=int, default=72, help="traffic: hours sent per stay")
    parser.add_argument("--workers", type=int, default=4, help="traffic: parallel requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-folder", default="physionet", help="data/<folder>/ in the artifact bucket")
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    try:
        if args.mode == "traffic":
            result = traffic(session, args.stack, args.region, split=args.split, stays=args.stays, seed=args.seed,
                             max_hours=args.max_hours, workers=args.workers, data_folder=args.data_folder)
            text = (f"### Predict demo: traffic\n\n{result['ok']} of {result['sent']} stays scored through `/score` "
                    f"({args.split}, seed {args.seed}). Nothing was saved except the service's own prediction log. "
                    "Now run **monitor** with `report` (it needs at least 200 requests).\n")
        else:
            location = run(session, args.stack, args.region, split=args.split, beds=args.beds, seed=args.seed,
                           data_folder=args.data_folder)
            text = f"### Predict demo\n\nWard replay ({args.split}): `{location}`\n\nPrivate: it holds PhysioNet stays.\n"
    except (ValueError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
