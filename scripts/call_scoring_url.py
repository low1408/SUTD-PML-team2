#!/usr/bin/env python3
"""Call the Lambda scoring API, signing the request with your AWS credentials (SigV4).

The function URL uses IAM authentication, so plain curl gets 403. This signs
the request the same way the hospital integration would:

    python scripts/call_scoring_url.py --url https://<id>.lambda-url.ap-southeast-1.on.aws GET /health
    python scripts/call_scoring_url.py --url ... GET /model
    python scripts/call_scoring_url.py --url ... POST /predict \\
        '{"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}]}]}'

Synthetic data only until authentication, network path and data handling are
approved for real use. Needs boto3 (aws extra).
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse


def signed_request(url: str, method: str, body: str | None, region: str | None, credentials=None):
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    host = urlparse(url).hostname or ""
    region = region or (host.split(".")[2] if host.endswith(".on.aws") and host.count(".") >= 4 else None)
    if not region:
        raise SystemExit("pass --region; it could not be read from the URL")
    credentials = credentials or boto3.Session().get_credentials()
    if credentials is None:
        raise SystemExit("no AWS credentials found (aws configure, or AWS_PROFILE)")
    payload = body.encode() if body is not None else b""
    headers = {"x-amz-content-sha256": hashlib.sha256(payload).hexdigest()}
    if body is not None:
        headers["content-type"] = "application/json"
    request = AWSRequest(method=method, url=url, data=body.encode() if body is not None else None, headers=headers)
    SigV4Auth(credentials, "lambda", region).add_auth(request)
    return request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="ApiBaseUrl output of the stack")
    parser.add_argument("--region", default=None)
    parser.add_argument("method", choices=["GET", "POST"])
    parser.add_argument("path", help="/health, /model, /score or /predict")
    parser.add_argument("body", nargs="?", default=None, help="JSON body for POST")
    args = parser.parse_args(argv)
    if args.body is not None:
        json.loads(args.body)  # fail early on malformed JSON
    url = args.url.rstrip("/") + args.path
    prepared = signed_request(url, args.method, args.body, args.region)
    request = urllib.request.Request(url, data=prepared.body, method=args.method, headers=dict(prepared.headers))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            print(json.dumps(json.loads(response.read()), indent=2))
            return 0
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode(errors='replace')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
