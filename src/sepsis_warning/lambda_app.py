"""AWS Lambda entry point: the same FastAPI app as ``serve``, behind a Lambda function URL.

Configuration (environment variables, set by ``infra/cloudformation/serverless.yaml``):

* ``MODEL_S3_URI`` and ``MODEL_SHA256``: the published bundle and its pinned hash
  (from ``python -m sepsis_warning publish``). The bundle is downloaded to /tmp
  and checked before it is loaded.
* ``MODEL_PATH`` (instead of S3): a bundle already inside the container, for
  local tests with the Lambda runtime emulator.

The model loads on the first request of a new execution environment and is
reused while that environment stays warm; requests stay stateless.
``ALERT_TABLE`` (optional) names the DynamoDB alert ledger, and
``ENVIRONMENT_NAME`` labels the telemetry.
"""

import os
import time
from pathlib import Path

_adapter = None
_failed_at = None
_failure = ""
RETRY_AFTER_SECONDS = 60


def _build():
    from mangum import Mangum

    from .api import create_app
    from .model_source import load_serving_model
    from .telemetry import Telemetry

    alert_store = None
    if os.environ.get("ALERT_TABLE"):
        from .alert_store import DynamoAlertStore

        alert_store = DynamoAlertStore(os.environ["ALERT_TABLE"])
    expected = os.environ.get("MODEL_SHA256") or None
    if os.environ.get("MODEL_S3_URI"):
        service, card, source = load_serving_model(
            model_s3=os.environ["MODEL_S3_URI"], expected_sha256=expected, download_dir=Path("/tmp/model"),
            alert_store=alert_store,
        )
    elif os.environ.get("MODEL_PATH"):
        service, card, source = load_serving_model(model=Path(os.environ["MODEL_PATH"]), expected_sha256=expected,
                                                   alert_store=alert_store)
    else:
        raise RuntimeError("set MODEL_S3_URI and MODEL_SHA256 (or MODEL_PATH for local tests)")
    print(f"Loaded {source} ({service.health()['model_name']})", flush=True)
    telemetry = Telemetry(environment=os.environ.get("ENVIRONMENT_NAME", "lambda"))
    return Mangum(create_app(service, card, telemetry), lifespan="off")


def _unavailable(reason: str) -> dict:
    return {"statusCode": 503, "headers": {"content-type": "application/json",
                                           "retry-after": str(RETRY_AFTER_SECONDS)},
            "body": '{"error":"model_unavailable","detail":"the model could not be loaded; retry later"}',
            "isBase64Encoded": False, "_reason": reason}


def handler(event, context):
    """A failed model load is remembered for a minute, so a broken release does not
    download and fail again on every request; callers get 503 with Retry-After."""
    global _adapter, _failed_at, _failure
    if _adapter is None:
        if _failed_at is not None and time.monotonic() - _failed_at < RETRY_AFTER_SECONDS:
            response = _unavailable(_failure)
            response.pop("_reason")
            return response
        try:
            _adapter = _build()
            _failed_at = None
        except Exception as error:
            _failed_at, _failure = time.monotonic(), f"{type(error).__name__}: {error}"
            print(f'{{"record_type":"model_load_failed","detail":{_failure!r}}}', flush=True)
            raise
    return _adapter(event, context)
