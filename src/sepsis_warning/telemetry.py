"""Structured telemetry for the scoring service: request metrics and a prediction log.

Everything is one JSON object per line on stdout, which CloudWatch Logs
collects on both Lambda and Fargate.

* Request lines use the CloudWatch Embedded Metric Format, so CloudWatch turns
  them into metrics (latency, hours scored, alerts, errors, input warnings)
  with no extra API calls and no agent.
* Prediction lines (``record_type: "prediction"``) are the input for drift
  monitoring: hashed stay ID, hour, probability, alert, and which measurements
  were missing. They never contain a measurement value or the stay ID as sent.

A log subscription filter sends the prediction lines to S3 (see
``infra/cloudformation/observability.yaml``); ``--prediction-log`` writes the
same lines to a local file instead.
"""

import datetime
import json
import sys
import threading

NAMESPACE = "SepsisWarning/Scoring"
METRICS = [("LatencyMs", "Milliseconds"), ("HoursScored", "Count"), ("StaysScored", "Count"),
           ("NewAlerts", "Count"), ("InputWarnings", "Count"), ("ClientErrors", "Count"), ("ServerErrors", "Count")]


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Telemetry:
    def __init__(self, stream=None, prediction_log=None, environment: str = "local"):
        self.stream = stream or sys.stdout
        self.environment = environment
        self._prediction_file = open(prediction_log, "a", encoding="utf-8") if prediction_log else None  # noqa: SIM115
        self._lock = threading.Lock()

    def _write(self, record: dict, to_prediction_file: bool = False) -> None:
        line = json.dumps(record, separators=(",", ":"), allow_nan=False)
        with self._lock:
            target = self._prediction_file if (to_prediction_file and self._prediction_file) else self.stream
            target.write(line + "\n")
            target.flush()

    def request(self, *, request_id: str, method: str, path: str, status: int, latency_ms: float,
                model_version: str | None, bundle_sha256: str | None, stats: dict | None = None) -> None:
        stats = stats or {}
        now = _now()
        self._write({
            "_aws": {
                "Timestamp": int(now.timestamp() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": NAMESPACE, "Dimensions": [["Environment"], ["Environment", "Path"]],
                    "Metrics": [{"Name": name, "Unit": unit} for name, unit in METRICS],
                }],
            },
            "record_type": "request", "ts": now.isoformat(timespec="milliseconds"),
            "Environment": self.environment, "Path": path, "method": method, "status": status,
            "request_id": request_id, "model_version": model_version,
            "bundle_sha256": bundle_sha256[:12] if bundle_sha256 else None,
            "LatencyMs": round(latency_ms, 2),
            "HoursScored": int(stats.get("hours", 0)), "StaysScored": int(stats.get("stays", 0)),
            "NewAlerts": int(stats.get("new_alerts", 0)), "InputWarnings": int(stats.get("warnings", 0)),
            "ClientErrors": int(400 <= status < 500), "ServerErrors": int(status >= 500),
        })

    def prediction(self, record: dict) -> None:
        self._write({"record_type": "prediction", "ts": _now().isoformat(timespec="seconds"),
                     "environment": self.environment, **record}, to_prediction_file=True)
