"""Reference client for the scoring API: timeouts, retries, a circuit breaker and a rule-based fallback.

A ward system should never be left without an answer because the model
service is slow, throttled or down. This client:

* signs requests with SigV4 when the API is the IAM-protected Lambda URL;
* gives each call a short timeout and retries 429 and 5xx responses and
  network errors with exponential backoff and full jitter (never 4xx: a
  rejected request will be rejected again);
* opens a circuit after repeated failures, so a struggling service is not
  hammered, and probes it again after a pause;
* can fall back to the SIRS rule (``baselines.py``), computed locally from the
  same stay history and clearly marked ``"fallback": true``.

Standard library only, plus boto3 when signing.
"""

import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
import pandas as pd

from .baselines import BASELINE_THRESHOLD, SIRS_COLUMNS, SIRS_CRITERIA, sirs_count
from .policy import alert_events

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ScoringUnavailable(RuntimeError):
    """The API did not answer after retries, or the circuit is open."""


class ScoringRejected(ValueError):
    """The API refused the request (4xx); retrying will not help."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


def sirs_fallback(stays: list[dict], cooldown_hours: int = 6) -> dict:
    """SIRS (two or more criteria) for each stay's latest hour, in the shape of ``/predict``."""
    predictions = []
    for stay in stays:
        frame = pd.DataFrame.from_records(stay["hours"]).reindex(columns=["ICULOS", *SIRS_COLUMNS]).astype(float)
        score = sirs_count(frame) / len(SIRS_CRITERIA)
        hours = frame["ICULOS"].to_numpy()
        events = alert_events(hours, score, BASELINE_THRESHOLD, cooldown_hours)
        alerts = hours[np.flatnonzero(events)]
        predictions.append({
            "patient_id": stay["patient_id"], "hour": float(hours[-1]), "hours_scored": len(hours),
            "sirs_criteria_met": round(score[-1] * len(SIRS_CRITERIA)),
            "above_threshold": bool(score[-1] >= BASELINE_THRESHOLD), "new_alert": bool(events[-1]),
            "last_alert_hour": float(alerts[-1]) if len(alerts) else None,
        })
    return {
        "fallback": True, "model_name": "sirs_rule_fallback", "model_version": None,
        "detail": "The scoring service was unavailable; these are SIRS rule results (2 or more criteria), "
                  "not model scores.",
        "predictions": predictions,
    }


class ScoringClient:
    def __init__(self, base_url: str, *, sign: bool = False, region: str | None = None, timeout: float = 5.0,
                 retries: int = 3, backoff: float = 0.5, max_backoff: float = 8.0, breaker_failures: int = 5,
                 breaker_reset: float = 30.0, opener=None, sleep=time.sleep, clock=time.monotonic, rng=None):
        self.base_url = base_url.rstrip("/")
        self.sign, self.region, self.timeout = sign, region, timeout
        self.retries, self.backoff, self.max_backoff = retries, backoff, max_backoff
        self.breaker_failures, self.breaker_reset = breaker_failures, breaker_reset
        self._open = opener or urllib.request.urlopen
        self._sleep, self._clock = sleep, clock
        self._rng = rng or random.Random()
        self._failures = 0
        self._opened_at = None

    # --- circuit breaker -------------------------------------------------
    @property
    def circuit_open(self) -> bool:
        if self._opened_at is None:
            return False
        # After the pause the circuit is half-open: one request may probe the service.
        return self._clock() - self._opened_at < self.breaker_reset

    def _record(self, success: bool) -> None:
        if success:
            self._failures, self._opened_at = 0, None
            return
        self._failures += 1
        if self._failures >= self.breaker_failures:
            self._opened_at = self._clock()

    # --- transport --------------------------------------------------------
    def _headers(self, url: str, method: str, body: bytes | None) -> dict:
        headers = {"x-request-id": uuid.uuid4().hex}
        if body is not None:
            headers["content-type"] = "application/json"
        if not self.sign:
            return headers
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        host = urllib.parse.urlparse(url).hostname or ""
        region = self.region or (host.split(".")[2] if host.endswith(".on.aws") and host.count(".") >= 4 else None)
        if not region:
            raise ValueError("pass region=; it could not be read from the URL")
        headers["x-amz-content-sha256"] = hashlib.sha256(body or b"").hexdigest()
        request = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(boto3.Session().get_credentials(), "lambda", region).add_auth(request)
        return dict(request.headers)

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if self.circuit_open:
            raise ScoringUnavailable("circuit open: the scoring service failed repeatedly; not calling it for now")
        url = self.base_url + path
        body = json.dumps(payload).encode() if payload is not None else None
        last = "no attempt made"
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, data=body, method=method,
                                                 headers=self._headers(url, method, body))
                with self._open(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                self._record(True)
                return result
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code not in RETRYABLE_STATUS:
                    self._record(True)  # the service is up; the request is wrong
                    raise ScoringRejected(error.code, detail) from error
                last = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last = f"{type(error).__name__}: {error}"
            self._record(False)
            if attempt < self.retries and not self.circuit_open:
                # Full jitter: spreads retries from many clients so they do not arrive together.
                self._sleep(self._rng.uniform(0, min(self.max_backoff, self.backoff * 2 ** attempt)))
            elif self.circuit_open:
                break
        raise ScoringUnavailable(f"{method} {path} failed after {attempt + 1} attempt(s): {last}")

    # --- API --------------------------------------------------------------
    def health(self) -> dict:
        return self.request("GET", "/health")

    def score(self, patient_id: str, hours: list[dict]) -> dict:
        return self.request("POST", "/score", {"patient_id": patient_id, "hours": hours})

    def predict(self, stays: list[dict]) -> dict:
        return self.request("POST", "/predict", {"stays": stays})

    def predict_or_fallback(self, stays: list[dict]) -> dict:
        """Model predictions, or SIRS rule results marked ``fallback`` when the service is unavailable."""
        try:
            return {"fallback": False, **self.predict(stays)}
        except ScoringUnavailable as error:
            result = sirs_fallback(stays)
            result["reason"] = str(error)
            return result
