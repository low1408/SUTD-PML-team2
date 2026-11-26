"""FastAPI wrapper around ``ScoringService`` (needs the ``serve`` extra).

    python -m sepsis_warning serve --model artifacts/demo/model.joblib

Endpoints:

* ``GET /health``: liveness and the loaded bundle's identity and manifest.
* ``GET /model``: the selected model, why it was selected and every candidate's
  validation metrics (never held-out test results), plus its registry version.
* ``POST /score``: one stay's full history, a score for every hour.
* ``POST /predict``: up to 64 stays (a ward), the current risk and alert state of each.

Every scoring response names the model version and bundle hash that produced
it, and each prediction has a stable ``prediction_id``. The ``X-Request-Id``
header is echoed (or generated). Validation and scoring live in ``service.py``
so the same rules apply with or without HTTP. No authentication here: the CLI
binds to 127.0.0.1, and AWS deployments put IAM in front.
"""

import time
import uuid
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from .service import (
    INPUT_COLUMNS,
    MAX_BODY_BYTES,
    MAX_HOURS,
    MAX_STAYS,
    InvalidInput,
    ModelMismatch,
    ScoringService,
    TooLarge,
)

# One typed, documented field per model input. Extra keys are passed through so the
# service can reject labels and unit codes with its own, clearer message.
HourRecord = create_model(
    "HourRecord",
    __config__=ConfigDict(extra="allow"),
    __doc__="One ICU hour. ICULOS is required; any other field may be omitted or null (not measured).",
    # strict: a JSON number or null only; true/false and numeric strings are refused.
    **cast(Any, {
        name: (float | None, Field(default=None, allow_inf_nan=False, strict=True))
        for name in INPUT_COLUMNS
    }),
)


class ScoreRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64, description="Opaque stay ID; not a real identifier")
    hours: list[HourRecord] = Field(  # type: ignore[valid-type]
        min_length=1, max_length=MAX_HOURS,
        description="Full stay history from the first ICU hour, one object per hour, null for missing",
    )


class PredictRequest(BaseModel):
    stays: list[ScoreRequest] = Field(min_length=1, max_length=MAX_STAYS,
                                      description="One entry per stay, each with its full history")


class Provenance(BaseModel):
    model_name: str
    model_version: str | None = Field(description="Registry version, when served from the registry or S3")
    bundle_sha256: str | None
    synthetic_model: bool
    threshold: float
    cooldown_hours: int
    scored_at: str
    alert_source: str = Field(description="'ledger' when delivered alerts are recorded, else 'stateless_replay'")
    disclaimer: str


class StayPrediction(BaseModel):
    patient_id: str
    prediction_id: str = Field(description="Same stay, hour and bundle always give the same ID")
    hour: float = Field(description="ICULOS of the latest hour sent")
    hours_scored: int
    probability: float
    risk_tier: str = Field(description="low, watch or alert, relative to the threshold; a demonstration scale")
    above_threshold: bool
    new_alert: bool = Field(description="An alert fires at this hour (once per alert, then a cooldown)")
    last_alert_hour: float | None
    input_warnings: list[str] = Field(description="Values outside the review range; scored, but worth a look")


class PredictResponse(Provenance):
    predictions: list[StayPrediction]


class HourScore(BaseModel):
    hour: float
    probability: float
    hourly_positive: int
    alert_event: int
    risk_tier: str
    prediction_id: str


class ScoreResponse(Provenance):
    patient_id: str
    input_warnings: list[str]
    hours: list[HourScore]


def _records(hours: list) -> list[dict]:
    # Only the fields that were sent, so "omitted" and "null" stay distinguishable from extra keys.
    return [hour.model_dump(exclude_unset=True) for hour in hours]


def route_label(request: Request) -> str:
    """The matched route template (``/score``), never the raw URL: scans of random paths must not create
    a new CloudWatch dimension value each, so unmatched requests all share one label."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def create_app(service: ScoringService, model_card: dict | None = None, telemetry=None) -> FastAPI:
    app = FastAPI(
        title="Sepsis early-warning research scorer",
        description="Research tool on a benchmark label. Not clinically validated.",
        version="0.2.0",
    )
    if telemetry is not None:
        service.on_prediction = telemetry.prediction

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.stats = {}
        started = time.perf_counter()
        declared = request.headers.get("content-length")
        too_large = int(declared) if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES else None
        if too_large is None and request.method in {"POST", "PUT", "PATCH"} and not (declared and declared.isdigit()):
            # No usable Content-Length (chunked transfer): count the bytes as they arrive and stop at the limit.
            received, chunks = 0, []
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_BODY_BYTES:
                    too_large = received
                    break
                chunks.append(chunk)
            if too_large is None:
                request._body = b"".join(chunks)  # pylint: disable=protected-access   # replayed to the route; the stream is not read twice
        if too_large is not None:
            response = JSONResponse(status_code=413, content={
                "error": "too_large",
                "detail": f"request body is over {MAX_BODY_BYTES} bytes ({too_large} read); send fewer stays per call",
            })
        else:
            response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        if telemetry is not None:
            telemetry.request(
                request_id=request_id, method=request.method, path=route_label(request), status=response.status_code,
                latency_ms=(time.perf_counter() - started) * 1000, model_version=service.model_version,
                bundle_sha256=service.bundle_sha256, stats=request.state.stats,
            )
        return response

    @app.exception_handler(InvalidInput)
    async def invalid_input(_, error: InvalidInput):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def malformed(_, error: RequestValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in error.errors()[:5]]
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": "; ".join(parts)})

    @app.exception_handler(TooLarge)
    async def too_large(_, error: TooLarge):
        return JSONResponse(status_code=413, content={"error": "too_large", "detail": str(error)})

    @app.exception_handler(ModelMismatch)
    async def model_mismatch(_, error: ModelMismatch):
        return JSONResponse(status_code=409, content={"error": "model_mismatch", "detail": str(error)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/model")
    def model() -> dict[str, Any]:
        health = service.health()
        return {
            "model_name": health["model_name"],
            "model_version": health["model_version"],
            "synthetic_model": health["synthetic_model"],
            "bundle_sha256": health["bundle_sha256"],
            "threshold": health["threshold"],
            "cooldown_hours": health["cooldown_hours"],
            "window_hours": health["window_hours"],
            "feature_count": health["feature_count"],
            "selection": model_card,  # None when served without its run's selection files
            "disclaimer": health["disclaimer"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> dict:
        result = service.predict([{"patient_id": stay.patient_id, "hours": _records(stay.hours)}
                                  for stay in body.stays])
        request.state.stats.update(
            stays=len(result["predictions"]), hours=sum(p["hours_scored"] for p in result["predictions"]),
            new_alerts=sum(p["new_alert"] for p in result["predictions"]),
            warnings=sum(len(p["input_warnings"]) for p in result["predictions"]),
        )
        return result

    @app.post("/score", response_model=ScoreResponse)
    def score(body: ScoreRequest, request: Request) -> dict:
        # Plain def: FastAPI runs it in a worker thread, so slow requests do not block others.
        result = service.score(body.patient_id, _records(body.hours))
        request.state.stats.update(
            stays=1, hours=len(result["hours"]), new_alerts=result["hours"][-1]["alert_event"],
            warnings=len(result["input_warnings"]),
        )
        return result

    return app
