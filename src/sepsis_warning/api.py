"""FastAPI wrapper around ``ScoringService`` (needs the ``serve`` extra).

    python -m sepsis_warning serve --model artifacts/demo/model.joblib

Endpoints:

* ``GET /health``: liveness and the loaded bundle's identity.
* ``GET /model``: the selected model, why it was selected and every candidate's
  validation metrics (never held-out test results), plus its registry version.
* ``POST /score``: one stay's full history, a score for every hour.
* ``POST /predict``: up to 64 stays (a ward), the current risk and alert state of each.

Validation and scoring live in
``service.py`` so the same rules apply with or without HTTP. No authentication:
the CLI binds to 127.0.0.1 and refuses other hosts unless told otherwise.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import MAX_HOURS, MAX_STAYS, InvalidInput, ModelMismatch, ScoringService


class ScoreRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64, description="Opaque stay ID; not a real identifier")
    hours: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_HOURS,
        description="Full stay history from the first ICU hour, one object per hour, null for missing",
    )


class PredictRequest(BaseModel):
    stays: list[ScoreRequest] = Field(min_length=1, max_length=MAX_STAYS,
                                      description="One entry per stay, each with its full history")


class StayPrediction(BaseModel):
    patient_id: str
    hour: float = Field(description="ICULOS of the latest hour sent")
    hours_scored: int
    probability: float
    above_threshold: bool
    new_alert: bool = Field(description="An alert fires at this hour (once per alert, then a cooldown)")
    last_alert_hour: float | None


class PredictResponse(BaseModel):
    model_name: str
    synthetic_model: bool
    threshold: float
    cooldown_hours: int
    predictions: list[StayPrediction]
    disclaimer: str


class HourScore(BaseModel):
    hour: float
    probability: float
    hourly_positive: int
    alert_event: int


class ScoreResponse(BaseModel):
    patient_id: str
    model_name: str
    synthetic_model: bool
    threshold: float
    cooldown_hours: int
    hours: list[HourScore]
    disclaimer: str


def create_app(service: ScoringService, model_card: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="Sepsis early-warning research scorer",
        description="Research tool on a benchmark label. Not clinically validated.",
        version="0.1.0",
    )

    @app.exception_handler(InvalidInput)
    async def invalid_input(_, error: InvalidInput):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(error)})

    @app.exception_handler(ModelMismatch)
    async def model_mismatch(_, error: ModelMismatch):
        return JSONResponse(status_code=409, content={"error": "model_mismatch", "detail": str(error)})

    @app.get("/health")
    def health() -> dict:
        return service.health()

    @app.get("/model")
    def model() -> dict:
        health = service.health()
        return {
            "model_name": health["model_name"],
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
    def predict(request: PredictRequest) -> dict:
        return service.predict([stay.model_dump() for stay in request.stays])

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> dict:
        # Plain def: FastAPI runs it in a worker thread, so slow requests do not block others.
        return service.score(request.patient_id, request.hours)

    return app
