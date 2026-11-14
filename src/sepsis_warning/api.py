"""FastAPI wrapper around ``ScoringService`` (needs the ``serve`` extra).

    python -m sepsis_warning serve --model artifacts/demo/model.joblib

Endpoints: ``GET /health`` and ``POST /score``. Validation and scoring live in
``service.py`` so the same rules apply with or without HTTP. No authentication:
the CLI binds to 127.0.0.1 and refuses other hosts unless told otherwise.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import MAX_HOURS, InvalidInput, ModelMismatch, ScoringService


class ScoreRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64, description="Opaque stay ID; not a real identifier")
    hours: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_HOURS,
        description="Full stay history from the first ICU hour, one object per hour, null for missing",
    )


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


def create_app(service: ScoringService) -> FastAPI:
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

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> dict:
        # Plain def: FastAPI runs it in a worker thread, so slow requests do not block others.
        return service.score(request.patient_id, request.hours)

    return app
