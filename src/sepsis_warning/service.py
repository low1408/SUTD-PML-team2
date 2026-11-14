"""Research scoring service core (L02), independent of the web framework.

Request semantics
-----------------
* Stateless. Every request carries one stay's full hourly history from its first
  ICU hour up to "now". The server keeps nothing between requests, so one
  patient can never affect another, and a restart loses nothing.
* The response scores every supplied hour with the frozen bundle: the same
  shared feature builder, saved feature columns, threshold and cooldown as
  ``replay``. Hour t only uses hours up to t, so scoring a prefix gives the same
  answers for those hours as scoring the whole stay.
* Labels, administrative unit codes and hospital IDs are rejected as inputs.
* Nothing is trained, downloaded or written per request.

This is a research tool on a benchmark label. It is not clinically validated and
has no authentication: bind it to localhost until the team agrees otherwise.
"""

import hashlib
import math
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, validate_record
from .features import build_features, select_saved_features
from .policy import apply_policy
from .scoring import BUNDLE_KEYS, file_sha256, load_bundle

INPUT_COLUMNS = DYNAMIC + STATIC
# Present in PhysioNet files but never model inputs.
REJECTED_COLUMNS = {"SepsisLabel", "Unit1", "Unit2", "patient_id", "hospital"}
MAX_HOURS = 1000
PATIENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
DISCLAIMER = "Research scores on a benchmark label; not clinically validated and not for patient care."


class InvalidInput(ValueError):
    """The request cannot be scored as sent (HTTP 422)."""


class ModelMismatch(RuntimeError):
    """The loaded bundle cannot be used with this feature code (HTTP 409 / startup failure)."""


class ScoringService:
    def __init__(self, bundle: dict, bundle_sha256: str | None = None):
        missing = BUNDLE_KEYS - set(bundle) if isinstance(bundle, dict) else BUNDLE_KEYS
        if missing:
            raise ModelMismatch(f"not a model bundle; missing {sorted(missing)}")
        self.bundle = bundle
        self.bundle_sha256 = bundle_sha256
        # threadpoolctl changes process-wide state, so predictions are
        # serialised; feature building stays per request and in parallel.
        self._predict_lock = threading.Lock()
        self._check_feature_contract()

    @classmethod
    def from_path(cls, path: Path, expected_sha256: str | None = None) -> "ScoringService":
        """Load a bundle this project produced. Joblib can run code: never load an untrusted file."""
        path = Path(path)
        try:
            bundle = load_bundle(path, expected_sha256)
        except ValueError as error:
            raise ModelMismatch(str(error)) from error
        return cls(bundle, expected_sha256 or file_sha256(path))

    def _check_feature_contract(self):
        probe = pd.DataFrame({name: [np.nan] for name in INPUT_COLUMNS})
        probe["ICULOS"] = 1.0
        try:
            select_saved_features(build_features(probe, self.bundle["window"]), self.bundle["feature_columns"])
        except ValueError as error:
            raise ModelMismatch(f"saved model does not match the current feature code: {error}") from error

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_name": self.bundle["model_name"],
            "synthetic_model": bool(self.bundle["synthetic"]),
            "window_hours": int(self.bundle["window"]),
            "feature_count": len(self.bundle["feature_columns"]),
            "threshold": float(self.bundle["threshold"]),
            "cooldown_hours": int(self.bundle["cooldown"]),
            "bundle_sha256": self.bundle_sha256,
            "feature_columns_sha256": hashlib.sha256(
                "\n".join(self.bundle["feature_columns"]).encode()
            ).hexdigest(),
            "disclaimer": DISCLAIMER,
        }

    def history_frame(self, patient_id, hours) -> pd.DataFrame:
        """Validate a request and return the stay as a frame. Raises InvalidInput."""
        if not isinstance(patient_id, str) or not PATIENT_ID.match(patient_id):
            raise InvalidInput("patient_id must be 1-64 characters of letters, digits, '_', '.', ':' or '-'")
        if not isinstance(hours, list) or not hours:
            raise InvalidInput("hours must be a non-empty list of hourly records")
        if len(hours) > MAX_HOURS:
            raise InvalidInput(f"at most {MAX_HOURS} hours per request")
        for index, record in enumerate(hours):
            if not isinstance(record, dict):
                raise InvalidInput(f"hours[{index}] must be an object")
            rejected = sorted(REJECTED_COLUMNS & set(record))
            if rejected:
                raise InvalidInput(f"hours[{index}]: {rejected} are not model inputs; remove them")
            unknown = sorted(set(record) - set(INPUT_COLUMNS))
            if unknown:
                raise InvalidInput(f"hours[{index}]: unknown fields {unknown[:5]}")
            for name, value in record.items():
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidInput(f"hours[{index}].{name} must be a number or null")
                if not math.isfinite(value):
                    raise InvalidInput(f"hours[{index}].{name} must be finite; send null for a missing value")
            if record.get("ICULOS") is None:
                raise InvalidInput(f"hours[{index}].ICULOS is required")
        frame = pd.DataFrame.from_records(hours).reindex(columns=INPUT_COLUMNS).astype(float)
        try:
            return validate_record(frame, f"patient {patient_id}", require_labels=False)
        except ValueError as error:
            raise InvalidInput(str(error)) from error

    def score(self, patient_id, hours) -> dict:
        frame = self.history_frame(patient_id, hours)
        try:
            features = select_saved_features(
                build_features(frame, self.bundle["window"]), self.bundle["feature_columns"]
            )
        except ValueError as error:
            raise ModelMismatch(str(error)) from error
        with self._predict_lock:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                probability = self.bundle["model"].predict_proba(features)[:, 1]
        table = pd.DataFrame({"patient_id": patient_id, "hour": frame["ICULOS"], "probability": probability})
        table = apply_policy(table, self.bundle["threshold"], self.bundle["cooldown"])
        return {
            "patient_id": patient_id,
            "model_name": self.bundle["model_name"],
            "synthetic_model": bool(self.bundle["synthetic"]),
            "threshold": float(self.bundle["threshold"]),
            "cooldown_hours": int(self.bundle["cooldown"]),
            "hours": [
                {
                    "hour": float(row.hour),
                    "probability": float(row.probability),
                    "hourly_positive": int(row.hourly_positive),
                    "alert_event": int(row.alert_event),
                }
                for row in table.itertuples(index=False)
            ],
            "disclaimer": DISCLAIMER,
        }
