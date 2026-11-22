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

import datetime
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .alert_store import LedgerConflict, ledger_events
from .data import DYNAMIC, STATIC, validate_record
from .features import build_features, select_saved_features
from .manifest import ManifestMismatch, check_runtime, code_sha256
from .policy import apply_policy
from .schema import range_problems
from .scoring import BUNDLE_KEYS, file_sha256, load_bundle

INPUT_COLUMNS = DYNAMIC + STATIC
# Present in PhysioNet files but never model inputs.
REJECTED_COLUMNS = {"SepsisLabel", "Unit1", "Unit2", "patient_id", "hospital"}
MAX_HOURS = 1000
MAX_STAYS = 64  # one ward per request
# Keeps a ward batch inside Lambda's 6 MB request and 30 s time limits.
MAX_TOTAL_HOURS = 8000
MAX_BODY_BYTES = 5_000_000
PATIENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
DISCLAIMER = "Research scores on a benchmark label; not clinically validated and not for patient care."


class InvalidInput(ValueError):
    """The request cannot be scored as sent (HTTP 422)."""


class ModelMismatch(RuntimeError):
    """The loaded bundle cannot be used with this feature code (HTTP 409 / startup failure)."""


class TooLarge(ValueError):
    """The request is bigger than one call may be (HTTP 413); split it."""


def risk_tier(probability: float, threshold: float) -> str:
    """Display tier relative to the frozen threshold. A demonstration scale, not a clinical one."""
    if probability >= threshold:
        return "alert"
    return "watch" if probability >= threshold / 2 else "low"


def _single_thread(model, depth: int = 0) -> None:
    """Make every nested estimator predict on one thread, so requests need no lock."""
    if depth > 6 or model is None or isinstance(model, (str, bytes, int, float, np.ndarray, pd.DataFrame)):
        return
    if isinstance(model, (list, tuple)):
        for item in model:
            _single_thread(item, depth + 1)
        return
    if not type(model).__module__.startswith("sklearn"):
        return
    if getattr(model, "n_jobs", None) not in (None, 1):
        model.n_jobs = 1
    for name in ("estimator", "steps", "calibrated_classifiers_", "estimators_"):
        child = getattr(model, name, None)
        if name == "estimators_" and child is not None and len(child) > 8:
            continue  # trees: the forest's own n_jobs is what matters
        _single_thread(child, depth + 1)


class ScoringService:
    def __init__(self, bundle: dict, bundle_sha256: str | None = None, *, model_version: str | None = None,
                 alert_store=None):
        missing = BUNDLE_KEYS - set(bundle) if isinstance(bundle, dict) else BUNDLE_KEYS
        if missing:
            raise ModelMismatch(f"not a model bundle; missing {sorted(missing)}")
        self.bundle = bundle
        self.bundle_sha256 = bundle_sha256
        self.manifest = bundle.get("manifest")
        if self.manifest is not None:
            try:
                check_runtime(self.manifest)
            except ManifestMismatch as error:
                raise ModelMismatch(str(error)) from error
        self.model_version = model_version
        self.alert_store = alert_store
        self.on_prediction = None  # optional callback(record) for the prediction log
        # One thread per prediction, set once for the process: no per-request lock, and
        # concurrent requests scale with workers or tasks instead of queueing.
        from threadpoolctl import threadpool_limits

        self._thread_limit = threadpool_limits(limits=1)
        _single_thread(bundle["model"])
        self._check_feature_contract()

    @classmethod
    def from_path(cls, path: Path, expected_sha256: str | None = None, **options) -> "ScoringService":
        """Load a bundle this project produced. Joblib can run code: never load an untrusted file."""
        path = Path(path)
        try:
            bundle = load_bundle(path, expected_sha256)
        except ValueError as error:
            raise ModelMismatch(str(error)) from error
        return cls(bundle, expected_sha256 or file_sha256(path), **options)

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
            "model_version": self.model_version,
            "alert_ledger": type(self.alert_store).__name__ if self.alert_store else None,
            "feature_columns_sha256": hashlib.sha256(
                "\n".join(self.bundle["feature_columns"]).encode()
            ).hexdigest(),
            # None for bundles made before manifests existed.
            "manifest": self.manifest,
            # False means features.py, policy.py or data.py changed since the model was fitted.
            "contract_code_matches": (
                self.manifest.get("contract_code_sha256") == code_sha256() if self.manifest else None
            ),
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
            frame = validate_record(frame, f"patient {patient_id}", require_labels=False)
        except ValueError as error:
            raise InvalidInput(str(error)) from error
        errors, warnings = range_problems(frame)
        if errors:
            raise InvalidInput("; ".join(errors[:5]))
        frame.attrs["input_warnings"] = warnings
        return frame

    def prediction_id(self, patient_id: str, hour: float) -> str:
        """Stable for the same stay, hour and bundle, so a retried request gives the same ID."""
        text = f"{self.bundle_sha256}:{patient_id}:{float(hour)!r}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _alerts(self, patient_id: str, hours, probability) -> tuple[np.ndarray, str]:
        """Alert flag per hour: from the ledger when there is one, else a stateless replay."""
        threshold, cooldown = self.bundle["threshold"], self.bundle["cooldown"]
        if self.alert_store is None:
            table = apply_policy(pd.DataFrame({"patient_id": patient_id, "hour": hours, "probability": probability}),
                                 threshold, cooldown)
            return table["alert_event"].to_numpy(), "stateless_replay"
        hours = np.asarray(hours, dtype=float)
        for _ in range(4):
            delivered, version = self.alert_store.get(patient_id)
            events, new_rows = ledger_events(hours, probability, threshold, cooldown, [a["hour"] for a in delivered])
            if not new_rows:
                return events, "ledger"
            now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
            new = [{"hour": float(hours[i]), "prediction_id": self.prediction_id(patient_id, hours[i]),
                    "model_version": self.model_version or (self.bundle_sha256 or "")[:12], "at": now}
                   for i in new_rows]
            try:
                self.alert_store.append(patient_id, new, version)
                return events, "ledger"
            except LedgerConflict:
                continue  # another request wrote first: read again, so no alert is recorded twice
        raise RuntimeError("alert ledger kept changing; try the request again")

    def score(self, patient_id, hours) -> dict:
        frame = self.history_frame(patient_id, hours)
        try:
            features = select_saved_features(
                build_features(frame, self.bundle["window"]), self.bundle["feature_columns"]
            )
        except ValueError as error:
            raise ModelMismatch(str(error)) from error
        probability = self.bundle["model"].predict_proba(features)[:, 1]
        hours = frame["ICULOS"].to_numpy(dtype=float)
        threshold = float(self.bundle["threshold"])
        events, source = self._alerts(patient_id, hours, probability)
        scored_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        result = {
            "patient_id": patient_id,
            "model_name": self.bundle["model_name"],
            "model_version": self.model_version,
            "bundle_sha256": self.bundle_sha256,
            "synthetic_model": bool(self.bundle["synthetic"]),
            "threshold": threshold,
            "cooldown_hours": int(self.bundle["cooldown"]),
            "scored_at": scored_at,
            "alert_source": source,
            "input_warnings": list(frame.attrs.get("input_warnings", [])),
            "hours": [
                {
                    "hour": float(hour),
                    "probability": float(p),
                    "hourly_positive": int(p >= threshold),
                    "alert_event": int(event),
                    "risk_tier": risk_tier(float(p), threshold),
                    "prediction_id": self.prediction_id(patient_id, hour),
                }
                for hour, p, event in zip(hours, probability, events, strict=True)
            ],
            "disclaimer": DISCLAIMER,
        }
        if self.on_prediction is not None:
            latest = frame.iloc[-1]
            self.on_prediction({
                "stay": hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16],
                "prediction_id": result["hours"][-1]["prediction_id"],
                "hour": float(hours[-1]), "hours_in_request": len(hours),
                "probability": float(probability[-1]), "above_threshold": bool(probability[-1] >= threshold),
                "new_alert": bool(events[-1]), "alert_source": source,
                "missing_now": [name for name in DYNAMIC if pd.isna(latest[name])],
                "input_warnings": len(result["input_warnings"]),
                "model_name": self.bundle["model_name"], "model_version": self.model_version,
                "bundle_sha256": (self.bundle_sha256 or "")[:12],
            })
        return result

    def predict(self, stays) -> dict:
        """Current risk for several stays at once (a ward), one entry per stay.

        Each stay carries its full history, exactly as for ``score``; the answer
        is for its latest hour. ``new_alert`` is true only in the hour an alert
        fires, so a caller polling every hour notifies once per alert.
        """
        if not isinstance(stays, list) or not stays:
            raise InvalidInput("stays must be a non-empty list")
        if len(stays) > MAX_STAYS:
            raise InvalidInput(f"at most {MAX_STAYS} stays per request")
        total = sum(len(stay.get("hours") or []) for stay in stays if isinstance(stay, dict))
        if total > MAX_TOTAL_HOURS:
            raise TooLarge(f"{total} hours in one request; at most {MAX_TOTAL_HOURS}. Send the ward in smaller batches")
        ids = [stay.get("patient_id") if isinstance(stay, dict) else None for stay in stays]
        duplicates = sorted({i for i in ids if isinstance(i, str) and ids.count(i) > 1})
        if duplicates:
            raise InvalidInput(f"patient_id values must be unique in one request: {duplicates[:5]}")
        predictions = []
        for index, stay in enumerate(stays):
            if not isinstance(stay, dict):
                raise InvalidInput(f"stays[{index}] must be an object")
            try:
                scored = self.score(stay.get("patient_id"), stay.get("hours"))
            except InvalidInput as error:
                raise InvalidInput(f"stays[{index}]: {error}") from error
            hours = scored["hours"]
            latest = hours[-1]
            alerts = [h["hour"] for h in hours if h["alert_event"]]
            predictions.append({
                "patient_id": scored["patient_id"],
                "prediction_id": latest["prediction_id"],
                "risk_tier": latest["risk_tier"],
                "input_warnings": scored["input_warnings"],
                "hour": latest["hour"],
                "hours_scored": len(hours),
                "probability": latest["probability"],
                "above_threshold": bool(latest["hourly_positive"]),
                "new_alert": bool(latest["alert_event"]),
                "last_alert_hour": alerts[-1] if alerts else None,
            })
        return {
            "model_name": self.bundle["model_name"],
            "model_version": self.model_version,
            "bundle_sha256": self.bundle_sha256,
            "synthetic_model": bool(self.bundle["synthetic"]),
            "threshold": float(self.bundle["threshold"]),
            "cooldown_hours": int(self.bundle["cooldown"]),
            "scored_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "alert_source": "ledger" if self.alert_store is not None else "stateless_replay",
            "predictions": predictions,
            "disclaimer": DISCLAIMER,
        }
