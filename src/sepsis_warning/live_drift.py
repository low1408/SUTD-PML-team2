"""Label-free drift checks on the live prediction log (the fast checks of ``monitoring.py``).

Input: prediction lines written by ``telemetry.py`` (hashed stay ID, hour,
probability, alert flag, which measurements were missing). They hold no
measurement values, so live checks cover what can be seen without them:

* **score shift**: PSI of the probabilities against the frozen reference;
* **alert rate**: new alerts per patient-day, as a ratio to the reference;
* **missingness shift**: per measurement, how much more or less often it is
  missing than in the reference (a changed lab panel or a broken feed).

Value-distribution drift and performance drift still need the offline replay
(``monitor-replay``) on stays with settled labels.

The reference is the file ``monitor-setup`` freezes for one bundle. A log
produced by another bundle is reported as ``reference_mismatch`` instead of
being compared. An alarm asks a person to look; nothing here retrains or
changes a threshold.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DYNAMIC
from .monitoring import bin_shares, psi

MIN_ROWS = 200
THRESHOLDS: dict[str, dict] = {
    "score_psi": {"warn": 0.10, "alarm": 0.25},
    # Alerts per patient-day relative to the reference: half or double is a warning.
    "alert_rate_ratio": {"warn": (0.5, 2.0), "alarm": (0.25, 4.0)},
    "max_missingness_shift": {"warn": 0.10, "alarm": 0.20},
}
LEVEL = {"ok": 0, "warn": 1, "alarm": 2}


def _json_objects(text: str):
    """Yield JSON objects from JSONL or from objects written back to back (Firehose)."""
    decoder, index = json.JSONDecoder(), 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return
        value, index = decoder.raw_decode(text, index)
        yield value


def _lines(data: bytes):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    for value in _json_objects(data.decode("utf-8")):
        if isinstance(value, dict) and "logEvents" in value:          # CloudWatch Logs -> Firehose batch
            for event in value["logEvents"]:
                try:
                    yield json.loads(event["message"])
                except (ValueError, KeyError):
                    continue
        elif isinstance(value, dict):
            yield value


def read_prediction_log(sources) -> pd.DataFrame:
    """Prediction records from files (``.jsonl``, ``.gz``, Firehose objects), folders, or raw bytes."""
    records: list[dict] = []
    for source in sources:
        if isinstance(source, bytes):
            chunks = [source]
        else:
            path = Path(source)
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            chunks = [file.read_bytes() for file in files]
        for chunk in chunks:
            records.extend(r for r in _lines(chunk) if r.get("record_type") == "prediction")
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    # A retried request logs the same prediction twice; count it once.
    return frame.drop_duplicates("prediction_id", keep="last").reset_index(drop=True)


def _status(name: str, value) -> str:
    limits: dict = THRESHOLDS[name]
    if value is None:
        return "ok"
    for level in ("alarm", "warn"):
        bound = limits[level]
        outside = not (bound[0] <= value <= bound[1]) if isinstance(bound, tuple) else value > bound
        if outside:
            return level
    return "ok"


def live_checks(predictions: pd.DataFrame, reference: dict) -> dict:
    """Compare a window of prediction records with the frozen reference."""
    rows = len(predictions)
    checks: dict[str, dict] = {}
    result: dict = {"rows": rows, "stays": int(predictions["stay"].nunique()) if rows else 0, "checks": checks}
    if rows:
        bundles = sorted(set(predictions["bundle_sha256"].dropna().astype(str)))
        result["bundles"] = bundles
        expected = str(reference.get("model_sha256", ""))[:12]
        if expected and any(b != expected for b in bundles):
            result["status"] = "reference_mismatch"
            result["detail"] = (f"the log holds bundle(s) {bundles}, but the reference was frozen for {expected}; "
                                "run monitor-setup for the deployed model")
            return result
    if rows < MIN_ROWS:
        result["status"] = "insufficient_data"
        result["detail"] = f"{rows} prediction rows; at least {MIN_ROWS} are needed"
        return result

    score = reference["score"]
    live_shares = bin_shares(predictions["probability"].to_numpy(dtype=float), np.asarray(score["edges"]))
    score_psi = psi(score["shares"], live_shares)
    live_rate = float(predictions["new_alert"].astype(bool).sum() / (rows / 24))
    reference_rate = score.get("alerts_per_patient_day") or 0.0
    ratio = live_rate / reference_rate if reference_rate > 0 else None

    missing_counts = pd.Series(0, index=DYNAMIC, dtype=float)
    for names in predictions["missing_now"]:
        missing_counts.loc[[n for n in names if n in missing_counts.index]] += 1
    shifts = {}
    for name in DYNAMIC:
        baseline = (reference.get("inputs", {}).get(name) or {}).get("missing_fraction")
        if baseline is not None:
            shifts[name] = float(missing_counts[name] / rows - baseline)
    worst = sorted(shifts.items(), key=lambda item: -abs(item[1]))[:5]
    max_shift = abs(worst[0][1]) if worst else None

    values = {"score_psi": score_psi, "alert_rate_ratio": ratio, "max_missingness_shift": max_shift}
    for name, value in values.items():
        checks[name] = {"value": value, "status": _status(name, value), "thresholds": THRESHOLDS[name]}
    result["alerts_per_patient_day"] = {"live": live_rate, "reference": reference_rate}
    result["largest_missingness_shifts"] = [{"input": name, "shift": shift} for name, shift in worst]
    result["status"] = max((c["status"] for c in checks.values()), key=lambda status: LEVEL[status])
    return result


def run_live_drift(sources, reference_path: Path, output: Path | None = None) -> dict:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    result = live_checks(read_prediction_log(sources), reference)
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "live_drift.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    return result


def emf_record(result: dict, environment: str) -> dict:
    """CloudWatch Embedded Metric Format line for a drift result (printed by the scheduled job)."""
    checks = result.get("checks", {})
    record = {
        "_aws": {"Timestamp": int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), "CloudWatchMetrics": [{
            "Namespace": "SepsisWarning/Drift", "Dimensions": [["Environment"]],
            "Metrics": [{"Name": "DriftLevel", "Unit": "None"}, {"Name": "ScorePSI", "Unit": "None"},
                        {"Name": "MaxMissingnessShift", "Unit": "None"}, {"Name": "PredictionRows", "Unit": "Count"}],
        }]},
        "record_type": "drift", "Environment": environment, "status": result["status"],
        # 0 ok, 1 warn, 2 alarm; -1 when the window could not be judged.
        "DriftLevel": LEVEL.get(result["status"], -1),
        "ScorePSI": (checks.get("score_psi") or {}).get("value") or 0.0,
        "MaxMissingnessShift": (checks.get("max_missingness_shift") or {}).get("value") or 0.0,
        "PredictionRows": result["rows"], "detail": result.get("detail"),
    }
    return record
