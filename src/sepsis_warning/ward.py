"""ICU ward replay: a self-contained page that plays stays hour by hour.

    python -m sepsis_warning ward --model artifacts/demo/model.joblib --data data/synthetic

Several stays are placed in beds with staggered admission times and scored
hour by hour with a frozen model bundle, either in this process or through the
running scoring API (``--api-url``). The page shows what a care team would
see: each bed's current risk, threshold crossings, alerts after the 6-hour
cooldown, and an alert feed, next to the real-world step each panel stands
for. Outcomes (the challenge label) stay hidden unless the viewer turns on
hindsight.

Rules kept here:

* Only train, calibration and validation stays by default. Internal-test and
  hospital-B stays need ``--after-freeze``, so test data is not looked at
  before the model is frozen.
* Beds are named "Bed 01" and so on; no record ID enters the page. A page
  built from PhysioNet stays still holds patient-level trajectories, so it is
  marked private and belongs in the Git-ignored ``artifacts/`` folder.
* Septic stays are over-represented so the demo shows alerts; the page states
  the real share next to the ward's share.
"""

import json
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, Patient
from .scoring import score_patients

SHOWN_VITALS = ["HR", "MAP", "Temp", "Resp", "O2Sat"]
SELECTION_SPLITS = ("train", "calibration", "validation")
HELD_OUT_SPLITS = ("internal_test", "external_test")
MAX_OFFSET_HOURS = 18
TEMPLATE = "ward_template.html"


def check_split(split: str, after_freeze: bool) -> None:
    if split in HELD_OUT_SPLITS and not after_freeze:
        raise ValueError(
            f"{split} is a test split; replay it only after the model is frozen (pass --after-freeze)"
        )
    if split not in SELECTION_SPLITS + HELD_OUT_SPLITS:
        raise ValueError(f"unknown split {split!r}")


def _first_label_row(frame: pd.DataFrame) -> int | None:
    rows = np.flatnonzero(frame["SepsisLabel"].to_numpy())
    return int(rows[0]) if len(rows) else None


def choose_stays(patients: list[Patient], beds: int, septic_share: float, max_hours: int,
                 seed: int) -> list[Patient]:
    """Pick a seeded mix of stays and trim each to ``max_hours`` rows.

    Septic stays are preferred when their inferred onset (first label + 6 h)
    falls inside the shown window and after the first hour, so an early
    warning is possible. Trimming keeps the first rows, and features at hour t
    use only hours up to t, so scores on the shown hours are unchanged.
    """
    if not 1 <= beds <= 24:
        raise ValueError("beds must be between 1 and 24")
    if not 0 <= septic_share <= 1:
        raise ValueError("septic share must be between 0 and 1")
    rng = np.random.default_rng(seed)
    septic, other = [], []
    for patient in patients:
        first = _first_label_row(patient.frame)
        if first is None:
            other.append(patient)
        elif 0 < first and first + 6 < max_hours:
            septic.append(patient)
    want_septic = min(len(septic), round(beds * septic_share))
    want_other = min(len(other), beds - want_septic)
    if want_septic + want_other == 0:
        raise ValueError("no stays to show")
    picked = [septic[i] for i in rng.choice(len(septic), want_septic, replace=False)] if want_septic else []
    picked += [other[i] for i in rng.choice(len(other), want_other, replace=False)] if want_other else []
    rng.shuffle(picked)
    return [replace(p, frame=p.frame.iloc[:max_hours].reset_index(drop=True)) for p in picked]


def score_locally(bundle: dict, stays: list[Patient]) -> list[pd.DataFrame]:
    """Hourly probability, decision and alert per stay, scored in this process."""
    scored = score_patients(bundle, stays, SHOWN_VITALS)
    return [scored.loc[scored["patient_id"] == stay.patient_id].reset_index(drop=True) for stay in stays]


def _post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score_via_api(api_url: str, stays: list[Patient], timeout: float = 30.0) -> tuple[list[pd.DataFrame], dict]:
    """Score each stay through ``POST /score``, as the hospital engine would.

    Sends only model inputs (no label, no record ID) and returns the same
    columns as ``score_locally`` plus the API's model details.
    """
    base = api_url.rstrip("/")
    frames, details = [], {}
    for index, stay in enumerate(stays, start=1):
        inputs = stay.frame[DYNAMIC + STATIC].astype(object).where(stay.frame[DYNAMIC + STATIC].notna(), None)
        response = _post_json(f"{base}/score", {"patient_id": f"bed-{index:02d}",
                                                "hours": inputs.to_dict(orient="records")}, timeout)
        hours = pd.DataFrame(response["hours"])
        frame = pd.DataFrame({
            "hour": hours["hour"], "probability": hours["probability"],
            "hourly_positive": hours["hourly_positive"], "alert_event": hours["alert_event"],
            "label": stay.frame["SepsisLabel"].to_numpy(dtype=int),
        })
        for name in SHOWN_VITALS:
            frame[name] = stay.frame[name].to_numpy()
        frames.append(frame)
        details = {"model_name": response["model_name"], "threshold": response["threshold"],
                   "cooldown_hours": response["cooldown_hours"], "synthetic_model": response["synthetic_model"]}
    return frames, details


def _rounded(values, digits):
    return [None if value is None or not np.isfinite(value) else round(float(value), digits) for value in values]


def build_payload(stays_scored: list[pd.DataFrame], *, model_name: str, threshold: float, cooldown: int,
                  synthetic: bool, split: str, scored_by: str, cohort_septic_share: float,
                  seed: int) -> dict:
    """Page data: one entry per bed, with staggered admission hours. No record IDs."""
    rng = np.random.default_rng(seed + 1)
    beds = []
    for index, frame in enumerate(stays_scored, start=1):
        vitals = frame[SHOWN_VITALS].ffill()  # last charted value, inside this stay only
        beds.append({
            "bed": f"Bed {index:02d}",
            "offset": int(rng.integers(0, MAX_OFFSET_HOURS + 1)),
            "icu_hour": [int(h) for h in frame["hour"]],
            "risk": _rounded(frame["probability"], 4),
            "above": [int(v) for v in frame["hourly_positive"]],
            "alert": [int(v) for v in frame["alert_event"]],
            "label": [int(v) for v in frame["label"]],
            "vitals": {name: _rounded(vitals[name], 1) for name in SHOWN_VITALS},
        })
    septic_beds = sum(any(bed["label"]) for bed in beds)
    return {
        "model_name": model_name, "threshold": float(threshold), "cooldown_hours": int(cooldown),
        "synthetic": bool(synthetic), "split": split, "scored_by": scored_by,
        "ward_hours": max(bed["offset"] + len(bed["risk"]) for bed in beds),
        "ward_septic_share": septic_beds / len(beds), "cohort_septic_share": float(cohort_septic_share),
        "alert_budget_per_day": 0.2, "beds": beds,
    }


def write_ward(path: Path, payload: dict) -> None:
    """Fill the page template; refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} already exists; choose a new output file")
    text = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name(TEMPLATE).read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__PAYLOAD__", text), encoding="utf-8")
