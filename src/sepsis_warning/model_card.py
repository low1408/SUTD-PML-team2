"""Model card for the served model: which model was selected and why (validation only).

Built from the files ``train``/``demo`` write next to ``model.joblib``
(``selection_metrics.json`` or ``metrics.json``, ``run_metadata.json``), or
from the same files downloaded from the registry. Held-out test results are
never included, so the API cannot leak them, and a card that disagrees with
the loaded bundle is refused.
"""

import json
from pathlib import Path

SELECTION_RULE = (
    "Highest validation normalized hourly utility at a threshold tuned to the false-alert budget; "
    "ties broken by fewer false alerts per non-septic patient-day, then higher average precision."
)
CANDIDATE_FIELDS = (
    "auroc", "average_precision", "brier_score", "normalized_hourly_utility",
    "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
    "median_inferred_lead_hours", "threshold",
)
TRAINING_FIELDS = ("seed", "data_mode", "window_hours", "alarm_budget", "feature_count",
                   "feature_columns_sha256", "git_commit")


class CardMismatch(ValueError):
    """The selection files do not describe the loaded bundle."""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def card_from_run_dir(run_dir: Path, bundle: dict, registry: dict | None = None) -> dict | None:
    """Card for ``bundle`` from its run folder; None if the folder has no selection files."""
    run_dir = Path(run_dir)
    selection = _read(run_dir / "selection_metrics.json") or _read(run_dir / "metrics.json")
    if selection is None:
        return None
    if selection.get("selected_model") != bundle["model_name"] or not _same(selection.get("threshold"), bundle["threshold"]):
        raise CardMismatch(
            f"selection files in {run_dir} describe {selection.get('selected_model')} at threshold "
            f"{selection.get('threshold')}, but the bundle is {bundle['model_name']} at {bundle['threshold']}"
        )
    metadata = _read(run_dir / "run_metadata.json") or {}
    registry = registry or _read(run_dir / "registry.json")  # written when a deployment stages a version
    candidates = [
        {"model": row["model"], "selected": row["model"] == bundle["model_name"],
         **{field: row.get(field) for field in CANDIDATE_FIELDS}}
        for row in selection.get("validation_models", [])
    ]
    candidates.sort(key=lambda row: (not row["selected"], -(row["normalized_hourly_utility"] or 0)))
    return {
        "selected_model": bundle["model_name"],
        "threshold": float(bundle["threshold"]),
        "cooldown_hours": int(bundle["cooldown"]),
        "validation_alarm_budget": selection.get("validation_alarm_budget"),
        "selection_rule": SELECTION_RULE,
        "evaluated_on": "validation split only; held-out test results are not served",
        "candidates": candidates,
        "training": {field: metadata.get(field) for field in TRAINING_FIELDS if field in metadata},
        "registry": registry,
    }


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return False
