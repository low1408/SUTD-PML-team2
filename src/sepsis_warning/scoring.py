"""Score whole cohorts with a saved model bundle (shared by monitoring and replay).

The bundle is the ``model.joblib`` written by ``train``/``demo``: calibrated
model, feature columns, feature window, threshold and cooldown. Loading a
Joblib file can execute code, so only load bundles this project produced and,
where a reference records one, check the file's SHA-256 first.
"""

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import Patient
from .features import META, feature_table
from .policy import apply_policy

BUNDLE_KEYS = {"model", "model_name", "feature_columns", "window", "threshold", "cooldown", "synthetic"}
# Patients per scoring batch; bounds feature-table memory on the full cohort.
BATCH_PATIENTS = 1000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: Path, expected_sha256: str | None = None) -> dict:
    """Load a local model bundle, refusing it if its hash does not match.

    Without an explicit hash, the one recorded in ``manifest.json`` next to the
    bundle is used when that file exists. Joblib can run code on load, so the
    hash is checked before the file is opened.
    """
    path = Path(path)
    source = "the expected hash"
    if expected_sha256 is None:
        from .manifest import recorded_sha256

        expected_sha256, source = recorded_sha256(path), "manifest.json beside it"
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"{path} has SHA-256 {actual}, but {source} says {expected_sha256}; "
                "use the frozen bundle, or rebuild whatever was pinned to it"
            )
    bundle = joblib.load(path)
    missing = BUNDLE_KEYS - set(bundle) if isinstance(bundle, dict) else BUNDLE_KEYS
    if missing:
        raise ValueError(f"{path} is not a model bundle; missing {sorted(missing)}")
    return bundle


def score_patients(bundle: dict, patients: list[Patient], input_columns: list[str] | None = None) -> pd.DataFrame:
    """Hourly scores for every patient with the frozen bundle.

    Returns ``patient_id``, ``hospital``, ``hour``, ``label``, ``probability``,
    ``hourly_positive`` and ``alert_event``, plus the raw ``input_columns``
    when requested (for input monitoring). Rows keep each patient's hour order.
    Nothing is refitted: the threshold and cooldown come from the bundle.
    """
    chunks = []
    for start in range(0, len(patients), BATCH_PATIENTS):
        batch = patients[start:start + BATCH_PATIENTS]
        table = feature_table(batch, bundle["window"])
        scored = table[META].copy()
        with threadpool_limits(limits=2):
            scored["probability"] = bundle["model"].predict_proba(table[bundle["feature_columns"]])[:, 1]
        scored = apply_policy(scored, bundle["threshold"], bundle["cooldown"])
        if input_columns:
            # feature_table concatenates patients in order and keeps row order,
            # so raw rows line up with the scored rows one to one.
            # read_patient already guarantees these columns exist.
            raw = pd.concat([patient.frame[input_columns] for patient in batch], ignore_index=True)
            scored = pd.concat([scored, raw], axis=1)
        chunks.append(scored)
    if not chunks:
        raise ValueError("No patients to score")
    return pd.concat(chunks, ignore_index=True)
