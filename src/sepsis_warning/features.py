"""Patient-local features: feature[t] depends only on observations up to t."""

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, Patient


def build_features(frame: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be positive")
    columns = {}
    hour = np.arange(len(frame), dtype=float)
    for name in DYNAMIC:
        raw = frame[name]
        observed = raw.notna().to_numpy()
        # No backfill or interpolation across future measurements.
        last = raw.ffill()
        last_seen_hour = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
        roll = raw.rolling(window, min_periods=1)
        columns[f"{name}__last"] = last
        columns[f"{name}__missing_now"] = (~observed).astype(float)
        columns[f"{name}__never_seen"] = last.isna().astype(float)
        columns[f"{name}__hours_since_seen"] = hour - last_seen_hour
        columns[f"{name}__mean_{window}h"] = roll.mean()
        columns[f"{name}__std_{window}h"] = roll.std(ddof=0)
        columns[f"{name}__change_{window}h"] = last - last.shift(window)
        columns[f"{name}__observations_{window}h"] = roll.count()
    for name in STATIC:
        columns[name] = frame[name].ffill()
    return pd.DataFrame(columns).astype(np.float32)


def feature_table(patients: list[Patient], window: int) -> pd.DataFrame:
    chunks = []
    for patient in patients:
        features = build_features(patient.frame, window)
        features["patient_id"] = patient.patient_id
        features["hospital"] = patient.hospital
        features["hour"] = patient.frame["ICULOS"].to_numpy()
        # The challenge has ALREADY shifted its labels six hours earlier.
        features["label"] = patient.frame["SepsisLabel"].to_numpy(dtype=int)
        chunks.append(features)
    return pd.concat(chunks, ignore_index=True)


META = ["patient_id", "hospital", "hour", "label"]
