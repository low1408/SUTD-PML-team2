"""Patient-local features: feature[t] depends only on observations up to t."""

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, Patient


def build_features(frame: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Features at hour t use this stay only, through hour t.

    The last observed value is carried forward until the stay ends or a new
    measurement arrives. There is no expiry. A missing value is never filled
    from a later hour. Hours since a measurement are ICU hours from ICULOS,
    so a stay that starts after hour 1, or skips an hour, still counts the
    ICU-hour gap.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if "ICULOS" not in frame.columns or frame["ICULOS"].isna().any():
        raise ValueError("ICULOS is required for every row")
    columns = {}
    hour = frame["ICULOS"].to_numpy(dtype=float)
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


def select_saved_features(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Keep the saved column list, in that order. Reject a stay that cannot supply it."""
    if not feature_columns:
        raise ValueError("saved model has no feature columns")
    missing = [name for name in feature_columns if name not in features.columns]
    if missing:
        shown = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(f"saved feature columns are missing from this stay: {shown}{extra}")
    return features.loc[:, feature_columns]


def _window_column(name: str) -> bool:
    return any(token in name for token in ("__mean_", "__std_", "__change_", "__observations_"))


def features_for_windows(frame: pd.DataFrame, windows: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    """One shared builder, called once per window. Column order is stable."""
    if not windows:
        raise ValueError("windows must be positive")
    built = [build_features(frame, window) for window in windows]
    base = [name for name in built[0].columns if not _window_column(name)]
    pieces = [built[0].loc[:, base]]
    for window, features in zip(windows, built):
        names = [name for name in features.columns if name.endswith(f"_{window}h")]
        pieces.append(features.loc[:, names])
    return pd.concat(pieces, axis=1)


def fit_training_fallback(frames: list[pd.DataFrame]) -> pd.Series:
    """Median of observed measurements. Pass training tables only.

    A column that is blank in every training row stays blank. Rows that are
    not included here cannot change the medians. Carry-forward still has no
    expiry; this function only supplies a value before the first observation.
    """
    if not frames:
        raise ValueError("training fallback needs at least one table")
    columns = [name for name in DYNAMIC if any(name in frame.columns for frame in frames)]
    if not columns:
        raise ValueError("training tables have no measurement columns")
    observed = pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)
    return observed.median()


def apply_training_fallback(frame: pd.DataFrame, fallback: pd.Series) -> pd.DataFrame:
    """Forward-fill inside each stay, then fill what is still missing.

    Does not refit the medians and does not modify `frame`. A later hour never
    fills an earlier blank. A value from one stay never fills another stay.
    """
    columns = [name for name in DYNAMIC if name in frame.columns]

    def fill_stay(stay: pd.DataFrame) -> pd.DataFrame:
        filled = stay.copy()
        for name in columns:
            filled[name] = filled[name].ffill()
            if name in fallback.index:
                filled[name] = filled[name].fillna(fallback[name])
        return filled

    if "patient_id" in frame.columns and frame["patient_id"].nunique() > 1:
        stays = [fill_stay(stay) for _, stay in frame.groupby("patient_id", sort=False)]
        return pd.concat(stays)
    return fill_stay(frame)
