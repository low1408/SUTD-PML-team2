"""H02: depth-2 log-signature features over a trailing window (uses past and current hours only).

Channels: time plus the HMM vitals (HR, MAP, Resp, O2Sat).
Time encoding: hour / (window - 1), so a full window runs from 0 to 1.
Scaling: each vital is standardised with the TRAIN mean and standard deviation.
Missing values: implausible readings become NaN, then Jawad's apply_training_fallback
(carry forward inside the stay, then the TRAIN median). Nothing from a later hour is used.
Depth: 2, as a log-signature: the increment of every channel, plus the Levy area of every
pair of channels (positive area for (a, b) means a tended to change before b).
Short history: hour t uses hours max(0, t - window + 1) .. t; the first hour gives zeros.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import apply_training_fallback, fit_training_fallback
from .hmm import VITALS, vital_observations

WINDOW = 6


def log_signature_depth2(path: np.ndarray) -> np.ndarray:
    """Increments, then Levy areas for channel pairs i < j, of one piecewise-linear path (points, channels)."""
    path = np.asarray(path, dtype=float)
    channels = path.shape[1]
    i, j = np.triu_indices(channels, k=1)
    if len(path) < 2:
        return np.zeros(channels + len(i))
    steps = np.diff(path, axis=0)                 # increment of each segment
    before = path[:-1] - path[0]                  # total change before each segment
    m = before.T @ steps                          # m[a, b] = sum over segments of (a so far) * (b in this segment)
    area = 0.5 * (m - m.T)
    return np.concatenate([path[-1] - path[0], area[i, j]])


def feature_names(columns=VITALS) -> list[str]:
    channels = ["time", *columns]
    i, j = np.triu_indices(len(channels), k=1)
    return [f"sig_inc_{c}" for c in channels] + [f"sig_area_{channels[a]}_{channels[b]}" for a, b in zip(i, j)]


@dataclass
class SignatureScaler:
    """Everything fitted on the TRAIN split: fallback medians, means and standard deviations."""

    fallback: pd.Series
    means: np.ndarray
    stds: np.ndarray
    columns: tuple = VITALS


def _clean(frame: pd.DataFrame, columns) -> pd.DataFrame:
    return pd.DataFrame(vital_observations(frame, columns), index=frame.index, columns=list(columns))


def fit_signature_scaler(train_frames: list[pd.DataFrame], columns=VITALS) -> SignatureScaler:
    """Pass TRAIN stays only."""
    cleaned = [_clean(frame, columns) for frame in train_frames]
    fallback = fit_training_fallback(cleaned)
    observed = np.vstack([frame.to_numpy() for frame in cleaned])
    means = np.nanmean(observed, axis=0)
    stds = np.nanstd(observed, axis=0)
    if np.any(np.isnan(means)) or np.any(stds <= 0):
        raise ValueError("every vital needs at least two different observed training values")
    return SignatureScaler(fallback, means, stds, tuple(columns))


def signature_features(frame: pd.DataFrame, scaler: SignatureScaler, window: int = WINDOW) -> pd.DataFrame:
    """Log-signature features for every hour of ONE stay, same index as `frame`.

    Same result as calling log_signature_depth2 on each trailing window, but computed with
    running sums so a whole stay takes one pass. tests/test_signature.py checks they agree.
    """
    if window < 1:
        raise ValueError("window must be at least 1 hour")
    filled = apply_training_fallback(_clean(frame, scaler.columns), scaler.fallback)
    vitals = (filled.to_numpy(dtype=float) - scaler.means) / scaler.stds
    hours = len(vitals)
    time = np.arange(hours, dtype=float) / max(window - 1, 1)
    path = np.column_stack([time, vitals])        # (hours, channels)

    steps = np.zeros_like(path)
    steps[1:] = np.diff(path, axis=0)
    previous = np.zeros_like(path)
    previous[1:] = path[:-1]
    running = np.cumsum(previous[:, :, None] * steps[:, None, :], axis=0)   # sum of X[l-1] (x) step[l]

    end = np.arange(hours)
    start = np.maximum(end - window + 1, 0)
    increment = path[end] - path[start]
    m = (running[end] - running[start]) - path[start][:, :, None] * increment[:, None, :]
    area = 0.5 * (m - m.transpose(0, 2, 1))
    i, j = np.triu_indices(path.shape[1], k=1)
    values = np.column_stack([increment, area[:, i, j]])
    return pd.DataFrame(values, index=frame.index, columns=feature_names(scaler.columns))
