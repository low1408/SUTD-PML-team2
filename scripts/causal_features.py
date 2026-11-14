"""Run the shared feature builder on a six-row example.

    python scripts/causal_features.py

No patient files are read. Training and replay call the same functions in
sepsis_warning.features and sepsis_warning.policy. This script does not keep
a second copy of the rolling features or the utility formula.
"""

import argparse

import numpy as np
import pandas as pd

from sepsis_warning.data import DYNAMIC, LABS, STATIC, Patient
from sepsis_warning.features import (
    apply_training_fallback,
    build_features,
    feature_table,
    features_for_windows,
    fit_training_fallback,
)
from sepsis_warning.policy import benchmark_utility, utility_components


WINDOWS = (3, 6, 12)


def example_frame():
    frame = pd.DataFrame(np.nan, index=range(6), columns=DYNAMIC + STATIC + ["SepsisLabel", "Unit1", "patient_id"])
    frame["patient_id"] = "a"
    frame["Unit1"] = [1, 1, 1, 0, 0, 0]
    frame["HR"] = [80, np.nan, np.nan, 90, 90, 100]
    frame["Lactate"] = [np.nan, np.nan, 1.5, np.nan, 2.0, np.nan]
    frame["SepsisLabel"] = [0, 0, 0, 0, 1, 1]
    frame["ICULOS"] = np.arange(1, 7)
    return frame


def hours_since_any_lab(features, lab_columns):
    """Earliest shared per-lab clock among labs that have been seen."""
    seen = [
        features[f"{name}__hours_since_seen"].where(features[f"{name}__never_seen"] == 0)
        for name in lab_columns
    ]
    return pd.concat(seen, axis=1).min(axis=1)


def utility_score(labels, decisions):
    positive, negative = utility_components(labels)
    decisions = np.asarray(decisions).astype(bool)
    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have the same length")
    return float(np.where(decisions, positive, negative).sum())


def leakage_check_corrupt_next_hour(frame):
    """Corrupting hour t+1 onward must leave features through hour t unchanged."""
    if len(frame) < 2:
        raise ValueError("need at least two hourly rows")
    t = max(0, len(frame) // 2 - 1)
    before = build_features(frame)
    corrupted = frame.copy()
    rows = corrupted.index[t + 1 :]
    corrupted.loc[rows, DYNAMIC] = 9999
    corrupted.loc[rows, "SepsisLabel"] = 1 - corrupted.loc[rows, "SepsisLabel"].astype(int)
    corrupted.loc[rows, "Unit1"] = 999
    after = build_features(corrupted)
    pd.testing.assert_frame_equal(before.iloc[: t + 1], after.iloc[: t + 1])
    return True


def record_isolation_check(left, right):
    """Features for one patient stay the same when another patient is in the table."""
    alone = feature_table([Patient("left", "A", left)], 6)
    together = feature_table([
        Patient("left", "A", left),
        Patient("right", "A", right),
    ], 6)
    columns = [name for name in alone.columns if name not in ("patient_id", "hospital")]
    built = together.loc[together["patient_id"] == "left", columns]
    pd.testing.assert_frame_equal(alone[columns].reset_index(drop=True), built.reset_index(drop=True))
    return True


def check_utility_examples():
    published = utility_score([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1])
    if abs(published - 3.388888888888889) > 1e-9:
        raise AssertionError(f"published utility example returned {published}")
    blank = np.zeros(4, dtype=int)
    if utility_score(blank, np.zeros(4, dtype=int)) != 0:
        raise AssertionError("an all-negative record should score 0 when no alert fires")
    if abs(utility_score(blank, np.ones(4, dtype=int)) - (-0.2)) > 1e-9:
        raise AssertionError("false alerts on a non-septic record should score -0.05 each hour")
    long_label = np.r_[np.zeros(18), np.ones(22)]
    positive, negative = utility_components(long_label)
    if not (positive[28:] == 0).all() or not (negative[28:] == 0).all():
        raise AssertionError("utility should stop three hours after the inferred onset")
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    frame = example_frame()
    features = features_for_windows(frame, WINDOWS)
    again = features_for_windows(frame, WINDOWS)
    pd.testing.assert_frame_equal(features, again)
    shared = build_features(frame, 6)
    pd.testing.assert_frame_equal(features[shared.columns], shared)
    train = frame.loc[:, ["HR", "Lactate", "Fibrinogen"]]
    fallback = fit_training_fallback([train])
    held_out = pd.DataFrame({
        "HR": [np.nan, np.nan],
        "Lactate": [np.nan, 4.0],
        "Fibrinogen": [np.nan, np.nan],
    })
    applied = apply_training_fallback(held_out, fallback)
    filled = apply_training_fallback(frame, fallback)
    score = check_utility_examples()
    timing = benchmark_utility()
    leakage_check_corrupt_next_hour(frame)
    other = frame.copy()
    other["HR"] = 50
    record_isolation_check(frame.drop(columns="patient_id"), other.drop(columns="patient_id"))
    print("Forward-filled HR:", filled["HR"].tolist())
    print("Hours since last lactate:", features["Lactate__hours_since_seen"].tolist())
    print("Lactate missing now:", features["Lactate__missing_now"].tolist())
    print("Hours since any lab:", hours_since_any_lab(features, LABS).tolist())
    print("HR mean over 3 hours:", [round(float(value), 2) for value in features["HR__mean_3h"]])
    print("Training median HR:", float(fallback["HR"]))
    print("Held-out HR after fallback:", applied["HR"].tolist())
    print("Unobserved training column stays blank:", bool(applied["Fibrinogen"].isna().all()))
    print(f"Feature columns: {len(shared.columns)} at float32")
    print(f"Utility score: {score:.4f}")
    print(
        "Utility benchmark: "
        f"{timing['patients']} patients x {timing['hours_per_patient']} hours "
        f"= {timing['patient_hours']} patient-hours in {timing['elapsed_seconds']:.4f} seconds"
    )
    print("Next-hour leakage check passed.")
    print("Record isolation check passed.")


if __name__ == "__main__":
    main()
