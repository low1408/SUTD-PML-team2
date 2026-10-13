"""Separate hourly model decisions from notifications with a cooldown."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def alert_events(hours, probabilities, threshold, cooldown_hours):
    if cooldown_hours < 0:
        raise ValueError("cooldown_hours must be nonnegative")
    events = np.zeros(len(hours), dtype=int)
    last_alert = -np.inf
    for i, (hour, probability) in enumerate(zip(hours, probabilities)):
        if probability >= threshold and hour - last_alert >= cooldown_hours:
            events[i] = 1
            last_alert = hour
    return events


def apply_policy(table: pd.DataFrame, threshold: float, cooldown: int):
    result = table.copy()
    result["hourly_positive"] = (result["probability"] >= threshold).astype(int)
    result["alert_event"] = 0
    for _, patient in result.groupby("patient_id", sort=False):
        patient = patient.sort_values("hour")
        result.loc[patient.index, "alert_event"] = alert_events(
            patient["hour"], patient["probability"], threshold, cooldown
        )
    return result


def utility_components(labels):
    """Default PhysioNet 2019 utility weights, from the published scorer.

    Reference: github.com/physionetchallenges/evaluation-2019
    Independently expressed here as utility for positive vs negative decisions.
    Hours use zero-based row offsets, as in the official scorer.
    """
    labels = np.asarray(labels)
    if not labels.any():
        return np.full(len(labels), -0.05), np.zeros(len(labels))
    onset = int(np.argmax(labels)) + 6
    delta = np.arange(len(labels)) - onset
    positive = np.where(
        delta <= -6, np.maximum((delta + 12) / 6, -0.05), (3 - delta) / 9
    )
    negative = np.where(delta <= -6, 0.0, -2 * (delta + 6) / 9)
    # The official utility ceases accumulating after onset + 3 hours.
    return np.where(delta <= 3, positive, 0), np.where(delta <= 3, negative, 0)


def normalized_utility(table):
    observed = best = inactive = 0.0
    for _, patient in table.groupby("patient_id", sort=False):
        patient = patient.sort_values("hour")
        positive, negative = utility_components(patient["label"])
        decisions = patient["hourly_positive"].to_numpy()
        observed += np.where(decisions, positive, negative).sum()
        best += np.maximum(positive, negative).sum()
        inactive += negative.sum()
    denominator = best - inactive
    return float((observed - inactive) / denominator) if denominator > 0 else None


def summarize(table):
    labels = table["label"].to_numpy()
    probability = table["probability"].to_numpy()
    nonseptic_hours = false_events = septic_patients = eligible = detected = 0
    left_censored = 0
    leads = []
    for _, patient in table.groupby("patient_id", sort=False):
        patient = patient.sort_values("hour")
        positive_rows = np.flatnonzero(patient["label"].to_numpy())
        if not len(positive_rows):
            nonseptic_hours += len(patient)
            false_events += int(patient["alert_event"].sum())
            continue
        septic_patients += 1
        if positive_rows[0] == 0:
            # Label boundary may have occurred before admission. Do not pretend
            # its first row + 6 is a precisely observed clinical onset.
            left_censored += 1
            continue
        onset_hour = float(patient.iloc[positive_rows[0]]["hour"]) + 6
        if onset_hour > float(patient["hour"].max()):
            continue
        eligible += 1
        useful = patient.loc[
            (patient["alert_event"] == 1)
            & (patient["hour"] >= onset_hour - 12)
            & (patient["hour"] < onset_hour)
        ]
        if not useful.empty:
            detected += 1
            leads.append(onset_hour - float(useful["hour"].min()))
    return {
        "patients": int(table["patient_id"].nunique()),
        "hours": int(len(table)),
        "positive_hour_fraction": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, probability)) if labels.any() else None,
        "auroc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None,
        "brier_score": float(brier_score_loss(labels, probability)),
        "normalized_hourly_utility": normalized_utility(table),
        "alert_events": int(table["alert_event"].sum()),
        "false_alert_events_nonseptic": false_events,
        "nonseptic_patient_days": nonseptic_hours / 24,
        "false_alert_events_per_nonseptic_patient_day": false_events / (nonseptic_hours / 24) if nonseptic_hours else None,
        "septic_patients": septic_patients,
        "eligible_septic_patients_for_lead_time": eligible,
        "left_censored_septic_patients": left_censored,
        "early_detection_recall": detected / eligible if eligible else None,
        "median_inferred_lead_hours": float(np.median(leads)) if leads else None,
    }


def tune_threshold(table, budget=0.2, cooldown=6):
    """Maximize hourly utility subject to an empirical validation alarm budget.

    The budget counts notifications in never-septic patients per observed
    nonseptic patient-day. It is an empirical constraint, not a guarantee at B.
    Cooldown makes alarm counts nonmonotone, so evaluate every grid candidate.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if table.groupby("patient_id")["label"].max().nunique() != 2:
        raise ValueError("Threshold validation requires septic and nonseptic patients")
    candidates = []
    # 1.01 is an explicit no-alert fallback, including for probability == 1.
    for threshold in np.r_[np.linspace(0, 1, 101), 1.01]:
        metrics = summarize(apply_policy(table, float(threshold), cooldown))
        metrics["threshold"] = float(threshold)
        rate = metrics["false_alert_events_per_nonseptic_patient_day"]
        if rate is not None and rate <= budget + 1e-12:
            candidates.append(metrics)
    # Tie-break toward fewer false notifications, then a higher threshold.
    return max(candidates, key=lambda m: (
        m["normalized_hourly_utility"],
        -m["false_alert_events_per_nonseptic_patient_day"], m["threshold"],
    ))
