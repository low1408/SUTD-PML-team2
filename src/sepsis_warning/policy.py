"""Separate hourly model decisions from notifications with a cooldown."""

import time

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


def utility_training_targets(table, normalize=True):
    """Turn the hourly utility into a weighted classification problem.

    For every row this returns the decision the utility prefers at that hour
    (``target``: 1 = alarm) and how much utility is lost by deciding the other
    way (``weight``). For any set of hourly decisions,

        sum(weight * (decision != target)) == best utility - achieved utility,

    so a classifier that minimises weighted error on these targets is
    minimising utility regret. Three things differ from training on ``label``
    with balanced class weights:

    * Hours 7 to 11 before onset are targets (an early alarm is rewarded there)
      although the six-hour-shifted label is still 0.
    * Hours more than three after onset have weight 0, as do the hours exactly
      12 before onset; the utility ignores them.
    * A false alarm costs 0.05 while a missed septic hour costs up to 2, which
      takes the place of class weighting. Fit with ``class_weight=None``.

    Uses labels, so call it on training rows only. ``normalize`` rescales the
    weights to mean 1 so regularisation settings keep their meaning. Returns
    arrays aligned with the rows of ``table``.
    """
    labels = table["label"].to_numpy()
    hours = table["hour"].to_numpy()
    target = np.zeros(len(table), dtype=int)
    weight = np.zeros(len(table))
    for rows in table.groupby("patient_id", sort=False).indices.values():
        rows = rows[np.argsort(hours[rows], kind="stable")]
        positive, negative = utility_components(labels[rows])
        target[rows] = positive > negative
        weight[rows] = np.abs(positive - negative)
    if normalize and weight.sum() > 0:
        weight = weight / weight.mean()
    return target, weight


# Stated timing workload: 500 stays of 48 hours. This is not a speed limit.
UTILITY_BENCHMARK_PATIENTS = 500
UTILITY_BENCHMARK_HOURS = 48


def utility_benchmark_table(
    n_patients: int = UTILITY_BENCHMARK_PATIENTS,
    hours: int = UTILITY_BENCHMARK_HOURS,
) -> pd.DataFrame:
    """Fixed hourly decisions for timing the published utility."""
    patient_id = np.repeat(np.arange(n_patients), hours)
    hour = np.tile(np.arange(hours), n_patients)
    label = np.zeros(n_patients * hours, dtype=int)
    hourly_positive = np.zeros(n_patients * hours, dtype=int)
    label[:hours] = np.r_[np.zeros(hours - 12, dtype=int), np.ones(12, dtype=int)]
    hourly_positive[:hours] = np.r_[np.zeros(hours - 18, dtype=int), np.ones(18, dtype=int)]
    return pd.DataFrame({
        "patient_id": patient_id,
        "hour": hour,
        "label": label,
        "hourly_positive": hourly_positive,
    })


def benchmark_utility(table: pd.DataFrame | None = None) -> dict:
    """Time normalized hourly utility. The result records seconds; it does not set a limit."""
    if table is None:
        table = utility_benchmark_table()
    started = time.perf_counter()
    score = normalized_utility(table)
    elapsed = time.perf_counter() - started
    return {
        "patients": int(table["patient_id"].nunique()),
        "hours_per_patient": int(table.groupby("patient_id").size().iloc[0]),
        "patient_hours": len(table),
        "elapsed_seconds": float(elapsed),
        "normalized_utility": score,
    }


def patient_utilities(table) -> np.ndarray:
    """Per patient: utility of the decisions made, of the best decisions, and of never alerting."""
    rows = []
    for _, patient in table.groupby("patient_id", sort=False):
        patient = patient.sort_values("hour")
        positive, negative = utility_components(patient["label"])
        decisions = patient["hourly_positive"].to_numpy()
        rows.append((np.where(decisions, positive, negative).sum(),
                     np.maximum(positive, negative).sum(), negative.sum()))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _normalize(sums) -> float | None:
    observed, best, inactive = sums
    denominator = best - inactive
    return float((observed - inactive) / denominator) if denominator > 0 else None


def normalized_utility(table):
    return _normalize(patient_utilities(table).sum(axis=0))


def utility_interval(table, n_boot: int = 200, seed: int = 0) -> dict:
    """Patient-level bootstrap of the normalized utility: standard error and a 95% interval.

    Whole patients are resampled, because hours within a stay are not independent.
    """
    per_patient = patient_utilities(table)
    rng = np.random.default_rng(seed)
    count = len(per_patient)
    values = []
    for _ in range(n_boot):
        value = _normalize(per_patient[rng.integers(0, count, count)].sum(axis=0))
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return {"utility_standard_error": None, "utility_ci_low": None, "utility_ci_high": None}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"utility_standard_error": float(np.std(values, ddof=1)),
            "utility_ci_low": float(low), "utility_ci_high": float(high)}


SELECTION_RULE = (
    "One-standard-error rule on the validation split: take the model with the highest normalized "
    "hourly utility at a threshold tuned to the false-alert budget, then choose the simplest model "
    "whose utility is within one bootstrap standard error of it (ties: higher utility, fewer false "
    "alerts per non-septic patient-day, higher average precision)."
)


def select_candidate(comparisons: list[dict], complexity: dict) -> tuple[dict, dict]:
    """Pick a model from validation results. Returns (selected row, how it was chosen)."""
    def rank(m):
        return (m["normalized_hourly_utility"], -m["false_alert_events_per_nonseptic_patient_day"],
                m["average_precision"])

    best = max(comparisons, key=rank)
    margin = best.get("utility_standard_error") or 0.0
    floor = best["normalized_hourly_utility"] - margin
    eligible = [m for m in comparisons if m["normalized_hourly_utility"] >= floor]
    selected = min(eligible, key=lambda m: (complexity.get(m["model"], 99), tuple(-v for v in rank(m))))
    return selected, {
        "rule": SELECTION_RULE, "best_by_utility": best["model"],
        "best_utility": best["normalized_hourly_utility"], "standard_error": margin,
        "within_one_standard_error": [m["model"] for m in eligible], "selected": selected["model"],
    }


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
        "hours": len(table),
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
    chosen = max(candidates, key=lambda m: (
        m["normalized_hourly_utility"],
        -m["false_alert_events_per_nonseptic_patient_day"], m["threshold"],
    ))
    # The interval is for the chosen threshold only; it does not include threshold-tuning noise.
    chosen.update(utility_interval(apply_policy(table, chosen["threshold"], cooldown)))
    return chosen


def calibration_curve_bins(labels, probability, n_bins: int = 10) -> tuple[list[dict], float | None]:
    """Compute uniform-width reliability curve bins over [0, 1] and ECE."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError("labels and probability must have the same length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    total = len(y)
    for idx in range(n_bins):
        low, high = float(edges[idx]), float(edges[idx + 1])
        # Include 1.0 in the final bin so edge probabilities are never dropped.
        mask = (p >= low) & (p <= high if idx == n_bins - 1 else p < high)
        count = int(mask.sum())
        if count > 0:
            positives = int(y[mask].sum())
            mean_pred = float(p[mask].mean())
            obs_freq = float(y[mask].mean())
            ece += (count / total) * abs(obs_freq - mean_pred)
        else:
            positives = 0
            mean_pred = None
            obs_freq = None
        bins.append({
            "bin_index": idx,
            "bin_lower": low,
            "bin_upper": high,
            "count": count,
            "positive_count": positives,
            "mean_predicted": mean_pred,
            "observed_frequency": obs_freq,
        })
    return bins, (float(ece) if total > 0 else None)


def calibration_diagnostics(labels, uncalibrated_probability, calibrated_probability, n_bins: int = 10) -> dict:
    """Record before/after Platt scaling Brier scores, ECE, and binned calibration curves."""
    y = np.asarray(labels, dtype=np.float64)
    p_before = np.asarray(uncalibrated_probability, dtype=np.float64)
    p_after = np.asarray(calibrated_probability, dtype=np.float64)
    brier_before = float(brier_score_loss(y, p_before))
    brier_after = float(brier_score_loss(y, p_after))
    bins_before, ece_before = calibration_curve_bins(y, p_before, n_bins=n_bins)
    bins_after, ece_after = calibration_curve_bins(y, p_after, n_bins=n_bins)
    return {
        "n_samples": int(len(y)),
        "n_positives": int(y.sum()),
        "bin_strategy": "uniform",
        "n_bins": int(n_bins),
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_delta": float(brier_after - brier_before),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "bins_before": bins_before,
        "bins_after": bins_after,
    }
