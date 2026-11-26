"""Two-speed drift monitoring for the frozen early-warning model (task M02).

Fast checks run on every window and need no labels: input schema,
missingness, input distributions (PSI), variables never seen in training,
score distribution (PSI) and alert rate.

Slow checks run only on stays whose labels have settled: calibration
(observed / expected), AUROC, normalized hourly utility, early-detection
recall, false-alert burden, and utility gaps between recorded-sex and age-band
groups.

Reference data and thresholds come from hospital A only and are frozen before
any held-out replay. Input distributions come from A train. Scores, alerts and
performance come from A validation scored by the frozen bundle. Each
threshold is the larger of a fixed floor and a high percentile of the same
statistic on no-shift windows resampled from A validation. The reference file
stores the configuration and the bundle's SHA-256, so a replay cannot quietly
change either.

A drift alarm asks a named owner to review. An unlabeled input or score shift
does not prove that performance dropped; only the slow checks speak to that.
Nothing here retrains, recalibrates or moves the threshold.

The windows are a simulated stream. PhysioNet records carry no calendar time,
so whole stays are put in a seeded random order and grouped into windows of a
fixed number of patient-hours.
"""

import importlib.metadata
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, load_patients, split_patients
from .policy import summarize
from .scoring import file_sha256, load_bundle, score_patients

REFERENCE_VERSION = 1
REFERENCE_FILE = "monitoring_reference.json"
REPLAY_SPLITS = ("internal_test", "external_test")
# Inputs whose distributions are monitored; ICULOS is the time index, not a measurement.
INPUT_COLUMNS = DYNAMIC + ["Age", "Gender", "HospAdmTime"]
# One value per stay: compared across stays, so long stays do not count more.
STAY_LEVEL = {"Age", "Gender", "HospAdmTime"}
REQUIRED_COLUMNS = DYNAMIC + STATIC
# Same bands as hospital_comparison, so monitoring and the bias audit agree.
AGE_BINS = [0, 45, 65, 80, np.inf]
AGE_LABELS = ["<45", "45-64", "65-79", "80+"]
GENDER_LABELS = {0: "female", 1: "male"}
# Added to alert rates (per patient-day) so a zero rate still has a finite ratio.
RATE_EPSILON = 0.01
# Floor for empty PSI bins, so log(0) never occurs.
PSI_FLOOR = 1e-4

OK, WARN, ALERT, INSUFFICIENT = "ok", "warn", "alert", "insufficient_data"
SEVERITY = {INSUFFICIENT: -1, OK: 0, WARN: 1, ALERT: 2}

# family: (speed, owner, what it watches)
CHECKS = {
    "schema": ("fast", "Edwin", "Required columns present, values numeric and finite, hours consecutive within each stay"),
    "missingness": ("fast", "Jawad", "Absolute change in the share of hours with no measurement, per variable, against A train"),
    "input_drift": ("fast", "Jawad", "PSI of observed values per variable against A train"),
    "new_variable": ("fast", "Jawad", "Variables never observed in A train now carry values"),
    "score_drift": ("fast", "Kaiyuan", "PSI of hourly probabilities against A validation"),
    "alert_rate": ("fast", "Kaiyuan", "Alerts per patient-day, all stays, against A validation (absolute log ratio)"),
    "calibration": ("slow", "Kaiyuan", "Change in log(observed / expected) positive-hour rate against A validation"),
    "discrimination": ("slow", "Mashie", "Hourly AUROC drop against A validation"),
    "utility": ("slow", "Mashie", "Normalized hourly utility drop against A validation"),
    "early_detection": ("slow", "Mashie", "Early-detection recall drop against A validation"),
    "alert_burden": ("slow", "Kaiyuan", "Increase in false alerts per non-septic patient-day (log ratio)"),
    "subgroup_gap": ("slow", "Kaiyuan", "Largest utility gap between recorded-sex or age-band groups"),
}
OWNER_ROLES = {
    "Edwin": "data and schema owner (E01, E03, E04)",
    "Jawad": "feature and missing-value owner (J01-J06)",
    "Kaiyuan": "model, calibration, alert-policy and fairness owner (K03-K06)",
    "Mashie": "evaluation and monitoring owner (M01, M02)",
}
# (warn floor, alert floor). None means the check never escalates to alert.
FLOORS = {
    "schema": (1, 1),
    "new_variable": (1, None),
    "missingness": (0.05, 0.10),
    "input_drift": (0.10, 0.20),
    "score_drift": (0.10, 0.20),
    "alert_rate": (math.log(1.5), math.log(2)),
    "calibration": (math.log(1.25), math.log(1.5)),
    "discrimination": (0.03, 0.05),
    "utility": (0.05, 0.10),
    "early_detection": (0.10, 0.20),
    "alert_burden": (math.log(1.5), math.log(2)),
    "subgroup_gap": (0.10, 0.20),
}
# Count checks with fixed thresholds: never widened by resampling, never delayed by persistence.
FIXED_FAMILIES = {"schema", "new_variable"}
FAST_STATS = (
    ["schema_errors", "new_variables", "score_psi", "alert_rate_log_ratio"]
    + [f"missing:{name}" for name in INPUT_COLUMNS]
    + [f"psi:{name}" for name in INPUT_COLUMNS]
)
SLOW_STATS = [
    "calibration_log_oe_change", "auroc_drop", "utility_drop",
    "early_recall_drop", "false_alert_log_ratio", "subgroup_utility_gap",
]
STAT_FAMILY = {
    "schema_errors": "schema", "new_variables": "new_variable", "score_psi": "score_drift",
    "alert_rate_log_ratio": "alert_rate", "calibration_log_oe_change": "calibration",
    "auroc_drop": "discrimination", "utility_drop": "utility", "early_recall_drop": "early_detection",
    "false_alert_log_ratio": "alert_burden", "subgroup_utility_gap": "subgroup_gap",
}
RESPONSES = {
    OK: "No action.",
    WARN: "Log it; review at the weekly team check.",
    ALERT: (
        "Owner reviews within one working day using the runbook in docs/monitoring-plan.md. "
        "No automatic retraining: retraining or a threshold change needs team approval, a new "
        "frozen bundle and a new monitoring reference."
    ),
    INSUFFICIENT: "Too little data in this window; do not read it as ok.",
}
LIMITATIONS = [
    "Research replay of retrospective PhysioNet records; not live clinical monitoring.",
    "Windows are a simulated stream: stays are shuffled with a fixed seed because records have no calendar time.",
    "Fast (unlabeled) checks show that inputs or scores changed; they do not by themselves show that performance dropped.",
    "Slow checks assume labels have settled; in the retrospective replay every stay is complete.",
    "Reference performance comes from A validation, where the threshold was chosen, so it is slightly optimistic.",
    "About 80 statistics are checked per fast window, so single-window warnings are expected by chance; fast alerts need two consecutive windows.",
    "Subgroups are recorded sex and age band only, with no person ID; small groups are reported as insufficient.",
    "Nothing in monitoring retrains, recalibrates or changes the alert threshold.",
]


@dataclass(frozen=True)
class MonitoringConfig:
    """Window sizes, sample minimums and threshold rules, frozen into the reference."""

    fast_window_hours: int = 2000
    slow_window_hours: int = 20000
    psi_bins: int = 10
    min_observed_values: int = 200
    min_observed_stays: int = 30
    min_slow_stays: int = 300
    min_slow_septic_stays: int = 20
    min_subgroup_septic_stays: int = 10
    null_windows: int = 200
    warn_quantile: float = 0.95
    alert_quantile: float = 0.99
    fast_persistence: int = 2
    seed: int = 42

    def __post_init__(self):
        positive = [
            "fast_window_hours", "slow_window_hours", "min_observed_values", "min_observed_stays", "min_slow_stays",
            "min_slow_septic_stays", "min_subgroup_septic_stays", "null_windows", "fast_persistence",
        ]
        if any(getattr(self, name) < 1 for name in positive) or self.psi_bins < 2:
            raise ValueError("window sizes, minimums, null_windows and persistence must be >= 1; psi_bins >= 2")
        if self.slow_window_hours < self.fast_window_hours:
            raise ValueError("slow_window_hours must be at least fast_window_hours")
        if not 0 < self.warn_quantile < self.alert_quantile < 1:
            raise ValueError("need 0 < warn_quantile < alert_quantile < 1")


def family_of(stat: str) -> str:
    if stat.startswith("missing:"):
        return "missingness"
    if stat.startswith("psi:"):
        return "input_drift"
    return STAT_FAMILY[stat]


# ---------------------------------------------------------------- statistics

def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray | None:
    """Inner bin edges at reference quantiles; None if nothing was observed."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of finite values in each bin: (-inf, e0), [e0, e1), ..., [ek, inf)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(expected, actual) -> float:
    """Population stability index: sum((a - e) * ln(a / e)) over bins."""
    expected = np.clip(np.asarray(expected, dtype=float), PSI_FLOOR, None)
    actual = np.clip(np.asarray(actual, dtype=float), PSI_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def alerts_per_patient_day(table: pd.DataFrame) -> float:
    """All notification events per observed patient-day; needs no labels."""
    return float(table["alert_event"].sum() / (len(table) / 24)) if len(table) else 0.0


def input_values(table: pd.DataFrame, name: str) -> np.ndarray:
    """Hourly values, or one value per stay for STAY_LEVEL inputs."""
    column = pd.to_numeric(table[name], errors="coerce")
    if name in STAY_LEVEL:
        column = column.groupby(table["patient_id"], sort=False).first()
    return column.to_numpy(dtype=float)


def schema_problems(window: pd.DataFrame) -> dict:
    missing = [name for name in REQUIRED_COLUMNS if name not in window.columns]
    non_numeric = infinite = 0
    for name in REQUIRED_COLUMNS:
        if name in missing:
            continue
        numeric = pd.to_numeric(window[name], errors="coerce")
        non_numeric += int((numeric.isna() & window[name].notna()).sum())
        infinite += int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    step = window.groupby("patient_id", sort=False)["hour"].diff()
    hour_gaps = int((step.notna() & (step != 1)).sum())
    return {
        "missing_columns": missing, "non_numeric_values": non_numeric,
        "infinite_values": infinite, "hour_gaps": hour_gaps,
    }


def fast_statistics(window: pd.DataFrame, reference: dict, config: MonitoringConfig) -> tuple[dict, dict]:
    """Label-free statistics for one window; also returns descriptive values."""
    problems = schema_problems(window)
    stats = {"schema_errors": len(problems["missing_columns"]) + problems["non_numeric_values"]
             + problems["infinite_values"] + problems["hour_gaps"]}
    new_variables = []
    for name in INPUT_COLUMNS:
        if name not in window.columns:
            stats[f"missing:{name}"] = stats[f"psi:{name}"] = None
            continue
        values = input_values(window, name)
        observed = np.isfinite(values)
        ref = reference["inputs"][name]
        stats[f"missing:{name}"] = abs(float(1 - observed.mean()) - ref["missing_fraction"])
        minimum = config.min_observed_stays if name in STAY_LEVEL else config.min_observed_values
        if ref["edges"] is None:
            stats[f"psi:{name}"] = None
            if observed.any():
                new_variables.append(name)
        elif observed.sum() < minimum:
            stats[f"psi:{name}"] = None
        else:
            stats[f"psi:{name}"] = psi(ref["shares"], bin_shares(values[observed], np.asarray(ref["edges"])))
    stats["new_variables"] = len(new_variables)
    score = reference["score"]
    stats["score_psi"] = psi(score["shares"], bin_shares(window["probability"], np.asarray(score["edges"])))
    rate = alerts_per_patient_day(window)
    stats["alert_rate_log_ratio"] = abs(math.log(
        (rate + RATE_EPSILON) / (score["alerts_per_patient_day"] + RATE_EPSILON)
    ))
    extras = {
        "alerts_per_patient_day": rate,
        "mean_probability": float(window["probability"].mean()),
        "schema_missing_columns": ";".join(problems["missing_columns"]),
        "schema_non_numeric_values": problems["non_numeric_values"],
        "schema_infinite_values": problems["infinite_values"],
        "schema_hour_gaps": problems["hour_gaps"],
        "new_variable_names": ";".join(new_variables),
    }
    return stats, extras


def subgroup_rows(table: pd.DataFrame, config: MonitoringConfig) -> list[dict]:
    """Utility and false-alert burden by recorded sex and age band (aggregates only)."""
    stays = table.groupby("patient_id", sort=True).agg(
        age=("Age", "first"), gender=("Gender", "first"), septic=("label", "max"),
    )
    attributes = {
        "sex": (stays["gender"].map(GENDER_LABELS), list(GENDER_LABELS.values())),
        "age_band": (pd.cut(stays["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(object), AGE_LABELS),
    }
    rows = []
    for attribute, (labels, groups) in attributes.items():
        for group in groups:
            members = labels.index[labels == group]
            septic = int(stays.loc[members, "septic"].sum())
            sufficient = config.min_subgroup_septic_stays <= septic < len(members)
            row = {
                "attribute": attribute, "group": group, "stays": len(members),
                "septic_stays": septic, "sufficient": bool(sufficient),
                "normalized_hourly_utility": None, "false_alerts_per_nonseptic_patient_day": None,
            }
            if sufficient:
                metrics = summarize(table[table["patient_id"].isin(members)])
                row["normalized_hourly_utility"] = metrics["normalized_hourly_utility"]
                row["false_alerts_per_nonseptic_patient_day"] = metrics["false_alert_events_per_nonseptic_patient_day"]
            rows.append(row)
    return rows


def subgroup_gap(rows: list[dict]) -> float | None:
    """Largest best-minus-worst utility within one attribute; None if no attribute has two sufficient groups."""
    gaps = []
    for attribute in {row["attribute"] for row in rows}:
        values = [r["normalized_hourly_utility"] for r in rows
                  if r["attribute"] == attribute and r["normalized_hourly_utility"] is not None]
        if len(values) >= 2:
            gaps.append(max(values) - min(values))
    return float(max(gaps)) if gaps else None


def performance_values(table: pd.DataFrame) -> dict:
    metrics = summarize(table)
    observed = float(table["label"].mean())
    expected = float(table["probability"].mean())
    return {
        "stays": metrics["patients"], "septic_stays": metrics["septic_patients"], "hours": metrics["hours"],
        "observed_positive_hour_rate": observed, "mean_probability": expected,
        "log_observed_expected": math.log(max(observed, 1e-6) / max(expected, 1e-6)),
        "brier_score": metrics["brier_score"], "average_precision": metrics["average_precision"],
        "auroc": metrics["auroc"], "normalized_hourly_utility": metrics["normalized_hourly_utility"],
        "early_detection_recall": metrics["early_detection_recall"],
        "false_alerts_per_nonseptic_patient_day": metrics["false_alert_events_per_nonseptic_patient_day"],
        "alerts_per_patient_day": alerts_per_patient_day(table),
    }


def _drop(reference, value):
    return None if reference is None or value is None else float(reference - value)


def slow_statistics(window: pd.DataFrame, reference: dict, config: MonitoringConfig) -> tuple[dict, dict, list[dict]]:
    """Label-dependent statistics; insufficient unless labels are settled and counts are large enough."""
    empty = dict.fromkeys(SLOW_STATS)
    if "label" not in window.columns or window["label"].isna().any():
        return empty, {"slow_note": "labels not settled"}, []
    stays = window.groupby("patient_id", sort=False)["label"].max()
    if len(stays) < config.min_slow_stays or stays.sum() < config.min_slow_septic_stays:
        return empty, {"slow_note": "too few stays or septic stays", "stays": len(stays),
                       "septic_stays": int(stays.sum())}, []
    window = window.assign(label=window["label"].astype(int))
    values = performance_values(window)
    ref = reference["performance"]
    rows = subgroup_rows(window, config)
    burden, ref_burden = values["false_alerts_per_nonseptic_patient_day"], ref["false_alerts_per_nonseptic_patient_day"]
    stats = {
        "calibration_log_oe_change": abs(values["log_observed_expected"] - ref["log_observed_expected"]),
        "auroc_drop": _drop(ref["auroc"], values["auroc"]),
        "utility_drop": _drop(ref["normalized_hourly_utility"], values["normalized_hourly_utility"]),
        "early_recall_drop": _drop(ref["early_detection_recall"], values["early_detection_recall"]),
        "false_alert_log_ratio": None if burden is None or ref_burden is None else max(
            0.0, math.log((burden + RATE_EPSILON) / (ref_burden + RATE_EPSILON))
        ),
        "subgroup_utility_gap": subgroup_gap(rows),
    }
    return stats, {"slow_note": "", **values}, rows


# ---------------------------------------------------------------- reference and thresholds

def stream_windows(table: pd.DataFrame, window_hours: int, seed: int) -> tuple[list[np.ndarray], int]:
    """Whole stays in seeded random order, cut into windows of at least window_hours rows.

    Returns row positions per window and the number of rows left in the
    unevaluated tail (a final window smaller than window_hours).
    """
    positions = table.groupby("patient_id", sort=True).indices
    ids = sorted(positions)
    windows, current, rows = [], [], 0
    for index in np.random.default_rng(seed).permutation(len(ids)):
        rows_of_stay = positions[ids[index]]
        current.append(rows_of_stay)
        rows += len(rows_of_stay)
        if rows >= window_hours:
            windows.append(np.concatenate(current))
            current, rows = [], 0
    return windows, rows


def null_thresholds(validation: pd.DataFrame, reference: dict, config: MonitoringConfig) -> dict:
    """Warn/alert thresholds: max(floor, percentile of the statistic on no-shift A-validation windows)."""
    samples = defaultdict(list)
    for speed, window_hours, compute, offset in [
        ("fast", config.fast_window_hours, fast_statistics, 10_000),
        ("slow", config.slow_window_hours, slow_statistics, 20_000),
    ]:
        collected = permutation = 0
        while collected < config.null_windows:
            windows, _ = stream_windows(validation, window_hours, config.seed + offset + permutation)
            if not windows:
                raise ValueError(
                    f"A validation has fewer than {window_hours} patient-hours, so no {speed} reference "
                    f"window can be formed; use a smaller --{speed}-window-hours or more data"
                )
            for rows in windows[: config.null_windows - collected]:
                for stat, value in compute(validation.iloc[rows], reference, config)[0].items():
                    if value is not None:
                        samples[stat].append(value)
                collected += 1
            permutation += 1
    if not any(samples.get(stat) for stat in SLOW_STATS):
        raise ValueError(
            f"No A-validation slow window reached {config.min_slow_stays} stays and "
            f"{config.min_slow_septic_stays} septic stays; use larger slow windows, lower minimums or more data"
        )
    thresholds = {}
    for stat in FAST_STATS + SLOW_STATS:
        family = family_of(stat)
        warn_floor, alert_floor = FLOORS[family]
        values = samples.get(stat, [])
        entry = {"family": family, "floor_warn": warn_floor, "floor_alert": alert_floor,
                 "null_windows": 0 if family in FIXED_FAMILIES else len(values),
                 "null_quantile_warn": None, "null_quantile_alert": None}
        warn, alert = warn_floor, alert_floor
        if values and family not in FIXED_FAMILIES:
            entry["null_quantile_warn"] = float(np.quantile(values, config.warn_quantile))
            entry["null_quantile_alert"] = float(np.quantile(values, config.alert_quantile))
            warn = max(warn_floor, entry["null_quantile_warn"])
            alert = max(alert_floor, entry["null_quantile_alert"], warn)
        thresholds[stat] = {"warn": float(warn), "alert": None if alert is None else float(alert), **entry}
    return thresholds


def build_reference(train_inputs: pd.DataFrame, validation: pd.DataFrame, config: MonitoringConfig,
                    provenance: dict | None = None) -> dict:
    """Frozen monitoring reference from A train inputs and the A-validation scored table."""
    inputs = {}
    for name in INPUT_COLUMNS:
        values = input_values(train_inputs, name)
        edges = quantile_edges(values, config.psi_bins)
        inputs[name] = {
            "unit": "stay" if name in STAY_LEVEL else "hour",
            "missing_fraction": float(1 - np.isfinite(values).mean()),
            "observed_values": int(np.isfinite(values).sum()),
            "edges": None if edges is None else edges.tolist(),
            "shares": None if edges is None else bin_shares(values, edges).tolist(),
        }
    score_edges = quantile_edges(validation["probability"], config.psi_bins)
    if score_edges is None:
        raise ValueError("A-validation probabilities are missing")
    performance = performance_values(validation.assign(label=validation["label"].astype(int)))
    reference = {
        "reference_version": REFERENCE_VERSION,
        **(provenance or {}),
        "config": asdict(config),
        "policy": {
            "automatic_retraining": False,
            "responses": RESPONSES,
            "label_settlement": (
                "Slow checks use only stays whose labels have settled (stay complete and the sepsis "
                "label confirmed). Retrospective PhysioNet stays are complete."
            ),
        },
        "checks": {family: {"speed": speed, "owner": owner, "owner_role": OWNER_ROLES[owner],
                            "watches": watches}
                   for family, (speed, owner, watches) in CHECKS.items()},
        "reference_data": {
            "inputs": f"hospital A train split ({train_inputs['patient_id'].nunique()} stays, {len(train_inputs)} hours)",
            "scores_alerts_performance": (
                f"hospital A validation split scored by the frozen bundle "
                f"({validation['patient_id'].nunique()} stays, {len(validation)} hours)"
            ),
        },
        "inputs": inputs,
        "score": {
            "edges": score_edges.tolist(),
            "shares": bin_shares(validation["probability"], score_edges).tolist(),
            "alerts_per_patient_day": performance["alerts_per_patient_day"],
        },
        "performance": performance,
        "limitations": LIMITATIONS,
    }
    reference["thresholds"] = null_thresholds(validation, reference, config)
    return reference


# ---------------------------------------------------------------- running the checks

def classify(value, threshold: dict) -> str:
    if value is None:
        return INSUFFICIENT
    if threshold["alert"] is not None and value >= threshold["alert"]:
        return ALERT
    if value >= threshold["warn"]:
        return WARN
    return OK


def run_monitoring(scored: pd.DataFrame, reference: dict) -> dict:
    """Fast and slow checks over a simulated stream; returns aggregate tables only."""
    config = MonitoringConfig(**reference["config"])
    thresholds = reference["thresholds"]
    results = {}
    for speed, window_hours, compute, stats_names in [
        ("fast", config.fast_window_hours, fast_statistics, FAST_STATS),
        ("slow", config.slow_window_hours, slow_statistics, SLOW_STATS),
    ]:
        windows, tail = stream_windows(scored, window_hours, config.seed)
        streak = defaultdict(int)
        records, subgroups = [], []
        for number, rows in enumerate(windows, start=1):
            window = scored.iloc[rows]
            output = compute(window, reference, config)
            stats, extras = output[0], output[1]
            record = {"window": number, "stays": int(window["patient_id"].nunique()), "hours": len(window), **extras}
            for stat in stats_names:
                value = stats[stat]
                status = classify(value, thresholds[stat])
                streak[stat] = streak[stat] + 1 if status == ALERT else 0
                persistence = config.fast_persistence if speed == "fast" else 1
                if status == ALERT and family_of(stat) not in FIXED_FAMILIES and streak[stat] < persistence:
                    status = WARN
                record[stat] = None if value is None else float(value)
                record[f"status:{stat}"] = status
            records.append(record)
            if speed == "slow":
                subgroups.extend({"window": number, **row} for row in output[2])
        results[speed] = pd.DataFrame(records)
        results[f"{speed}_tail_hours"] = tail
        if speed == "slow":
            results["subgroups"] = pd.DataFrame(subgroups)
    return results


def family_status(record: pd.Series | dict, family: str, stats_names: list[str]) -> str:
    statuses = [record[f"status:{s}"] for s in stats_names if family_of(s) == family]
    evaluated = [s for s in statuses if s != INSUFFICIENT]
    return max(evaluated, key=SEVERITY.get) if evaluated else INSUFFICIENT


def summarize_results(results: dict, scored: pd.DataFrame, reference: dict) -> dict:
    """Aggregate summary for the report and slides; contains no patient-level rows."""
    config = MonitoringConfig(**reference["config"])
    families = {}
    for family, (speed, owner, _watches) in CHECKS.items():
        table = results[speed]
        names = FAST_STATS if speed == "fast" else SLOW_STATS
        statuses = [family_status(row, family, names) for _, row in table.iterrows()] if len(table) else []
        alerting = [i + 1 for i, s in enumerate(statuses) if s == ALERT]
        families[family] = {
            "speed": speed, "owner": owner, "windows": len(statuses),
            **{status: statuses.count(status) for status in (OK, WARN, ALERT, INSUFFICIENT)},
            "first_alert_window": alerting[0] if alerting else None,
            "response": RESPONSES[ALERT if alerting else (WARN if WARN in statuses else OK)],
        }
    fast = results["fast"]
    variables = []
    for name in INPUT_COLUMNS:
        psi_values = fast[f"psi:{name}"].dropna() if len(fast) else pd.Series(dtype=float)
        missing_values = fast[f"missing:{name}"].dropna() if len(fast) else pd.Series(dtype=float)
        alert_share = float((fast[f"status:psi:{name}"] == ALERT).mean()) if len(fast) else 0.0
        variables.append({
            "variable": name,
            "reference_missing_fraction": reference["inputs"][name]["missing_fraction"],
            "median_missing_change": float(missing_values.median()) if len(missing_values) else None,
            "missing_alert_share": float((fast[f"status:missing:{name}"] == ALERT).mean()) if len(fast) else 0.0,
            "evaluable_psi_windows": len(psi_values),
            "median_psi": float(psi_values.median()) if len(psi_values) else None,
            "psi_alert_threshold": reference["thresholds"][f"psi:{name}"]["alert"],
            "median_psi_over_alert_threshold": (
                float(psi_values.median()) / reference["thresholds"][f"psi:{name}"]["alert"]
                if len(psi_values) else None
            ),
            "psi_alert_share": alert_share,
        })
    labelled = "label" in scored.columns and not scored["label"].isna().any()
    pooled = performance_values(scored.assign(label=scored["label"].astype(int))) if labelled else None
    slow = results["slow"]
    slow_median = {}
    if pooled is not None and len(slow) and "auroc" in slow:
        for key in ["normalized_hourly_utility", "brier_score", "auroc", "early_detection_recall",
                    "false_alerts_per_nonseptic_patient_day", "log_observed_expected"]:
            column = slow[key].dropna() if key in slow else pd.Series(dtype=float)
            slow_median[key] = float(column.median()) if len(column) else None
    return {
        "windows": {
            "fast": len(fast), "slow": len(slow),
            "fast_tail_hours_not_evaluated": results["fast_tail_hours"],
            "slow_tail_hours_not_evaluated": results["slow_tail_hours"],
        },
        "families": families,
        "persistently_drifted_inputs": sorted(
            v["variable"] for v in variables if v["psi_alert_share"] >= 0.5
        ),
        "inputs_with_any_psi_alert": sorted(v["variable"] for v in variables if v["psi_alert_share"] > 0),
        "variables": variables,
        "alert_rate": {
            "reference_alerts_per_patient_day": reference["score"]["alerts_per_patient_day"],
            "replay_alerts_per_patient_day": alerts_per_patient_day(scored),
            "median_window_alerts_per_patient_day": float(fast["alerts_per_patient_day"].median()) if len(fast) else None,
        },
        "performance": {
            "reference_a_validation": reference["performance"],
            "replay_pooled": pooled,
            "replay_median_slow_window": slow_median or None,
        },
        "subgroups_pooled": subgroup_rows(scored.assign(label=scored["label"].astype(int)), config) if labelled else [],
        "automatic_retraining": False,
        "interpretation": (
            "Fast-check alarms mean the inputs or scores no longer look like hospital A; they are a "
            "prompt to investigate, not proof that performance dropped. Slow-check alarms compare "
            "settled-label performance with the A-validation reference."
        ),
        "limitations": LIMITATIONS,
    }


# ---------------------------------------------------------------- report

def _top_drift(summary: dict, count: int) -> list[dict]:
    """Variables ranked by median PSI relative to their own alert threshold (above 1 = past the alert line)."""
    scored = [v for v in summary["variables"] if v["median_psi_over_alert_threshold"] is not None]
    return sorted(scored, key=lambda v: v["median_psi_over_alert_threshold"], reverse=True)[:count]


def _fmt(value, digits=3):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def write_report(output: Path, summary: dict, reference: dict, recreate: str) -> None:
    perf = summary["performance"]
    ref, pooled, median = perf["reference_a_validation"], perf["replay_pooled"] or {}, perf["replay_median_slow_window"] or {}
    lines = [
        f"# Monitoring replay: {summary['split']}",
        "",
        "**Research replay of retrospective records; not live clinical monitoring.** "
        + summary["interpretation"],
        "",
        f"- Data mode: `{summary['data_mode']}`. Model: `{reference['model_name']}`, "
        f"SHA-256 `{reference['model_sha256'][:12]}`. Reference SHA-256 `{summary['reference_sha256'][:12]}`.",
        f"- Fast windows: {summary['windows']['fast']} of {reference['config']['fast_window_hours']} patient-hours; "
        f"slow windows: {summary['windows']['slow']} of {reference['config']['slow_window_hours']} patient-hours.",
        "- Automatic retraining: **off**. Every alert goes to the named owner for review.",
        "",
        "## Checks",
        "",
        "| Check | Speed | Owner | Windows | OK | Warn | Alert | Insufficient | First alert window |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary["families"].items():
        lines.append(
            f"| {family} | {row['speed']} | {row['owner']} | {row['windows']} | {row[OK]} | {row[WARN]} | "
            f"{row[ALERT]} | {row[INSUFFICIENT]} | {row['first_alert_window'] or '-'} |"
        )
    top = _top_drift(summary, 10)
    lines += [
        "",
        "## Inputs with the largest distribution shift (fast)",
        "",
        f"Persistently drifted (PSI alert in at least half of the windows): "
        f"{len(summary['persistently_drifted_inputs'])} "
        f"({', '.join(summary['persistently_drifted_inputs']) or 'none'}).",
        "",
        "Ranked by median PSI divided by the variable's own alert threshold (above 1 means past the alert line).",
        "",
        "| Variable | Median PSI | PSI alert threshold | Ratio | Windows alerting | Missing in A train | Median missingness change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in top:
        lines.append(
            f"| {v['variable']} | {_fmt(v['median_psi'])} | {_fmt(v['psi_alert_threshold'])} | "
            f"{_fmt(v['median_psi_over_alert_threshold'], 2)} | {100 * v['psi_alert_share']:.0f}% | {100 * v['reference_missing_fraction']:.1f}% | "
            f"{_fmt(v['median_missing_change'])} |"
        )
    rate = summary["alert_rate"]
    lines += [
        "",
        "## Alert rate (fast, no labels)",
        "",
        f"Alerts per patient-day, all stays: A validation {_fmt(rate['reference_alerts_per_patient_day'])} "
        f"-> replay {_fmt(rate['replay_alerts_per_patient_day'])} "
        f"(median window {_fmt(rate['median_window_alerts_per_patient_day'])}).",
        "",
        "## Performance (slow, settled labels)",
        "",
        "| Metric | A validation (reference) | Replay, pooled | Replay, median slow window |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("normalized_hourly_utility", "Normalized hourly utility"), ("auroc", "Hourly AUROC"),
        ("brier_score", "Brier score (prevalence-dependent)"), ("log_observed_expected", "log(observed / expected)"),
        ("early_detection_recall", "Early-detection recall"),
        ("false_alerts_per_nonseptic_patient_day", "False alerts per non-septic patient-day"),
    ]:
        lines.append(f"| {label} | {_fmt(ref.get(key))} | {_fmt(pooled.get(key))} | {_fmt(median.get(key))} |")
    lines += [
        "",
        "## Subgroups, pooled over the replay",
        "",
        "| Attribute | Group | Stays | Septic stays | Utility | False alerts / non-septic day |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["subgroups_pooled"]:
        note = "" if row["sufficient"] else " (too few septic stays)"
        lines.append(
            f"| {row['attribute']} | {row['group']}{note} | {row['stays']} | {row['septic_stays']} | "
            f"{_fmt(row['normalized_hourly_utility'])} | {_fmt(row['false_alerts_per_nonseptic_patient_day'])} |"
        )
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in summary["limitations"]]
    lines += ["", "Recreate:", "", f"    {recreate}", ""]
    (output / "monitoring_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_overview(output: Path, results: dict, summary: dict, reference: dict, recreate: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    top = _top_drift(summary, 12)[::-1]
    ax = axes[0, 0]
    ax.barh([v["variable"] for v in top], [v["median_psi_over_alert_threshold"] for v in top], color="#2a6fdb")
    ax.axvline(1, color="#d96f2c", lw=1.2, label="alert threshold")
    ax.set_xlabel("Median PSI / the variable's alert threshold (vs A train)")
    ax.set_title("Input drift, top variables")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fast, slow = results["fast"], results["slow"]
    reference_rate = reference["score"]["alerts_per_patient_day"]
    ax = axes[0, 1]
    if len(fast):
        ax.plot(fast["window"], fast["alerts_per_patient_day"], "-o", ms=3, color="#2a6fdb", label="replay window")
    ax.axhline(reference_rate, color="grey", ls="--", label="A validation")
    band = reference["thresholds"]["alert_rate_log_ratio"]["alert"]
    upper = (reference_rate + RATE_EPSILON) * math.exp(band) - RATE_EPSILON
    lower = max(0.0, (reference_rate + RATE_EPSILON) * math.exp(-band) - RATE_EPSILON)
    ax.axhline(upper, color="#d96f2c", lw=1, label="alert band")
    ax.axhline(lower, color="#d96f2c", lw=1)
    ax.set_xlabel("Fast window")
    ax.set_ylabel("Alerts per patient-day")
    ax.set_title("Alert rate (no labels)")
    ax.legend(frameon=False, fontsize=8)

    performance = reference["performance"]
    for ax, key, label, stat in [
        (axes[1, 0], "normalized_hourly_utility", "Normalized hourly utility", "utility_drop"),
        (axes[1, 1], "log_observed_expected", "log(observed / expected)", "calibration_log_oe_change"),
    ]:
        if len(slow) and key in slow and slow[key].notna().any():
            ax.plot(slow["window"], slow[key], "-o", ms=4, color="#2a6fdb", label="replay window")
        else:
            ax.text(0.5, 0.5, "no slow window had enough settled labels", ha="center", transform=ax.transAxes)
        if performance.get(key) is not None:
            ax.axhline(performance[key], color="grey", ls="--", label="A validation")
            alert = reference["thresholds"][stat]["alert"]
            lower = performance[key] - alert
            ax.axhline(lower, color="#d96f2c", lw=1, label="alert line")
            if key == "log_observed_expected":
                ax.axhline(performance[key] + alert, color="#d96f2c", lw=1)
        ax.set_xlabel("Slow window (settled labels)")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Monitoring replay, {summary['split']} ({summary['data_mode']}); aggregates only")
    fig.text(0.99, 0.005, f"Recreate: {recreate}", ha="right", va="bottom", fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output / "monitoring_overview.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require_empty(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")


def _load_cohorts(data: Path, seed: int):
    synthetic = (data / "SYNTHETIC.json").exists()
    patients = load_patients(data)
    if not synthetic and any("synthetic_" in p.patient_id for p in patients):
        raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
    return split_patients(patients, seed), synthetic


def reference_from_cohorts(cohorts: dict, bundle: dict, config: MonitoringConfig, provenance: dict) -> dict:
    """Build the reference from the A train and A validation cohorts only.

    ``cohorts`` may hold other splits, but only ``train`` and ``validation``
    are read: held-out test stays never shape the monitoring reference.
    """
    train_inputs = pd.concat(
        [p.frame[INPUT_COLUMNS].assign(patient_id=p.patient_id) for p in cohorts["train"]], ignore_index=True
    )
    validation = score_patients(bundle, cohorts["validation"], REQUIRED_COLUMNS)
    return build_reference(train_inputs, validation, config, provenance)


def setup_monitoring(data: Path, model: Path, output: Path, config: MonitoringConfig, split_seed: int = 42) -> dict:
    """Freeze the monitoring reference before any internal-test or hospital-B replay."""
    _require_empty(output)
    splits, synthetic = _load_cohorts(data, split_seed)
    bundle = load_bundle(model)
    if bool(bundle["synthetic"]) != synthetic:
        raise ValueError("The bundle and the data disagree on whether they are synthetic")
    provenance = {
        "created_by": "python -m sepsis_warning monitor-setup",
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "split_seed": split_seed,
        "model_sha256": file_sha256(model),
        "model_name": bundle["model_name"],
        "threshold": float(bundle["threshold"]),
        "cooldown_hours": int(bundle["cooldown"]),
        "feature_window_hours": int(bundle["window"]),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]},
    }
    reference = reference_from_cohorts({"train": splits["train"], "validation": splits["validation"]},
                                       bundle, config, provenance)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / REFERENCE_FILE, reference)
    return reference


def _write_cross_source_section(report: Path, cross: dict) -> None:
    """Append the M01 comparison to the replay report (aggregates only)."""
    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    def ci(interval):
        return "n/a" if interval.get("low") is None else f"{interval['low']:.3f} to {interval['high']:.3f}"

    a, b = "reference_A_validation", "replay_B_external_test"
    rows = ["", "## Hospital A validation against hospital B (cross-source)", "",
            f"{cross['method']}", "",
            "| Metric | A validation | 95% interval | B external test | 95% interval | B minus A (95% interval) |",
            "|---|---:|---|---:|---|---|"]
    for metric in ("normalized_hourly_utility", "auroc", "average_precision",
                   "false_alert_events_per_nonseptic_patient_day", "early_detection_recall"):
        rows.append(
            f"| {metric} | {fmt(cross['metrics_point'][a][metric])} | "
            f"{ci(cross['metrics_interval_95'][a][metric])} | {fmt(cross['metrics_point'][b][metric])} | "
            f"{ci(cross['metrics_interval_95'][b][metric])} | "
            f"{fmt(cross['difference_B_minus_A'][metric])} ({ci(cross['difference_interval_95'][metric])}) |")
    timing = cross["timing_limitations"][b]
    rows += ["", "Timing limitations on hospital B: "
             f"{timing['stays']} stays, {timing['septic_stays']} septic, "
             f"{timing['left_censored_septic_stays']} left-censored, "
             f"{timing['septic_stays_ending_before_onset']} ending before inferred onset, "
             f"{timing['stays_eligible_for_early_detection']} eligible for early detection.",
             "", cross["caveat"], ""]
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


def replay_monitoring(data: Path, model: Path, reference_path: Path, split: str, output: Path) -> dict:
    """Run both speeds over a held-out split with the frozen reference; writes aggregates only."""
    if split not in REPLAY_SPLITS:
        raise ValueError(f"split must be one of {REPLAY_SPLITS}")
    _require_empty(output)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference.get("reference_version") != REFERENCE_VERSION:
        raise ValueError(f"{reference_path} is not a version-{REFERENCE_VERSION} monitoring reference")
    bundle = load_bundle(model, expected_sha256=reference["model_sha256"])
    splits, synthetic = _load_cohorts(data, reference["split_seed"])
    data_mode = "synthetic_fixture" if synthetic else "physionet"
    if data_mode != reference["data_mode"]:
        raise ValueError(f"Reference was built on {reference['data_mode']} data, not {data_mode}")
    scored = score_patients(bundle, splits[split], REQUIRED_COLUMNS)
    if len(scored) < reference["config"]["fast_window_hours"]:
        raise ValueError(
            f"{split} has {len(scored)} patient-hours, fewer than one fast window "
            f"({reference['config']['fast_window_hours']}); rebuild the reference with smaller windows"
        )
    results = run_monitoring(scored, reference)
    cross = None
    if split == "external_test":
        # M01: the same metrics on hospital A validation (the frozen model's reference population),
        # with stay-grouped uncertainty. Thresholds and cooldown are the frozen ones; nothing is retuned.
        from .cross_source import cross_source_report

        reference_scored = score_patients(bundle, splits["validation"], REQUIRED_COLUMNS)
        cross = cross_source_report(reference_scored, scored)
    summary = {
        "split": split, "data_mode": data_mode,
        "reference_sha256": file_sha256(Path(reference_path)), "model_sha256": reference["model_sha256"],
        **summarize_results(results, scored, reference),
    }
    recreate = (
        f"python -m sepsis_warning monitor-replay --data {data} --model {model} "
        f"--reference {reference_path} --split {split} --output {output}"
    )
    output.mkdir(parents=True, exist_ok=True)
    results["fast"].to_csv(output / "fast_windows.csv", index=False)
    results["slow"].to_csv(output / "slow_windows.csv", index=False)
    results["subgroups"].to_csv(output / "slow_subgroups.csv", index=False)
    if cross is not None:
        summary["cross_source"] = cross
    _write_json(output / "summary.json", summary)
    if cross is not None:
        _write_json(output / "cross_source.json", cross)
    write_report(output, summary, reference, recreate)
    if cross is not None:
        _write_cross_source_section(output / "monitoring_report.md", cross)
    plot_overview(output, results, summary, reference,
                  f"python -m sepsis_warning monitor-replay --split {split} (full command in monitoring_report.md)")
    return summary
