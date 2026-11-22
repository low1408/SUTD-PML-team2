"""Hospital B replay (M01): cross-source comparison with record-grouped uncertainty.

Used by ``monitor-replay --split external_test`` after the model is frozen. It
answers the questions the plan (M01) asks:

* **Cross-source metrics**: the same metrics for hospital A validation (the
  frozen model's own reference population) and for hospital B, side by side,
  with the difference.
* **Record-grouped uncertainty**: a bootstrap over whole stays (never over
  hours), so hours within one stay do not look like independent evidence. The
  interval for the difference resamples each source independently, because the
  two sources are different populations.
* **Timing limitations**: how many stays have an inferred onset that could be
  scored, how many are left-censored (onset before admission), and how many
  end before the inferred onset. Early-detection recall is only meaningful on
  the eligible stays, and the report says how many there are.

Nothing here changes the threshold or the cooldown, and nothing is written at
hour or stay level. Only aggregates are returned.
"""

import numpy as np
import pandas as pd

from .policy import summarize

METRICS = ("normalized_hourly_utility", "auroc", "average_precision",
           "false_alert_events_per_nonseptic_patient_day", "early_detection_recall")
DEFAULT_BOOTSTRAPS = 200
SEED = 42


def _metrics(table: pd.DataFrame) -> dict:
    result = summarize(table)
    return {name: result.get(name) for name in METRICS}


def _stay_groups(table: pd.DataFrame) -> list[pd.DataFrame]:
    return [group for _, group in table.groupby("patient_id", sort=False)]


def _bootstrap(groups: list[pd.DataFrame], draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Metric distributions from resampling whole stays with replacement."""
    values = {name: [] for name in METRICS}
    for _ in range(draws):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat(
            [groups[index].assign(patient_id=f"{groups[index]['patient_id'].iloc[0]}#{copy}")
             for copy, index in enumerate(picked)],
            ignore_index=True,
        )
        result = _metrics(sample)
        for name in METRICS:
            if result[name] is not None:
                values[name].append(result[name])
    return {name: np.asarray(series, dtype=float) for name, series in values.items()}


def _interval(samples: np.ndarray) -> dict:
    if len(samples) < 20:
        return {"low": None, "high": None, "standard_error": None, "draws": len(samples)}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "standard_error": float(np.std(samples, ddof=1)),
            "draws": len(samples)}


def timing_limitations(table: pd.DataFrame) -> dict:
    """What the label timing means for this source: which stays can be scored for early detection."""
    stays = 0
    septic = left_censored = ends_before_onset = eligible = 0
    for group in _stay_groups(table):
        stays += 1
        group = group.sort_values("hour")
        positive = np.flatnonzero(group["label"].to_numpy())
        if not len(positive):
            continue
        septic += 1
        if positive[0] == 0:
            left_censored += 1
            continue
        onset = float(group.iloc[positive[0]]["hour"]) + 6
        if onset > float(group["hour"].max()):
            ends_before_onset += 1
            continue
        eligible += 1
    return {
        "stays": stays, "septic_stays": septic,
        "left_censored_septic_stays": left_censored,
        "septic_stays_ending_before_onset": ends_before_onset,
        "stays_eligible_for_early_detection": eligible,
        "note": ("The sepsis label is shifted six hours earlier by the challenge. Early detection counts only "
                 "stays with an observed onset inside the stay; left-censored stays are excluded, and recall is "
                 "reported with its denominator."),
    }


def cross_source_report(reference_table: pd.DataFrame, replay_table: pd.DataFrame,
                        draws: int = DEFAULT_BOOTSTRAPS, seed: int = SEED) -> dict:
    """Side-by-side metrics for hospital A validation and the replayed split, with grouped intervals."""
    rng = np.random.default_rng(seed)
    sources = {"reference_A_validation": reference_table, "replay_B_external_test": replay_table}
    point, intervals, timing, draws_by_source = {}, {}, {}, {}
    for name, table in sources.items():
        point[name] = _metrics(table)
        draws_by_source[name] = _bootstrap(_stay_groups(table), draws, rng)
        intervals[name] = {metric: _interval(values) for metric, values in draws_by_source[name].items()}
        timing[name] = timing_limitations(table)
    difference, diff_interval = {}, {}
    for metric in METRICS:
        a, b = point["reference_A_validation"][metric], point["replay_B_external_test"][metric]
        difference[metric] = None if a is None or b is None else float(b - a)
        # Paired draws: the i-th resample of B minus the i-th resample of A. Each source was resampled on its own.
        b_draws = draws_by_source["replay_B_external_test"][metric]
        a_draws = draws_by_source["reference_A_validation"][metric]
        size = min(len(a_draws), len(b_draws))
        diff_interval[metric] = _interval(b_draws[:size] - a_draws[:size]) if size else _interval(np.array([]))
    return {
        "method": (f"Patient-level bootstrap over whole stays, {draws} draws, seed {seed}. Each source is resampled "
                   "on its own; the difference interval uses the paired draws."),
        "metrics_point": point,
        "metrics_interval_95": intervals,
        "difference_B_minus_A": difference,
        "difference_interval_95": diff_interval,
        "timing_limitations": timing,
        "caveat": ("Different hospitals, different populations: a gap is a description of this replay, not a "
                   "statement that the model would perform the same or worse elsewhere. No threshold was changed "
                   "on hospital B."),
    }
