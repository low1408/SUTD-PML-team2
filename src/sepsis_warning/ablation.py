"""H03: metrics and helpers for the with/without temporal-feature comparison.

NumPy and pandas only. The arms, the fixed settings and what counts as "helps"
are written down in docs/h03-ablation-plan.md.
"""

import numpy as np
import pandas as pd

# Which extra feature groups each arm adds to the shared features.
ARMS = {"A": (), "B": ("hmm",), "C": ("signature",), "D": ("hmm", "signature")}
ARM_NAMES = {"A": "shared features", "B": "+ HMM", "C": "+ signature", "D": "+ both"}
EARLY_HOURS = 6


def arm_columns(base: list[str], groups: dict[str, list[str]], arm: str) -> list[str]:
    """Shared columns first, in their own order, then each added group."""
    columns = list(base)
    for group in ARMS[arm]:
        columns += groups[group]
    return columns


def average_precision(labels, scores) -> float:
    """Area under the precision-recall curve, as a step sum over distinct scores."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_each_score = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    true_pos = np.cumsum(y)[last_of_each_score]
    flagged = last_of_each_score + 1
    precision = true_pos / flagged
    recall = true_pos / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auroc(labels, scores) -> float:
    """Probability a random positive hour scores above a random negative hour (ties count half)."""
    y = np.asarray(labels, dtype=bool)
    positives, negatives = y.sum(), (~y).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def brier(labels, probabilities) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    return float(np.mean((p - y) ** 2))


def score_table(table: pd.DataFrame) -> dict:
    """Hour-level metrics for one arm. `table` has label, probability and hour."""
    early = table["hour"] <= EARLY_HOURS
    return {
        "hours": int(len(table)),
        "positive_hour_fraction": float(table["label"].mean()),
        "auprc": average_precision(table["label"], table["probability"]),
        "auroc": auroc(table["label"], table["probability"]),
        "brier": brier(table["label"], table["probability"]),
        "auprc_first_6h": average_precision(table.loc[early, "label"], table.loc[early, "probability"]),
    }


def paired_bootstrap(predictions: dict[str, pd.DataFrame], reference: str = "A",
                     resamples: int = 1000, seed: int = 0) -> dict[str, dict]:
    """AUPRC difference from `reference`, resampling whole stays, the same stays for every arm.

    Every table must hold the same rows in the same order (patient_id, label, probability).
    Returns, for each other arm, the difference on all data and a 95% percentile interval.
    """
    first = predictions[reference]
    for arm, table in predictions.items():
        if not np.array_equal(table["patient_id"].to_numpy(), first["patient_id"].to_numpy()):
            raise ValueError(f"arm {arm} rows do not line up with arm {reference}")
    rows_by_stay = list(first.groupby("patient_id", sort=False).indices.values())
    labels = first["label"].to_numpy()
    scores = {arm: table["probability"].to_numpy() for arm, table in predictions.items()}
    rng = np.random.default_rng(seed)
    differences = {arm: [] for arm in predictions if arm != reference}
    for _ in range(resamples):
        picked = rng.integers(0, len(rows_by_stay), len(rows_by_stay))
        rows = np.concatenate([rows_by_stay[k] for k in picked])
        base = average_precision(labels[rows], scores[reference][rows])
        for arm in differences:
            differences[arm].append(average_precision(labels[rows], scores[arm][rows]) - base)
    result = {}
    full_base = average_precision(labels, scores[reference])
    for arm, values in differences.items():
        low, high = np.nanpercentile(values, [2.5, 97.5])
        result[arm] = {
            "auprc_difference": average_precision(labels, scores[arm]) - full_base,
            "ci_low": float(low), "ci_high": float(high),
            "helps": bool(low > 0),
        }
    return result
