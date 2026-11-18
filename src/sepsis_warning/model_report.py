"""One script for the report's results: metrics table, PR curves and the other figures.

    python -m sepsis_warning model-report --run artifacts/real --output reports/results

Input is a finished ``train`` folder (``validation_scores.csv``, ``validation_models.csv``,
``selection_metrics.json``, ``candidate_details.json``). The script compares every candidate
and the SIRS baseline on the **validation split only**; it fits nothing and reads no test data.

Optional inputs
* ``--tuning DIR``     a ``tune`` folder: adds the cross-validation figure.
* ``--evaluation DIR`` an ``evaluate`` folder (the model's one frozen test evaluation): adds the
  test-split table and PR curves. It only reads that folder's output; it never scores test data.

Outputs (all in ``--output``)
* ``metrics_table.csv`` and ``metrics_table.md``: accuracy, precision, recall, F1, specificity,
  average precision, AUROC, Brier score, hourly utility, false alerts per non-septic patient-day,
  early-detection recall, confusion counts, and stay-grouped bootstrap intervals for the
  threshold metrics. Every model is judged at its own validation-tuned threshold.
* One PNG per figure, each with its recreate command printed on it.
* ``figures_manifest.json`` and ``FIGURES.md``: every figure's title, what it shows, the files it
  was made from and the exact command that recreates it.
* ``environment.json``: Python and package versions, platform and CPU count (the brief asks for
  seeds and hardware notes).

Intervals resample whole stays (never hours). Accuracy is reported because the brief asks for
it, but with about 2% septic hours a model that never alerts already scores about 98%, so
precision, recall, F1 and the utility score carry the comparison.
"""

import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .baselines import hourly_classification_metrics

SCORE_PREFIX = "p_"
BASELINE = "sirs_rule"
BASELINE_SCORE = "sirs_score"
INTERVAL_METRICS = ("accuracy", "precision", "recall", "f1")
TABLE_COLUMNS = [
    "model", "role", "threshold", "accuracy", "precision", "recall", "f1", "specificity", "average_precision",
    "auroc", "brier_score", "normalized_hourly_utility", "false_alert_events_per_nonseptic_patient_day",
    "early_detection_recall", "tp", "fp", "fn", "tn", "no_predicted_positives",
    *[f"{m}_ci_{side}" for m in INTERVAL_METRICS for side in ("low", "high")],
]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]


class ReportInputError(ValueError):
    """The run folder is missing something the report needs."""


# --- numbers ---------------------------------------------------------------------------------------------


def _read_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    needed = ["validation_scores.csv", "validation_models.csv", "selection_metrics.json"]
    missing = [name for name in needed if not (run_dir / name).is_file()]
    if missing:
        raise ReportInputError(
            f"{run_dir} is missing {missing}; run `python -m sepsis_warning train ...` with the current code first")
    details_path = run_dir / "candidate_details.json"
    return {
        "scores": pd.read_csv(run_dir / "validation_scores.csv"),
        "models": pd.read_csv(run_dir / "validation_models.csv").set_index("model"),
        "selection": json.loads((run_dir / "selection_metrics.json").read_text(encoding="utf-8")),
        "details": json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else {},
    }


def _patient_counts(frame: pd.DataFrame, score: str, threshold: float) -> np.ndarray:
    """Per-stay [TP, FP, FN, TN] hour counts at ``threshold``, for stay-grouped resampling."""
    codes, _ = pd.factorize(frame["patient_id"])
    labels = frame["label"].to_numpy(dtype=bool)
    decisions = frame[score].to_numpy(dtype=float) >= threshold
    stays = int(codes.max()) + 1
    counts = np.zeros((stays, 4))
    for column, mask in enumerate((labels & decisions, ~labels & decisions, labels & ~decisions, ~labels & ~decisions)):
        counts[:, column] = np.bincount(codes, weights=mask.astype(float), minlength=stays)
    return counts


def _rates(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, precision, recall and F1 from [TP, FP, FN, TN] sums (rows are draws)."""
    tp, fp, fn, tn = (sums[..., i] for i in range(4))

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)

    return {"accuracy": divide(tp + tn, tp + fp + fn + tn), "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn)}


def bootstrap_intervals(frame: pd.DataFrame, score: str, threshold: float, draws: int, seed: int) -> dict:
    """95% intervals from resampling whole stays. Empty when ``draws`` is 0."""
    if draws <= 0:
        return {}
    counts = _patient_counts(frame, score, threshold)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(counts), size=(draws, len(counts)))
    rates = _rates(counts[picks].sum(axis=1))
    result = {}
    for metric in INTERVAL_METRICS:
        low, high = np.percentile(rates[metric], [2.5, 97.5])
        result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = float(low), float(high)
    return result


def _row(frame: pd.DataFrame, score: str, name: str, role: str, threshold: float, extra: dict,
         draws: int, seed: int) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    values = frame[score].to_numpy(dtype=float)
    hourly = hourly_classification_metrics(labels, values >= threshold)
    both_classes = len(np.unique(labels)) == 2
    row = {
        "model": name, "role": role, "threshold": float(threshold),
        "accuracy": hourly["hourly_accuracy"], "precision": hourly["hourly_precision"],
        "recall": hourly["hourly_recall"], "f1": hourly["hourly_f1"], "specificity": hourly["hourly_specificity"],
        "average_precision": float(average_precision_score(labels, values)) if both_classes else None,
        "auroc": float(roc_auc_score(labels, values)) if both_classes else None,
        "tp": hourly["hourly_tp"], "fp": hourly["hourly_fp"], "fn": hourly["hourly_fn"], "tn": hourly["hourly_tn"],
        "no_predicted_positives": hourly["no_predicted_positives"],
        **extra,
    }
    row.update(bootstrap_intervals(frame, score, threshold, draws, seed))
    return row


def metrics_table(run: dict, draws: int = 200, seed: int = 42) -> pd.DataFrame:
    """One row per candidate plus the SIRS baseline, all at their validation-tuned thresholds."""
    scores, models, selection = run["scores"], run["models"], run["selection"]
    selected = selection["selected_model"]
    rows = []
    for column in [c for c in scores.columns if c.startswith(SCORE_PREFIX)]:
        name = column[len(SCORE_PREFIX):]
        if name not in models.index:
            continue
        info = models.loc[name]
        rows.append(_row(scores, column, name, "selected" if name == selected else "candidate",
                         float(info["threshold"]), {
                             "brier_score": info.get("brier_score"),
                             "normalized_hourly_utility": info.get("normalized_hourly_utility"),
                             "false_alert_events_per_nonseptic_patient_day":
                                 info.get("false_alert_events_per_nonseptic_patient_day"),
                             "early_detection_recall": info.get("early_detection_recall"),
                         }, draws, seed))
    sirs = (selection.get("reference_baselines") or {}).get("sirs")
    if sirs and BASELINE_SCORE in scores:
        rows.append(_row(scores, BASELINE_SCORE, BASELINE, "baseline", float(sirs["threshold"]), {
            "brier_score": None,   # the SIRS score is not a probability
            "normalized_hourly_utility": sirs.get("normalized_hourly_utility"),
            "false_alert_events_per_nonseptic_patient_day": sirs.get("false_alert_events_per_nonseptic_patient_day"),
            "early_detection_recall": sirs.get("early_detection_recall"),
        }, draws, seed))
    if not rows:
        raise ReportInputError("validation_scores.csv holds no scored candidates")
    table = pd.DataFrame(rows)
    return table.reindex(columns=TABLE_COLUMNS)


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def markdown_table(table: pd.DataFrame) -> str:
    """The comparison table as Markdown, ready for the report."""
    def interval(row, metric):
        low, high = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
        return "" if low is None or pd.isna(low) else f" ({_fmt(low)}-{_fmt(high)})"

    lines = ["| Model | Threshold | Accuracy | Precision | Recall | F1 | Avg. precision | AUROC | Hourly utility | False alerts / non-septic patient-day |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in table.iterrows():
        label = f"**{row['model']}** (selected)" if row["role"] == "selected" else (
            f"{row['model']} (baseline)" if row["role"] == "baseline" else row["model"])
        lines.append(
            f"| {label} | {_fmt(row['threshold'], 2)} | {_fmt(row['accuracy'])}{interval(row, 'accuracy')} | "
            f"{_fmt(row['precision'])}{interval(row, 'precision')} | {_fmt(row['recall'])}{interval(row, 'recall')} | "
            f"{_fmt(row['f1'])}{interval(row, 'f1')} | {_fmt(row['average_precision'])} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['normalized_hourly_utility'])} | {_fmt(row['false_alert_events_per_nonseptic_patient_day'])} |")
    lines += ["", "Hourly decisions on the validation split, each model at its own validation-tuned threshold. "
              "Brackets are 95% intervals from resampling whole stays. A model that never alerts has precision 0 by "
              "convention (no predicted positives)."]
    return "\n".join(lines) + "\n"


# --- figures ---------------------------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _footer(fig, command: str) -> None:
    import textwrap

    wrapped = "\n".join(textwrap.wrap(f"Recreate: {command}", 95, subsequent_indent="    ", break_long_words=False))
    fig.text(0.01, -0.01, wrapped, fontsize=6.5, color="#555555", ha="left", va="top")


def _save(fig, path: Path, command: str) -> None:
    _footer(fig, command)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _thin(values: np.ndarray, limit: int = 1500) -> np.ndarray:
    if len(values) <= limit:
        return np.arange(len(values))
    return np.unique(np.linspace(0, len(values) - 1, limit).astype(int))


def figure_pr_curves(run: dict, table: pd.DataFrame, path: Path, command: str) -> None:
    plt = _pyplot()
    scores = run["scores"]
    labels = scores["label"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for index, (_, row) in enumerate(table.iterrows()):
        column = BASELINE_SCORE if row["role"] == "baseline" else f"{SCORE_PREFIX}{row['model']}"
        precision, recall, _ = precision_recall_curve(labels, scores[column].to_numpy(dtype=float))
        keep = _thin(precision)
        style = "--" if row["role"] == "baseline" else "-"
        ax.plot(recall[keep], precision[keep], style, color=COLORS[index % len(COLORS)],
                lw=2.2 if row["role"] == "selected" else 1.4,
                label=f"{row['model']} (AP {_fmt(row['average_precision'])})")
        ax.plot(row["recall"], row["precision"], "o", color=COLORS[index % len(COLORS)], ms=6, mec="black")
    ax.axhline(labels.mean(), color="#999999", lw=0.8, ls=":", label=f"chance ({labels.mean():.3f} septic hours)")
    ax.set_xlabel("Recall (septic-labelled hours flagged)")
    ax.set_ylabel("Precision (flagged hours that are septic-labelled)")
    ax.set_title("Precision-recall curves on the validation split\n(dots: each model at its tuned alert threshold)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    _save(fig, path, command)


def figure_metric_bars(table: pd.DataFrame, path: Path, command: str) -> None:
    plt = _pyplot()
    metrics = ["precision", "recall", "f1", "average_precision", "normalized_hourly_utility"]
    names = [row["model"] for _, row in table.iterrows()]
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(names) + 3), 4.6))
    positions = np.arange(len(names))
    for offset, metric in enumerate(metrics):
        values = [0.0 if v is None or pd.isna(v) else float(v) for v in table[metric]]
        ax.bar(positions + offset * width, values, width, label=metric.replace("_", " "))
    ax.set_xticks(positions + 0.4 - width / 2)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (validation split)")
    ax.set_title("Models and SIRS baseline at their validation-tuned thresholds")
    ax.legend(fontsize=7.5, ncols=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path, command)


def figure_calibration(run: dict, table: pd.DataFrame, path: Path, command: str, bins: int = 10) -> None:
    plt = _pyplot()
    scores = run["scores"]
    labels = scores["label"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls=":", label="perfect calibration")
    for index, (_, row) in enumerate(table.iterrows()):
        if row["role"] == "baseline":
            continue
        predicted = scores[f"{SCORE_PREFIX}{row['model']}"].to_numpy(dtype=float)
        edges = np.unique(np.quantile(predicted, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, len(edges) - 2)
        xs = [predicted[which == b].mean() for b in range(len(edges) - 1) if (which == b).any()]
        ys = [labels[which == b].mean() for b in range(len(edges) - 1) if (which == b).any()]
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.4, color=COLORS[index % len(COLORS)],
                label=f"{row['model']} (Brier {_fmt(row['brier_score'], 4)})")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Mean predicted probability (quantile bins)")
    ax.set_ylabel("Observed septic-hour rate")
    ax.set_title("Calibration on the validation split")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, path, command)


def figure_threshold_tradeoff(run: dict, row: pd.Series, path: Path, command: str) -> None:
    plt = _pyplot()
    scores = run["scores"]
    labels = scores["label"].to_numpy(dtype=bool)
    values = scores[f"{SCORE_PREFIX}{row['model']}"].to_numpy(dtype=float)
    grid = np.unique(np.quantile(values, np.linspace(0.5, 0.9999, 120)))
    sums = np.array([[(labels & (values >= t)).sum(), (~labels & (values >= t)).sum(),
                      (labels & (values < t)).sum(), (~labels & (values < t)).sum()] for t in grid], dtype=float)
    rates = _rates(sums)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for metric, color in (("precision", "#d62728"), ("recall", "#1f77b4"), ("f1", "#2ca02c")):
        ax.plot(grid, rates[metric], color=color, label=metric)
    ax.axvline(row["threshold"], color="black", ls="--", lw=1.0, label=f"chosen threshold {row['threshold']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Alert threshold on the calibrated probability (log scale)")
    ax.set_ylabel("Hourly metric")
    ax.set_title(f"{row['model']}: precision, recall and F1 against the threshold (validation split)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path, command)


def figure_confusion(table: pd.DataFrame, path: Path, command: str) -> None:
    plt = _pyplot()
    chosen = table[table["role"].isin(["selected", "baseline"])]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.4 * len(chosen), 4.0), squeeze=False)
    for ax, (_, row) in zip(axes[0], chosen.iterrows(), strict=True):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        shares = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(matrix[i, j]):,}\n({shares[i, j]:.1%} of row)", ha="center", va="center",
                        fontsize=8, color="white" if shares[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["not septic-labelled", "septic-labelled"])
        ax.set_title(f"{row['model']} ({row['role']})", fontsize=10)
    fig.suptitle("Hourly confusion matrices on the validation split", fontsize=11)
    _save(fig, path, command)


def figure_loss_curves(run: dict, path: Path, command: str) -> str | None:
    """Training-loss curves for models that record one. Returns a reason when there are none."""
    curves = {name: info["loss_history"] for name, info in run["details"].items() if info.get("loss_history")}
    if not curves:
        return ("no candidate recorded a training-loss history; only the hand-written NumPy models do "
                "(train with --include-numpy-models)")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for index, (name, values) in enumerate(curves.items()):
        ax.plot(range(1, len(values) + 1), values, "-o", ms=3, color=COLORS[index % len(COLORS)], label=name)
    ax.set_xlabel("Iteration (boosting round or Newton step)")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss per iteration (hand-written NumPy models)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path, command)
    return None


def figure_tuning(tuning_dir: Path, path: Path, command: str) -> None:
    plt = _pyplot()
    results = pd.read_csv(Path(tuning_dir) / "tuning_results.csv")
    models = list(dict.fromkeys(results["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        rows = results[results["model"] == model].reset_index(drop=True)
        ax.errorbar(rows["trial"], rows["cv_ap_mean"], yerr=rows["cv_ap_std"], fmt="o", color="#1f77b4", capsize=2)
        ax.plot(0, rows["cv_ap_mean"].iloc[0], "s", color="#d62728", ms=9, mfc="none", label="reference settings")
        best = rows["cv_ap_mean"].idxmax()
        ax.plot(rows["trial"][best], rows["cv_ap_mean"][best], "*", color="#2ca02c", ms=14, label="best")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Configuration (0 = reference)")
        ax.set_ylabel("Cross-validated average precision")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Hyperparameter search: patient-grouped cross-validation on the training split", fontsize=11)
    _save(fig, path, command)


def figure_failures(run: dict, row: pd.Series, path: Path, command: str, examples: int = 3) -> None:
    """Anonymised trajectories of stays where the selected model went wrong (stay-level data)."""
    plt = _pyplot()
    scores = run["scores"].sort_values(["patient_id", "hour"])
    column = f"{SCORE_PREFIX}{row['model']}"
    threshold = float(row["threshold"])
    per_stay = scores.groupby("patient_id").agg(septic=("label", "max"), peak=(column, "max"))
    alerted = scores[scores[column] >= threshold].groupby("patient_id").size()
    per_stay["alerts"] = alerted.reindex(per_stay.index).fillna(0)
    missed = per_stay[(per_stay["septic"] == 1) & (per_stay["alerts"] == 0)].sort_values("peak").head(examples)
    false = per_stay[(per_stay["septic"] == 0)].sort_values("alerts", ascending=False).head(examples)
    picks = [("missed septic stay", s) for s in missed.index] + [("false-alert stay", s) for s in false.index]
    if not picks:
        picks = [("highest-scoring stay", per_stay["peak"].idxmax())]
    cols = min(3, len(picks))
    rows_needed = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_needed, cols, figsize=(4.4 * cols, 2.9 * rows_needed), squeeze=False)
    for number, (ax, (kind, stay)) in enumerate(zip(axes.ravel(), picks, strict=False), start=1):
        frame = scores[scores["patient_id"] == stay]
        ax.plot(frame["hour"], frame[column], color="#1f77b4", lw=1.4, label="model probability")
        ax.axhline(threshold, color="black", ls="--", lw=0.9, label="alert threshold")
        onset = frame.loc[frame["label"] == 1, "hour"]
        if len(onset):
            ax.axvspan(onset.iloc[0], frame["hour"].max(), color="#d62728", alpha=0.15, label="septic-labelled hours")
        ax.set_yscale("log")
        ax.set_title(f"Stay {chr(64 + number)}: {kind}", fontsize=9)
        ax.set_xlabel("ICU hour")
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle(f"Where {row['model']} goes wrong (anonymised stays, validation split)", fontsize=11)
    _save(fig, path, command)


def evaluation_tables(evaluation_dir: Path, run: dict, draws: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Test-split metrics for the frozen model, from the files `evaluate` wrote (never scores anything)."""
    evaluation_dir = Path(evaluation_dir)
    metrics_file = evaluation_dir / "evaluation_metrics.json"
    predictions_file = evaluation_dir / "held_out_predictions.csv"
    if not metrics_file.is_file() or not predictions_file.is_file():
        raise ReportInputError(f"{evaluation_dir} is not an `evaluate` folder (needs evaluation_metrics.json and "
                               "held_out_predictions.csv); the test splits are scored there, once")
    evaluation = json.loads(metrics_file.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_file)
    threshold = float(evaluation["threshold"])
    rows = []
    for split, frame in predictions.groupby("split"):
        rows.append(_row(frame, "probability", evaluation["selected_model"], split, threshold, {
            key: evaluation["splits"][split].get(key) for key in
            ("brier_score", "normalized_hourly_utility", "false_alert_events_per_nonseptic_patient_day",
             "early_detection_recall")}, draws, seed))
    return pd.DataFrame(rows).reindex(columns=TABLE_COLUMNS), {"predictions": predictions, "evaluation": evaluation}


def figure_test_pr(held: dict, table: pd.DataFrame, path: Path, command: str) -> None:
    plt = _pyplot()
    predictions = held["predictions"]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for index, (_, row) in enumerate(table.iterrows()):
        frame = predictions[predictions["split"] == row["role"]]
        precision, recall, _ = precision_recall_curve(frame["label"], frame["probability"])
        keep = _thin(precision)
        ax.plot(recall[keep], precision[keep], color=COLORS[index], label=f"{row['role']} (AP {_fmt(row['average_precision'])})")
        ax.plot(row["recall"], row["precision"], "o", color=COLORS[index], mec="black")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Frozen model on the test splits, scored once ({held['evaluation']['selected_model']})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path, command)


# --- orchestration ---------------------------------------------------------------------------------------


def environment_notes() -> dict:
    import importlib.metadata

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "packages": packages}


def build_report(run_dir: Path, output: Path, *, tuning_dir: Path | None = None, evaluation_dir: Path | None = None,
                 bootstrap: int = 200, seed: int = 42, command: str | None = None) -> dict:
    """Write the table, figures, manifest and notes. Returns the manifest."""
    run_dir, output = Path(run_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    run = _read_run(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = command or f"python -m sepsis_warning model-report --run {run_dir} --output {output}"
    if command is None:
        if tuning_dir:
            base += f" --tuning {tuning_dir}"
        if evaluation_dir:
            base += f" --evaluation {evaluation_dir}"
        base += f" --bootstrap {bootstrap} --seed {seed}"

    table = metrics_table(run, bootstrap, seed)
    table.to_csv(output / "metrics_table.csv", index=False)
    (output / "metrics_table.md").write_text(markdown_table(table), encoding="utf-8")
    selected = table[table["role"] == "selected"].iloc[0]
    inputs = ["validation_scores.csv", "validation_models.csv", "selection_metrics.json"]

    figures: list[dict] = []
    skipped: list[dict] = []

    def add(figure_id: str, filename: str, title: str, shows: str, files: list[str], stay_level: bool = False):
        figures.append({"id": figure_id, "file": filename, "title": title, "shows": shows,
                        "made_from": files, "recreate": base, "contains_stay_level_data": stay_level})

    figure_pr_curves(run, table, output / "fig_pr_curves.png", base)
    add("F1", "fig_pr_curves.png", "Precision-recall curves, all models and the SIRS baseline",
        "Hourly precision against recall for every candidate and the baseline, with each model's tuned operating "
        "point marked.", inputs)
    figure_metric_bars(table, output / "fig_metrics_bars.png", base)
    add("F2", "fig_metrics_bars.png", "Model comparison", "Precision, recall, F1, average precision and hourly "
        "utility for every model and the baseline.", inputs)
    figure_threshold_tradeoff(run, selected, output / "fig_threshold_tradeoff.png", base)
    add("F3", "fig_threshold_tradeoff.png", "Threshold trade-off of the selected model",
        "Precision, recall and F1 as the alert threshold moves; the chosen threshold is marked.", inputs)
    figure_calibration(run, table, output / "fig_calibration.png", base)
    add("F4", "fig_calibration.png", "Calibration", "Observed septic-hour rate against predicted probability.", inputs)
    figure_confusion(table, output / "fig_confusion.png", base)
    add("F5", "fig_confusion.png", "Confusion matrices", "Hourly confusion matrices of the selected model and the "
        "SIRS baseline.", inputs)
    reason = figure_loss_curves(run, output / "fig_loss_curves.png", base)
    if reason:
        skipped.append({"id": "F6", "title": "Training-loss curves", "reason": reason})
    else:
        add("F6", "fig_loss_curves.png", "Training-loss curves", "Training loss per iteration for the NumPy models.",
            ["candidate_details.json"])
    figure_failures(run, selected, output / "fig_failure_examples.png", base)
    add("F7", "fig_failure_examples.png", "Examples of the selected model going wrong",
        "Probability trajectories of missed septic stays and false-alert stays (anonymised). Individual stay "
        "trajectories: check this file before it leaves the private artifacts folder.", inputs, stay_level=True)
    if tuning_dir:
        figure_tuning(Path(tuning_dir), output / "fig_tuning.png", base)
        add("F8", "fig_tuning.png", "Hyperparameter search", "Cross-validated average precision of every "
            "configuration tried, with the reference and best marked.", [str(Path(tuning_dir) / "tuning_results.csv")])
    test_table = None
    if evaluation_dir:
        test_table, held = evaluation_tables(Path(evaluation_dir), run, bootstrap, seed)
        test_table.to_csv(output / "test_metrics_table.csv", index=False)
        (output / "test_metrics_table.md").write_text(markdown_table(test_table.assign(
            model=test_table["model"] + " / " + test_table["role"], role="candidate")), encoding="utf-8")
        figure_test_pr(held, test_table, output / "fig_test_pr.png", base)
        add("F9", "fig_test_pr.png", "Frozen model on the test splits", "Precision-recall curves for the internal "
            "test and hospital B, from the model's one frozen evaluation.",
            [str(Path(evaluation_dir) / "held_out_predictions.csv")])

    notes = environment_notes()
    (output / "environment.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_dir": str(run_dir), "seed": seed, "bootstrap_draws": bootstrap, "selected_model": str(selected["model"]),
        "split_used": "validation" + (" + the frozen test evaluation" if evaluation_dir else ""),
        "metrics_files": ["metrics_table.csv", "metrics_table.md"] + (["test_metrics_table.csv",
                                                                        "test_metrics_table.md"] if test_table is not None else []),
        "figures": figures, "skipped": skipped, "environment": notes,
    }
    (output / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "FIGURES.md").write_text(figures_markdown(manifest), encoding="utf-8")
    return manifest


def figures_markdown(manifest: dict) -> str:
    lines = ["# Figures and tables", "",
             f"Made from `{manifest['run_dir']}` ({manifest['split_used']}), seed {manifest['seed']}, "
             f"{manifest['bootstrap_draws']} bootstrap draws over whole stays.", "",
             "Every figure is recreated by the command below the table; each figure also prints it on the image.", "",
             "| ID | File | What it shows | Made from | Stay-level data |", "|---|---|---|---|---|"]
    for figure in manifest["figures"]:
        lines.append(f"| {figure['id']} | `{figure['file']}` | {figure['shows']} | "
                     f"{', '.join(f'`{f}`' for f in figure['made_from'])} | "
                     f"{'yes, keep private' if figure['contains_stay_level_data'] else 'no (aggregates)'} |")
    for skipped in manifest["skipped"]:
        lines.append(f"| {skipped['id']} | (not made) | {skipped['title']}: {skipped['reason']} | | |")
    commands = sorted({figure["recreate"] for figure in manifest["figures"]})
    lines += ["", "## Recreate", "", "```bash", *commands, "```", "",
              "Tables: `metrics_table.md` (copy into the report) and `metrics_table.csv`. "
              "Environment and hardware notes: `environment.json`.", ""]
    return "\n".join(lines)
