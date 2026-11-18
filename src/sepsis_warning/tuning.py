"""Hyperparameter search with patient-grouped cross-validation (the brief's "hyper-parameter tuning").

What it does
------------
For each tunable model in ``models.toml`` the search tries the hand-picked settings
(the reference) and a random sample of the settings in ``tuning_space.toml``. Every
configuration is scored with stratified, patient-grouped K-fold cross-validation on
the **training split only**: all hours of a patient stay in one fold, so no stay is
split between the part a model learns from and the part it is scored on.

What it deliberately does not touch
-----------------------------------
* The calibration, validation and both test splits are never read. ``tune`` takes
  only ``cohorts["train"]``; a test checks this.
* Imputation and scaling live inside each estimator's pipeline, so they are fitted
  on the training folds only, never on the held-out fold.
* The alert threshold is not tuned here. That still happens once, on the validation
  split, in ``train`` (policy.tune_threshold).

Selection metric
----------------
Mean average precision (AP) over the folds, which is threshold-free and works on the
raw scores, so no calibration or alert policy is involved. The best configuration per
model is written to ``tuned_models.toml``, which ``train --models-config`` reads. The
final model choice among the tuned candidates is still made on the validation split.

Honest reporting
----------------
Trying many configurations and keeping the best flatters the cross-validation score.
The number of trials, folds and seed are recorded, the reference score is kept next to
the best one, and the unbiased numbers are the validation and test results.
"""

import hashlib
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .features import META, feature_table
from .modeling import BUILDERS, load_model_config

DEFAULT_SPACE = Path(__file__).with_name("tuning_space.toml")
SELECTION_METRIC = "cv_ap_mean"
RESULT_COLUMNS = [
    "model", "trial", "is_reference", "params", "cv_ap_mean", "cv_ap_std", "cv_auroc_mean", "cv_auroc_std",
    "folds_scored", "fit_seconds",
]


def load_space(path: Path | None = None) -> tuple[dict, str]:
    """Return (space, SHA-256 of the file). Every parameter must list at least one value."""
    path = Path(path or DEFAULT_SPACE)
    raw = path.read_bytes()
    space = tomllib.loads(raw.decode("utf-8"))
    if not space:
        raise ValueError(f"{path} defines no search space")
    for model, parameters in space.items():
        if not parameters:
            raise ValueError(f"{path}: model {model!r} lists no parameters")
        for name, values in parameters.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{path}: {model}.{name} must be a non-empty list of values")
    return space, hashlib.sha256(raw).hexdigest()


def sample_trials(space: dict, base_params: dict, trials: int, rng: np.random.Generator) -> list[dict]:
    """Distinct parameter sets to try: the reference first, then up to ``trials`` others.

    A space smaller than ``trials`` is searched completely (grid search); a larger one
    is sampled without replacement (random search). The reference counts separately.
    """
    if trials < 0:
        raise ValueError("trials must be non-negative")
    names = sorted(space)
    sizes = [len(space[name]) for name in names]
    total = int(np.prod(sizes))
    chosen: list[dict] = [dict(base_params)]
    seen = {json.dumps(chosen[0], sort_keys=True)}
    for index in rng.permutation(total):
        if len(chosen) > trials:
            break
        digits, rest = {}, int(index)
        for name, size in zip(names, sizes, strict=True):
            rest, digit = divmod(rest, size)
            digits[name] = space[name][digit]
        params = {**base_params, **digits}
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            chosen.append(params)
    return chosen


def grouped_folds(table: pd.DataFrame, folds: int, seed: int):
    """Stratified patient-grouped folds: (train_index, test_index) over the table's rows."""
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(table[META[:1]], table["label"], groups=table["patient_id"]))


def cross_validate(builder_kind: str, params: dict, table: pd.DataFrame, columns: list[str],
                   folds: list, seed: int) -> dict:
    """Mean and spread of AP and AUROC over folds for one configuration."""
    average_precision, auroc = [], []
    started = time.perf_counter()
    for train_index, test_index in folds:
        train, test = table.iloc[train_index], table.iloc[test_index]
        if test["label"].nunique() < 2 or train["label"].nunique() < 2:
            continue  # a fold with one class cannot be scored; counted in folds_scored
        model = BUILDERS[builder_kind](seed, dict(params))
        with threadpool_limits(limits=2):
            model.fit(train[columns], train["label"])
            scores = model.predict_proba(test[columns])[:, 1]
        average_precision.append(average_precision_score(test["label"], scores))
        auroc.append(roc_auc_score(test["label"], scores))
    if len(average_precision) < 2:
        raise ValueError("fewer than two folds contain both classes; use more stays or fewer folds")
    return {
        "cv_ap_mean": float(np.mean(average_precision)), "cv_ap_std": float(np.std(average_precision, ddof=1)),
        "cv_auroc_mean": float(np.mean(auroc)), "cv_auroc_std": float(np.std(auroc, ddof=1)),
        "folds_scored": len(average_precision), "fit_seconds": round(time.perf_counter() - started, 2),
    }


def run_search(train_patients, *, models_config: dict | None = None, space: dict | None = None,
               window: int = 6, folds: int = 3, trials: int = 12, seed: int = 42, max_stays: int | None = None,
               only: list[str] | None = None, progress=print) -> tuple[pd.DataFrame, dict, dict]:
    """Search every tunable model. Returns (results table, best-per-model summary, tuned models config).

    ``train_patients`` must be the training cohort and nothing else.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if models_config is None:
        models_config, _ = load_model_config()
    if space is None:
        space, _ = load_space()
    patients = list(train_patients)
    if max_stays is not None and max_stays < len(patients):
        order = np.random.default_rng(seed).permutation(len(patients))[:max_stays]
        patients = [patients[i] for i in sorted(order)]
    table = feature_table(patients, window)
    columns = [name for name in table if name not in META]
    fold_indexes = grouped_folds(table, folds, seed)

    tunable = [name for name in models_config if name in space and (only is None or name in only)]
    if not tunable:
        raise ValueError("no model in the config has a search space; check --models and the space file")
    rows, best = [], {}
    for model_index, name in enumerate(tunable):
        entry = models_config[name]
        rng = np.random.default_rng([seed, model_index])
        configurations = sample_trials(space[name], entry.get("params", {}), trials, rng)
        progress(f"Tuning {name}: {len(configurations)} configurations x {folds} folds "
                 f"on {len(patients)} stays ({len(table)} hours)")
        model_rows = []
        for trial, params in enumerate(configurations):
            scores = cross_validate(entry["kind"], params, table, columns, fold_indexes, seed)
            row = {"model": name, "trial": trial, "is_reference": trial == 0,
                   "params": json.dumps(params, sort_keys=True), **scores}
            model_rows.append(row)
            progress(f"  {name} #{trial:<2} AP {scores['cv_ap_mean']:.4f} +/- {scores['cv_ap_std']:.4f} "
                     f"AUROC {scores['cv_auroc_mean']:.4f} ({scores['fit_seconds']}s)"
                     f"{'  [reference]' if trial == 0 else ''}")
        rows.extend(model_rows)
        winner = max(model_rows, key=lambda r: (r["cv_ap_mean"], r["cv_auroc_mean"], -r["trial"]))
        reference = model_rows[0]
        best[name] = {
            "params": json.loads(winner["params"]), "trial": winner["trial"],
            "cv_ap_mean": winner["cv_ap_mean"], "cv_ap_std": winner["cv_ap_std"],
            "reference_params": json.loads(reference["params"]), "reference_cv_ap_mean": reference["cv_ap_mean"],
            "ap_gain_over_reference": winner["cv_ap_mean"] - reference["cv_ap_mean"],
            "configurations_tried": len(model_rows),
            "winner_is_reference": winner["trial"] == 0,
        }
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    tuned = {name: dict(entry) for name, entry in models_config.items()}
    for name, summary in best.items():
        tuned[name] = {**tuned[name], "params": summary["params"]}
    meta = {
        "selection_metric": SELECTION_METRIC, "folds": folds, "trials_per_model": trials, "seed": seed,
        "stays_used": len(patients), "hours_used": int(len(table)), "window_hours": window,
        "septic_stays": int(sum(p.septic for p in patients)),
        "note": ("Cross-validation on the training split only; calibration, validation and test splits were "
                 "not read. The best cross-validation score is optimistic because it is the best of several "
                 "configurations; the validation and test results are the unbiased ones."),
    }
    return results, {"models": best, **meta}, tuned


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string for our values
    raise TypeError(f"cannot write {value!r} to TOML")


def write_models_toml(path: Path, config: dict, header: str = "") -> None:
    """Write a models.toml-compatible file (tomllib reads TOML but cannot write it)."""
    lines = [f"# {line}" for line in header.splitlines()] + ([""] if header else [])
    for name, entry in config.items():
        lines.append(f"[{name}]")
        for key in ("kind", "enabled", "complexity"):
            if key in entry:
                lines.append(f"{key} = {_toml_value(entry[key])}")
        if entry.get("params"):
            inline = ", ".join(f"{key} = {_toml_value(value)}" for key, value in entry["params"].items())
            lines.append(f"params = {{ {inline} }}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output: Path, results: pd.DataFrame, summary: dict, tuned: dict, *, models_config_sha256: str,
                  space_sha256: str, data_note: str) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "tuning_results.csv", index=False)
    summary = {**summary, "models_config_sha256": models_config_sha256, "space_sha256": space_sha256,
               "data": data_note}
    (output / "tuning_best.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_models_toml(output / "tuned_models.toml", tuned, header=(
        "Tuned candidate settings from `python -m sepsis_warning tune`.\n"
        f"Search: {summary['trials_per_model']} sampled configurations per model, {summary['folds']}-fold "
        f"patient-grouped cross-validation on the training split, seed {summary['seed']}.\n"
        "Use with: python -m sepsis_warning train --models-config <this file> ..."))
