"""H03: with/without ablation of the HMM and signature features (docs/h03-ablation-plan.md).

    python scripts/h03_ablation.py --data data/physionet/combined.parquet \
        --splits data/physionet/splits.csv --hmm artifacts/hmm/hmm_model.json

Arms: A shared features, B + HMM, C + signature, D + both. Every arm uses the
same split, the same shared columns in the same order, the same K01 NumPy
logistic regression, the same C grid chosen the same way, Platt calibration
on `calibration` with the model frozen, and a threshold tuned on `validation`
under the false-alert budget with the cooldown.

Only `train`, `calibration` and `validation` are read. Internal and external
test stays are never featurised or scored. Writes aggregate results only (no
patient-level predictions) to --output, and refuses to overwrite.

Needs sepsis_warning.numpy_models (K01/K04, branch kaiyuan/model-from-scratch)
and an HMM fitted on train by scripts/fit_hmm.py. Use --limit 300 for a quick run.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.ablation import ARM_NAMES, ARMS, arm_columns, paired_bootstrap, score_table
from sepsis_warning.features import apply_training_fallback, features_for_windows, fit_training_fallback
from sepsis_warning.hmm import VITALS, hmm_state_features, load_hmm
from sepsis_warning.policy import tune_threshold
from sepsis_warning.signature import WINDOW, fit_signature_scaler, signature_features
from sepsis_warning.splits import load_split_patients

try:
    from sepsis_warning.numpy_models import NumpyLogisticRegression, calibrate_frozen
except ImportError as error:  # K01 is not merged yet
    raise SystemExit(
        "sepsis_warning.numpy_models is missing. H03 needs the K01 NumPy logistic regression "
        "(branch kaiyuan/model-from-scratch)."
    ) from error

SPLITS_USED = ("train", "calibration", "validation")
WINDOWS = (3, 6, 12)
GROUPS = ("shared", "hmm", "signature")


def build_split(patients, hmm_model, scaler, fallback):
    """Feature table for one split plus seconds spent on each feature group."""
    chunks = []
    seconds = dict.fromkeys(GROUPS, 0.0)
    for patient in patients:
        frame = patient.frame
        started = time.perf_counter()
        source = apply_training_fallback(frame, fallback) if fallback is not None else frame
        shared = features_for_windows(source, WINDOWS)
        seconds["shared"] += time.perf_counter() - started

        started = time.perf_counter()
        hmm = hmm_state_features(frame, hmm_model).astype(np.float32)
        seconds["hmm"] += time.perf_counter() - started

        started = time.perf_counter()
        signature = signature_features(frame, scaler, WINDOW).astype(np.float32)
        seconds["signature"] += time.perf_counter() - started

        table = pd.concat([shared, hmm, signature], axis=1)
        table["patient_id"] = patient.patient_id
        table["hour"] = frame["ICULOS"].to_numpy()
        table["label"] = frame["SepsisLabel"].to_numpy(dtype=int)
        chunks.append(table)
    return pd.concat(chunks, ignore_index=True), seconds, (shared.columns, hmm.columns, signature.columns)


def run_arm(arm, columns, tables, c_grid, budget, cooldown):
    """Fit every C on train, calibrate on calibration, pick C by validation AUPRC."""
    train, calibration, validation = (tables[name] for name in SPLITS_USED)
    tried = []
    for C in c_grid:
        started = time.perf_counter()
        base = NumpyLogisticRegression(C=C, class_weight="balanced").fit(train[columns], train["label"])
        fit_seconds = time.perf_counter() - started
        model = calibrate_frozen(base, calibration, columns)
        scored = validation[["patient_id", "hour", "label"]].copy()
        scored["probability"] = model.predict_proba(validation[columns])[:, 1]
        metrics = score_table(scored)
        tried.append({
            "C": C, "fit_seconds": fit_seconds, "newton_iterations": int(base.n_iter_),
            "converged": bool(base.converged_), "calibration_a": model.calibrator.a_,
            "calibration_b": model.calibrator.b_, "metrics": metrics, "scored": scored,
        })
        print(f"  arm {arm} C={C:g}: validation AUPRC {metrics['auprc']:.4f}, "
              f"{base.n_iter_} Newton steps, {fit_seconds:.1f}s")
    best = max(tried, key=lambda t: (np.nan_to_num(t["metrics"]["auprc"], nan=-1.0), -t["C"]))
    policy = tune_threshold(best["scored"], budget=budget, cooldown=cooldown)
    summary = {key: value for key, value in best.items() if key != "scored"}
    summary.update({
        "arm": arm, "name": ARM_NAMES[arm], "columns": len(columns),
        "c_grid_results": [{k: v for k, v in t.items() if k not in ("scored",)} for t in tried],
        "policy": policy,
    })
    return summary, best["scored"]


def refuse_existing(paths):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {', '.join(existing)}; choose a new --output")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--hmm", type=Path, default=Path("artifacts/hmm/hmm_model.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/h03"))
    parser.add_argument("--arms", default="ABCD", help="subset of ABCD; A is always included")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.3],
                        help="same grid for every arm, chosen by validation AUPRC")
    parser.add_argument("--budget", type=float, default=0.2, help="false alerts per nonseptic patient-day")
    parser.add_argument("--cooldown", type=int, default=6)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--fallback", action="store_true",
                        help="apply the J02 training-median fallback to raw values before the shared "
                             "features (same in every arm; off keeps the missingness indicators)")
    parser.add_argument("--limit", type=int, default=0, help="first N stays of each split (0 = all)")
    args = parser.parse_args()

    arms = sorted(set("A" + args.arms.upper()))
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms {sorted(unknown)}")
    results_path, arms_path = args.output / "h03_results.json", args.output / "h03_arms.csv"
    refuse_existing([results_path, arms_path])

    cohorts = load_split_patients(args.data, args.splits)
    cohorts = {name: cohorts[name][: args.limit] if args.limit else cohorts[name] for name in SPLITS_USED}
    for name, patients in cohorts.items():
        hospitals = {p.hospital for p in patients}
        if hospitals - {"A"}:
            raise SystemExit(f"{name} contains hospitals {sorted(hospitals)}; H03 is hospital A only")
        print(f"{name}: {len(patients)} stays, {sum(len(p.frame) for p in patients)} hours")

    hmm_model, hmm_columns = load_hmm(args.hmm)
    if list(hmm_columns) != list(VITALS):
        raise SystemExit(f"{args.hmm} was fitted on {hmm_columns}, expected {list(VITALS)}")
    hmm_meta = json.loads(args.hmm.read_text())
    if hmm_meta.get("split") != "train":
        raise SystemExit(f"{args.hmm} does not say it was fitted on the train split")

    train_frames = [p.frame for p in cohorts["train"]]
    scaler = fit_signature_scaler(train_frames)
    fallback = fit_training_fallback(train_frames) if args.fallback else None

    tables, feature_seconds = {}, {}
    for name in SPLITS_USED:
        started = time.perf_counter()
        tables[name], feature_seconds[name], (shared, hmm, signature) = build_split(
            cohorts[name], hmm_model, scaler, fallback)
        print(f"{name}: features in {time.perf_counter() - started:.0f}s")
    groups = {"hmm": list(hmm), "signature": list(signature)}
    base = list(shared)

    results, predictions = {}, {}
    for arm in arms:
        columns = arm_columns(base, groups, arm)
        print(f"arm {arm} ({ARM_NAMES[arm]}): {len(columns)} columns")
        results[arm], predictions[arm] = run_arm(arm, columns, tables, args.c_grid, args.budget, args.cooldown)

    print(f"paired bootstrap, {args.resamples} resamples of validation stays")
    differences = paired_bootstrap(predictions, "A", args.resamples, args.bootstrap_seed) if len(arms) > 1 else {}

    stays = {name: len(cohorts[name]) for name in SPLITS_USED}
    per_1000 = {group: 1000 * sum(s[group] for s in feature_seconds.values()) / sum(stays.values())
                for group in GROUPS}
    rows = []
    for arm in arms:
        r, p = results[arm], results[arm]["policy"]
        rows.append({
            "arm": arm, "name": r["name"], "columns": r["columns"], "C": r["C"],
            "auprc": r["metrics"]["auprc"], "auroc": r["metrics"]["auroc"], "brier": r["metrics"]["brier"],
            "auprc_first_6h": r["metrics"]["auprc_first_6h"],
            "normalized_hourly_utility": p["normalized_hourly_utility"], "threshold": p["threshold"],
            "early_detection_recall": p["early_detection_recall"],
            "false_alerts_per_nonseptic_day": p["false_alert_events_per_nonseptic_patient_day"],
            "median_lead_hours": p["median_inferred_lead_hours"],
            "auprc_diff_vs_A": differences.get(arm, {}).get("auprc_difference", 0.0 if arm == "A" else None),
            "ci_low": differences.get(arm, {}).get("ci_low"), "ci_high": differences.get(arm, {}).get("ci_high"),
            "helps": differences.get(arm, {}).get("helps"),
            "fit_seconds": r["fit_seconds"],
            "feature_seconds_per_1000_stays": per_1000["shared"]
            + sum(per_1000[g] for g in ARMS[arm]),
        })
    summary = pd.DataFrame(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    refuse_existing([results_path, arms_path])
    summary.to_csv(arms_path, index=False)
    results_path.write_text(json.dumps({
        "plan": "docs/h03-ablation-plan.md", "data": str(args.data), "splits": str(args.splits),
        "hmm": str(args.hmm), "hmm_fit": {k: hmm_meta.get(k) for k in ("split", "stays", "hours", "iterations")},
        "stays": stays, "limit": args.limit, "windows": list(WINDOWS), "signature_window": WINDOW,
        "fallback": args.fallback, "c_grid": args.c_grid, "budget": args.budget, "cooldown": args.cooldown,
        "resamples": args.resamples, "bootstrap_seed": args.bootstrap_seed,
        "feature_seconds": feature_seconds, "feature_seconds_per_1000_stays": per_1000,
        "arms": results, "auprc_difference_vs_A": differences,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }, indent=2, default=float))

    shown = summary[["arm", "name", "columns", "auprc", "auroc", "brier", "auprc_first_6h",
                     "normalized_hourly_utility", "early_detection_recall", "auprc_diff_vs_A", "ci_low", "ci_high"]]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved {arms_path} and {results_path}")


if __name__ == "__main__":
    main()
