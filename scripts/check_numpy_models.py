"""Validation-only check: the NumPy models next to their scikit-learn references.

    python scripts/check_numpy_models.py --data data/physionet/combined.parquet --splits data/physionet/splits.csv

Each model is fitted on the train split, Platt-calibrated on the calibration
split, and scored on the validation split with the same threshold search,
alarm budget and cooldown as `train`. Internal and external test stays are
never built into features or scored. Writes one row of aggregate metrics per
model to reports/model_check/validation_numpy_vs_sklearn.csv.
About 20 minutes on a laptop, mostly building features.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from sepsis_warning.features import META, feature_table
from sepsis_warning.gradient_boosting import NumpyGradientBoosting
from sepsis_warning.logistic import NumpyLogisticRegression, calibrate_frozen
from sepsis_warning.modeling import candidate_models, fit_calibrated
from sepsis_warning.policy import tune_threshold
from sepsis_warning.random_forest import NumpyRandomForest
from sepsis_warning.splits import load_split_patients


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_check/validation_numpy_vs_sklearn.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--cooldown", type=int, default=6)
    parser.add_argument("--numpy-only", action="store_true", help="Skip the scikit-learn references")
    args = parser.parse_args()

    splits = load_split_patients(args.data, args.splits)
    tables = {n: feature_table(splits[n], args.window) for n in ["train", "calibration", "validation"]}
    columns = [c for c in tables["train"] if c not in META]
    print({n: len(t) for n, t in tables.items()}, len(columns), "features", flush=True)

    reference = candidate_models(args.seed)
    models = [
        ("sklearn", "logistic_regression", reference["logistic_regression"]),
        ("numpy", "logistic_regression", NumpyLogisticRegression(C=0.3)),
        ("sklearn", "hist_gradient_boosting", reference["hist_gradient_boosting"]),
        ("numpy", "gradient_boosting", NumpyGradientBoosting(seed=args.seed)),
        ("sklearn", "random_forest", reference["random_forest"]),
        ("numpy", "random_forest", NumpyRandomForest(seed=args.seed)),
    ]
    if args.numpy_only:
        models = [m for m in models if m[0] == "numpy"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for implementation, name, estimator in models:
        print(f"Training {implementation} {name}...", flush=True)
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            if implementation == "numpy":
                estimator.fit(tables["train"][columns], tables["train"]["label"])
                model = calibrate_frozen(estimator, tables["calibration"], columns)
            else:
                model = fit_calibrated(estimator, tables["train"], tables["calibration"], columns)
            validation = tables["validation"][META].copy()
            validation["probability"] = model.predict_proba(tables["validation"][columns])[:, 1]
        seconds = time.perf_counter() - started
        metrics = tune_threshold(validation, args.budget, args.cooldown)
        rows.append({"implementation": implementation, "model": name, "seconds": round(seconds), **metrics})
        print(f"  {seconds:.0f}s utility={metrics['normalized_hourly_utility']:.3f} "
              f"auroc={metrics['auroc']:.3f} "
              f"false/day={metrics['false_alert_events_per_nonseptic_patient_day']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
