"""Run the sketch end to end, or score an individual patient history."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import generate_synthetic, load_patients, read_patient, split_patients
from .features import META, build_features, feature_table
from .modeling import candidate_models, fit_calibrated
from .policy import apply_policy, summarize, tune_threshold
from .report import write_report


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic = (args.data / "SYNTHETIC.json").exists()
    print("Data mode:", "SYNTHETIC SOFTWARE FIXTURE" if synthetic else "PhysioNet records", flush=True)
    patients = load_patients(args.data)
    if not synthetic and any("synthetic_" in p.patient_id for p in patients):
        raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
    splits = split_patients(patients, args.seed)
    manifest = []
    for name, cohort in splits.items():
        print(f"{name}: {len(cohort)} patients", flush=True)
        for patient in cohort:
            path = args.data / f"training_set{patient.hospital}" / f"{patient.patient_id.split(':', 1)[1]}.psv"
            manifest.append({
                "patient_id": patient.patient_id, "hospital": patient.hospital,
                "split": name, "septic": patient.septic,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(manifest).to_csv(args.output / "split_manifest.csv", index=False)
    # Only source training/calibration/validation features are built before
    # selecting the model and threshold. Test outcomes never guide selection.
    tables = {name: feature_table(splits[name], args.window) for name in ["train", "calibration", "validation"]}
    columns = [c for c in tables["train"] if c not in META]
    comparisons = []
    fitted = {}
    for name, estimator in candidate_models(args.seed).items():
        print(f"Training {name}...", flush=True)
        with threadpool_limits(limits=2):
            model = fit_calibrated(estimator, tables["train"], tables["calibration"], columns, is_constant=name == "constant_prior")
            validation = tables["validation"][META].copy()
            validation["probability"] = model.predict_proba(tables["validation"][columns])[:, 1]
        metrics = tune_threshold(validation, args.budget, args.cooldown)
        comparisons.append({"model": name, **metrics})
        fitted[name] = model
    selected = max(comparisons, key=lambda m: (
        m["normalized_hourly_utility"], -m["false_alert_events_per_nonseptic_patient_day"],
        m["average_precision"],
    ))
    name, threshold = selected["model"], selected["threshold"]
    pd.DataFrame(comparisons).to_csv(args.output / "validation_models.csv", index=False)
    bundle = {
        "model": fitted[name], "model_name": name, "feature_columns": columns,
        "window": args.window, "threshold": threshold, "cooldown": args.cooldown,
        "synthetic": synthetic,
    }
    joblib.dump(bundle, args.output / "model.joblib")
    held_out = {}
    predictions = []
    for split in ["internal_test", "external_test"]:
        table = feature_table(splits[split], args.window)
        predicted = table[META].copy()
        with threadpool_limits(limits=2):
            predicted["probability"] = fitted[name].predict_proba(table[columns])[:, 1]
        predicted = apply_policy(predicted, threshold, args.cooldown)
        held_out[split] = summarize(predicted)
        predicted["split"] = split
        predictions.append(predicted)
    summary = {
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "selected_model": name, "threshold": threshold,
        "validation_alarm_budget": args.budget, "cooldown_hours": args.cooldown,
        "validation_models": comparisons, "held_out": held_out,
    }
    config = {
        "seed": args.seed, "window_hours": args.window, "budget": args.budget,
        "cooldown_hours": args.cooldown, "data_root": str(args.data.resolve()),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "synthetic": synthetic, "feature_columns": columns,
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "metrics.json", summary)
    replay = pd.concat(predictions, ignore_index=True)
    replay.to_csv(args.output / "held_out_predictions.csv", index=False)
    write_report(args.output / "replay.html", replay, summary, threshold, synthetic)
    print(f"Selected: {name}; threshold={threshold:.2f}", flush=True)
    print(f"Replay: {(args.output / 'replay.html').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate an explicitly synthetic fixture and run it")
    demo.add_argument("--data", type=Path, default=Path("data/synthetic"))
    for command in [demo, sub.add_parser("train", help="Run on downloaded training_setA/B")]:
        if command is not demo:
            command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, default=Path("artifacts/demo" if command is demo else "artifacts/real"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--window", type=int, default=6)
        command.add_argument("--budget", type=float, default=0.2)
        command.add_argument("--cooldown", type=int, default=6)
    replay = sub.add_parser("replay", help="Score a patient PSV with a saved model")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--patient", type=Path, required=True)
    replay.add_argument("--output", type=Path, default=Path("artifacts/patient_replay.csv"))
    args = parser.parse_args()
    try:
        if args.command == "demo":
            generate_synthetic(args.data, args.seed)
            write_json(args.data / "SYNTHETIC.json", {"purpose": "Software test only", "seed": args.seed})
        if args.command in ["demo", "train"]:
            if args.window < 1 or args.budget < 0 or args.cooldown < 0:
                raise ValueError("window must be positive; budget and cooldown must be nonnegative")
            run_experiment(args)
        else:
            bundle = joblib.load(args.model)
            frame = read_patient(args.patient, require_labels=False)
            features = build_features(frame, bundle["window"])
            table = pd.DataFrame({"patient_id": args.patient.stem, "hour": frame["ICULOS"]})
            with threadpool_limits(limits=2):
                table["probability"] = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]
            table = apply_policy(table, bundle["threshold"], bundle["cooldown"])
            # Replay supports histories without a SepsisLabel column.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.output, index=False)
            print(args.output.resolve())
    except ValueError as error:
        parser.error(str(error))
