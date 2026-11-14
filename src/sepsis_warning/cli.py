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

from .baselines import (
    BASELINE_THRESHOLD, BASELINES, auroc, average_precision, baseline_table,
    hourly_classification_metrics, plot_sirs_threshold_tradeoff, sirs_threshold_sweep,
)
from .data import generate_synthetic, load_patients, read_patient, split_patients
from .hospital_comparison import run_comparison
from .features import META, build_features, feature_table, select_saved_features
from .modeling import candidate_models, fit_calibrated
from .missingness import run_missingness_analysis
from .monitoring import REPLAY_SPLITS, MonitoringConfig, replay_monitoring, setup_monitoring
from .policy import apply_policy, summarize, tune_threshold
from .prepare import prepare_parquet, write_summary
from .report import write_report
from .scoring import load_bundle
from .splits import write_splits
from .tracking import check_tracking_uri, log_run, require_mlflow, run_metadata
from .ward import (
    HELD_OUT_SPLITS, SELECTION_SPLITS, build_payload, check_split, choose_stays,
    score_locally, score_via_api, write_ward,
)
from .schema import extreme_value_counts, validate_hourly
from .sirs_failures import failure_tables
from .splits import load_split_patients, write_splits


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_cohorts(args):
    """Return (cohorts, synthetic) for `baseline` and `sirs-failures`.

    A Parquet file as --data uses the saved splits.csv. A folder of .psv files
    still reads every record and re-splits with --seed.
    """
    if args.data.is_file():
        splits_path = args.splits or args.data.with_name("splits.csv")
        if not splits_path.exists():
            raise ValueError(f"{splits_path} not found; run `split` first")
        cohorts = load_split_patients(args.data, splits_path)
        ids = [p.patient_id for cohort in cohorts.values() for p in cohort]
        flagged = sum("synthetic_" in patient_id for patient_id in ids)
        if flagged not in (0, len(ids)):
            raise ValueError(f"{args.data} mixes synthetic and real records")
        synthetic = bool(flagged)
    else:
        synthetic = (args.data / "SYNTHETIC.json").exists()
        patients = load_patients(args.data)
        if not synthetic and any("synthetic_" in p.patient_id for p in patients):
            raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
        cohorts = split_patients(patients, args.seed)
    print("Data mode:", "SYNTHETIC SOFTWARE FIXTURE" if synthetic else "PhysioNet records", flush=True)
    return cohorts, synthetic


def run_experiment(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    splits, synthetic = load_cohorts(args)
    manifest = []
    for name, cohort in splits.items():
        print(f"{name}: {len(cohort)} patients", flush=True)
        for patient in cohort:
            pid = patient.patient_id.split(":", 1)[1] if ":" in patient.patient_id else patient.patient_id
            if not args.data.is_file():
                path = args.data / f"training_set{patient.hospital}" / f"{pid}.psv"
                sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            else:
                path = args.data.parent / f"training_set{patient.hospital}" / f"{pid}.psv"
                if path.exists():
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    sha = hashlib.sha256(patient.patient_id.encode("utf-8")).hexdigest()
            manifest.append({
                "patient_id": patient.patient_id, "hospital": patient.hospital,
                "split": name, "septic": patient.septic,
                "sha256": sha,
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
    splits_summary = args.splits_summary
    if splits_summary is None and args.data.is_file():
        splits_path = getattr(args, "splits", None) or args.data.with_name("splits.csv")
        candidate = splits_path.with_name(f"{splits_path.name}.summary.json")
        if candidate.exists():
            try:
                candidate_seed = json.loads(candidate.read_text(encoding="utf-8")).get("seed")
                if candidate_seed is not None and int(candidate_seed) == int(args.seed):
                    splits_summary = candidate
            except Exception:
                pass
    write_json(args.output / "run_metadata.json", run_metadata(
        args.seed, synthetic=synthetic, window=args.window, budget=args.budget,
        cooldown=args.cooldown, feature_columns=columns,
        splits_summary=splits_summary, dvc_lock=args.dvc_lock,
    ))
    replay = pd.concat(predictions, ignore_index=True)
    replay.to_csv(args.output / "held_out_predictions.csv", index=False)
    write_report(args.output / "replay.html", replay, summary, threshold, synthetic)
    print(f"Selected: {name}; threshold={threshold:.2f}", flush=True)
    print(f"Replay: {(args.output / 'replay.html').resolve()}", flush=True)
    if args.mlflow_uri:
        run_id = log_run(args.output, args.mlflow_uri)
        print(f"MLflow run: {run_id}", flush=True)


BASELINE_SPLITS = ["validation", "internal_test", "external_test"]
BASELINE_COLUMNS = [
    "model", "split", "patients", "septic_patients", "hours", "positive_hour_fraction",
    "hourly_precision", "hourly_recall", "hourly_f1", "hourly_specificity", "hourly_accuracy",
    "hourly_alert_rate", "no_predicted_positives",
    "normalized_hourly_utility", "alert_events",
    "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
    "median_inferred_lead_hours", "average_precision", "auroc",
    "eligible_septic_patients_for_lead_time", "left_censored_septic_patients",
    "hourly_tp", "hourly_fp", "hourly_fn", "hourly_tn",
]
SWEEP_COLUMNS = [
    "alert_at_criteria", "hourly_precision", "hourly_recall", "hourly_f1", "hourly_specificity",
    "hourly_accuracy", "hourly_alert_rate", "normalized_hourly_utility",
    "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
    "hourly_tp", "hourly_fp", "hourly_fn", "hourly_tn",
]


def run_baselines(args):
    """Score SIRS and always-no on validation, internal test and hospital B."""
    # The saved split and a seed-42 re-split are the same stays, so rows line up with `train`.
    splits, _ = load_cohorts(args)
    rows = []
    for split in BASELINE_SPLITS:
        for name in BASELINES:
            table = apply_policy(baseline_table(splits[split], name), BASELINE_THRESHOLD, args.cooldown)
            metrics = summarize(table)
            metrics.update(hourly_classification_metrics(table["label"], table["hourly_positive"]))
            # Hand-written ranking metrics replace the sklearn ones from summarize.
            metrics["auroc"] = auroc(table["label"], table["probability"])
            metrics["average_precision"] = average_precision(table["label"], table["probability"])
            rows.append({"model": name, "split": split, **metrics})
    result = pd.DataFrame(rows)[BASELINE_COLUMNS]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "baseline_metrics.csv"
    result.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(result[BASELINE_COLUMNS[:2] + [
            "hourly_precision", "hourly_recall", "hourly_f1", "normalized_hourly_utility",
            "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
            "auroc", "average_precision",
        ]].round(3).to_string(index=False))
    print(f"Baseline metrics: {path.resolve()}", flush=True)
    # Threshold trade-off (course L4) on VALIDATION ONLY. Descriptive: the rule
    # stays at >= 2 criteria; test splits are never used to look at thresholds.
    validation = baseline_table(splits["validation"], "sirs")
    sweep = sirs_threshold_sweep(validation, args.cooldown)[SWEEP_COLUMNS]
    sweep_path = args.output / "sirs_threshold_sweep_validation.csv"
    sweep.to_csv(sweep_path, index=False)
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_sirs_threshold_tradeoff(
        sweep, float(validation["label"].mean()), 0.2, args.figures / "08_sirs_threshold_tradeoff.png",
        f"python -m sepsis_warning baseline --data {args.data.as_posix()} --output artifacts/baseline",
    )
    print(sweep[["alert_at_criteria", "hourly_precision", "hourly_recall", "hourly_f1",
                 "hourly_specificity", "false_alert_events_per_nonseptic_patient_day"]].round(3).to_string(index=False))
    print(f"Threshold sweep (validation): {sweep_path.resolve()}", flush=True)
    return result


def run_server(args):
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_non_local_host:
        raise ValueError(f"--host {args.host} exposes an unauthenticated API; pass --allow-non-local-host if agreed")
    try:
        import uvicorn

        from .api import create_app
    except ImportError as error:
        raise RuntimeError("the API needs the serve extra; run `pip install -e .[serve]`") from error
    from .service import ModelMismatch, ScoringService

    try:
        service = ScoringService.from_path(args.model, args.model_sha256)
    except ModelMismatch as error:
        raise ValueError(str(error)) from error
    health = service.health()
    print(f"Serving {health['model_name']} (bundle {health['bundle_sha256'][:12]}) on {args.host}:{args.port}", flush=True)
    if health["synthetic_model"]:
        print("WARNING: synthetic fixture model; scores are meaningless", flush=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)
def run_sirs_failures(args):
    """Aggregate SIRS failure patterns. Validation split only, by design."""
    cohorts, _ = load_cohorts(args)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in failure_tables(cohorts["validation"], args.cooldown).items():
        table.to_csv(args.output / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}", flush=True)
    print(f"\nTables: {args.output.resolve()}", flush=True)


def run_validate(args):
    """Check every stay against the hard rules, then count review flags on train only."""
    for path, step in [(args.data, "prepare"), (args.splits, "split")]:
        if not path.exists():
            raise ValueError(f"{path} not found; run `{step}` first")
    frame = pd.read_parquet(args.data)
    validate_hourly(frame)
    print(f"All {frame.groupby(['hospital', 'patient_id']).ngroups} stays and {len(frame)} rows pass the hard rules.", flush=True)
    splits = pd.read_csv(args.splits, dtype=str)
    train = splits.loc[splits["split"] == "train", ["hospital", "patient_id"]]
    table = extreme_value_counts(frame.merge(train, on=["hospital", "patient_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "extreme_value_review_train.csv"
    table.to_csv(path, index=False)
    print(f"\nReview flags, training stays only (nothing rejected):\n{table.to_string(index=False)}")
    print(f"\nReview table: {path.resolve()}", flush=True)


DATA_HELP = "combined.parquet (uses the saved split), or a folder of training_setA/B (re-splits with --seed)"
SPLITS_HELP = "Saved split for a Parquet --data; defaults to splits.csv next to it"


def run_ward(args):
    """Write the ICU ward replay page for a saved model (see ward.py)."""
    check_split(args.split, args.after_freeze)
    if args.output.exists():
        raise ValueError(f"{args.output} already exists; choose a new output file")
    cohorts, synthetic = load_cohorts(args)
    cohort = cohorts[args.split]
    stays = choose_stays(cohort, args.beds, args.septic_share, args.max_hours, args.seed)
    if args.api_url:
        frames, details = score_via_api(args.api_url, stays)
        if bool(details["synthetic_model"]) != synthetic:
            raise ValueError("The API's model and the data disagree on whether they are synthetic")
        model_name, threshold, cooldown = details["model_name"], details["threshold"], details["cooldown_hours"]
        scored_by = f"POST /score on {args.api_url}, one request per bed, as the hospital engine would send"
    else:
        bundle = load_bundle(args.model)
        if bool(bundle["synthetic"]) != synthetic:
            raise ValueError("The model bundle and the data disagree on whether they are synthetic")
        frames = score_locally(bundle, stays)
        model_name, threshold, cooldown = bundle["model_name"], bundle["threshold"], bundle["cooldown"]
        scored_by = "Saved model scored in-process, with the same feature code as the API"
    payload = build_payload(
        frames, model_name=model_name, threshold=threshold, cooldown=cooldown, synthetic=synthetic,
        split=args.split, scored_by=scored_by, seed=args.seed,
        cohort_septic_share=sum(p.septic for p in cohort) / len(cohort),
    )
    write_ward(args.output, payload)
    if not synthetic:
        print("Contains PhysioNet stays: keep it private and out of Git.", flush=True)
    print(f"Ward replay: {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate an explicitly synthetic fixture and run it")
    demo.add_argument("--data", type=Path, default=Path("data/synthetic"))
    for command in [demo, sub.add_parser("train", help="Run on downloaded training_setA/B or combined.parquet")]:
        if command is not demo:
            command.add_argument("--data", type=Path, required=True, help=DATA_HELP)
            command.add_argument("--splits", type=Path, default=None, help=SPLITS_HELP)
        command.add_argument("--output", type=Path, default=Path("artifacts/demo" if command is demo else "artifacts/real"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--window", type=int, default=6)
        command.add_argument("--budget", type=float, default=0.2)
        command.add_argument("--cooldown", type=int, default=6)
        command.add_argument(
            "--splits-summary", type=Path, default=None,
            help="splits.csv.summary.json from `split`; its seed and hashes are logged and the seed must match",
        )
        command.add_argument("--dvc-lock", type=Path, default=None, help="dvc.lock whose output hashes are logged")
        command.add_argument(
            "--mlflow-uri", default=None,
            help="Local MLflow store, e.g. file:artifacts/mlruns (needs the tracking extra)",
        )
    replay = sub.add_parser("replay", help="Score a patient PSV with a saved model")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--patient", type=Path, required=True)
    replay.add_argument("--output", type=Path, default=Path("artifacts/patient_replay.csv"))
    missingness = sub.add_parser(
        "missingness",
        help="Audit training/validation missingness and compare fixed baselines",
    )
    missingness.add_argument("--data", type=Path, required=True)
    missingness.add_argument(
        "--output", type=Path, default=Path("artifacts/missingness")
    )
    missingness.add_argument("--seed", type=int, default=42)
    prepare = sub.add_parser(
        "prepare", help="Validate PSV files and combine them into one Parquet file"
    )
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument(
        "--output", type=Path, default=Path("data/physionet/combined.parquet")
    )
    baseline = sub.add_parser(
        "baseline", help="Score the SIRS and always-no baselines on every evaluation split"
    )
    baseline.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    baseline.add_argument("--splits", type=Path, help=SPLITS_HELP)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--cooldown", type=int, default=6)
    baseline.add_argument("--figures", type=Path, default=Path("reports/figures"))
    validate = sub.add_parser(
        "validate", help="Check the combined Parquet against the Pandera rules (needs the checks extra)"
    )
    validate.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    validate.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"))
    validate.add_argument("--output", type=Path, default=Path("artifacts/data_checks"))
    failures = sub.add_parser(
        "sirs-failures", help="Aggregate patterns of where SIRS fails (validation split only)"
    )
    failures.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"), help=DATA_HELP)
    failures.add_argument("--splits", type=Path, help=SPLITS_HELP)
    failures.add_argument("--output", type=Path, default=Path("artifacts/baseline/failures"))
    failures.add_argument("--seed", type=int, default=42)
    failures.add_argument("--cooldown", type=int, default=6)
    split = sub.add_parser("split", help="Save the patient split once as splits.csv")
    split.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    split.add_argument("--output", type=Path, default=Path("data/physionet/splits.csv"))
    split.add_argument("--seed", type=int, default=42)
    compare = sub.add_parser(
        "compare-hospitals",
        help="Descriptive hospital A vs B figures and summary table from the combined Parquet",
    )
    compare.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    compare.add_argument("--output", type=Path, default=Path("reports/figures"))
    compare.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"),
                         help="Hospital A is restricted to this file's train split")
    defaults = MonitoringConfig()
    monitor_setup = sub.add_parser(
        "monitor-setup",
        help="Freeze the two-speed monitoring reference from A train/validation (run before any test replay)",
    )
    monitor_setup.add_argument("--data", type=Path, required=True)
    monitor_setup.add_argument("--model", type=Path, required=True, help="Frozen model.joblib from train/demo")
    monitor_setup.add_argument("--output", type=Path, default=Path("artifacts/monitoring/reference"))
    monitor_setup.add_argument("--seed", type=int, default=42, help="Patient split seed used by train")
    monitor_setup.add_argument("--fast-window-hours", type=int, default=defaults.fast_window_hours)
    monitor_setup.add_argument("--slow-window-hours", type=int, default=defaults.slow_window_hours)
    monitor_setup.add_argument("--min-observed-values", type=int, default=defaults.min_observed_values,
                               help="Minimum observed hourly values for a variable's PSI in one fast window")
    monitor_setup.add_argument("--min-observed-stays", type=int, default=defaults.min_observed_stays,
                               help="Minimum stays for the PSI of Age, Gender and HospAdmTime in one fast window")
    monitor_setup.add_argument("--min-slow-stays", type=int, default=defaults.min_slow_stays)
    monitor_setup.add_argument("--min-slow-septic-stays", type=int, default=defaults.min_slow_septic_stays)
    monitor_setup.add_argument("--min-subgroup-septic-stays", type=int, default=defaults.min_subgroup_septic_stays)
    monitor_setup.add_argument("--null-windows", type=int, default=defaults.null_windows)
    monitor_replay = sub.add_parser(
        "monitor-replay",
        help="Replay a held-out split through the frozen monitoring reference (aggregate outputs only)",
    )
    monitor_replay.add_argument("--data", type=Path, required=True)
    monitor_replay.add_argument("--model", type=Path, required=True)
    monitor_replay.add_argument("--reference", type=Path, required=True, help="monitoring_reference.json")
    monitor_replay.add_argument("--split", choices=REPLAY_SPLITS, default="external_test")
    monitor_replay.add_argument("--output", type=Path, default=Path("artifacts/monitoring/replay"))
    serve = sub.add_parser("serve", help="Local research scoring API for a saved model (needs the serve extra)")
    serve.add_argument("--model", type=Path, required=True, help="model.joblib produced by train/demo")
    serve.add_argument("--model-sha256", default=None, help="Refuse to start unless the bundle has this SHA-256")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--allow-non-local-host", action="store_true",
        help="Bind beyond localhost. The API has no authentication; agree this with the team first",
    )
    ward = sub.add_parser("ward", help="Write an ICU ward replay page for a saved model (research demo)")
    ward.add_argument("--model", type=Path, help="model.joblib produced by train/demo (not needed with --api-url)")
    ward.add_argument("--api-url", default=None, help="Score through a running API instead, e.g. http://127.0.0.1:8000")
    ward.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    ward.add_argument("--splits", type=Path, help=SPLITS_HELP)
    ward.add_argument("--split", choices=SELECTION_SPLITS + HELD_OUT_SPLITS, default="validation")
    ward.add_argument("--after-freeze", action="store_true",
                      help="Allow internal_test or external_test; only once the model is frozen")
    ward.add_argument("--beds", type=int, default=12)
    ward.add_argument("--septic-share", type=float, default=0.3,
                      help="Share of beds given to stays that become septic (over-represented for the demo)")
    ward.add_argument("--max-hours", type=int, default=72, help="Hours shown per stay")
    ward.add_argument("--seed", type=int, default=42)
    ward.add_argument("--output", type=Path, default=Path("artifacts/ward/ward.html"))
    args = parser.parse_args()
    try:
        if args.command in ["demo", "train"] and args.mlflow_uri:
            # Fail before generating data or training, not after it.
            check_tracking_uri(args.mlflow_uri)
            require_mlflow()
        if args.command == "demo":
            generate_synthetic(args.data, args.seed)
            write_json(args.data / "SYNTHETIC.json", {"purpose": "Software test only", "seed": args.seed})
        if args.command in ["demo", "train"]:
            if args.window < 1 or args.budget < 0 or args.cooldown < 0:
                raise ValueError("window must be positive; budget and cooldown must be nonnegative")
            run_experiment(args)
        elif args.command == "replay":
            bundle = joblib.load(args.model)
            frame = read_patient(args.patient, require_labels=False)
            features = select_saved_features(build_features(frame, bundle["window"]), bundle["feature_columns"])
            table = pd.DataFrame({"patient_id": args.patient.stem, "hour": frame["ICULOS"]})
            with threadpool_limits(limits=2):
                table["probability"] = bundle["model"].predict_proba(features)[:, 1]
            table = apply_policy(table, bundle["threshold"], bundle["cooldown"])
            # Replay supports histories without a SepsisLabel column.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.output, index=False)
            print(args.output.resolve())
        elif args.command == "serve":
            run_server(args)
        elif args.command == "ward":
            if not args.model and not args.api_url:
                raise ValueError("pass --model, or --api-url for a running API")
            if args.max_hours < 1:
                raise ValueError("max-hours must be positive")
            run_ward(args)
        elif args.command == "baseline":
            if args.cooldown < 0:
                raise ValueError("cooldown must be nonnegative")
            run_baselines(args)
        elif args.command == "validate":
            run_validate(args)
        elif args.command == "sirs-failures":
            if args.cooldown < 0:
                raise ValueError("cooldown must be nonnegative")
            run_sirs_failures(args)
        elif args.command == "split":
            if not args.data.exists():
                raise ValueError(f"{args.data} not found; run `prepare` first")
            summary = write_splits(args.data, args.output, args.seed)
            for name, count in summary["counts"].items():
                print(f"{name:14s} {count['stays']:6d} stays, {count['septic']:5d} septic")
            print(f"Splits: {args.output.resolve()}")
        elif args.command == "compare-hospitals":
            if not args.data.exists():
                raise ValueError(f"{args.data} not found; run `prepare` first")
            if not args.splits.exists():
                raise ValueError(f"{args.splits} not found; run `split` first")
            summary = run_comparison(args.data, args.output, args.splits)
            print(summary.to_string())
            print(f"Figures and README: {args.output.resolve()}")
        elif args.command == "missingness":
            synthetic = (args.data / "SYNTHETIC.json").exists()
            patients = load_patients(args.data)
            if not synthetic and any(
                "synthetic_" in patient.patient_id for patient in patients
            ):
                raise ValueError(
                    "Synthetic filenames found without SYNTHETIC.json marker"
                )
            summary = run_missingness_analysis(
                patients,
                args.output,
                seed=args.seed,
                synthetic=synthetic,
            )
            print(
                f"Missingness report: {(args.output / 'missingness_report.html').resolve()}"
            )
            print(
                f"Records used: train={summary['training_records']}, "
                f"validation={summary['validation_records']}"
            )
        elif args.command == "monitor-setup":
            config = MonitoringConfig(
                fast_window_hours=args.fast_window_hours, slow_window_hours=args.slow_window_hours,
                min_observed_values=args.min_observed_values, min_observed_stays=args.min_observed_stays,
                min_slow_stays=args.min_slow_stays,
                min_slow_septic_stays=args.min_slow_septic_stays,
                min_subgroup_septic_stays=args.min_subgroup_septic_stays, null_windows=args.null_windows,
            )
            reference = setup_monitoring(args.data, args.model, args.output, config, split_seed=args.seed)
            print(f"Data mode: {reference['data_mode']}; model {reference['model_name']} "
                  f"(SHA-256 {reference['model_sha256'][:12]})")
            print("Reference: A train inputs; A validation scores, alerts and performance. No test data used.")
            print(f"Monitoring reference: {(args.output / 'monitoring_reference.json').resolve()}")
        elif args.command == "monitor-replay":
            summary = replay_monitoring(args.data, args.model, args.reference, args.split, args.output)
            print("Research replay only; not live clinical monitoring. No automatic retraining.")
            for family, row in summary["families"].items():
                print(f"{family:16s} {row['speed']:4s} owner={row['owner']:8s} "
                      f"ok={row['ok']:3d} warn={row['warn']:3d} alert={row['alert']:3d} "
                      f"insufficient={row['insufficient_data']:3d}")
            print(f"Report: {(args.output / 'monitoring_report.md').resolve()}")
        else:
            summary = prepare_parquet(args.data, args.output)
            summary_path = args.output.with_name(
                f"{args.output.name}.summary.json"
            )
            write_summary(summary_path, summary)
            print(f"Combined Parquet: {args.output.resolve()}")
            print(
                f"Records: {summary['records']}; hourly rows: "
                f"{summary['hourly_rows']}"
            )
    except (ValueError, FileExistsError, RuntimeError) as error:
        # FileExistsError: prepare refuses to overwrite; show a message, not a traceback.
        parser.error(str(error))
