"""Run metadata for reproducible experiments, with optional MLflow logging (L01).

``run_metadata`` always works with the base install and is written next to every
``train``/``demo`` run as ``run_metadata.json``. ``log_run`` needs the optional
``tracking`` extra (``pip install -e .[tracking]``) and logs a *selection* run:
one parent run with the data, split and code fingerprints, and one nested run
per candidate model with its validation metrics and, where the model records
one, its training-loss curve. Only aggregate files from an allow-list are
logged: never patient records, patient-level predictions, replay reports or
model files. Held-out test results are left out of selection runs.

Use a SQLite store (``sqlite:///artifacts/mlflow.db``). MLflow 3 refuses the
old ``file:`` store, and the model registry (``registry.py``) needs a database.
Artifacts are kept next to the database, inside the Git-ignored ``artifacts/``.
"""

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlparse

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow"]

# Aggregate outputs only. held_out_predictions.csv, split_manifest.csv,
# replay.html and model.joblib are patient-level or executable and stay local.
# metrics.json is logged as selection_metrics.json, without held-out results.
LOGGABLE_ARTIFACTS = {"config.json", "selection_metrics.json", "validation_models.csv",
                      "run_metadata.json", "candidate_details.json"}
DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow.db"
SELECTION_EXPERIMENT = "sepsis-warning-selection"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names=PACKAGES) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(root: Path | None = None) -> str | None:
    """Current commit, or None outside a Git checkout. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def split_evidence(summary_path: Path, seed: int) -> dict:
    """Seed and hashes from Edwin's ``splits.csv.summary.json``; refuse a seed mismatch."""
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("seed", "splits_csv_sha256", "source_parquet_sha256"):
        if key not in summary:
            raise ValueError(f"{summary_path} has no '{key}'; regenerate it with the `split` command")
    if int(summary["seed"]) != int(seed):
        raise ValueError(
            f"run seed {seed} does not match split seed {summary['seed']} in {summary_path}; "
            "the logged split would not describe this run"
        )
    return {
        "split_seed": int(summary["seed"]),
        "splits_csv_sha256": summary["splits_csv_sha256"],
        "source_parquet_sha256": summary["source_parquet_sha256"],
        "split_unit": summary.get("unit"),
        "split_counts": summary.get("counts"),
        "splits_summary_sha256": _sha256(summary_path),
    }


def dvc_evidence(lock_path: Path) -> dict:
    """Output hashes from ``dvc.lock`` (``{stage: {path: md5}}``). Needs PyYAML, which DVC installs."""
    lock_path = Path(lock_path)
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("reading dvc.lock needs PyYAML; install the DVC extra first") from error
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    stages = lock.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"{lock_path} has no stages")
    hashes = {}
    for stage, entry in stages.items():
        outs = {}
        for out in (entry or {}).get("outs", []) or []:
            digest = out.get("md5") or out.get("hash")
            if out.get("path") and digest:
                outs[out["path"]] = digest
        if outs:
            hashes[stage] = outs
    if not hashes:
        raise ValueError(f"{lock_path} lists no output hashes")
    return {"dvc_lock_sha256": _sha256(lock_path), "dvc_outputs": hashes}


def run_metadata(
    seed: int,
    *,
    synthetic: bool,
    window: int,
    budget: float,
    cooldown: int,
    feature_columns: list[str],
    splits_summary: Path | None = None,
    dvc_lock: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Everything needed to say which data, split, code and settings produced a run."""
    metadata = {
        "seed": int(seed),
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "window_hours": int(window),
        "alarm_budget": float(budget),
        "cooldown_hours": int(cooldown),
        "feature_count": len(feature_columns),
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
        "git_commit": git_commit(repo_root),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    if splits_summary is not None:
        metadata.update(split_evidence(splits_summary, seed))
    if dvc_lock is not None:
        metadata.update(dvc_evidence(dvc_lock))
    return metadata


def check_tracking_uri(uri: str) -> str:
    """Allow local file/sqlite stores and localhost servers only, so nothing leaves the machine."""
    parsed = urlparse(uri)
    # MLflow reads bare "databricks"/"databricks-uc" as a hosted workspace, not a folder.
    if parsed.scheme == "" and (uri.lower().startswith("databricks") or uri.lower() == "uc"):
        raise ValueError(f"tracking URI {uri!r} is a hosted MLflow workspace; that needs team approval first")
    if parsed.scheme in ("", "file", "sqlite"):
        return uri
    if parsed.scheme in ("http", "https") and parsed.hostname in LOCAL_HOSTS:
        return uri
    raise ValueError(
        f"tracking URI {uri!r} is not local; a shared or cloud MLflow server needs team approval first"
    )


def _flat_params(metadata: dict) -> dict:
    params = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            continue
        params[key] = "" if value is None else str(value)[:500]
    for name, version in metadata.get("packages", {}).items():
        params[f"package.{name}"] = str(version)
    return params


def _numeric(values: dict, prefix: str) -> dict:
    """Finite numbers only, with a prefix; booleans and None are skipped."""
    flat = {}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def artifact_root(tracking_uri: str) -> str | None:
    """Folder for run artifacts: next to a SQLite database, else the store's default.

    MLflow otherwise writes artifacts to ``./mlruns`` in the current folder,
    which is not Git-ignored at the repository root.
    """
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "sqlite":
        return None
    database = Path(tracking_uri[len("sqlite:///"):])
    return (database.parent / "mlartifacts").resolve().as_uri()


def prepare_store(tracking_uri: str, experiment: str, mlflow_module):
    """Point MLflow at a local store and an experiment whose artifacts stay in that folder."""
    check_tracking_uri(tracking_uri)
    if urlparse(tracking_uri).scheme == "sqlite":
        Path(tracking_uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    mlflow_module.set_tracking_uri(tracking_uri)
    if hasattr(mlflow_module, "set_registry_uri"):
        mlflow_module.set_registry_uri(tracking_uri)
    root = artifact_root(tracking_uri)
    if root and hasattr(mlflow_module, "get_experiment_by_name") and \
            mlflow_module.get_experiment_by_name(experiment) is None:
        mlflow_module.create_experiment(experiment, artifact_location=root)
    mlflow_module.set_experiment(experiment)


def require_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow is not installed; run `pip install -e .[tracking]`") from error
    return mlflow


def log_run(run_dir: Path, tracking_uri: str, experiment: str = SELECTION_EXPERIMENT,
            mlflow_module=None) -> str:
    """Log a finished ``train``/``demo`` directory as a selection run. Returns the parent run id.

    The parent run holds the fingerprints, the selected model and its validation
    metrics; each candidate gets a nested run with its validation metrics and,
    if ``candidate_details.json`` has one, its training-loss curve. Held-out
    test metrics are never logged here.
    """
    run_dir = Path(run_dir)
    check_tracking_uri(tracking_uri)
    if mlflow_module is None:
        mlflow_module = require_mlflow()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    details_path = run_dir / "candidate_details.json"
    details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.exists() else {}
    selection = {key: value for key, value in metrics.items() if key != "held_out"}
    (run_dir / "selection_metrics.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    candidates = metrics.get("validation_models") or []
    selected = next((row for row in candidates if row.get("model") == metrics.get("selected_model")), {})

    prepare_store(tracking_uri, experiment, mlflow_module)
    with mlflow_module.start_run(run_name=f"train-{metadata.get('data_mode', 'run')}") as run:
        mlflow_module.set_tags({
            "data_mode": metadata["data_mode"],
            "selected_model": metrics.get("selected_model", ""),
            "run_kind": "selection",
            "research_only": "true",
        })
        mlflow_module.log_params(_flat_params(metadata))
        mlflow_module.log_metrics({"threshold": float(metrics["threshold"]),
                                   **_numeric(selected, "validation.")})
        for name in sorted(LOGGABLE_ARTIFACTS):
            if (run_dir / name).exists():
                mlflow_module.log_artifact(str(run_dir / name))
        for row in candidates:
            name = row.get("model", "candidate")
            info = details.get(name, {})
            with mlflow_module.start_run(run_name=name, nested=True):
                mlflow_module.set_tags({
                    "candidate": name, "run_kind": "candidate",
                    "selected": str(name == metrics.get("selected_model")).lower(),
                    "implementation": info.get("implementation", "unknown"),
                })
                params = {"threshold": row.get("threshold")}
                params.update({f"model.{key}": value for key, value in (info.get("params") or {}).items()})
                mlflow_module.log_params({key: "" if value is None else str(value)[:500]
                                          for key, value in params.items()})
                values = _numeric(row, "validation.")
                if "fit_seconds" in info:
                    values["fit_seconds"] = float(info["fit_seconds"])
                mlflow_module.log_metrics(values)
                for step, loss in enumerate(info.get("loss_history") or []):
                    mlflow_module.log_metric("training_loss", float(loss), step=step)
        return run.info.run_id
