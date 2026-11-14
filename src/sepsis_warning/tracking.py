"""Run metadata for reproducible experiments, with optional MLflow logging (L01).

``run_metadata`` always works with the base install and is written next to every
``train``/``demo`` run as ``run_metadata.json``. ``log_run`` needs the optional
``tracking`` extra (``pip install -e .[tracking]``) and only logs aggregate
files from an allow-list: never patient records, patient-level predictions,
replay reports or model files.

Model registry promotion is out of scope here and needs team review.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlparse

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow"]

# Aggregate outputs only. held_out_predictions.csv, split_manifest.csv,
# replay.html and model.joblib are patient-level or executable and stay local.
LOGGABLE_ARTIFACTS = {"config.json", "metrics.json", "validation_models.csv", "run_metadata.json"}

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


def _numeric_metrics(metrics: dict) -> dict:
    flat = {}
    for split, values in (metrics.get("held_out") or {}).items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
                flat[f"{split}.{key}"] = float(value)
    return flat


def require_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow is not installed; run `pip install -e .[tracking]`") from error
    return mlflow


def log_run(run_dir: Path, tracking_uri: str, experiment: str = "sepsis-warning", mlflow_module=None) -> str:
    """Log a finished run directory to MLflow. Returns the MLflow run id."""
    run_dir = Path(run_dir)
    check_tracking_uri(tracking_uri)
    if mlflow_module is None:
        mlflow_module = require_mlflow()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment)
    with mlflow_module.start_run() as run:
        mlflow_module.set_tags({
            "data_mode": metadata["data_mode"],
            "selected_model": metrics.get("selected_model", ""),
            "research_only": "true",
        })
        mlflow_module.log_params(_flat_params(metadata))
        mlflow_module.log_metrics({"threshold": float(metrics["threshold"]), **_numeric_metrics(metrics)})
        for name in sorted(LOGGABLE_ARTIFACTS):
            if (run_dir / name).exists():
                mlflow_module.log_artifact(str(run_dir / name))
        return run.info.run_id
