"""Model manifest: what a bundle is, what built it and what it needs to run.

``train`` embeds the manifest in the bundle (``bundle["manifest"]``) and writes
it as ``manifest.json`` next to ``model.joblib``. The JSON file also carries the
bundle's SHA-256, which cannot be stored inside the file it describes.

A pickled scikit-learn model can fail, or score differently without an error,
under another scikit-learn release, so a bundle is refused when the runtime's
scikit-learn major.minor differs from the one that fitted it.
"""

import datetime
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

MANIFEST_FORMAT = 1
MANIFEST_FILE = "manifest.json"
# The code that defines what the model's inputs and alerts mean.
CONTRACT_MODULES = ("data.py", "features.py", "policy.py")
STRICT_PACKAGES = ("scikit-learn",)
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


class ManifestMismatch(ValueError):
    """The bundle was built for a different runtime."""


def code_sha256() -> str:
    """Hash of the feature and policy source this process is running."""
    digest = hashlib.sha256()
    for name in CONTRACT_MODULES:
        # Normalised line endings, so Windows and Linux checkouts agree.
        digest.update(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def package_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def build_manifest(*, model_name, threshold, cooldown, window, synthetic, feature_columns,
                   metadata: dict, validation: dict, models_config_sha256: str | None) -> dict:
    """``metadata`` is ``run_metadata`` (seed, split, data and git fingerprints)."""
    lineage = {key: metadata.get(key) for key in (
        "seed", "data_mode", "alarm_budget", "git_commit", "split_seed", "splits_csv_sha256",
        "source_parquet_sha256", "dvc_lock_sha256",
    ) if key in metadata}
    return {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "model_name": model_name,
        "threshold": float(threshold), "cooldown_hours": int(cooldown), "window_hours": int(window),
        "synthetic": bool(synthetic),
        "feature_count": len(feature_columns),
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
        "contract_code_sha256": code_sha256(),
        "models_config_sha256": models_config_sha256,
        "python": platform.python_version(),
        "packages": package_versions(),
        "lineage": lineage,
        "validation": {key: validation.get(key) for key in (
            "auroc", "average_precision", "brier_score", "normalized_hourly_utility",
            "utility_standard_error", "utility_ci_low", "utility_ci_high",
            "false_alert_events_per_nonseptic_patient_day", "early_detection_recall", "patients",
        )},
    }


def write_manifest(run_dir: Path, manifest: dict, bundle_sha256: str) -> Path:
    path = Path(run_dir) / MANIFEST_FILE
    path.write_text(json.dumps({**manifest, "bundle_sha256": bundle_sha256}, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def recorded_sha256(bundle_path: Path) -> str | None:
    """The bundle hash recorded in ``manifest.json`` beside it, if there is one."""
    path = Path(bundle_path).with_name(MANIFEST_FILE)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("bundle_sha256")


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def check_runtime(manifest: dict, runtime: dict | None = None) -> None:
    """Refuse a bundle fitted under another scikit-learn major.minor, or an unknown format."""
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ManifestMismatch(f"model manifest format {manifest.get('format')!r} is not supported "
                               f"(this code reads format {MANIFEST_FORMAT})")
    runtime = runtime or package_versions()
    for name in STRICT_PACKAGES:
        built = (manifest.get("packages") or {}).get(name)
        if built and _minor(built) != _minor(runtime[name]):
            raise ManifestMismatch(
                f"the model was fitted with {name} {built}, but this runtime has {runtime[name]}; "
                f"install {name} {_minor(built)}.x or retrain"
            )
