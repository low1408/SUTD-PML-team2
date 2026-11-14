"""Model registry for frozen bundles (L01): register, promote, resolve.

Needs the optional ``tracking`` extra and a local SQLite store
(``sqlite:///artifacts/mlflow.db``); the registry does not work with the old
``file:`` store.

* ``register_bundle`` logs a run's ``model.joblib`` as an MLflow model,
  registers a new version of ``sepsis-warning`` and points the ``candidate``
  alias at it. The bundle's SHA-256 and the data, split and feature
  fingerprints are stored as version tags.
* ``promote`` moves an alias (``production`` by default) to a version. It
  needs the name of the person who reviewed the version, and records the
  version it replaced so a rollback is one more ``promote``.
* ``resolve_model`` turns ``models:/sepsis-warning@production`` (or
  ``models:/sepsis-warning/3``) into a local bundle path plus its expected
  SHA-256, which ``serve`` checks before loading.

Joblib files can run code when loaded, so a bundle is only loaded after its
hash matches the one recorded when it was registered. Everything stays in the
local store; nothing here talks to a remote server.
"""

import datetime
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .data import DYNAMIC, STATIC
from .scoring import file_sha256, load_bundle
from .tracking import check_tracking_uri, package_versions, prepare_store

MODEL_NAME = "sepsis-warning"
MODEL_EXPERIMENT = "sepsis-warning-selection"
BUNDLE_FILE = "model.joblib"
INPUT_COLUMNS = DYNAMIC + STATIC


class BundleModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around a frozen bundle, scoring the same way as the API.

    Input: hourly rows with a ``stay_id`` column plus the model inputs
    (``ICULOS`` required, other measurements may be missing). Output: one row
    per input hour with ``probability``, ``hourly_positive`` and ``alert_event``.
    """

    def load_context(self, context):
        from .service import ScoringService  # avoids a circular import at module load

        expected = context.model_config["bundle_sha256"]
        self.service = ScoringService.from_path(Path(context.artifacts["bundle"]), expected)

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        if "stay_id" not in frame:
            raise ValueError("model_input needs a stay_id column")
        results = []
        for stay_id, rows in frame.groupby("stay_id", sort=False):
            records = rows.drop(columns=["stay_id"]).astype(object).where(rows.drop(columns=["stay_id"]).notna(), None)
            scored = self.service.score(str(stay_id), records.to_dict(orient="records"))
            hours = pd.DataFrame(scored["hours"])
            hours.insert(0, "stay_id", stay_id)
            results.append(hours)
        return pd.concat(results, ignore_index=True)


def _client(tracking_uri: str) -> MlflowClient:
    check_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def register_bundle(run_dir: Path, tracking_uri: str, run_id: str | None = None,
                    name: str = MODEL_NAME, alias: str = "candidate") -> str:
    """Log ``run_dir/model.joblib`` as an MLflow model and register it. Returns the version.

    ``run_id`` attaches the model to an existing selection run (from
    ``tracking.log_run``); otherwise a new run is created.
    """
    run_dir = Path(run_dir)
    bundle_path = run_dir / BUNDLE_FILE
    sha = file_sha256(bundle_path)
    bundle = load_bundle(bundle_path, sha)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    prepare_store(tracking_uri, MODEL_EXPERIMENT, mlflow)
    requirements = [f"{package}=={version}" for package, version in package_versions().items() if version]

    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model", python_model=BundleModel(), artifacts={"bundle": str(bundle_path)},
            model_config={"bundle_sha256": sha}, pip_requirements=requirements,
        )
    tags = {
        "bundle_sha256": sha,
        "model_name": bundle["model_name"],
        "threshold": str(bundle["threshold"]),
        "cooldown_hours": str(bundle["cooldown"]),
        "window_hours": str(bundle["window"]),
        "synthetic": str(bool(bundle["synthetic"])).lower(),
        "data_mode": metadata.get("data_mode", ""),
        "git_commit": metadata.get("git_commit") or "",
        "feature_columns_sha256": metadata.get("feature_columns_sha256", ""),
        "splits_csv_sha256": metadata.get("splits_csv_sha256") or "",
        "research_only": "true",
    }
    version = mlflow.register_model(info.model_uri, name, tags=tags)
    client = _client(tracking_uri)
    client.update_model_version(name, version.version, description=(
        f"{bundle['model_name']} bundle, threshold {bundle['threshold']}, cooldown {bundle['cooldown']} h. "
        "Research model on a benchmark label; not for patient care."
    ))
    client.set_registered_model_alias(name, alias, version.version)
    return str(version.version)


def promote(tracking_uri: str, version: str, reviewed_by: str, name: str = MODEL_NAME,
            alias: str = "production") -> dict:
    """Point ``alias`` at ``version`` after a named review. Returns what changed."""
    if not reviewed_by or not reviewed_by.strip():
        raise ValueError("promotion needs the name of the reviewer (--reviewed-by)")
    client = _client(tracking_uri)
    target = client.get_model_version(name, str(version))
    target_version = str(target.version)
    try:
        previous = str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        previous = None
    if previous == target_version:
        raise ValueError(f"{name}@{alias} already points at version {target_version}")
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    client.set_model_version_tag(name, target_version, f"{alias}_reviewed_by", reviewed_by.strip())
    client.set_model_version_tag(name, target_version, f"{alias}_promoted_at", now)
    client.set_model_version_tag(name, target_version, f"previous_{alias}", previous or "none")
    client.set_registered_model_alias(name, alias, target_version)
    return {"name": name, "alias": alias, "version": target_version, "previous": previous,
            "reviewed_by": reviewed_by.strip(), "promoted_at": now}


def parse_model_uri(model_uri: str) -> tuple[str, str | None, str | None]:
    """``models:/name@alias`` or ``models:/name/version`` -> (name, alias, version)."""
    if not model_uri.startswith("models:/"):
        raise ValueError("model URI must look like models:/sepsis-warning@production or models:/sepsis-warning/3")
    path = model_uri[len("models:/"):]
    if "@" in path:
        name, alias = path.split("@", 1)
        return name, alias, None
    name, _, version = path.partition("/")
    if not version:
        raise ValueError("model URI needs an @alias or a /version")
    return name, None, version


def resolve_model(model_uri: str, tracking_uri: str, download_dir: Path | None = None) -> tuple[Path, str, dict]:
    """Download a registered bundle; return (local path, expected SHA-256, version tags).

    The caller loads the bundle with the returned hash, so a file that changed
    after registration is refused.
    """
    name, alias, version = parse_model_uri(model_uri)
    client = _client(tracking_uri)
    found = client.get_model_version_by_alias(name, alias) if alias else client.get_model_version(name, version)
    expected = found.tags.get("bundle_sha256")
    if not expected:
        raise ValueError(f"{name} version {found.version} has no bundle_sha256 tag; refusing to load it")
    target = Path(download_dir or tempfile.mkdtemp(prefix="sepsis-model-"))
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{found.version}",
                                                     dst_path=str(target)))
    matches = list(local.rglob(BUNDLE_FILE))
    if len(matches) != 1:
        raise ValueError(f"expected one {BUNDLE_FILE} in {name} version {found.version}, found {len(matches)}")
    return matches[0], expected, {"version": str(found.version), **found.tags}
