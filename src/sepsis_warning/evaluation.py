"""Frozen evaluation: score the test splits once per model, never during training.

``train`` selects on validation only and does not touch ``internal_test`` or
hospital B (``external_test``). After a model has been reviewed and frozen,
``evaluate`` scores those splits one time and records that it happened:

* a registered version gets the tag ``frozen_evaluated_at`` and cannot be
  evaluated again;
* a local bundle gets ``frozen_evaluation.json`` beside it, with the same effect.

Repeated looks at a test set make its numbers optimistic, so there is no
override for real data. Synthetic fixtures may be re-evaluated: their numbers
carry no evidence either way.

The split must be the one the model was trained with: the saved split's hash
(or, for folder data, the seed) is compared with the model's manifest, because a
different split could put training patients into the test set.
"""

import datetime
import json
from argparse import Namespace
from pathlib import Path

from .policy import summarize
from .report import write_report
from .scoring import file_sha256, load_bundle, score_patients

EVALUATION_EXPERIMENT = "sepsis-warning-evaluation"
EVALUATED_TAG = "frozen_evaluated_at"
MARKER_FILE = "frozen_evaluation.json"
TEST_SPLITS = ("internal_test", "external_test")


class AlreadyEvaluated(ValueError):
    """This model's test results already exist; a second look is refused."""


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def check_same_split(manifest: dict | None, data: Path, splits: Path | None, seed: int) -> None:
    """Refuse a split that differs from the one recorded when the model was trained."""
    if not manifest:
        raise ValueError("the bundle has no manifest, so its training split cannot be confirmed; retrain it")
    lineage = manifest.get("lineage") or {}
    if Path(data).is_file():
        splits_path = Path(splits or Path(data).with_name("splits.csv"))
        recorded = lineage.get("splits_csv_sha256")
        if recorded is None:
            raise ValueError("the model's manifest has no split hash; it was not trained on a saved split "
                             "with its summary (pass --splits-summary to train)")
        actual = file_sha256(splits_path)
        if actual != recorded:
            raise ValueError(f"{splits_path} (SHA-256 {actual[:12]}…) is not the split the model was trained "
                             f"with ({recorded[:12]}…); test stays could include training patients")
    elif lineage.get("seed") is not None and int(lineage["seed"]) != int(seed):
        raise ValueError(f"folder data is re-split with --seed; the model was trained with seed "
                         f"{lineage['seed']}, not {seed}")


def refuse_repeat(already: str | None, synthetic: bool, identity: str) -> None:
    """Real-data models are evaluated once. Synthetic fixtures may be repeated."""
    if already and not synthetic:
        raise AlreadyEvaluated(
            f"{identity} was already evaluated on the test splits at {already}. Test results are "
            "produced once per frozen model; train and register a new version to evaluate again."
        )


def run_frozen_evaluation(*, data: Path, output: Path, splits: Path | None = None, seed: int = 42,
                          model: Path | None = None, model_uri: str | None = None,
                          mlflow_uri: str | None = None, model_sha256: str | None = None) -> dict:
    from .cli import load_cohorts  # shared loader; imported here to avoid a cycle

    if (model is None) == (model_uri is None):
        raise ValueError("give exactly one of --model or --model-uri")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")

    client = version = name = None
    if model_uri:
        from .registry import _client, parse_model_uri, resolve_model

        path, expected, tags = resolve_model(model_uri, mlflow_uri)
        name, version = parse_model_uri(model_uri)[0], tags["version"]
        client = _client(mlflow_uri)
        bundle = load_bundle(path, expected)
        already = tags.get(EVALUATED_TAG)
        identity = f"{name} version {version}"
    else:
        path = Path(model)
        bundle = load_bundle(path, model_sha256)
        marker = path.with_name(MARKER_FILE)
        already = json.loads(marker.read_text(encoding="utf-8")).get("evaluated_at") if marker.is_file() else None
        identity = str(path)
    synthetic = bool(bundle["synthetic"])
    refuse_repeat(already, synthetic, identity)

    cohorts, data_synthetic = load_cohorts(Namespace(data=Path(data), splits=splits, seed=seed))
    if data_synthetic != synthetic:
        raise ValueError("The model bundle and the data disagree on whether they are synthetic")
    check_same_split(bundle.get("manifest"), data, splits, seed)

    output.mkdir(parents=True, exist_ok=True)
    scored, results = [], {}
    for split in TEST_SPLITS:
        table = score_patients(bundle, cohorts[split])
        results[split] = summarize(table)
        table["split"] = split
        scored.append(table)
    evaluated_at = _now()
    sha = file_sha256(path)
    summary = {
        "evaluated_at": evaluated_at, "model": identity, "bundle_sha256": sha,
        "selected_model": bundle["model_name"], "threshold": float(bundle["threshold"]),
        "cooldown_hours": int(bundle["cooldown"]), "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "splits": results,
    }
    (output / "evaluation_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
    import pandas as pd

    replay = pd.concat(scored, ignore_index=True)
    replay.to_csv(output / "held_out_predictions.csv", index=False)
    write_report(output / "replay.html", replay, summary, bundle["threshold"], synthetic)

    # Record the look before anything else can fail, so it cannot be repeated quietly.
    if client is not None:
        client.set_model_version_tag(name, version, EVALUATED_TAG, evaluated_at)
        _log_to_mlflow(mlflow_uri, name, version, sha, summary)
    else:
        path.with_name(MARKER_FILE).write_text(json.dumps(
            {"evaluated_at": evaluated_at, "bundle_sha256": sha, "output": str(output.resolve())}, indent=2) + "\n",
            encoding="utf-8")
    return summary


def _log_to_mlflow(tracking_uri: str, name: str, version: str, sha: str, summary: dict) -> None:
    """Aggregate test metrics only, in their own experiment, away from selection runs."""
    import mlflow

    from .tracking import _numeric, prepare_store

    prepare_store(tracking_uri, EVALUATION_EXPERIMENT, mlflow)
    with mlflow.start_run(run_name=f"evaluate-{name}-v{version}"):
        mlflow.set_tags({"run_kind": "frozen_evaluation", "model_name": name, "model_version": str(version),
                         "bundle_sha256": sha, "data_mode": summary["data_mode"], "research_only": "true"})
        mlflow.log_param("threshold", summary["threshold"])
        for split, values in summary["splits"].items():
            mlflow.log_metrics(_numeric(values, f"{split}."))
