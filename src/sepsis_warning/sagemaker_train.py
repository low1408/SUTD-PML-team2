"""SageMaker AI training job entry point: ``train`` inside the standard /opt/ml layout.

SageMaker starts the training image with the argument ``train``. This module:

1. reads hyperparameters from ``/opt/ml/input/config/hyperparameters.json``
   (all values arrive as strings): ``seed``, ``window``, ``budget``,
   ``cooldown``, ``mlflow_tracking_uri`` (the MLflow App ARN),
   ``register_model`` ("true"/"false"), ``git_commit``, ``require_gates`` (do
   not register a model that fails the promotion gates), ``mode`` (``train``,
   ``prepare`` to build the data in S3, ``tune`` for the grouped-CV search,
   ``promote`` to move ``@production`` after the gates, ``monitor_reference`` to
   build the Evidently reference of the served model, or ``evaluate_publish`` for
   the pipeline step after approval);
2. reads the ``data`` channel, ``/opt/ml/input/data/data/``: either
   ``combined.parquet`` with ``splits.csv`` (and optionally
   ``splits.csv.summary.json`` and ``dvc.lock``), or a folder with
   ``training_setA``/``training_setB`` (the synthetic fixture);
3. runs the same ``run_experiment`` as ``python -m sepsis_warning train``:
   the data checks run first, every candidate is fitted, and selection uses
   the validation split only (the test splits are never scored here);
4. writes the run to ``/opt/ml/model``, which SageMaker uploads to the
   encrypted artifact bucket as ``model.tar.gz``, and, when a tracking URI is
   given, logs the selection run and registers the bundle as ``@candidate``.

Pipeline modes (see ``infra/cloudformation/pipeline.yaml``): ``prepare`` generates the synthetic fixture or
downloads PhysioNet, builds ``combined.parquet`` and ``splits.csv`` and uploads them to the data prefix;
``tune`` runs the patient-grouped search on the train split and uploads ``tuned_models.toml`` to
``tuning/<run_id>/``; ``train`` then reads that file from the optional ``config`` channel and, after training,
writes the metrics table and figures (``model-report``) to ``reports/<run_id>/``.

On failure the reason goes to ``/opt/ml/output/failure``, which SageMaker
shows as the job's failure reason. ``SM_ROOT`` overrides ``/opt/ml`` for tests.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from argparse import Namespace
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("SM_ROOT", "/opt/ml"))


def _flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _params(root: Path) -> dict:
    config = root / "input" / "config" / "hyperparameters.json"
    return json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}


def build_args(root: Path) -> Namespace:
    params = _params(root)
    tuned = root / "input" / "data" / "config" / "tuned_models.toml"
    channel = root / "input" / "data" / "data"
    if (channel / "combined.parquet").is_file():
        data, splits = channel / "combined.parquet", channel / "splits.csv"
        if not splits.is_file():
            raise ValueError("the data channel has combined.parquet but no splits.csv; upload the team's saved split")
    elif (channel / "training_setA").is_dir():
        data, splits = channel, None
    else:
        raise ValueError(f"no combined.parquet or training_setA in the data channel {channel}")
    summary = channel / "splits.csv.summary.json"
    dvc_lock = channel / "dvc.lock"
    tracking = params.get("mlflow_tracking_uri") or None
    if params.get("git_commit"):
        os.environ["SEPSIS_GIT_COMMIT"] = params["git_commit"]
    return Namespace(
        data=data, splits=splits, output=root / "model",
        seed=int(params.get("seed", 42)), window=int(params.get("window", 6)),
        budget=float(params.get("budget", 0.2)), cooldown=int(params.get("cooldown", 6)),
        splits_summary=summary if summary.is_file() else None,
        dvc_lock=dvc_lock if dvc_lock.is_file() else None,
        mlflow_uri=tracking, register_model=_flag(params.get("register_model", "false")) and tracking is not None,
        models_config=tuned if tuned.is_file() else None, include_numpy_models=_flag(params.get("include_numpy_models", "false")),
        skip_data_checks=_flag(params.get("skip_data_checks", "false")),
        require_gates=_flag(params.get("require_gates", "false")),
        require_real_data=_flag(params.get("require_real_data", "false")),
    )


# Only these files of a frozen evaluation may leave the step: no patient rows, no replay page.
AGGREGATE_EVALUATION_FILES = ("evaluation_metrics.json",)


def promote_model(root: Path, params: dict) -> dict:
    """Approval step: gates, a named reviewer, then ``@production`` moves to the version.

    Runs as its own task with a different role from the training task, so on AWS the principal that
    registered the version cannot also promote it (see ``registry.promote``). Hyperparameters:
    ``mlflow_tracking_uri``, ``model_version`` and ``reviewed_by`` (required), optionally ``alias``
    (default production) and ``require_real_data``.
    """
    from .registry import MODEL_NAME, promote
    from .tracking import check_tracking_uri, require_mlflow

    tracking, version, reviewer = params.get("mlflow_tracking_uri"), params.get("model_version"), params.get("reviewed_by")
    if not tracking or not version or not reviewer:
        raise ValueError("mode promote needs mlflow_tracking_uri, model_version and reviewed_by")
    check_tracking_uri(tracking)
    require_mlflow()
    change = promote(tracking, str(version), reviewer, name=params.get("model_name") or MODEL_NAME,
                     alias=params.get("alias") or "production",
                     require_real_data=_flag(params.get("require_real_data", "false")))
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "model" / "promotion.json").write_text(json.dumps(change, indent=2), encoding="utf-8")
    print(f"Promoted {change['name']} version {change['version']} to @{change['alias']} "
          f"(reviewer {change['reviewed_by']}, approver {change['approved_by']})", flush=True)
    return change


def monitor_reference(root: Path, params: dict, args: Namespace, client=None) -> list[str]:
    """Build the Evidently reference for the model that is (or will be) served, from hospital A validation only.

    The frozen model scores the validation stays in batch; the result, reduced to the columns the live prediction
    log can also provide (``evidently_frames``), goes to ``monitoring/evidently/<bundle>/``. The test splits are never
    read. Hyperparameters: ``mlflow_tracking_uri``, ``publish_bucket`` and optionally ``model_uri``
    (default ``models:/sepsis-warning@production``).
    """
    import datetime

    from .data import DYNAMIC
    from .evidently_frames import reference_frame
    from .registry import resolve_model
    from .scoring import load_bundle, score_patients
    from .splits import load_split_patients

    tracking, bucket = params.get("mlflow_tracking_uri"), params.get("publish_bucket")
    if not tracking or not bucket:
        raise ValueError("mode monitor_reference needs mlflow_tracking_uri and publish_bucket")
    if args.splits is None:
        raise ValueError("mode monitor_reference needs combined.parquet with splits.csv in the data channel")
    model_uri = params.get("model_uri") or "models:/sepsis-warning@production"
    with tempfile.TemporaryDirectory(prefix="sepsis-reference-") as scratch:
        path, sha, _ = resolve_model(model_uri, tracking, Path(scratch))
        bundle = load_bundle(path, sha)
        cohorts = load_split_patients(args.data, args.splits)
        reference = reference_frame(score_patients(bundle, cohorts["validation"], DYNAMIC))
    output = root / "model" / "reference"
    output.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output / "reference.parquet", index=False)
    (output / "reference.json").write_text(json.dumps({
        "bundle_sha256": sha, "model_name": bundle["model_name"], "threshold": float(bundle["threshold"]),
        "rows": int(len(reference)), "alert_rate": float(reference["new_alert"].mean()), "split": "validation",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    uris = _upload(output, bucket, f"monitoring/evidently/{sha[:12]}", client=client)
    print(f"Evidently reference for bundle {sha[:12]} ({len(reference)} hourly rows): " + ", ".join(uris), flush=True)
    return uris


def evaluate_and_publish(root: Path, params: dict, args: Namespace) -> dict:
    """Pipeline step after approval: the one frozen evaluation, then a write-once publish to S3.

    Hyperparameters: ``mlflow_tracking_uri``, ``publish_bucket`` and optionally
    ``model_uri`` (default ``models:/sepsis-warning@production``).
    """
    from .evaluation import AlreadyEvaluated, run_frozen_evaluation
    from .model_source import check_promoted, publish_model

    tracking, bucket = params.get("mlflow_tracking_uri"), params.get("publish_bucket")
    if not tracking or not bucket:
        raise ValueError("mode evaluate_publish needs mlflow_tracking_uri and publish_bucket")
    model_uri = params.get("model_uri") or "models:/sepsis-warning@production"
    # Before the one-time test evaluation: the version must have been promoted by a named reviewer.
    check_promoted(model_uri, tracking)
    # The evaluation writes patient-level test predictions and a replay page. SageMaker archives everything
    # under /opt/ml/model into model.tar.gz, so those files go to a scratch folder that is deleted at the end
    # of the step; only the aggregate metrics file is kept with the job's output.
    (root / "model" / "evaluation").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sepsis-evaluation-") as scratch:
        scratch_output = Path(scratch) / "evaluation"
        try:
            run_frozen_evaluation(data=args.data, splits=args.splits, seed=args.seed, model_uri=model_uri,
                                  mlflow_uri=tracking, output=scratch_output)
        except AlreadyEvaluated as error:
            print(f"Frozen evaluation already recorded ({error}); publishing without a second look.", flush=True)
        for name in AGGREGATE_EVALUATION_FILES:
            if (scratch_output / name).is_file():
                shutil.copy2(scratch_output / name, root / "model" / "evaluation" / name)
    published = publish_model(model_uri, tracking, bucket)
    (root / "model" / "publish.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    print(f"Published {published['model_s3_uri']} sha256 {published['sha256']}", flush=True)
    return published


S3_CLIENT = None  # tests set a stand-in; in a job boto3 uses the SageMaker role


def _upload(directory: Path, bucket: str, prefix: str, names: set | None = None, client=None) -> list[str]:
    """Copy files to s3://bucket/prefix/ (the bucket's default KMS encryption applies). Returns the S3 URIs."""
    client = client or S3_CLIENT
    if client is None:
        import boto3

        client = boto3.client("s3")
    uris = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and (names is None or path.name in names):
            key = f"{prefix.strip('/')}/{path.relative_to(directory).as_posix()}"
            client.upload_file(str(path), bucket, key)
            uris.append(f"s3://{bucket}/{key}")
    return uris


def fetch_raw_from_s3(bucket: str, prefix: str, destination: Path, limit: int, seed: int, client=None, workers: int = 16) -> int:
    """Read the PhysioNet records from our own bucket (copied there by `aws_deploy.sh fetch-data`).

    A same-region S3 read, not a web download: ``prefix`` holds training_setA/ and training_setB/. ``limit`` caps
    stays per hospital (0 = all), sampled with ``seed``. Returns the number of files fetched.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor

    client = client or S3_CLIENT
    if client is None:
        import boto3

        client = boto3.client("s3")
    rng, jobs = random.Random(seed), []
    for hospital in ("A", "B"):
        folder = f"{prefix.strip('/')}/training_set{hospital}/"
        keys = sorted(item["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=folder)
                      for item in page.get("Contents", []) if item["Key"].endswith(".psv"))
        if not keys:
            raise ValueError(f"no records under s3://{bucket}/{folder}; run the pipeline workflow (it copies the "
                             "PhysioNet data first) or `bash scripts/aws_deploy.sh fetch-data`")
        if limit:
            keys = sorted(rng.sample(keys, min(limit, len(keys))))
        target = destination / f"training_set{hospital}"
        target.mkdir(parents=True, exist_ok=True)
        jobs += [(key, target / key.rsplit("/", 1)[-1]) for key in keys]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: client.download_file(bucket, job[0], str(job[1])), jobs))
    return len(jobs)


MARKER = "_SUCCESS"


def _s3(client=None):
    client = client or S3_CLIENT
    if client is None:
        import boto3

        client = boto3.client("s3")
    return client


def read_marker(client, bucket: str, prefix: str) -> dict | None:
    """The ``_SUCCESS`` record a finished prepare step leaves next to the data, or None."""
    try:
        body = client.get_object(Bucket=bucket, Key=f"{prefix.strip('/')}/{MARKER}")["Body"].read()
    except Exception as error:  # noqa: BLE001 - only "not there" means None
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound") or type(error).__name__ == "NoSuchKey":
            return None
        raise
    return json.loads(body)


def prepare_data(root: Path, params: dict, client=None) -> list[str]:
    """Pipeline step 1: build the data in S3 so nothing has to be prepared on a laptop.

    ``data_source`` is ``synthetic`` (the software fixture; ``synthetic_stays`` sets its size), ``physionet`` (the
    public Challenge 2019 training sets, read from ``raw_prefix`` in our bucket where the pipeline workflow copied
    them from the open-data bucket) or ``physionet-web`` (downloads them over HTTPS instead). ``limit`` caps stays
    per hospital, 0 = all. The records
    are combined into ``combined.parquet`` with the saved patient split ``splits.csv`` and uploaded to
    ``data_prefix`` of ``publish_bucket``.
    """
    from .prepare import prepare_parquet
    from .splits import write_splits

    bucket = params.get("publish_bucket")
    if not bucket:
        raise ValueError("mode prepare needs publish_bucket")
    source, seed = params.get("data_source", "synthetic"), int(params.get("seed", 42))
    prefix, limit = params.get("data_prefix", "data/physionet"), int(params.get("limit", 0))
    # The real data never changes, so it is prepared once. A rerun with the same settings does nothing, and a
    # rehearsal on the synthetic fixture can never overwrite it (force=true is the deliberate way around that).
    existing = read_marker(_s3(client), bucket, prefix)
    if existing and not _flag(params.get("force", "false")):
        def real(name):
            return str(name).startswith("physionet")

        if real(existing.get("data_source")) and not real(source):
            raise ValueError(f"s3://{bucket}/{prefix}/ holds real PhysioNet data (prepared {existing.get('prepared_at')}); "
                             "refusing to overwrite it with the synthetic fixture. Use a different data prefix, or force=true")
        if (existing.get("data_source"), existing.get("limit"), existing.get("seed")) == (source, limit, seed):
            print(f"s3://{bucket}/{prefix}/ already holds {source} data prepared {existing.get('prepared_at')} "
                  f"({existing.get('files', {}).get('combined.parquet', {}).get('sha256', '')[:12]}); nothing to do", flush=True)
            return []
    raw = root / "work" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if source == "synthetic":
        from .data import generate_synthetic

        stays = int(params.get("synthetic_stays", 600))
        generate_synthetic(raw, seed=seed, source_size=stays, external_size=max(1, stays // 2))
    elif source == "physionet":
        count = fetch_raw_from_s3(bucket, params.get("raw_prefix", "raw/physionet/training"), raw,
                                  int(params.get("limit", 0)), seed, client=client)
        print(f"Read {count} records from s3://{bucket}/{params.get('raw_prefix', 'raw/physionet/training')}", flush=True)
    elif source == "physionet-web":
        from .download import download

        download(raw, limit=int(params.get("limit", 0)), seed=seed, workers=8)
    else:
        raise ValueError(f"data_source must be synthetic, physionet or physionet-web, not {source!r}")
    built = root / "work" / "built"
    built.mkdir(parents=True, exist_ok=True)
    prepare_parquet(raw, built / "combined.parquet")
    write_splits(built / "combined.parquet", built / "splits.csv", seed)
    manifest = raw / "download_manifest.json"
    if manifest.is_file():
        (built / "download_manifest.json").write_bytes(manifest.read_bytes())
    import datetime
    import hashlib

    files = {name: {"sha256": hashlib.sha256((built / name).read_bytes()).hexdigest(), "bytes": (built / name).stat().st_size}
             for name in ("combined.parquet", "splits.csv")}
    (built / MARKER).write_text(json.dumps({
        "data_source": source, "limit": limit, "seed": seed, "files": files,
        "raw_records": sum(1 for _ in (raw).rglob("*.psv")),
        "prepared_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    uris = _upload(built, bucket, prefix, client=client)
    print(f"Prepared {source} data: " + ", ".join(uris), flush=True)
    return uris


def tune_data(root: Path, params: dict, args: Namespace, client=None) -> Path:
    """Pipeline step 2: patient-grouped hyperparameter search on the train split only (``trials`` 0 = keep defaults)."""
    from . import tuning
    from .cli import run_tune
    from .modeling import load_model_config

    trials = int(params.get("trials", 6))
    output = root / "model" / "tuning"
    if trials <= 0:
        config, _ = load_model_config(None)
        output.mkdir(parents=True, exist_ok=True)
        tuning.write_models_toml(output / "tuned_models.toml", config, header="Reference settings; no search was run (trials=0).")
        print("trials=0: the reference settings are passed on unchanged", flush=True)
    else:
        run_tune(Namespace(
            data=args.data, splits=args.splits, output=output, seed=args.seed, window=args.window,
            folds=int(params.get("folds", 3)), trials=trials,
            max_stays=int(params["max_stays"]) if params.get("max_stays") else None, models=None, models_config=None,
            space_config=None, mlflow_uri=args.mlflow_uri))
    bucket, run_id = params.get("publish_bucket"), params.get("run_id")
    if bucket and run_id:
        print("Uploaded: " + ", ".join(_upload(output, bucket, f"tuning/{run_id}", client=client)), flush=True)
    return output


def write_run_report(root: Path, params: dict, args: Namespace, client=None) -> list[str]:
    """After training: the metrics table and figures for the reviewer, in S3 under ``reports/<run_id>/``.

    Reads only the files `train` wrote (validation split); a failure here is logged and never fails the job.
    """
    from .model_report import build_report

    try:
        output = args.output / "report"
        tuning_dir = root / "input" / "data" / "config"
        build_report(args.output, output, tuning_dir=tuning_dir if (tuning_dir / "tuning_results.csv").is_file() else None,
                     bootstrap=int(params.get("bootstrap", 200)), seed=args.seed,
                     command="python -m sepsis_warning model-report --run artifacts/real --output reports/results "
                             "--tuning artifacts/tuning --bootstrap 200")
        bucket, run_id = params.get("publish_bucket"), params.get("run_id")
        uris = _upload(output, bucket, f"reports/{run_id}", client=client) if bucket and run_id else []
        print(f"Report written to {output}" + (f" and uploaded to s3://{bucket}/reports/{run_id}/" if uris else ""), flush=True)
        return uris
    except Exception:  # noqa: BLE001 - the model is already trained; the reviewer can run model-report by hand
        print("Could not write the report (training is unaffected):", flush=True)
        traceback.print_exc()
        return []


def main(argv=None) -> int:
    root = _root()
    try:
        from .cli import run_experiment
        from .tracking import check_tracking_uri, require_mlflow

        params = _params(root)
        mode = params.get("mode", "train")
        if mode == "prepare":  # needs no input data: it makes it
            prepare_data(root, params)
            return 0
        if mode == "promote":  # works on the registry only
            promote_model(root, params)
            return 0
        args = build_args(root)
        if args.mlflow_uri:
            check_tracking_uri(args.mlflow_uri)
            require_mlflow()
        if mode == "monitor_reference":
            monitor_reference(root, params, args)
            return 0
        if mode == "evaluate_publish":
            evaluate_and_publish(root, params, args)
            return 0
        if mode == "tune":
            tune_data(root, params, args)
            return 0
        print(f"Training on {args.data} (seed {args.seed}); outputs to {args.output}", flush=True)
        run_experiment(args)
        if _flag(params.get("report", "true")):
            write_run_report(root, params, args)
        return 0
    except Exception as error:  # noqa: BLE001 - SageMaker needs the reason in a file
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output" / "failure").write_text(f"{type(error).__name__}: {error}"[:1000], encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
