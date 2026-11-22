# MLflow tracking and model registry (L01)

Everything stays in one local SQLite file, `artifacts/mlflow.db`, with run and
model files next to it in `artifacts/mlartifacts/`. Both are Git-ignored. Only
local stores are accepted (`sqlite:`, `file:` or a localhost server); a shared
or cloud server needs team approval first.

```bash
pip install -e ".[tracking]"
```

Use `sqlite:///artifacts/mlflow.db`. MLflow 3 refuses the old `file:` store
(it is in maintenance mode), and the model registry needs a database.

## Selection runs

```bash
python -m sepsis_warning train --data data/physionet/combined.parquet --output artifacts/run-001 \
  --splits-summary data/physionet/splits.csv.summary.json \
  --mlflow-uri sqlite:///artifacts/mlflow.db
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db     # http://127.0.0.1:5000
```

Experiment `sepsis-warning-selection`, one parent run per `train` or `demo`:

| Run | What is logged |
|---|---|
| Parent | Tags: data mode, selected model, `research_only`. Params: seed, split and DVC hashes, feature-contract hash, git commit, package versions. Metrics: the selected model's **validation** metrics and threshold. Artifacts: `config.json`, `selection_metrics.json`, `validation_models.csv`, `run_metadata.json`, `candidate_details.json`. |
| One nested run per candidate | Tags: candidate name, implementation (`numpy`, `sklearn` or `reference`), whether it was selected. Params: the model's settings and its validation threshold. Metrics: validation AUROC, AP, Brier, utility, false alerts per non-septic patient-day, early detection, lead time, fit time. Step metric `training_loss` when the model records one (the NumPy models' Newton and boosting losses). |

Never logged: held-out test results (internal test, hospital B), patient-level
predictions, the split manifest, replay or ward pages, raw data. `train`
does not score the test splits at all, so selection runs can only hold
validation numbers. The run also logs `manifest.json` (versions, hashes and
validation metrics of the selected bundle).

## Registry

```bash
# Register the selected bundle as a new version; it becomes @candidate
python -m sepsis_warning train ... --mlflow-uri sqlite:///artifacts/mlflow.db --register-model

# After the team has reviewed that version (model card, tests, validation results)
python -m sepsis_warning promote --version 3 --reviewed-by "Kaiyuan"

# Serve whatever @production points at; the bundle's SHA-256 is checked before loading
pip install -e ".[serve]"
python -m sepsis_warning serve --model-uri models:/sepsis-warning@production
```

| Piece | Meaning |
|---|---|
| Registered model `sepsis-warning` | One version per registered bundle: calibrated model, feature list, window, threshold and cooldown. |
| Version tags | `bundle_sha256`, model name, threshold, cooldown, window, synthetic flag, data mode, git commit, feature-contract and split hashes. |
| `@candidate` | Set automatically to the newest registered bundle. Nothing serves it by default. |
| `@production` | Moved only by `promote`, which runs the promotion gates (`gates.py`: beats the no-model reference and SIRS at the same alert budget, within the budget, calibrated; `--require-real-data` also refuses synthetic models), needs `--reviewed-by`, and records the approver's identity (`production_approved_by`; on AWS it must differ from `registered_by`). The version records `production_reviewed_by`, `production_promoted_at` and `previous_production`. |
| Rollback | `promote --version <previous_production> --reviewed-by <name>`; `promote` prints the exact command. |

A registered version is the freeze record: it fixes the model, features,
calibration and threshold, with hashes that show which data, split and code
produced it. Internal-test and hospital-B results come from that version
once, in the separate `sepsis-warning-evaluation` experiment:

```bash
python -m sepsis_warning evaluate --data data/physionet/combined.parquet \
  --model-uri models:/sepsis-warning/3 --output artifacts/evaluation-v3
```

The version is tagged `frozen_evaluated_at`; a second `evaluate` on it is
refused. The split's hash must match the one in the model's manifest.

The model can also be loaded with MLflow's standard loader, which checks the
same hash and scores through the API's validation and feature code:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/sepsis-warning@production")
scores = model.predict(hourly_rows)   # columns: stay_id, ICULOS and the model inputs
```

## Safety notes

- A bundle is a Joblib file, which can run code when loaded. It is loaded only
  after its SHA-256 matches the `bundle_sha256` recorded at registration.
- Accepted stores are local SQLite or files, a localhost server, and the
  team's own SageMaker AI MLflow App (below). Any other hosted server needs
  team approval first.
- The scoring images do not include MLflow. In AWS, `publish` copies the
  registered bundle to S3 and the service loads it with its pinned hash.

## On AWS: the serverless SageMaker MLflow App

`infra/cloudformation/mlops.yaml` creates a SageMaker AI **MLflow App**:
serverless MLflow with no hourly server and no additional charge beyond S3
storage. Its ARN is the tracking URI, so the commands above work unchanged:

```bash
pip install -e ".[tracking,sagemaker]"     # sagemaker-mlflow signs requests with your AWS credentials
ARN=$(aws cloudformation describe-stacks --stack-name sepsis-research \
      --query "Stacks[0].Outputs[?OutputKey=='MlflowAppArn'].OutputValue" --output text)
python -m sepsis_warning promote --version 3 --reviewed-by "Kaiyuan" --mlflow-uri "$ARN"
python -m sepsis_warning publish --mlflow-uri "$ARN" --bucket <ArtifactBucketName>
aws sagemaker create-presigned-mlflow-app-url --arn "$ARN"   # link to the MLflow UI
```

SageMaker training jobs (`scripts/sagemaker_train.py`) log the selection run
and register `@candidate` there directly. The same rules apply: validation
metrics only, no patient-level files, and `@production` moves only through a
named review. See [aws-serverless.md](aws-serverless.md).
