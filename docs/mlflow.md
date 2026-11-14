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
still writes held-out results to its local `metrics.json`; they are removed
from the logged `selection_metrics.json` so model choice in MLflow only ever
sees validation numbers.

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
| `@production` | Moved only by `promote`, which needs `--reviewed-by`. The version records `production_reviewed_by`, `production_promoted_at` and `previous_production`. |
| Rollback | `promote --version <previous_production> --reviewed-by <name>`; `promote` prints the exact command. |

A registered version is the freeze record: it fixes the model, features,
calibration and threshold, with hashes that show which data, split and code
produced it. Internal-test and hospital-B evaluation should use that version
once, in a separate `evaluation` experiment (M01), not in selection runs.

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
- The model files are written to the local store only. Moving the store to a
  shared or cloud server (for example SageMaker managed MLflow) needs team
  approval and a decision on access control.
- The Docker image does not include MLflow. To serve from the registry inside a
  container, resolve the model first and mount the bundle, or add the
  `tracking` extra to the image.
