# Early sepsis warning across hospitals

Rough project sketch and runnable classical-ML scaffold for the selected proposal.
The question is: **can an hourly early-warning model retain useful performance
at another hospital under a limited notification budget?**

The first version runs with an explicitly synthetic software fixture. It produces
a model comparison, saved model, held-out predictions, and an offline patient
replay. Synthetic scores demonstrate software behavior only. Real-data model
quality has not been established.

## System sketch

```mermaid
flowchart LR
    A[Hourly patient records] --> B[Past and current measurements only]
    B --> C[Classical classifier]
    C --> D[Probability calibration]
    D --> E[Hourly classification]
    E --> F[Threshold and notification cooldown]
    F --> G[Replay and alert metrics]
```

Input: one PhysioNet `.psv` file per patient, containing hourly vitals, laboratory
measurements and demographics. Output: a probability at every hour, an hourly
binary decision, and a separate notification event when the cooldown permits it.
An alert is a research output, not a diagnosis or treatment recommendation.

**Target semantics matter:** `SepsisLabel` becomes positive six hours before the
dataset-defined onset and stays positive afterward. We preserve this target
without shifting it again. The score estimates the probability of the challenge
label, not a validated probability of a first sepsis onset in the next six hours.
The inferred onset used for lead time is first-positive-label hour plus six;
records positive from their first row are excluded from our lead-time analysis.

## Run the synthetic demo

Use Python 3.12 or newer. The pinned `requirements.txt` is tested on 3.12 and 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m sepsis_warning demo
python -m unittest discover -s tests -v
```

Open `artifacts/demo/replay.html` in a browser. Select an internal-test or
external-test patient, play the hourly sequence, and inspect probabilities,
threshold crossings, notifications, and retrospective positive labels.

The default fixture has 160 hospital-A patients and 80 hospital-B patients. It
uses an artificial 20% patient prevalence and simplified trends, with a modest
measurement shift at B. It is intentionally easy and must not be used to claim
clinical accuracy, realistic domain adaptation, or real hospital performance.

Demo generation and experiment outputs require empty destinations. To repeat:

```bash
python -m sepsis_warning demo --data data/demo2 --output artifacts/demo2 --seed 43
```

The Codex workspace currently has an already-installed `.venv`; the commands
above are the portable setup for teammates.

## Test the whole pipeline locally (macOS or Linux)

Train, select, register, review, serve and call the model on synthetic data,
then run the production-like stack. Needs Python 3.12+ (`brew install
python@3.12` on a Mac) and, for steps 8-9, Docker Desktop.

```bash
# 1. Set up
git clone https://github.com/low1408/SUTD-PML-team2.git && cd SUTD-PML-team2
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[serve,test,tracking]"

# 2. Tests (about 2-3 minutes)
MPLBACKEND=Agg python -m unittest discover -s tests

# 3. Train every candidate, select one on validation, register it as @candidate
python -m sepsis_warning demo --output artifacts/demo \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model
open artifacts/demo/replay.html                   # xdg-open on Linux

# 4. Look at the selection: one parent run, one nested run per candidate (second terminal)
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5001   # http://127.0.0.1:5001

# 5. Review, then promote the selected version to @production
python -m sepsis_warning promote --version 1 --reviewed-by "Your Name"

# 6. Serve the production model (checks its SHA-256 first); leave it running
python -m sepsis_warning serve --model-uri models:/sepsis-warning@production
```

In another terminal (with `source .venv/bin/activate`):

```bash
# 7. Call the prediction API
curl -s localhost:8000/health
curl -s localhost:8000/model                      # selected model, why, all candidates (validation only)
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118, "Temp": 38.9}]}'
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118, "Temp": 38.9}]},
                 {"patient_id": "bed-2", "hours": [{"ICULOS": 1, "HR": 75}]}]}'
open http://127.0.0.1:8000/docs                   # interactive API docs
python scripts/latency.py --url http://127.0.0.1:8000 --output artifacts/latency.json
python -m sepsis_warning ward --api-url http://127.0.0.1:8000 --data data/synthetic \
  --output artifacts/ward/ward-api.html && open artifacts/ward/ward-api.html
```

Stop the server (Ctrl+C), then run the production-like stack: a load balancer,
two Fargate-sized tasks on a private network and the `@production` model
([docs/local-production.md](docs/local-production.md)).

```bash
# 8. Start it and run the production checks
bash scripts/local_prod.sh up                     # http://127.0.0.1:8080
bash scripts/local_prod.sh test

# 9. Release a second version with a rolling update, then roll back
python -m sepsis_warning train --data data/synthetic --output artifacts/run-002 --seed 43 \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model
python -m sepsis_warning promote --version 2 --reviewed-by "Your Name"
bash scripts/local_prod.sh deploy models:/sepsis-warning@production
bash scripts/local_prod.sh deploy models:/sepsis-warning/1
bash scripts/local_prod.sh down
```

Notes:

- On an Apple Silicon Mac, `PLATFORM=linux/arm64 bash scripts/local_prod.sh up`
  is much faster; the default `linux/amd64` matches Fargate but runs emulated.
- `demo` needs an empty data folder; to train again on the same synthetic data
  use `train --data data/synthetic` with a new `--output`, as in step 9.
- To serve straight after training without the registry:
  `serve --model artifacts/demo/model.joblib` (`/model` still shows the selection).
- Start over: `rm -rf artifacts data/synthetic`.
- With real data, replace step 3 with the `train` command in
  [Run tracking and the research API](#run-tracking-and-the-research-api-l01-l02).
  Keep real data and models out of Git.

## Use real PhysioNet data

Source: [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).
Two public training cohorts contain 20,336 and 20,000 patients. Use the current
[training directory](https://physionet.org/files/challenge-2019/1.0.0/training/)
rather than the older archive link on the challenge page, which may be broken.
Follow the dataset's CC BY 4.0 terms and attribution requirements.

```text
data/physionet/
  training_setA/
    p000001.psv
    ...
  training_setB/
    p100001.psv
    ...
```

Full steps, from a clean checkout to the SIRS results:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_data.py --limit 0 --output data/physionet
python -m sepsis_warning prepare --data data/physionet --output data/physionet/combined.parquet
python -m sepsis_warning split --data data/physionet/combined.parquet --output data/physionet/splits.csv
python -m sepsis_warning baseline --data data/physionet/combined.parquet --output artifacts/baseline
```

To train the models, which read the `.psv` folder:

```bash
python -m sepsis_warning train --data data/physionet --output artifacts/real
```

Check `split` against the stay counts in [Patient split](#patient-split). For the
notebook, also run `python -m pip install -e ".[notebook]"`.

Hardware used for the reported results, and how long each step took:

| Item | Value |
|---|---|
| Machine | Windows 11 laptop, Intel Core i9-14900HX (24 cores), 32 GB RAM |
| GPU | Present but not used; no code here needs one |
| Disk | 323 MB raw files on this machine, 19 MB Parquet |
| `prepare` | About 4.5 minutes |
| `split` | About 30 seconds |
| `baseline` | About 4 to 5.5 minutes |
| `sirs-failures` | About 1.5 minutes |
| Download and full `train` | Not timed |

Seeds: the split and the models use `--seed 42`.

`--limit 500` downloads a sample instead of all records. A small random subset may
have too few sepsis patients for reliable splitting; increase the sample if the
CLI reports that.
Full feature tables are currently built in memory and may consume several GB.
Start with a subset; add partitioned feature caching before a full-cohort study.
Raw data and generated artifacts are excluded by `.gitignore`.

## SIRS baseline (the score every model must beat)

```bash
python -m sepsis_warning baseline --data data/physionet/combined.parquet --output artifacts/baseline
```

Writes `artifacts/baseline/baseline_metrics.csv`: one row per baseline per
split (`validation`, `internal_test`, `external_test` = hospital B). With a
Parquet `--data`, `baseline` uses the saved `splits.csv` next to it (or
`--splits <path>`). A folder of `.psv` files as `--data` still works: it reads
every record and re-splits with `--seed`, which gives the same stays as `train`,
so model rows can be added to the same table later.

| Baseline | Rule |
|---|---|
| `sirs` | Alert hour when **≥ 2** of: Temp > 38 or < 36 °C; HR > 90; Resp > 20 or PaCO2 < 32 mmHg; WBC > 12 or < 4 ×10⁹/L. |
| `always_no` | Never alerts. Shows the score of doing nothing. |

- **Causal.** Each hour uses the last value seen at or before that hour (forward
  fill only). Future hours never change earlier scores.
- **Missing = normal.** A variable never observed does not count as abnormal.
- Band neutrophils (part of the clinical WBC criterion) are not in the dataset.
- Neither baseline is trained, so they use no training data.
- Alerts go through the same `policy.apply_policy` (6 h cooldown) and
  `policy.summarize` as the models. Precision, recall, F1, AUROC and AP are
  hourly and hand-written in `baselines.py` (no sklearn metrics; tests check
  them against sklearn). For `always_no`, precision is undefined (blank).
- `probability` for SIRS is criteria met ÷ 4, so the ranking metrics (AP, AUROC)
  are meaningful but it is not a calibrated probability.

**What this baseline is and is not.** It is a transparent research comparator
for the benchmark label. It does **not** diagnose sepsis, and it is not full
clinical SIRS:

- Band neutrophils (> 10% bands) are not recorded, so the white-cell criterion
  uses WBC only.
- PaCO2 is only available when a blood gas was taken (91% of A hours and 98% of
  B hours have none), so the breathing criterion mostly relies on Resp.
- Old values are carried forward without a time limit. A WBC measured a day
  ago still counts now.
- Clinical SIRS also assumes a suspected infection. The rule here is applied to
  every ICU hour.

**Results on the full data** (seed 42; `artifacts/baseline/baseline_metrics.csv`):

| Split | Precision | Recall | F1 | Specificity | Accuracy | Utility | False alerts / non-septic day |
|---|---|---|---|---|---|---|---|
| Validation (A) | 0.033 | 0.536 | 0.063 | 0.655 | 0.652 | 0.084 | 1.98 |
| Internal test (A) | 0.033 | 0.529 | 0.063 | 0.661 | 0.658 | 0.087 | 1.97 |
| External test (B) | 0.028 | 0.511 | 0.052 | 0.742 | 0.739 | −0.014 | 1.52 |
| Always-no (validation) | 0 (nothing flagged) | 0 | 0 | 1.000 | **0.978** | 0 | 0 |

Metrics follow the course's confusion-matrix definitions (L9), hand-written in
NumPy: precision TP/(TP+FP), recall TP/(TP+FN), specificity TN/(TN+FP),
F1 2TP/(2TP+FP+FN), accuracy (TP+TN)/all. Accuracy is reported because the
brief asks for it, but it is misleading here: always-no is 97.8% "accurate"
while catching no sepsis at all. When nothing is flagged, precision is 0/0; it
is reported as 0 and marked in `no_predicted_positives`.

**Threshold trade-off (validation only).** The same command also scores SIRS
at ≥ 1, 2, 3 and 4 criteria on the validation split
(`artifacts/baseline/sirs_threshold_sweep_validation.csv`,
`reports/figures/08_sirs_threshold_tradeoff.png`). Raising the threshold gives
fewer false alarms but misses more sepsis (course L4). Only ≥ 4 stays under the
0.2 false-alerts-per-day budget, and it then catches just 3% of septic hours.
This is descriptive: the rule stays at the standard ≥ 2 and the test splits are
not used to look at thresholds.

| Alert at | Precision | Recall | F1 | Specificity | False alerts / day |
|---|---|---|---|---|---|
| ≥ 1 | 0.025 | 0.823 | 0.049 | 0.293 | 3.43 |
| **≥ 2 (rule)** | 0.033 | 0.536 | 0.063 | 0.655 | 1.98 |
| ≥ 3 | 0.044 | 0.220 | 0.074 | 0.895 | 0.70 |
| ≥ 4 | 0.056 | 0.033 | 0.042 | 0.988 | 0.10 |

Group-level patterns of where SIRS fails (validation split only) are in
[docs/sirs-failure-patterns.md](docs/sirs-failure-patterns.md).

`baseline` also redraws `reports/figures/08_sirs_threshold_tradeoff.png`. Pass
`--figures <folder>` to write it somewhere else.

## Patient split

```bash
python -m sepsis_warning split --data data/physionet/combined.parquet --output data/physionet/splits.csv
```

Writes `splits.csv` (`patient_id`, `hospital`, `septic`, `split`) and
`splits.csv.summary.json` (seed, SHA-256 of the Parquet input and of the CSV,
counts). Both stay in the git-ignored `data/` folder.

| Split | Source | ICU stays | Septic stays |
|---|---|---|---|
| train | 55% of A | 11,183 | 984 |
| calibration | 15% of A | 3,051 | 269 |
| validation | 15% of A | 3,051 | 268 |
| internal_test | 15% of A | 3,051 | 269 |
| external_test | all of B | 20,000 | 1,142 |

- Stratified by "ever septic", so each A split has the same 8.8% septic share.
- The split depends only on the seed. Records are sorted by ID before
  splitting, so the load order (folder or Parquet) cannot change it.
- **The unit is an ICU stay, not a person.** The data has no person ID, so one
  person could have stays in two different splits.

Code: [src/sepsis_warning/baselines.py](src/sepsis_warning/baselines.py).
Tests: [tests/test_baselines.py](tests/test_baselines.py).

## Data checks and versioning

These are optional installs, so nothing else depends on them.

```bash
python -m pip install -e ".[checks]"
python -m sepsis_warning validate
```

`validate` checks every stay against the written hard rules in
`src/sepsis_warning/schema.py`: required columns, numeric types, no infinite
values, labels that are 0 or 1 and never drop back to 0, and consecutive hours.
Blank clinical values are allowed. It also counts readings outside broad review
limits on the training stays only. Those limits are this project's own, and they
never reject data. Reports show counts only, and small counts appear as `<10`.

DVC records which data files belong with this code: see
[docs/data-versioning.md](docs/data-versioning.md).

## What is implemented

| Component | First implementation |
|---|---|
| Data checks | Required schema, binary nondecreasing labels, consecutive hours |
| Features | Last observed value, current missingness, never-observed flag, hours since observation, rolling mean/std/count, six-hour change |
| Models | Constant-prior baseline, logistic regression, random forest, histogram gradient boosting |
| Missing values | Forward fill within a patient; training-only median imputation for logistic regression and random forest |
| Calibration | Sigmoid calibration on a separate source-patient cohort |
| Alert selection | Threshold grid, fixed six-hour cooldown, validation alarm budget |
| Evaluation | Average precision, AUROC, Brier score, normalized hourly utility, alert count, early-detection recall, inferred warning lead time |
| Serving sketch | Batch replay from a saved model; labels are not required |
| Monitoring | Two-speed drift checks (fast: schema, missingness, input and score PSI, alert rate; slow: calibration, AUROC, utility, early recall, false alerts, subgroup gaps) with thresholds frozen from hospital A only |
| Reproducibility | Seed, versions, file hashes, patient split manifest, model and predictions |

Source hospital A is split approximately 55% training, 15% calibration,
15% validation and 15% internal testing, stratified at patient level. Hospital B
is reserved for external testing. No patient-hour row split is used. ICU unit
administrative codes and patient/hospital IDs do not enter the model.

Fit preprocessing and model on training patients. Fit calibration on calibration
patients. Choose model and probability threshold on validation patients. Freeze
all choices before evaluating internal-test and hospital-B patients. Do not
adjust the threshold after inspecting B and still call B an untouched test.

The default validation budget is **0.2 false alert events per nonseptic
patient-day**. This is a demonstration choice, not a clinician-approved budget.
The denominator includes only patients whose complete record has no positive
labels. It does not count excessively early alerts in eventual sepsis patients;
the hourly utility separately penalizes those predictions. Exposure is the
observed hourly record length divided by 24, not a calendar-day count. The budget
may be exceeded at the held-out hospital. Report that degradation rather than
retuning on its labels.

Normalized utility follows the default published PhysioNet 2019 equations and
uses **hourly_positive**, not cooldown-suppressed **alert_event**. Cooldown
notifications are a separate operational metric. No positive patients means
the normalized utility is undefined and is reported as null. The prototype's
average precision is sklearn's metric, not a claim of official leaderboard AUPRC.

## Design and presentation

- [Design page](docs/design/sepsis-warning-pipeline.html): data split, training, model selection, alert policy, ICU call contract, AWS deployment and MLOps loop. Open it in a browser.
- [Final presentation](docs/presentation/sepsis-early-warning.pptx): 22 slides with speaker notes. Source, assets and rebuild steps are in [docs/presentation](docs/presentation/README.md).
- [Monitoring plan](docs/monitoring-plan.md) (M02): two-speed checks, reference data, thresholds, owners and runbook.
- [Risk register](docs/risk-register.md) (M05): leakage, data quality, source shift, privacy, fairness, alert fatigue, licensing and operational risks, with evidence and remaining exposure.
- [ICU ward replay](docs/ward-replay.md): plays stays hour by hour as a care team would see them; synthetic version in `docs/design/ward-replay-synthetic.html`.

## Two-speed monitoring (M02)

Freeze a monitoring reference from hospital A, then replay a held-out split
through it. The reference reads only A train (inputs) and A validation scored
by the frozen bundle (scores, alerts, performance); thresholds are the larger
of a fixed floor and the 95th/99th percentile on no-shift A-validation
windows. Nothing retrains automatically.

```bash
python -m sepsis_warning monitor-setup --data data/physionet --model artifacts/real/model.joblib \
  --output artifacts/monitoring/reference
python -m sepsis_warning monitor-replay --data data/physionet --model artifacts/real/model.joblib \
  --reference artifacts/monitoring/reference/monitoring_reference.json \
  --split external_test --output artifacts/monitoring/replay-b
```

Run the replay only after the model, features, calibration and alert policy
are frozen. The replay refuses a bundle whose SHA-256 differs from the
reference and writes aggregates only (`monitoring_report.md`, `summary.json`,
per-window CSVs, `monitoring_overview.png`). Fast (unlabeled) alarms are a
prompt to investigate, not proof of performance loss. The synthetic smoke run
and the meaning of every check are in [docs/monitoring-plan.md](docs/monitoring-plan.md).

## Run tracking and the research API (L01, L02)

Every `train` and `demo` run writes `run_metadata.json`: seed, feature-contract
hash, git commit and package versions. Pass the split summary so the split's
seed and hashes are logged too; a different seed is refused.

```bash
pip install -e ".[tracking]"
python -m sepsis_warning train --data data/physionet/combined.parquet --output artifacts/run-001 \
  --splits-summary data/physionet/splits.csv.summary.json \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model
```

MLflow gets a selection run: the fingerprints and the selected model's
validation metrics, plus one nested run per candidate with its validation
metrics, settings, fit time and training-loss curve. Held-out test results,
predictions, manifests and replay reports are never logged. `--register-model`
also registers the selected bundle as a new version of `sepsis-warning` with the
`@candidate` alias; `promote` moves `@production` after a named review, and
`serve --model-uri models:/sepsis-warning@production` serves it after checking
its SHA-256. Use the SQLite store: MLflow 3 refuses the old `file:` store and the
registry needs a database. Details in [docs/mlflow.md](docs/mlflow.md). Add
`--dvc-lock dvc.lock` to record the data version.

```bash
pip install -e ".[serve]"
python -m sepsis_warning serve --model artifacts/demo/model.joblib   # 127.0.0.1:8000
python -m sepsis_warning serve --model-uri models:/sepsis-warning@production   # or from the registry
curl -s localhost:8000/health
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"patient_id": "demo-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 101, "Temp": 38.4}]}'
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | none | Status and the loaded bundle: model, hash, threshold, cooldown, feature count |
| `GET /model` | none | The selected model, the selection rule and every candidate's validation metrics, the training fingerprints, and the registry version and reviewer when served from the registry. Never held-out test results |
| `POST /score` | One stay's full history | A probability, threshold crossing and alert flag for every hour |
| `POST /predict` | Up to 64 stays (a ward), each with its full history | For each stay, its latest hour: probability, whether it is above the threshold, `new_alert` (true only in the hour an alert fires, so hourly polling notifies once) and the last alert hour |

`/docs` serves interactive documentation. `serve` refuses to start if the run
folder's selection files describe a different model or threshold from the bundle.

Each request sends one stay's full history from its first ICU hour; the server
keeps no state between requests. Missing fields mean "not measured". Labels,
unit codes and hospital IDs are rejected (422). A bundle that does not fit the
current feature code is refused at start-up or with 409. There is no
authentication, so the command refuses non-local hosts unless
`--allow-non-local-host` is passed. Scores are research output, not clinical advice.

To test the AWS layout on a laptop (load balancer, two Fargate-sized tasks on
a private network, model from the registry's `@production` alias, rolling
deploy and rollback), see [docs/local-production.md](docs/local-production.md):

```bash
bash scripts/local_prod.sh up && bash scripts/local_prod.sh test
```

## Files and artifacts

```text
src/sepsis_warning/
  data.py             # patient schema, splits, synthetic fixture
  features.py         # causal feature construction
  modeling.py         # classical models and calibration
  policy.py           # thresholds, notifications, metrics
  cli.py              # training and replay commands
  report.py           # offline report generation
  report_template.html
  scoring.py          # score whole cohorts with a saved bundle (shared scorer)
  monitoring.py       # two-speed drift monitoring (M02)
scripts/download_data.py
tests/test_invariants.py
tests/test_monitoring.py
```

The experiment creates:

- `validation_models.csv`: model comparison using validation patients only.
- `metrics.json`: validation results and selected model's held-out results.
- `split_manifest.csv`: patient-to-split mapping and raw record SHA256 hashes.
- `config.json`: feature names, seed, package versions, and operating settings.
- `model.joblib`: saved calibrated classifier, features, threshold, cooldown.
- `held_out_predictions.csv`: hourly outcomes, scores, decisions and notifications.
- `replay.html`: self-contained interactive replay of held-out predictions.

Batch inference from a saved local model:

```bash
python -m sepsis_warning replay \
  --model artifacts/demo/model.joblib \
  --patient data/synthetic/training_setB/synthetic_0000.psv \
  --output artifacts/patient_replay.csv
```

This computes patient-local feature histories in batch. The prefix-invariance
tests check that scoring an earlier hour does not change when later measurements
are appended. The implementation is not yet an incremental streaming service.

## Work remaining for the course project

1. Run full real-data experiments, including A-to-B and a separately prespecified
   B-to-A study. Treat dataset cohorts as the public hospital sources, not as
   identifiable hospitals.
2. Perform patient-grouped hyperparameter tuning within source training data.
   Current estimator settings are starting values, not a completed search.
3. Add patient-bootstrap confidence intervals, calibration plots, precision-recall
   plots, and a budget-versus-early-recall curve.
4. Audit performance by age and recorded gender, with subgroup sample sizes.
   Compare missingness features on/off to test sensitivity to care practices.
5. Examine false alarms, missed cases, very early alarms, and measurement outages.
   Document censored onset assumptions and post-onset rows in the challenge task.
6. Add feature caching for the complete dataset and a lightweight API if needed.
   Monitoring of missingness, calibration, cohort shift and alert volume is
   implemented (M02, `monitor-setup`/`monitor-replay`); service latency (L05) remains.

Suggested team split: data/splits; features/models; calibration/alert policy;
evaluation/bias audit; replay/reproducibility/report integration. Each person
should also review another person's component.

## Sources

- Reyna et al., [Early Prediction of Sepsis, PhysioNet 2019](https://physionet.org/content/challenge-2019/1.0.0/).
- [Official challenge scoring implementation](https://github.com/physionetchallenges/evaluation-2019).
- [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
