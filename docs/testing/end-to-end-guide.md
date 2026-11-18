# End-to-end guide: real data locally, then AWS

Run Part A first. Do not start Part B until Part A passes. Every step lists what a pass looks like, so a
failure is obvious. Real data, models and real-data ward pages stay in the git-ignored `data/` and `artifacts/`
folders. Never commit them, and never paste the project brief (it contains a cluster password).

Order of decisions, enforced by the code: **tune and train use only train, calibration and validation. The two
test splits (hospital A internal test, hospital B external test) are scored once, by `evaluate`, after the
model is frozen.**

## Part A. Local, real PhysioNet data

### A0. Set up (once)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[serve,test,tracking,checks]"
export MPLBACKEND=Agg
MPLBACKEND=Agg python -m unittest discover -s tests      # about 12 minutes; must end with OK
```

Pass: `OK`. Need about 4 GB of free RAM for the full cohort and 1 GB of disk for the data.

### A1. Get and check the data

```bash
python scripts/download_data.py --limit 0 --seed 42 --output data/physionet     # all 40,336 stays
python -m sepsis_warning prepare  --data data/physionet --output data/physionet/combined.parquet   # ~5 min
python -m sepsis_warning split    --data data/physionet/combined.parquet --output data/physionet/splits.csv
python -m sepsis_warning validate --data data/physionet/combined.parquet \
  --splits data/physionet/splits.csv --output artifacts/data-checks
```

Pass: 20,336 stays in set A and 20,000 in set B; the split has train about 11,183, calibration and validation
and internal test about 3,051 each (README, "Patient split"); `validate` reports no failed check.
First time, add `--limit 500 --output data/physionet-sample` to rehearse in minutes.

### A2. Baseline the models must beat

```bash
python -m sepsis_warning baseline --data data/physionet/combined.parquet --output artifacts/baseline \
  --figures reports/figures/baseline
```

Pass: `baseline_metrics.csv` exists with the SIRS rule and the always-no rule. Their utility is the floor.

### A3. Hyperparameter search (grouped cross-validation, logged to MLflow)

```bash
python -m sepsis_warning tune --data data/physionet/combined.parquet --output artifacts/tuning \
  --trials 12 --folds 3 --mlflow-uri sqlite:///artifacts/mlflow.db
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5001      # second terminal
```

Pass: `artifacts/tuning/tuning_results.csv` has the reference row plus the sampled trials for each model;
`tuned_models.toml` exists; in the MLflow UI, experiment `sepsis-warning-tuning` shows a `tune` parent run with one
nested run per configuration. No validation or test metric appears there. Start small with `--trials 3 --folds 3
--max-stays 3000`; the full search takes longer (random forest and boosting dominate).

### A4. Train with the tuned settings and register the result

```bash
python -m sepsis_warning train --data data/physionet/combined.parquet --output artifacts/real \
  --models-config artifacts/tuning/tuned_models.toml --include-numpy-models \
  --splits-summary data/physionet/splits.csv.summary.json \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model --require-gates
```

Pass: it prints the selected model and its threshold, ends with "Test splits were not scored", and registers
`sepsis-warning` version N as `@candidate`. If `--require-gates` refuses, read the failed gate; do not weaken it.

### A5. Metrics table and figures (validation only)

```bash
python -m sepsis_warning model-report --run artifacts/real --output reports/results --tuning artifacts/tuning
```

Pass: `reports/results/metrics_table.md` lists every model plus `sirs_rule` with accuracy, precision, recall, F1,
average precision, AUROC, hourly utility and intervals; the PNGs exist; `FIGURES.md` gives each figure's recreate
command. Check that the selected model beats `sirs_rule` on utility at the same alert budget. Copy the table and
figures into `reports/final/report.md` section 7.

### A6. Review, promote, serve and call the API

```bash
python -m sepsis_warning promote --version N --reviewed-by "Your Name" --require-real-data
python -m sepsis_warning serve --model-uri models:/sepsis-warning@production &
curl -s localhost:8000/health
curl -s localhost:8000/model          # the selected model, why, every candidate (validation only)
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118, "Temp": 38.9}]}]}'
python scripts/latency.py --url http://127.0.0.1:8000 --output artifacts/latency.json
```

Pass: `/health` shows the bundle hash and `synthetic_model: false`; `/predict` returns a probability between 0 and 1;
a body containing `SepsisLabel` returns 422. Stop the server.

### A7. Production-like stack (Docker)

```bash
bash scripts/local_prod.sh up        # load balancer and two tasks on http://127.0.0.1:8080
bash scripts/local_prod.sh test      # about 30 s; every check must pass
bash scripts/local_prod.sh down
```

Pass: the script prints no FAIL: load balancing, 422 on labels, non-root read-only tasks, wrong-hash task
refuses to start, one task down still serves.

### A8. Freeze, then score the test splits once

Team rule: write the freeze decision (model, threshold, features) in the team channel first. After that:

```bash
python -m sepsis_warning evaluate --data data/physionet/combined.parquet \
  --model artifacts/real/model.joblib --output artifacts/evaluation
python -m sepsis_warning model-report --run artifacts/real --output reports/results_test \
  --tuning artifacts/tuning --evaluation artifacts/evaluation
```

Pass: `artifacts/evaluation/evaluation_metrics.json` has internal-test and external-test metrics;
`reports/results_test/test_metrics_table.md` and `fig_test_pr.png` exist. A second `evaluate` on the same model is
refused; that is the point. Report hospital A and hospital B separately.

### A9. Monitoring rehearsal

```bash
python -m sepsis_warning monitor-setup --data data/physionet/combined.parquet \
  --model artifacts/real/model.joblib --output artifacts/monitoring
python -m sepsis_warning monitor-replay --data data/physionet/combined.parquet --model artifacts/real/model.joblib \
  --reference artifacts/monitoring/monitoring_reference.json --split external_test --after-freeze \
  --output artifacts/monitoring-b
```

Pass: replaying hospital B reports the drift you expect from A to B (different prevalence and sampling).

## Part B. AWS

Account and region: use the account you own. Both Singapore (`ap-southeast-1`) and Sydney (`ap-southeast-2`) are
supported; the SageMaker MLflow App is available in both (`docs/aws-serverless.md`). Pick one and use it for every
command below. Everything is serverless and bills only while it
runs or stores data. Never put access keys in the repository; the GitHub route uses OIDC.

### B0. Prerequisites

AWS CLI v2 logged in as an administrator (CloudShell is fine), Docker, and `pip install -e ".[aws,tracking,sagemaker]"`.

```bash
export AWS_REGION=ap-southeast-2 ENVIRONMENT=dev     # or ap-southeast-1 for Singapore BUDGET_EMAIL=you@example.com
aws sts get-caller-identity           # confirm it is the account you intend
```

### B1. Connect GitHub (once per account and region)

```bash
export GITHUB_REPO=<owner>/SUTD-PML-team2
bash scripts/bootstrap_github_oidc.sh
```

Run it once per region you want (Singapore and Sydney each need their own run). Then add required reviewers to the
`staging` and `prod` GitHub environments. From then on you can deploy from GitHub instead of the shell: Actions →
**deploy** → Run workflow → pick the environment and the region. B2 to B5 below can all run that way for the
stack, images, publish and serve steps. See `docs/deploy-to-any-account.md`.

### B2. Create the stack and images

```bash
bash scripts/aws_deploy.sh stack      # S3, KMS, ECR, SageMaker MLflow App, roles, pipeline, alarms, budget
bash scripts/aws_deploy.sh images     # builds and pushes lambda, training and scoring images
bash scripts/aws_deploy.sh outputs
```

Pass: stack status `CREATE_COMPLETE`; `outputs` lists the bucket, the MLflow App ARN and the repositories.

### B3. Rehearse with synthetic data first (cheap, proves the plumbing)

```bash
python -m sepsis_warning demo --data data/synthetic --output artifacts/aws-demo     # makes the synthetic .psv files
python -m sepsis_warning prepare --data data/synthetic --output data/synthetic/combined.parquet
python -m sepsis_warning split   --data data/synthetic/combined.parquet --output data/synthetic/splits.csv
bash scripts/aws_deploy.sh data data/synthetic          # uploads combined.parquet and splits.csv
ALLOW_SYNTHETIC=1 bash scripts/aws_deploy.sh pipeline   # train -> gates -> approval -> evaluate -> publish
bash scripts/aws_deploy.sh approve <token>              # token arrives by email; the approver is not the trainer
```

### B4. Real data and training on SageMaker

Upload only after the team agrees the data use is allowed. Then:

```bash
bash scripts/aws_deploy.sh data data/physionet           # combined.parquet + splits.csv (+ summary) to the bucket
bash scripts/aws_deploy.sh train                         # SageMaker spot training job; registers @candidate
bash scripts/aws_deploy.sh mlflow-ui                     # short-lived link to the MLflow App
```

Pass: the training job reaches Completed; the MLflow App shows the selection run and registered version.
Run `model-report` on a downloaded copy of the run folder (from `s3://<bucket>/training-output/<job>/output/model.tar.gz`)
to produce the same table and figures as Part A. The production route is the Step Functions pipeline
(`bash scripts/aws_deploy.sh pipeline`, then `approve`), which also evaluates and publishes; the manual steps below
are for a first look. Promote only after a reviewer other than the trainer checks it:

```bash
bash scripts/aws_deploy.sh promote <version> "<reviewer name>"
bash scripts/aws_deploy.sh publish                       # write-once copy to s3://<bucket>/models/vN/
bash scripts/aws_deploy.sh serve                         # Lambda behind an IAM-protected function URL
```

### B5. Test it in AWS

```bash
bash scripts/aws_deploy.sh call GET /health      # bundle hash must equal the published model's hash
bash scripts/aws_deploy.sh call GET /model
bash scripts/aws_deploy.sh call POST /predict '{"stays":[{"patient_id":"bed-1","hours":[{"ICULOS":1,"HR":92}]}]}'
API=$(aws cloudformation describe-stacks --stack-name "sepsis-$ENVIRONMENT" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue | [0]" --output text)
python -m sepsis_warning ward --api-url "$API" --sign --data data/physionet/combined.parquet --split validation --output artifacts/ward/ward-aws.html
```

Pass: health shows the expected hash and `synthetic_model: false`; unsigned requests get 403; a body with a label gets
422; `/predict` matches the local result for the same stay (same probability to rounding). Check the CloudWatch
dashboard and the alarms in the console, and that the monthly budget alert exists.

### B6. Monitoring on AWS

```bash
bash scripts/aws_deploy.sh reference artifacts/monitoring/monitoring_reference.json   # switches on the daily drift job
```

### B7. Tear down

From GitHub: Actions → **teardown** → pick the environment and region, scope "plan only" first, then "everything"
with the stack name typed in the confirm box. From the shell:

```bash
bash scripts/aws_deploy.sh destroy                              # plan only: lists what would go
CONFIRM_DESTROY=sepsis-$ENVIRONMENT bash scripts/aws_deploy.sh destroy    # deletes the stack and the retained resources
bash scripts/aws_deploy.sh down                                 # alternative: stack only, keeps data, keys and images
```

Pass: the plan lists the bucket, repositories, table, log groups and KMS key; after "everything" the stack is gone,
the buckets and repositories no longer exist, and the KMS key shows "pending deletion" (7 days). Check the console
for anything left, and the Budgets page for the next day's cost.

## When something fails

| Symptom | First thing to check |
|---|---|
| `tune`/`train` "not empty" | Use a new `--output` folder; the CLI never overwrites |
| `train` refuses on gates | Read the gate that failed in `selection_metrics.json`; fix the cause |
| `evaluate` refuses | The model was already evaluated, or the split does not match the one it was trained on |
| `model-report` "is missing" | Re-run `train` with this branch's code; older runs lack `validation_scores.csv` |
| AWS `AccessDenied` | Wrong profile or region; run `aws sts get-caller-identity` |
| Role trust error in GitHub | `GITHUB_REPO` must match the repository running the workflow, in the matching environment |
| Budget error | Set `BUDGET_EMAIL` or set `MonthlyBudgetUsd` to 0 in `infra/params/<env>.json` |
