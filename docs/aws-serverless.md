# Legacy design: SageMaker training, SageMaker MLflow and Lambda scoring

> **Not the design that is deployed.** This page describes the first, SageMaker-based design, which is still in the templates
> behind `TrainingBackend=sagemaker` and `EnableSageMaker`. All environments use ECS Fargate for training and MLflow and
> Evidently on ECS. For the current design read [RUN-ORDER.md](RUN-ORDER.md), [ecs-runbook.md](ecs-runbook.md) and
> [ecs-training.md](ecs-training.md). The Lambda scoring part below is still accurate.

The default AWS design runs only when it is used. Nothing in the default stack
bills by the hour: no load balancer, NAT gateway, VPC endpoint, notebook,
tracking server or idle container.

| Step | Service | How it is serverless |
|---|---|---|
| Experiment tracking and model registry | **SageMaker AI MLflow App** (`AWS::SageMaker::MlflowApp`) | Serverless MLflow at no additional charge; you pay for its S3 storage. Replaces the hourly MLflow tracking server |
| Training and selection | **SageMaker AI training job** started by `scripts/sagemaker_train.py` | One instance for the length of the job, on **managed spot** capacity by default, with a hard maximum runtime |
| Scoring API | **AWS Lambda** container image with a **function URL** | Scales to zero; billed per request and millisecond; optional reserved concurrency caps parallel use |
| Storage | S3 (KMS), ECR, DynamoDB on demand | Pay per GB or per request; lifecycle rules expire old versions, images and training outputs |
| Alerts | CloudWatch alarms, SNS email, AWS Budget | Budget alert exists from the first deploy |

The always-on private design from the deck (internal load balancer and Fargate
in private subnets behind a hospital VPN) is still available as
`ServingMode=fargate`.

**Status:** written, statically checked (cfn-lint, botocore request
validation, unit tests, local container runs of the Lambda and training
images) and checked against the AWS documentation (list below). Not yet
deployed to an AWS account. Deploying needs the L06 owner's
approval of the account, region, budget and data plan, and real patient data
must not be uploaded until licence, privacy and residency are confirmed.

## How it fits together

```text
laptop / CI                              AWS account (ap-southeast-1)
-----------                              ----------------------------
docker build --target training  ──push──▶ ECR  <env>/training
docker build --target lambda    ──push──▶ ECR  <env>/scoring-lambda
data/physionet/{combined.parquet,         S3   <bucket>/data/physionet/   (KMS)
  splits.csv, summary}          ──copy──▶
scripts/sagemaker_train.py      ──start─▶ SageMaker training job (spot, 1 instance, max 1 h)
                                            reads data/, runs `train`, writes training-output/
                                            logs to + registers in ──────▶ MLflow App  (@candidate)
promote --reviewed-by           ────────────────────────────────────────▶ MLflow App  (@production)
publish                         ──copy──▶ S3   <bucket>/models/vN/model.joblib (+ validation summary)
stack update (ModelS3Uri, ModelSha256) ─▶ Lambda function (image + pinned model)
hospital system / call_scoring_url.py ── SigV4-signed HTTPS ──▶ function URL ─▶ /health /model /score /predict
```

The Lambda function and the Fargate task load the bundle from S3 and refuse it
unless its SHA-256 matches the pinned `ModelSha256`, so a changed object is
never loaded.

## Commands

Prerequisites: AWS CLI v2 with credentials for the account, Docker, and
`pip install -e ".[aws,tracking,sagemaker]"`. Attach the stack's
`TrainerPolicyArn`, `ApproverPolicyArn` and `DeployerPolicyArn` outputs to different
people (or roles), so the person who trains is not the one who approves.

```bash
export AWS_REGION=ap-southeast-1 STACK=sepsis-research
export BUDGET_USD=20 BUDGET_EMAIL=you@example.com        # monthly budget alert

bash scripts/aws_deploy.sh stack          # S3, KMS, ECR, MLflow App, roles, alarms, budget
bash scripts/aws_deploy.sh images         # build + push lambda, training and scoring images (tag = git sha)
bash scripts/aws_deploy.sh data           # upload combined.parquet, splits.csv and its summary (after approval)
bash scripts/aws_deploy.sh train          # spot training job; registers the selected model as @candidate
bash scripts/aws_deploy.sh mlflow-ui      # short-lived link to the MLflow UI: compare runs, review the candidate
bash scripts/aws_deploy.sh promote 1 "Reviewer Name"
bash scripts/aws_deploy.sh publish        # copies @production to S3, records the hash
bash scripts/aws_deploy.sh serve          # deploys the Lambda scoring API with that model
bash scripts/aws_deploy.sh call GET /model
bash scripts/aws_deploy.sh call POST /predict \
  '{"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118}]}]}'
```

To release a new model: `train`, review in the MLflow UI, `promote`, `publish`,
`serve`. To roll back: `promote` the previous version, then `publish` and
`serve` again. `train --on-demand` skips spot capacity when a result is needed
quickly. `SERVING_MODE=fargate` deploys the always-on private design instead.

The same commands work without AWS for local runs: `train`/`demo` with
`--mlflow-uri sqlite:///artifacts/mlflow.db`, and the MLflow App ARN in place
of the SQLite URI once the stack exists.

## Pipeline, releases, environments and monitoring

Added in phases 3 and 4 of [mlops-audit.md](mlops-audit.md). Like the rest, these
are linted and unit-tested but **not yet run in an AWS account**.

| Piece | Template or file | What it does |
|---|---|---|
| Training pipeline | `pipeline.yaml` (Step Functions) | Train on spot with the data checks and promotion gates (a failing model is not registered) → email and wait for an approver → the one frozen evaluation → write-once publish → email. `aws_deploy.sh pipeline`, then `promote` and `approve <token>` |
| Promotion gates | `gates.py`, `promote` | Beats the no-model reference with a positive interval, beats SIRS at the same alert budget, within the alert budget, calibrated, and (with `--require-real-data`) not synthetic. The approver's AWS identity is recorded and must differ from the registrant's |
| Three duties | `mlops.yaml` | `TrainerPolicyArn`, `ApproverPolicyArn`, `DeployerPolicyArn` replace the single operator policy |
| Write-once models | `artifacts.yaml`, `publish` | `models/` accepts only conditional writes and no deletes; a different file under an existing version is refused. Access logs go to a log bucket |
| Safe releases | `serverless.yaml` (SAM) | Each change publishes a Lambda version behind the `live` alias and shifts traffic as a canary (`DeploymentType`); error or latency alarms roll it back |
| Alert ledger | `alert_store.py`, DynamoDB table | Delivered alerts are recorded, so a model change cannot re-alert inside the cooldown |
| Environments | `infra/params/{dev,staging,prod}.json` | `ENVIRONMENT=dev bash scripts/aws_deploy.sh stack`; one AWS account each; prod needs `CONFIRM_PROD=yes` or the workflow's reviewers |
| Deploy workflow | `.github/workflows/deploy.yml` | Manual start: choose the environment (dev, staging, prod) and the region (Singapore or Sydney). GitHub OIDC (no stored keys), a smoke test after the deploy. Inactive until the bootstrap has set `AWS_ACCOUNT_ID` for the environment |
| Telemetry | `telemetry.py` | Request metrics in the `SepsisWarning/Scoring` namespace and prediction lines with hashed stay IDs and no measurement values |
| Prediction log and drift | `observability.yaml`, `live_drift.py` | Prediction lines go to S3 through a log subscription and Firehose; a daily Lambda runs the label-free checks; the `<env>-drift` alarm emails the owners. `aws_deploy.sh reference <file>` switches it on |
| Retraining trigger | `observability.yaml` | With `RetrainImageTag`, a drift alarm starts the pipeline, which still stops at the approver |
| Private entry | `apigw.yaml` | With `PrivateApiVpcEndpointId`: private API Gateway (one VPC endpoint only), IAM auth, throttling, WAF rate limit |
| Runbooks | [runbooks/](runbooks/README.md) | Rollback, scoring errors, drift alarm, proposed service objectives |

### Deploy workflow setup (one time per account and environment)

The workflow signs in to AWS with GitHub's OIDC token, and the IAM role must
trust the exact token subject. Because the job runs in a GitHub environment, the
subject is `repo:<owner>/<name>:environment:<env>` (for example
`repo:leothelucifer/SUTD-PML-team2:environment:dev`), not `...:ref:refs/heads/main`.
A role made with the console wizard usually trusts the branch form or a different
repository (a fork has its own owner), and the run fails with
`Not authorized to perform sts:AssumeRoleWithWebIdentity`.

With admin credentials for the target account:

```bash
ENVIRONMENT=dev bash scripts/aws_deploy.sh github-oidc <owner>/<name>
```

This deploys `infra/cloudformation/github-oidc.yaml` (the OIDC provider if the
account has none, and the role `github-deploy-<env>`) and prints the role ARN.
In GitHub, under Settings → Environments → `<env>`, set the variable
`AWS_DEPLOY_ROLE_ARN` to it. Repeat per environment. The workflow's
"Show the OIDC subject" step prints the subject it sends, to compare with the
role's trust policy.

Extra cost of these: Firehose and the drift Lambda are pennies at this volume;
Step Functions bills per state transition (a few per run); the private API
needs an execute-api VPC endpoint (about $15 a month for two zones) and WAF
(about $6 a month), so it is off unless asked for.

## Rough monthly cost

Estimates for `ap-southeast-1`, light research use (one ward polled hourly, a
few training runs a month). Check them in the AWS Pricing Calculator before
deploying; prices change and these are not quotes.

| Item | Serverless default | Notes |
|---|---|---|
| Lambda scoring | about $0 | ~720 requests and ~720 GB-seconds a month fit in Lambda's always-free monthly allowance |
| MLflow App | $0 + S3 storage | No hourly charge (a Small tracking server would cost roughly $430 a month if left on) |
| Training jobs | cents per run | `ml.m5.xlarge` costs about $0.25-0.30 an hour on demand; spot is usually much less, and a run takes minutes |
| S3, ECR | about $1-2 | A few GB of data, models and images; old versions and images expire |
| KMS key | $1 | One customer-managed key |
| CloudWatch alarms and logs | under $1 | 3 alarms; logs kept 30 days |
| **Total** | **roughly $2-5 a month** | plus training runs |
| Fargate mode instead | roughly $90-100 a month before traffic | Load balancer, three interface VPC endpoints in two zones, one task running all the time |

Costs that remain even when idle: the KMS key, stored data and images, and
alarms. `aws_deploy.sh down` deletes the stack; the bucket, key, repositories,
table and log groups are retained on purpose and must be emptied and deleted by
hand when the project ends.

## Security notes

- **Authentication.** The function URL uses `AWS_IAM`: every request must be
  SigV4-signed by a principal allowed to call it (`scripts/call_scoring_url.py`
  shows how). An outside principal, such as a hospital integration role, is
  allowed with `AllowedInvokerArn`. The deck's OAuth2 API Gateway and VPN are
  not built; the Fargate mode keeps the network-private option.
- **Network.** In Lambda mode the function is not in a VPC. Requests reach it
  over HTTPS on AWS's public endpoint and are rejected without valid
  signatures; the function reaches S3 with its IAM role.
- **Data.** The artifact bucket is private, versioned, KMS-encrypted and denies
  non-TLS access. Training outputs (`model.tar.gz`) contain the full run,
  including patient-level validation predictions (no test-split results), so they stay in this bucket
  and expire after `TrainingOutputRetentionDays`. The approval step's frozen evaluation writes its
  patient-level test predictions and replay page to a scratch folder that is deleted when the step
  ends: only `evaluation_metrics.json` and `publish.json` go into that job's `model.tar.gz`. Nothing
  patient-level is logged to MLflow, and `/model` serves validation results only.
- **Least privilege.** The training role reads `data/`, writes
  `training-output/` and calls only this MLflow App. The Lambda role reads only
  `models/`. The MLflow App's role reaches only `mlflow/`.

## Checked against the AWS documentation (2026-11-15)

| Assumption | What the docs say | Result |
|---|---|---|
| `AWS::SageMaker::MlflowApp` properties | `ArtifactStoreUri`, `Name`, `RoleArn` required; `ModelRegistrationMode`, `Tags`, `WeeklyMaintenanceWindowStart` (`Sun:18:00` pattern) optional; `Fn::GetAtt Arn` | Template matches |
| MLflow Apps in Singapore and Sydney | Asia Pacific (Singapore) and Asia Pacific (Sydney) are both on the MLflow Apps region list; the artifact bucket must be in the same region | OK; both regions are supported, set `AWS_REGION` to `ap-southeast-1` or `ap-southeast-2` |
| MLflow version | Apps run MLflow 3.16 (Python 3.10+) and update themselves | Training image pins `mlflow==3.16.1` (build arg `MLFLOW_VERSION`) |
| App role | Trust `sagemaker.amazonaws.com`; permissions `s3:Get*`, `s3:Put*`, `s3:List*` (+ Model Registry actions only for automatic registration) | Role now matches, scoped to the bucket and `mlflow/` |
| Client permission | The data-plane action for MLflow Apps is `sagemaker:CallMlflowAppApi` | Training role and the trainer, approver and deployer policies grant it |
| Download limit | SageMaker MLflow has a 200 MB download size limit | Bundles are well under 1 MB on synthetic data; check real ones |
| Training container | Started as `docker run image train`; hyperparameters in `/opt/ml/input/config/hyperparameters.json` as strings; channels in `/opt/ml/input/data/<channel>`; `/opt/ml/model` uploaded as one tar file; first 1,024 characters of `/opt/ml/output/failure` become `FailureReason` | `sagemaker_train.py` matches |
| Training role | Documented minimum: CloudWatch metrics, logs, S3 get/put/list on input and output, ECR pull; KMS when keys are set; trust guarded with `aws:SourceArn` on `training-job/*` | Role matches; trust condition now as documented |
| Managed spot | `MaxWaitTimeInSeconds` must be larger than `MaxRuntimeInSeconds`; checkpointing recommended unless jobs are short | Script keeps wait at least 10 minutes above runtime; a 1-hour job without checkpoints restarts from scratch if interrupted |
| Function URL auth | `AWS_IAM` requires SigV4; callers need **both** `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`; another account needs a resource policy as well, with `lambda:InvokedViaFunctionUrl` | Template and deployer policy match |
| Container image | Non-AWS base images work with a runtime interface client; read-only file system except `/tmp`; single-architecture images only; same-account ECR access through the repository policy (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`) | Image and repository policy match; KMS-encrypted ECR needs no extra Lambda permission (ECR decrypts through its own grants) |
| Init phase | Limited to 10 seconds | Model loads on first request, not at init (about 2 s locally) |
| Reserved concurrency | Free; you can reserve only the account's unreserved concurrency minus 100 | Now optional (default 0 = not set): new accounts with a low limit cannot reserve any |

Sources: AWS docs pages for MLflow Apps (setup, IAM, CLI), the CloudFormation
MlflowApp reference, SageMaker training containers and roles, managed spot
training, StoppingCondition, Lambda function URLs (auth and invocation),
Lambda container images and concurrency, and ECR encryption at rest.
